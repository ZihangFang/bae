from functools import partial
import torch
import torch.distributed as dist
from pypose.optim import LevenbergMarquardt as ppLM
import pypose as pp
from torch.utils._pytree import tree_flatten, tree_map

from ..autograd.graph import jacobian, jacobian_components
from ..autograd.function import TrackingTensor
from ..sparse.py_ops import diagonal_op_, inv_op
from ..sparse.spgemm import CuSparse
from ..utils.linear_operator import NormalMatVec
from ..utils.parameter import parameter_update_shape

import warp as wp
from warp import sparse
from warp.optim import linear
from bae.sparse.warp_wrappers import format_vec_for_bsr, torchbsr2wp, wp2torchbsr

try:
    from torch.distributed.tensor import DTensor, Shard
except ImportError:
    DTensor = ()
    Shard = ()


def _contains_dtensor(value) -> bool:
    leaves, _ = tree_flatten(value)
    return any(isinstance(leaf, DTensor) for leaf in leaves)


def _localize_dtensors(value):
    return tree_map(
        lambda leaf: leaf.to_local() if isinstance(leaf, DTensor) else leaf,
        value,
    )


def _damped_blocks(blocks, damping, minimum, maximum):
    result = blocks.clone()
    if result.numel():
        diagonal = torch.diagonal(result, dim1=-2, dim2=-1)
        diagonal.copy_(
            diagonal.clamp(min=minimum, max=maximum) * (1.0 + damping)
        )
    return result


class LM(ppLM):
    def __init__(self, *args, matrix_free_normal: bool = False, **kwargs):
        self.matrix_free_normal = matrix_free_normal
        super(LM, self).__init__(*args, **kwargs)
        self.mm = CuSparse()

    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            R = list(self.model(input))
            R = R[0]
            J = jacobian(R, pg['params'])
            if isinstance(R, TrackingTensor):
                R = R.tensor()
            J = torch.cat([j.to_sparse_coo() for j in J], dim=-1).to_sparse_csr()

            self.last = self.loss = self.loss if hasattr(self, 'loss') else self.model.loss(input, target)
            self.reject_count = 0

            if self.matrix_free_normal:
                diag = NormalMatVec._compute_diag(J).clamp(min=pg['min'], max=pg['max'])
                A = NormalMatVec(J, damping=0.0, diag=diag)
                rhs = -(A._get_Jt() @ R.view(-1, 1))
                diag_scale = 1.0
            else:
                J_T = J.mT.to_sparse_csr()
                rhs = -J_T @ R.view(-1, 1)
                A = self.mm(J_T, J)
                del J_T
                diagonal_op_(A, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))

            while self.last <= self.loss:
                if self.matrix_free_normal:
                    diag_scale *= 1.0 + pg['damping']
                    A.set_damping(diag_scale - 1.0)
                else:
                    diagonal_op_(A, op=partial(torch.mul, other=1+pg['damping']))
                try:
                    D = self.solver(A, rhs)
                except Exception as e:
                    print(e, "\nLinear solver failed. Breaking optimization step...")
                    break
                self.update_parameter(pg['params'], D)
                self.loss = self.model.loss(input, target)
                print("Loss:", self.loss, "Last Loss:", self.last, "Reject Count:", self.reject_count, "Damping:", pg['damping'])
                self.strategy.update(pg, last=self.last, loss=self.loss, J=J, D=D, R=R.view(-1, 1))
                if self.last < self.loss and self.reject_count < self.reject:  # reject step
                    self.update_parameter(params=pg['params'], step=-D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break
        return self.loss

    def update_parameter(self, params, step):
        with torch.no_grad():
            numels = []
            for param in params:
                if param.requires_grad:
                    numels.append(torch.Size(parameter_update_shape(param)).numel())
            steps = step.split(numels)
            for (param, d) in zip(params, steps):
                if param.requires_grad:
                    step_view = d.view(parameter_update_shape(param))
                    if getattr(param, 'trim_SE3_grad', False):
                        # Update the base Tensor instead of assigning through a
                        # TrackingTensor/LieTensor view. Mixed-subclass
                        # __setitem__ dispatch is not traceable by Dynamo.
                        param_tensor = torch.Tensor(param)
                        updated_pose = (
                            pp.se3(step_view[..., :6]).Exp()
                            * pp.SE3(param_tensor[..., :7])
                        )
                        param_tensor[..., :7].copy_(updated_pose.tensor())
                        if param.shape[-1] > 7:
                            param_tensor[..., 7:].add_(step_view[..., 6:])
                    else:
                        param.add_(step_view)


class Schur(LM):
    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        parameters = tuple(
            parameter
            for group in self.param_groups
            for parameter in group["params"]
        )
        if _contains_dtensor((parameters, input, target)):
            return self._distributed_step(input, target=target, weight=weight)

        for pg in self.param_groups:
            self.reject_count = 0
            weight = self.weight if weight is None else weight
            R = self.model(input, target)[0]
            J = jacobian(R, pg['params'])

            self.last = self.loss = self.loss if hasattr(self, 'loss') else self.model.loss(input, target)
            J0wp = torchbsr2wp(J[0])
            J1wp = torchbsr2wp(J[1])
            J0twp = sparse.bsr_transposed(J0wp)
            J1twp = sparse.bsr_transposed(J1wp)
            U = sparse.bsr_mm(J0twp, J0wp)
            V = sparse.bsr_mm(J1twp, J1wp)

            if self.matrix_free_normal:
                del J0twp, J1twp
            else:
                W = sparse.bsr_mm(J0twp, J1wp)
                Wt = sparse.bsr_transposed(W)
                del J0twp, J1twp

            Upt = wp2torchbsr(U)
            Vpt = wp2torchbsr(V)
            diagonal_op_(Upt, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
            diagonal_op_(Vpt, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
            R_flat = R.reshape(-1).contiguous()
            Rwp = format_vec_for_bsr(R_flat, (J0wp.block_shape[1], J0wp.block_shape[0]))
            Ic = sparse.bsr_mv(J0wp, Rwp, alpha=-1.0, transpose=True)
            Ip = sparse.bsr_mv(J1wp, Rwp, alpha=-1.0, transpose=True)
            rhs_c = wp.empty_like(Ic)
            rhs_p = wp.empty_like(Ip)
            scratch_pts2 = wp.empty_like(Ip)

            if self.matrix_free_normal:
                scratch_obs = wp.empty_like(Rwp)
                scratch_pts = wp.empty_like(Ip)

            solver_tol = getattr(self.solver, "tol", None)
            solver_maxiter = getattr(self.solver, "maxiter", 0) or 0

            while self.last <= self.loss:
                damp = partial(torch.mul, other=1+pg['damping'])
                diagonal_op_(Upt, op=damp)
                diagonal_op_(Vpt, op=damp)

                V_i = torchbsr2wp(inv_op(Vpt))

                if self.matrix_free_normal:
                    def schur_matvec(x, y, z, alpha, beta, _V_i=V_i):
                        sparse.bsr_mv(J0wp, x, y=scratch_obs, beta=0.0)
                        sparse.bsr_mv(J1wp, scratch_obs, y=scratch_pts, beta=0.0, transpose=True)
                        sparse.bsr_mv(_V_i, scratch_pts, y=scratch_pts2, beta=0.0)
                        sparse.bsr_mv(J1wp, scratch_pts2, y=scratch_obs, beta=0.0)
                        if z.ptr != y.ptr and beta != 0.0:
                            wp.copy(src=y, dest=z)
                        sparse.bsr_mv(J0wp, scratch_obs, y=z, alpha=-alpha, beta=beta, transpose=True)
                        sparse.bsr_mv(U, x, y=z, alpha=alpha, beta=1.0)

                    schur_op = linear.LinearOperator(
                        shape=U.shape, dtype=U.values.dtype, device=U.device,
                        matvec=schur_matvec,
                    )
                    schur_M = linear.preconditioner(U)

                    wp.copy(src=Ic, dest=rhs_c)
                    sparse.bsr_mv(V_i, Ip, y=scratch_pts2, beta=0.0)
                    sparse.bsr_mv(J1wp, scratch_pts2, y=scratch_obs, beta=0.0)
                    sparse.bsr_mv(J0wp, scratch_obs, y=rhs_c, alpha=-1.0, beta=1.0, transpose=True)
                else:
                    WV_i = sparse.bsr_mm(W, V_i)
                    WVi_Wt = sparse.bsr_mm(WV_i, Wt)
                    U_clone_torch = torch.sparse_bsr_tensor(
                        crow_indices=Upt.crow_indices().clone(),
                        col_indices=Upt.col_indices().clone(),
                        values=Upt.values().clone(),
                        size=Upt.shape, device=Upt.device, dtype=Upt.dtype,
                    )
                    schur_op = sparse.bsr_axpy(WVi_Wt, torchbsr2wp(U_clone_torch), alpha=-1.0)
                    schur_M = linear.preconditioner(schur_op)
                    wp.copy(src=Ic, dest=rhs_c)
                    sparse.bsr_mv(V_i, Ip, y=scratch_pts2, beta=0.0)
                    sparse.bsr_mv(W, scratch_pts2, y=rhs_c, alpha=-1.0, beta=1.0)

                D_c = wp.zeros_like(rhs_c)
                linear.cg(
                    A=schur_op,
                    b=rhs_c,
                    x=D_c,
                    tol=solver_tol,
                    maxiter=solver_maxiter,
                    M=schur_M,
                )

                
                wp.copy(src=Ip, dest=rhs_p)
                
                if self.matrix_free_normal:
                    sparse.bsr_mv(J0wp, D_c, y=scratch_obs, beta=0.0)
                    sparse.bsr_mv(J1wp, scratch_obs, y=rhs_p,
                                  alpha=-1.0, beta=1.0, transpose=True)
                else:
                    sparse.bsr_mv(Wt, D_c, y=rhs_p, alpha=-1.0, beta=1.0)

                D_p = wp.zeros_like(rhs_p)
                linear.cg(
                    A=V,
                    b=rhs_p,
                    x=D_p,
                    tol=solver_tol,
                    maxiter=solver_maxiter,
                    M=linear.preconditioner(V),
                )

                D_c_t = wp.to_torch(D_c).flatten()
                D_p_t = wp.to_torch(D_p).flatten()
                D = torch.cat([D_c_t, D_p_t])
                self.update_parameter(pg['params'], D)
                self.loss = self.model.loss(input, target)
                print("Loss:", self.loss, "Last Loss:", self.last, "Reject Count:", self.reject_count, "Damping:", pg['damping'])

                self.strategy.update(
                    pg,
                    last=self.last,
                    loss=self.loss,
                    J=J,
                    Jwp=[J0wp, J1wp],
                    D=[D_c_t, D_p_t],
                    R=R_flat.view(-1, 1),
                )

                if self.last < self.loss and self.reject_count < self.reject:  # reject step
                    self.update_parameter(params=pg['params'], step=-D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break

        return self.loss

    def _validate_distributed_inputs(self, input, target):
        if not self.matrix_free_normal:
            raise ValueError(
                "DTensor Schur execution requires matrix_free_normal=True."
            )
        if not dist.is_initialized():
            raise RuntimeError(
                "DTensor Schur execution requires torch.distributed initialization."
            )
        if len(self.param_groups) != 1:
            raise ValueError(
                "Distributed Schur expects one optimizer group containing the "
                "camera and point parameters."
            )
        params = tuple(self.param_groups[0]["params"])
        if len(params) != 2 or not all(isinstance(param, DTensor) for param in params):
            raise ValueError(
                "Distributed Schur expects exactly two DTensor parameters: "
                "cameras followed by points."
            )

        camera, point = params
        mesh = camera.device_mesh
        if mesh.ndim != 1:
            raise ValueError("Distributed Schur requires a one-dimensional mesh.")
        for name, parameter in (("camera", camera), ("point", point)):
            if parameter.device_mesh != mesh:
                raise ValueError(
                    "Camera and point parameters must use the same DeviceMesh."
                )
            if (
                len(parameter.placements) != 1
                or not isinstance(parameter.placements[0], Shard)
                or parameter.placements[0].dim != 0
            ):
                raise ValueError(
                    f"The {name} parameter must use placements=[Shard(0)]."
                )

        input_leaves, _ = tree_flatten((input, target))
        sharded_inputs = [
            leaf for leaf in input_leaves if isinstance(leaf, DTensor)
        ]
        if not sharded_inputs:
            raise ValueError(
                "Distributed Schur requires row-sharded observation inputs."
            )
        local_rows = None
        for tensor in sharded_inputs:
            if tensor.device_mesh != mesh:
                raise ValueError(
                    "Observation inputs and parameters must use the same DeviceMesh."
                )
            if (
                len(tensor.placements) != 1
                or not isinstance(tensor.placements[0], Shard)
                or tensor.placements[0].dim != 0
            ):
                raise ValueError(
                    "Observation-related DTensors must be row-sharded with Shard(0)."
                )
            if tensor.ndim:
                rows = tensor.to_local().shape[0]
                if local_rows is None:
                    local_rows = rows
                elif rows != local_rows:
                    raise ValueError(
                        "Observation-related DTensors must share one row partition."
                    )
        return params

    def _distributed_evaluate(self, local_input, local_target, params):
        from ..distributed.context import DistributedTraceContext

        if not hasattr(self, "_compiled_distributed_residual"):
            def residual_and_components(local_input, local_target):
                residual = self.model(local_input, local_target)[0]
                components = jacobian_components(residual, params)
                return residual, components

            local_camera = params[0].to_local()
            backend = "inductor" if local_camera.is_cuda else "eager"
            self._compiled_distributed_residual = torch.compile(
                residual_and_components,
                backend=backend,
                fullgraph=True,
            )

        with DistributedTraceContext(params):
            return self._compiled_distributed_residual(
                local_input, local_target
            )

    def _distributed_evaluate_residual(self, local_input, local_target, params):
        """Re-evaluate only the residual for trust-region acceptance."""
        from ..distributed.context import DistributedTraceContext

        if not hasattr(self, "_compiled_distributed_residual_only"):
            def residual_only(local_input, local_target):
                return self.model(local_input, local_target)[0]

            local_camera = params[0].to_local()
            backend = "inductor" if local_camera.is_cuda else "eager"
            self._compiled_distributed_residual_only = torch.compile(
                residual_only,
                backend=backend,
                fullgraph=True,
            )

        with DistributedTraceContext(params):
            return self._compiled_distributed_residual_only(
                local_input, local_target
            )

    @staticmethod
    def _distributed_component_data(residual, components):
        if len(components) != 2:
            raise RuntimeError(
                "Distributed Schur requires camera and point Jacobian components."
            )
        if any(len(group) != 1 for group in components):
            raise RuntimeError(
                "Distributed Schur requires exactly one camera block and one "
                "point block per observation row."
            )

        camera, point = components[0][0], components[1][0]
        observation_count = residual.shape[0]
        for name, component in (("camera", camera), ("point", point)):
            row_counts = component.crow_indices[1:] - component.crow_indices[:-1]
            if (
                component.values.shape[0] != observation_count
                or component.col_indices.numel() != observation_count
                or row_counts.numel() != observation_count
                or not bool(torch.all(row_counts == 1).item())
            ):
                raise RuntimeError(
                    "Distributed Schur requires one "
                    f"{name} Jacobian block per observation row."
                )
        return (
            camera.values.contiguous(),
            camera.col_indices.to(torch.long).contiguous(),
            point.values.contiguous(),
            point.col_indices.to(torch.long).contiguous(),
        )

    @staticmethod
    def _distributed_global_sum(value, process_group):
        result = value.clone()
        if dist.get_world_size(process_group) > 1:
            dist.all_reduce(
                result, op=dist.ReduceOp.SUM, group=process_group
            )
        return result

    @staticmethod
    def _distributed_apply_step(parameter, step):
        from ..distributed.context import parameter_metadata

        local = parameter.to_local()
        if local.numel() == 0:
            return
        metadata = parameter_metadata(parameter)
        if (
            getattr(parameter, "trim_SE3_grad", False)
            or metadata.trim_se3_grad
        ):
            updated_pose = (
                pp.se3(step[..., :6]).Exp()
                * pp.SE3(local[..., :7])
            )
            local[..., :7].copy_(updated_pose.tensor())
            if local.shape[-1] > 7:
                local[..., 7:].add_(step[..., 6:])
        elif metadata.ltype == pp.SE3_type:
            updated = pp.se3(step).Exp() * pp.SE3(local)
            local.copy_(updated.tensor())
        elif metadata.ltype == pp.SO3_type:
            updated = pp.so3(step).Exp() * pp.SO3(local)
            local.copy_(updated.tensor())
        else:
            local.add_(step.view_as(local))

    @staticmethod
    def _distributed_update_strategy(pg, strategy, last, loss, denominator):
        if loss >= last or denominator <= 0:
            quality = -1.0
        else:
            quality = float(((last - loss) / denominator).item())

        pg["radius"] = 1.0 / pg["damping"]
        if quality > pg["high"]:
            pg["radius"] = pg["up"] * pg["radius"]
            pg["down"] = strategy.down
        elif quality <= pg["low"]:
            pg["radius"] = pg["radius"] * pg["down"]
            pg["down"] = pg["down"] * pg["factor"]
        pg["down"] = max(strategy.min, min(pg["down"], strategy.max))
        pg["radius"] = max(pg["min"], min(pg["radius"], pg["max"]))
        pg["damping"] = 1.0 / pg["radius"]

    @torch.no_grad()
    def _distributed_step(self, input, target=None, weight=None):
        from ..distributed.context import parameter_metadata
        from ..distributed.ops import (
            cached_gather_plan,
            ghost_gather,
            owner_reduce_scatter,
        )
        from ..distributed.plan import Ownership
        from ..distributed.schur import (
            DistributedBlockDiagonalOperator,
            DistributedSchurCameraOperator,
            apply_block_matrix,
            inverse_diagonal_blocks,
        )

        camera, point = self._validate_distributed_inputs(input, target)
        if not hasattr(self.solver, "_operator_forward"):
            raise TypeError(
                "Distributed Schur currently requires bae.utils.pysolvers.PCG."
            )
        local_input = _localize_dtensors(input)
        local_target = _localize_dtensors(target)
        pg = self.param_groups[0]
        process_group = camera.device_mesh.get_group()

        residual, components = self._distributed_evaluate(
            local_input, local_target, (camera, point)
        )
        Jc, camera_ids, Jp, point_ids = self._distributed_component_data(
            residual, components
        )
        residual_blocks = residual.reshape(residual.shape[0], -1)

        camera_ownership = Ownership.from_parameter(
            parameter_metadata(camera)
        )
        point_ownership = Ownership.from_parameter(parameter_metadata(point))
        camera_plan = cached_gather_plan(camera)
        point_plan = cached_gather_plan(point)

        U = owner_reduce_scatter(
            torch.einsum("ori,orj->oij", Jc, Jc),
            camera_ids,
            camera_ownership,
        )
        V = owner_reduce_scatter(
            torch.einsum("ori,orj->oij", Jp, Jp),
            point_ids,
            point_ownership,
        )
        Ic = owner_reduce_scatter(
            -torch.einsum("ori,or->oi", Jc, residual_blocks),
            camera_ids,
            camera_ownership,
        )
        Ip = owner_reduce_scatter(
            -torch.einsum("ori,or->oi", Jp, residual_blocks),
            point_ids,
            point_ownership,
        )

        self.reject_count = 0
        self.last = self._distributed_global_sum(
            residual.square().sum(), process_group
        )
        self.loss = self.last

        while self.last <= self.loss:
            U_effective = _damped_blocks(
                U, pg["damping"], pg["min"], pg["max"]
            )
            V_effective = _damped_blocks(
                V, pg["damping"], pg["min"], pg["max"]
            )
            V_inverse = (
                torch.linalg.inv(V_effective)
                if V_effective.numel()
                else V_effective
            )

            point_scaled = apply_block_matrix(V_inverse, Ip)
            point_eval = ghost_gather(point_scaled, point_plan)
            point_obs = point_eval.index_select(
                0, point_plan.observation_positions
            )
            camera_rhs_correction = owner_reduce_scatter(
                torch.einsum(
                    "ori,or->oi",
                    Jc,
                    torch.einsum("ori,oi->or", Jp, point_obs),
                ),
                camera_ids,
                camera_ownership,
            )
            camera_rhs = Ic - camera_rhs_correction

            camera_operator = DistributedSchurCameraOperator(
                camera_jacobians=Jc,
                point_jacobians=Jp,
                camera_global_ids=camera_ids,
                point_global_ids=point_ids,
                owned_u_blocks=U_effective,
                owned_v_inverse_blocks=V_inverse,
                camera_plan=camera_plan,
                point_plan=point_plan,
                camera_ownership=camera_ownership,
                point_ownership=point_ownership,
            )
            camera_preconditioner = DistributedBlockDiagonalOperator(
                inverse_diagonal_blocks(U_effective), process_group
            )
            camera_step = self.solver(
                camera_operator,
                camera_rhs,
                M=camera_preconditioner,
            )

            camera_eval = ghost_gather(camera_step, camera_plan)
            camera_obs = camera_eval.index_select(
                0, camera_plan.observation_positions
            )
            point_rhs_correction = owner_reduce_scatter(
                torch.einsum(
                    "ori,or->oi",
                    Jp,
                    torch.einsum("ori,oi->or", Jc, camera_obs),
                ),
                point_ids,
                point_ownership,
            )
            point_rhs = Ip - point_rhs_correction
            # Point blocks and their vectors are owner-local and independent.
            # Preserve the configured PCG approximation while keeping its
            # inner products local; global reductions cannot couple otherwise
            # independent point owners and only add synchronization latency.
            point_operator = DistributedBlockDiagonalOperator(
                V_effective,
                process_group,
                owner_local_inner=True,
            )
            point_preconditioner = DistributedBlockDiagonalOperator(
                inverse_diagonal_blocks(V_effective),
                process_group,
                owner_local_inner=True,
            )
            point_step = self.solver(
                point_operator,
                point_rhs,
                M=point_preconditioner,
            )

            camera_before = camera.to_local().clone()
            point_before = point.to_local().clone()
            self._distributed_apply_step(camera, camera_step)
            self._distributed_apply_step(point, point_step)

            new_residual = self._distributed_evaluate_residual(
                local_input, local_target, (camera, point)
            )
            self.loss = self._distributed_global_sum(
                new_residual.square().sum(), process_group
            )

            point_step_eval = ghost_gather(point_step, point_plan)
            point_step_obs = point_step_eval.index_select(
                0, point_plan.observation_positions
            )
            linearized_step = torch.einsum(
                "ori,oi->or", Jc, camera_obs
            ) + torch.einsum("ori,oi->or", Jp, point_step_obs)
            denominator = self._distributed_global_sum(
                -torch.sum(
                    linearized_step
                    * (2.0 * residual_blocks + linearized_step)
                ),
                process_group,
            )
            self._distributed_update_strategy(
                pg, self.strategy, self.last, self.loss, denominator
            )

            if self.loss > self.last:
                camera.to_local().copy_(camera_before)
                point.to_local().copy_(point_before)
                self.loss = self.last
                self.reject_count += 1
                if self.reject_count <= self.reject:
                    continue
            break

        return self.loss
