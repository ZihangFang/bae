from functools import partial
import torch
from pypose.optim import LevenbergMarquardt as ppLM
import pypose as pp
from torch.utils._pytree import tree_flatten, tree_map

from ..autograd.graph import (
    BSRJacobianData,
    jacobian,
    jacobian_components,
)
from ..autograd.function import TrackingTensor
from ..sparse.py_ops import diagonal_op_, inv_op
from ..sparse.spgemm import CuSparse
from ..utils.linear_operator import (
    ComponentBlockDiagonalPreconditioner,
    ComponentDiagonalPreconditioner,
    ComponentJacobianOperator,
    ComponentNormalMatVec,
)
from ..utils.parameter import parameter_update_shape

import warp as wp
from warp import sparse
from warp.optim import linear
from bae.sparse.warp_wrappers import format_vec_for_bsr, torchbsr2wp, wp2torchbsr

def _observation_count(value) -> int | None:
    """Infer the shared leading observation dimension from a pytree."""
    leaves, _ = tree_flatten(value)
    sizes = [
        int(leaf.shape[0])
        for leaf in leaves
        if isinstance(leaf, torch.Tensor) and leaf.ndim > 0
    ]
    return max(sizes) if sizes else None


def _slice_observation_rows(value, start: int, end: int, count: int):
    """Slice row-aligned Tensor leaves while preserving broadcast constants."""
    return tree_map(
        lambda leaf: (
            leaf[start:end]
            if (
                isinstance(leaf, torch.Tensor)
                and leaf.ndim > 0
                and leaf.shape[0] == count
            )
            else leaf
        ),
        value,
    )


def _observation_index_sources(value, count: int):
    """Return integer vectors that may directly back component columns."""
    leaves, _ = tree_flatten(value)
    return tuple(
        leaf
        for leaf in leaves
        if (
            isinstance(leaf, torch.Tensor)
            and leaf.ndim == 1
            and leaf.shape[0] == count
            and leaf.dtype
            in (
                torch.int32,
                torch.int64,
            )
        )
    )


def _plain_tensor(value):
    return value.tensor() if isinstance(value, TrackingTensor) else value


class _ChunkedComponentAccumulator:
    """Preallocate and fill fixed-row-sparsity Jacobian components."""

    def __init__(
        self,
        residual,
        components,
        observation_count: int,
        chunk_rows: int,
        column_sources=(),
    ):
        residual = _plain_tensor(residual)
        if residual.ndim == 0 or residual.shape[0] != chunk_rows:
            raise ValueError(
                "Chunked matrix-free LM requires one leading residual row "
                "per observation."
            )
        self.observation_count = observation_count
        self.residual = residual.new_empty(
            (observation_count, *residual.shape[1:])
        )
        self._layouts = []
        self._component_buffers = []

        for group in components:
            group_layouts = []
            group_buffers = []
            for component in group:
                if component.crow_indices.numel() != chunk_rows + 1:
                    raise ValueError(
                        "A Jacobian component row layout does not match its "
                        "residual chunk."
                    )
                counts = (
                    component.crow_indices[1:]
                    - component.crow_indices[:-1]
                )
                blocks_per_row = (
                    int(counts[0].item()) if counts.numel() else 0
                )
                if not bool(torch.all(counts == blocks_per_row).item()):
                    raise ValueError(
                        "Chunked matrix-free LM currently requires each "
                        "Jacobian component to have a fixed number of blocks "
                        "per observation row. Set evaluation_chunk_size=None "
                        "to use the unchunked path."
                    )
                expected = chunk_rows * blocks_per_row
                if (
                    component.col_indices.numel() != expected
                    or component.values.shape[0] != expected
                ):
                    raise ValueError(
                        "Jacobian component block counts are inconsistent."
                    )

                total_blocks = observation_count * blocks_per_row
                columns = None
                if blocks_per_row == 1:
                    for candidate in column_sources:
                        candidate_chunk = candidate[:chunk_rows]
                        if (
                            candidate_chunk.shape
                            == component.col_indices.shape
                            and candidate_chunk.stride()
                            == component.col_indices.stride()
                            and candidate_chunk.data_ptr()
                            == component.col_indices.data_ptr()
                        ):
                            columns = candidate
                            break
                owns_columns = columns is None
                if owns_columns:
                    columns = component.col_indices.new_empty(
                        total_blocks
                    )
                values = component.values.new_empty(
                    (total_blocks, *component.values.shape[1:])
                )
                group_layouts.append(
                    (
                        blocks_per_row,
                        component.size[1],
                        component.values.shape[-2],
                    )
                )
                group_buffers.append((columns, values, owns_columns))
            self._layouts.append(tuple(group_layouts))
            self._component_buffers.append(tuple(group_buffers))

    def append(self, start: int, end: int, residual, components) -> None:
        residual = _plain_tensor(residual)
        chunk_rows = end - start
        if (
            residual.shape[0] != chunk_rows
            or residual.shape[1:] != self.residual.shape[1:]
        ):
            raise ValueError(
                "Residual shape changed between observation chunks."
            )
        self.residual[start:end].copy_(residual)
        if len(components) != len(self._layouts):
            raise ValueError(
                "Jacobian component groups changed between chunks."
            )

        for group, layouts, buffers in zip(
            components, self._layouts, self._component_buffers
        ):
            if len(group) != len(layouts):
                raise ValueError(
                    "Jacobian component count changed between chunks."
                )
            for component, layout, buffer in zip(group, layouts, buffers):
                blocks_per_row, column_size, row_block_size = layout
                counts = (
                    component.crow_indices[1:]
                    - component.crow_indices[:-1]
                )
                expected_blocks = chunk_rows * blocks_per_row
                if (
                    component.crow_indices.numel() != chunk_rows + 1
                    or component.size[1] != column_size
                    or component.values.shape[-2] != row_block_size
                    or component.col_indices.numel() != expected_blocks
                    or component.values.shape[0] != expected_blocks
                    or not bool(
                        torch.all(counts == blocks_per_row).item()
                    )
                ):
                    raise ValueError(
                        "Jacobian component layout changed between chunks."
                    )
                block_start = start * blocks_per_row
                block_end = end * blocks_per_row
                columns, values, owns_columns = buffer
                if owns_columns:
                    columns[block_start:block_end].copy_(
                        component.col_indices
                    )
                else:
                    expected_columns = columns[start:end]
                    if (
                        expected_columns.shape
                        != component.col_indices.shape
                        or expected_columns.stride()
                        != component.col_indices.stride()
                        or expected_columns.data_ptr()
                        != component.col_indices.data_ptr()
                    ):
                        raise ValueError(
                            "A zero-copy Jacobian column source changed "
                            "between chunks."
                        )
                values[block_start:block_end].copy_(component.values)

    def finish(self):
        groups = []
        for layouts, buffers in zip(
            self._layouts, self._component_buffers
        ):
            group = []
            for layout, buffer in zip(layouts, buffers):
                blocks_per_row, column_size, row_block_size = layout
                columns, values, _ = buffer
                crow = torch.arange(
                    self.observation_count + 1,
                    device=columns.device,
                    dtype=columns.dtype,
                )
                if blocks_per_row != 1:
                    crow.mul_(blocks_per_row)
                group.append(
                    BSRJacobianData(
                        crow,
                        columns,
                        values,
                        (
                            self.observation_count * row_block_size,
                            column_size,
                        ),
                    )
                )
            groups.append(tuple(group))
        return self.residual, tuple(groups)

class LM(ppLM):
    def __init__(
        self,
        *args,
        matrix_free_normal: bool = False,
        evaluation_chunk_size: int | None = 250_000,
        compile_evaluation: bool = False,
        **kwargs,
    ):
        if evaluation_chunk_size is not None and evaluation_chunk_size <= 0:
            raise ValueError("evaluation_chunk_size must be positive or None")
        self.matrix_free_normal = matrix_free_normal
        self.evaluation_chunk_size = evaluation_chunk_size
        self.compile_evaluation = compile_evaluation
        super(LM, self).__init__(*args, **kwargs)
        self.mm = CuSparse()

    def _matrix_free_evaluator(self, params):
        if not self.compile_evaluation:
            def eager(input, target):
                residual = self.model(input, target)[0]
                return residual, jacobian_components(residual, params)

            return eager

        if not hasattr(self, "_compiled_matrix_free_evaluator"):
            from ..utils.pypose_ambient_grad import (
                install_pypose_ambient_grad_monkeypatch,
            )

            install_pypose_ambient_grad_monkeypatch()

            def residual_and_components(input, target):
                residual = self.model(input, target)[0]
                components = jacobian_components(residual, params)
                return residual, components

            first_parameter = params[0]
            backend = (
                "inductor" if first_parameter.is_cuda else "eager"
            )
            self._compiled_matrix_free_evaluator = torch.compile(
                residual_and_components,
                backend=backend,
                fullgraph=True,
            )
        return self._compiled_matrix_free_evaluator

    def _matrix_free_evaluate(self, input, target, params):
        observation_count = _observation_count((input, target))
        chunk_size = self.evaluation_chunk_size
        evaluate = self._matrix_free_evaluator(params)
        if (
            chunk_size is None
            or observation_count is None
            or observation_count <= chunk_size
        ):
            residual, components = evaluate(input, target)
            return _plain_tensor(residual), components

        accumulator = None
        column_sources = _observation_index_sources(
            input, observation_count
        )
        for start in range(0, observation_count, chunk_size):
            end = min(start + chunk_size, observation_count)
            chunk_input = _slice_observation_rows(
                input, start, end, observation_count
            )
            chunk_target = _slice_observation_rows(
                target, start, end, observation_count
            )
            residual, components = evaluate(
                chunk_input, chunk_target
            )
            if accumulator is None:
                accumulator = _ChunkedComponentAccumulator(
                    residual,
                    components,
                    observation_count,
                    end - start,
                    column_sources,
                )
            accumulator.append(start, end, residual, components)

        if accumulator is None:
            raise ValueError(
                "Chunked matrix-free LM received no observation rows."
            )
        return accumulator.finish()

    def _matrix_free_loss(self, input, target):
        observation_count = _observation_count((input, target))
        chunk_size = self.evaluation_chunk_size
        if (
            chunk_size is None
            or observation_count is None
            or observation_count <= chunk_size
        ):
            return self.model.loss(input, target)

        loss = None
        for start in range(0, observation_count, chunk_size):
            end = min(start + chunk_size, observation_count)
            chunk_input = _slice_observation_rows(
                input, start, end, observation_count
            )
            chunk_target = _slice_observation_rows(
                target, start, end, observation_count
            )
            residual = _plain_tensor(
                self.model(chunk_input, chunk_target)[0]
            )
            chunk_loss = self.model.kernel[0](
                residual.square().sum(-1)
            ).sum()
            loss = chunk_loss if loss is None else loss + chunk_loss
        if loss is None:
            raise ValueError(
                "Chunked matrix-free LM received no observation rows."
            )
        return loss

    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            if self.matrix_free_normal:
                active_params = tuple(
                    param for param in pg['params'] if param.requires_grad
                )
                R, components = self._matrix_free_evaluate(
                    input, target, active_params
                )
            else:
                R = list(self.model(input))
                R = R[0]
                J = jacobian(R, pg['params'])
            if isinstance(R, TrackingTensor):
                R = R.tensor()

            self.last = self.loss = (
                self.loss
                if hasattr(self, 'loss')
                else (
                    self.model.kernel[0](
                        R.square().sum(-1)
                    ).sum()
                    if self.matrix_free_normal
                    else self.model.loss(input, target)
                )
            )
            self.reject_count = 0

            if self.matrix_free_normal:
                parameter_sizes = tuple(
                    torch.Size(parameter_update_shape(param)).numel()
                    for param in active_params
                )
                J = ComponentJacobianOperator(
                    components,
                    parameter_sizes,
                    R.numel(),
                )
                # The operator retains value/column views but not BSR row
                # pointers. Release the component wrappers immediately.
                del components
                diag = J.diagonal().clamp(
                    min=pg['min'], max=pg['max']
                )
                A = ComponentNormalMatVec(J, damping=0.0, diag=diag)
                rhs = -J.rmatvec(R.reshape(-1, 1))
                block_diagonal = None
                diag_scale = 1.0
            else:
                J = torch.cat(
                    [j.to_sparse_coo() for j in J], dim=-1
                ).to_sparse_csr()
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
                    if self.matrix_free_normal:
                        # Very lightly damped BA blocks can be numerically
                        # singular even when the full normal operator is not.
                        # Keep the preconditioner positive definite in that
                        # regime; once damping is material, use block Jacobi.
                        if float(A.damping) < 1e-4:
                            preconditioner = (
                                ComponentDiagonalPreconditioner(
                                    A.diagonal()
                                )
                            )
                        else:
                            if block_diagonal is None:
                                block_diagonal = J.block_diagonal()
                            preconditioner = ComponentBlockDiagonalPreconditioner(
                                block_diagonal,
                                parameter_sizes,
                                diag,
                                A.damping,
                            )
                        D = self.solver(A, rhs, M=preconditioner)
                    else:
                        D = self.solver(A, rhs)
                except Exception as e:
                    print(e, "\nLinear solver failed. Breaking optimization step...")
                    break
                self.update_parameter(pg['params'], D)
                self.loss = (
                    self._matrix_free_loss(input, target)
                    if self.matrix_free_normal
                    else self.model.loss(input, target)
                )
                print("Loss:", self.loss, "Last Loss:", self.last, "Reject Count:", self.reject_count, "Damping:", pg['damping'])
                self.strategy.update(
                    pg,
                    last=self.last,
                    loss=self.loss,
                    J=J,
                    D=D,
                    R=R.view(-1, 1),
                )
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
            step_iterator = iter(steps)
            for param in params:
                if param.requires_grad:
                    d = next(step_iterator)
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
