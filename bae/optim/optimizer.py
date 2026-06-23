import copy
from functools import partial
import torch
from pypose.optim import LevenbergMarquardt as ppLM
import pypose as pp

from ..autograd.graph import jacobian
from ..autograd.function import TrackingTensor
from ..sparse.py_ops import diagonal_op_, inv_op
from ..sparse.spgemm import CuSparse
from ..utils.linear_operator import NormalMatVec
from ..utils.parameter import parameter_update_shape

import warp as wp
from warp import sparse
from warp.optim import linear
from bae.sparse.warp_wrappers import format_vec_for_bsr, torchbsr2wp, wp2torchbsr


class LM(ppLM):
    def __init__(self, *args, matrix_free_normal: bool = False, check_unused: bool = False, **kwargs):
        self.matrix_free_normal = matrix_free_normal
        self.check_unused = check_unused
        super(LM, self).__init__(*args, **kwargs)
        self.mm = CuSparse()
        self._solver_template = copy.deepcopy(self.solver) if check_unused else None
        self._jacobian_sparsity_signature = None

    @staticmethod
    def _compare_signature(lhs, rhs):
        if lhs is None or rhs is None:
            return lhs is rhs
        if len(lhs) != len(rhs):
            return False
        return all(
            lshape == rshape and torch.equal(lcrow_indices, rcrow_indices) and torch.equal(lcol_indices, rcol_indices)
            for (lshape, lcrow_indices, lcol_indices), (rshape, rcrow_indices, rcol_indices) in zip(lhs, rhs)
        )

    def _update_signature(self, jacobians):
        signature = [
            (tuple(j.shape), j.crow_indices().clone(), j.col_indices().clone())
            for j in jacobians
        ]
        if self._jacobian_sparsity_signature is not None and not self._compare_signature(
            self._jacobian_sparsity_signature,
            signature,
        ):
            self.mm = CuSparse()
            if self._solver_template is not None:
                self.solver = copy.deepcopy(self._solver_template)
        self._jacobian_sparsity_signature = signature

    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            R = list(self.model(input))
            R = R[0]
            if self.check_unused:
                J = jacobian(R, pg['params'], check_unused=True)
                self._update_signature(J)
            else:
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
        step_flat = step.reshape(-1)
        offset = 0

        for param in params:
            if not param.requires_grad:
                continue

            if self.check_unused:
                active_indices = param.active_indices.to(dtype=torch.long, device=param.device)
                local_shape = (active_indices.numel(),) + tuple(parameter_update_shape(param)[1:])
                chunk_size = torch.Size(local_shape).numel()
                chunk = step_flat[offset:offset + chunk_size]
                offset += chunk_size
                if active_indices.numel() == 0:
                    continue
                step_view = chunk.view(local_shape)
                if getattr(param, 'trim_SE3_grad', False):
                    param[active_indices, :7] = pp.SE3(param[active_indices, :7]).add_(pp.se3(step_view[..., :6]))
                    if param.shape[-1] > 7:
                        param[active_indices, 7:] += step_view[..., 6:]
                else:
                    active = param[active_indices]
                    param[active_indices] = active.add_(step_view.view(active.shape))
                continue

            chunk_size = torch.Size(parameter_update_shape(param)).numel()
            chunk = step_flat[offset:offset + chunk_size]
            offset += chunk_size
            step_view = chunk.view(parameter_update_shape(param))
            if getattr(param, 'trim_SE3_grad', False):
                param[..., :7] = pp.SE3(param[..., :7]).add_(pp.se3(step_view[..., :6]))
                if param.shape[-1] > 7:
                    param[:, 7:] += step_view[..., 6:]
            else:
                param.add_(step_view)
        if offset != step_flat.numel():
            raise RuntimeError(
                f"Step vector size mismatch: consumed {offset} elements from a step of length "
                f"{step_flat.numel()}."
            )


class Schur(LM):
    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            self.reject_count = 0
            weight = self.weight if weight is None else weight
            R = self.model(input, target)[0]
            if self.check_unused:
                J = jacobian(R, pg['params'], check_unused=True)
                self._update_signature(J)
            else:
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