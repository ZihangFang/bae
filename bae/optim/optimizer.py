import torch
import pypose as pp
from functools import partial
from pypose.optim import LevenbergMarquardt as ppLM
from .triton_kernel import (
    sparse_bsr_mm, sparse_bsr_mv,
    sparse_bsr_transposed, sparse_bsr_axpy,
    BlockJacobi, cg,
)
from ..autograd.graph import jacobian
from ..autograd.function import TrackingTensor
from ..sparse.py_ops import diagonal_op_, inv_op
from ..sparse.spgemm import CuSparse
from ..utils.linear_operator import NormalMatVec
from ..utils.parameter import parameter_update_shape


class LM(ppLM):
    def __init__(self, *args, matrix_free_normal: bool = False, **kwargs):
        self.matrix_free_normal = matrix_free_normal
        super(LM, self).__init__(*args, **kwargs)
        self.mm = CuSparse()

    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            R = self.model(input)[0]
            J_list = jacobian(R, pg['params'])

            if isinstance(R, TrackingTensor):
                R = R.tensor()

            J = torch.cat([j.to_sparse_coo() for j in J_list], dim=-1).to_sparse_csr()
            del J_list

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
        numels = []
        for param in params:
            if param.requires_grad:
                numels.append(torch.Size(parameter_update_shape(param)).numel())
        steps = step.split(numels)
        for (param, d) in zip(params, steps):
            if param.requires_grad:
                step_view = d.view(parameter_update_shape(param))
                if getattr(param, 'trim_SE3_grad', False):
                    param[..., :7] = pp.SE3(param[..., :7]).add_(pp.se3(step_view[..., :6]))
                    if param.shape[-1] > 7:
                        param[:, 7:] += step_view[..., 6:]
                else:
                    param.add_(d.view(param.shape))


class Schur(LM):
    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            self.reject_count = 0
            weight = self.weight if weight is None else weight
            R = self.model(input, target)[0]
            J = jacobian(R, pg['params'])
            
            if isinstance(R, TrackingTensor):
                R = R.tensor()
            else:
                R = R.detach()
            torch.cuda.empty_cache()

            self.last = self.loss = self.loss if hasattr(self, 'loss') else self.model.loss(input, target)

            J0 = J[0]
            J1 = J[1]
            if self.matrix_free_normal:
                J0t = sparse_bsr_transposed(J0)
                U = sparse_bsr_mm(J0t, J0)
                del J0t
                J1t = sparse_bsr_transposed(J1)
                V = sparse_bsr_mm(J1t, J1)
                del J1t
            else:
                J0t = sparse_bsr_transposed(J0)
                J1t = sparse_bsr_transposed(J1)
                U = sparse_bsr_mm(J0t, J0)
                V = sparse_bsr_mm(J1t, J1)
                W = sparse_bsr_mm(J0t, J1)
                Wt = sparse_bsr_transposed(W)
                del J0t, J1t

            diagonal_op_(U, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
            diagonal_op_(V, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
            R_flat = R.reshape(-1).contiguous()
            Ic = sparse_bsr_mv(J0, R_flat, alpha=-1.0, transpose=True)
            Ip = sparse_bsr_mv(J1, R_flat, alpha=-1.0, transpose=True)
            rhs_c = torch.empty_like(Ic)
            rhs_p = torch.empty_like(Ip)
            scratch_pts2 = torch.empty_like(Ip)
            schur_Ap_buf = torch.empty_like(Ic)
            v_Ap_buf = torch.empty_like(Ip)
            D_c = torch.empty_like(Ic)
            D_p = torch.empty_like(Ip)
            cg_r_buf_c = torch.empty_like(Ic)
            cg_p_buf_c = torch.empty_like(Ic)
            cg_r_buf_p = torch.empty_like(Ip)
            cg_p_buf_p = torch.empty_like(Ip)

            if self.matrix_free_normal:
                scratch_obs = torch.empty_like(R_flat)
                scratch_pts = torch.empty_like(Ip)
                z_buf = torch.empty_like(Ic)

            solver_tol = getattr(self.solver, "tol", None) or 1e-5
            solver_maxiter = getattr(self.solver, "maxiter", 0) or 0
            mm_cache_WV_i = {} if not self.matrix_free_normal else None
            mm_cache_WVi_Wt = {} if not self.matrix_free_normal else None
            axpy_cache_schur = {} if not self.matrix_free_normal else None
            v_M = BlockJacobi(V)
            schur_M = BlockJacobi(U) if self.matrix_free_normal else None

            while self.last <= self.loss:
                damp = partial(torch.mul, other=1+pg['damping'])
                diagonal_op_(U, op=damp)
                diagonal_op_(V, op=damp)
                V_i = inv_op(V)

                if self.matrix_free_normal:
                    def schur_matvec(p, _V_i=V_i, _z=schur_Ap_buf):
                        sparse_bsr_mv(J0, p, y=scratch_obs, beta=0.0)
                        sparse_bsr_mv(J1, scratch_obs, y=scratch_pts, beta=0.0, transpose=True)
                        sparse_bsr_mv(_V_i, scratch_pts, y=scratch_pts2, beta=0.0)
                        sparse_bsr_mv(J1, scratch_pts2, y=scratch_obs, beta=0.0)
                        sparse_bsr_mv(J0, scratch_obs, y=_z, alpha=-1.0, beta=0.0, transpose=True)
                        sparse_bsr_mv(U, p, y=_z, alpha=1.0, beta=1.0)
                        return _z

                    matvec_fn = schur_matvec
                    schur_M.refresh(U)
                    rhs_c.copy_(Ic)
                    sparse_bsr_mv(V_i, Ip, y=scratch_pts2, beta=0.0)
                    sparse_bsr_mv(J1, scratch_pts2, y=scratch_obs, beta=0.0)
                    sparse_bsr_mv(J0, scratch_obs, y=rhs_c, alpha=-1.0, beta=1.0, transpose=True)
                else:
                    WV_i = sparse_bsr_mm(W, V_i, topology_cache=mm_cache_WV_i)
                    WVi_Wt = sparse_bsr_mm(WV_i, Wt, topology_cache=mm_cache_WVi_Wt)
                    del WV_i
                    schur_op = sparse_bsr_axpy(WVi_Wt, U, alpha=-1.0,
                                               topology_cache=axpy_cache_schur)
                    del WVi_Wt
                    matvec_fn = lambda p, _S=schur_op, _y=schur_Ap_buf: \
                        sparse_bsr_mv(_S, p, y=_y, beta=0.0)
                    if schur_M is None:
                        schur_M = BlockJacobi(schur_op)
                    else:
                        schur_M.refresh(schur_op)
                    rhs_c.copy_(Ic)
                    sparse_bsr_mv(V_i, Ip, y=scratch_pts2, beta=0.0)
                    sparse_bsr_mv(W, scratch_pts2, y=rhs_c, alpha=-1.0, beta=1.0)

                D_c.zero_()
                cg(matvec_fn, rhs_c, x=D_c, M=schur_M,
                   tol=solver_tol, maxiter=solver_maxiter,
                   r_buf=cg_r_buf_c, p_buf=cg_p_buf_c)

                rhs_p.copy_(Ip)
                if self.matrix_free_normal:
                    sparse_bsr_mv(J0, D_c, y=scratch_obs, beta=0.0)
                    sparse_bsr_mv(J1, scratch_obs, y=rhs_p, alpha=-1.0, beta=1.0, transpose=True)
                else:
                    sparse_bsr_mv(Wt, D_c, y=rhs_p, alpha=-1.0, beta=1.0)

                v_M.refresh(V)
                D_p.zero_()
                cg(lambda p, _V=V, _y=v_Ap_buf: sparse_bsr_mv(_V, p, y=_y, beta=0.0),
                   rhs_p, x=D_p, M=v_M,
                   tol=solver_tol, maxiter=solver_maxiter,
                   r_buf=cg_r_buf_p, p_buf=cg_p_buf_p)

                D_c_t = D_c.flatten()
                D_p_t = D_p.flatten()
                D = torch.cat([D_c_t, D_p_t])
                self.update_parameter(pg['params'], D)
                self.loss = self.model.loss(input, target)
                print("Loss:", self.loss, "Last Loss:", self.last, "Reject Count:", self.reject_count, "Damping:", pg['damping'])

                self.strategy.update(
                    pg,
                    last=self.last,
                    loss=self.loss,
                    J=J,
                    Jwp=[J0, J1],
                    D=[D_c_t, D_p_t],
                    R=R_flat.view(-1, 1),
                )

                if self.last < self.loss and self.reject_count < self.reject:  # reject step
                    self.update_parameter(params=pg['params'], step=-D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break

        return self.loss

