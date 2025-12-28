from functools import partial
import math
import torch
from pypose.optim import LevenbergMarquardt as ppLM
import pypose as pp

from warp.optim import linear
from bae.sparse.warp_wrappers import format_vec_for_bsr, torchbsr2wp, wp2torchbsr
from ..autograd.graph import backward, construct_sbt
from ..autograd.function import TrackingTensor
from ..sparse.py_ops import diagonal_op_, inv_op
from ..sparse.spgemm import CuSparse
from ..utils.linear_operator import NormalMatVec

def jacobian(output, params):
    assert output.optrace[id(output)][0] == 'map', "The last operation in compute graph being indexing transform is not meaningful"
    backward(output)
    res = []
    for param in params:
        if hasattr(param, 'jactrace'):
            if getattr(param, 'trim_SE3_grad', False):
                if isinstance(param.jactrace, tuple):
                    values = param.jactrace[1]
                elif isinstance(param.jactrace, torch.Tensor) and param.jactrace.layout == torch.sparse_bsr:
                    values = param.jactrace.values()
                else:
                    values = param.jactrace

                if values.shape[-1] == 7:
                    values = values[..., :6]
                else:
                    values = torch.cat([values[..., :6], values[..., 7:]], dim=-1)
                
                if isinstance(param.jactrace, tuple):
                    param.jactrace = (param.jactrace[0], values)
                elif isinstance(param.jactrace, torch.Tensor) and param.jactrace.layout == torch.sparse_bsr:
                    param.jactrace = torch.sparse_bsr_tensor(
                        col_indices=param.jactrace.col_indices(), 
                        crow_indices=param.jactrace.crow_indices(),
                        values=values,
                        size=(param.jactrace.shape[0], param.shape[0] * values.shape[-1]),
                        device=param.device,
                    )
                else:
                    param.jactrace = values
            if type(param.jactrace) is tuple:
                param.jactrace = construct_sbt(param.jactrace[1], param.shape[0], param.jactrace[0], type=torch.sparse_bsr)
            res.append(param.jactrace)
            delattr(param, 'jactrace')
            
    return res



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
            J_list = jacobian(R, pg['params'])
            if isinstance(R, TrackingTensor):
                R = R.tensor()
            J = torch.cat([j.to_sparse_coo() for j in J_list], dim=-1)

            self.last = self.loss = self.loss if hasattr(self, 'loss') else self.model.loss(input, target)
            self.reject_count = 0
            J = J.to_sparse_csr()

            if self.matrix_free_normal:
                diag = NormalMatVec._compute_diag(J).clamp(min=pg['min'], max=pg['max'])
                A = NormalMatVec(J, damping=0.0, diag=diag)
                rhs = -(A._get_Jt() @ R.view(-1, 1))
                diag_scale = 1.0
            else:
                J_T = J.mT.to_sparse_csr()
                rhs = -J_T @ R.view(-1, 1)
                A = self.mm(J_T, J)
                diagonal_op_(A, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))

            while self.last <= self.loss:
                if self.matrix_free_normal:
                    diag_scale *= 1.0 + pg['damping']
                    A.set_damping(diag_scale - 1.0)
                else:
                    diagonal_op_(A, op=partial(torch.mul, other=1+pg['damping']))
                try:
                    D = self.solver(A, rhs)
                    D = D[:, None]
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
                if getattr(param, 'trim_SE3_grad', False):
                    numels.append(math.prod(param.shape[:-1]) * (param.shape[-1] - 1))
                else:
                    numels.append(param.numel())
        steps = step.split(numels)
        for (param, d) in zip(params, steps):
            if param.requires_grad:
                if getattr(param, 'trim_SE3_grad', False):
                    param[..., :7] = pp.SE3(param[..., :7]).add_(pp.se3(d.view(param.shape[0], -1)[..., :6]))
                    if param.shape[-1] > 7:
                        param[:, 7:] += d.view(param.shape[0], -1)[:, 6:]
                else:
                    param.add_(d.view(param.shape))

import warp as wp
from warp import sparse
class Schur(LM):
    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            self.reject_count = 0
            weight = self.weight if weight is None else weight
            R = self.model(input, target)

            R = R[0]
            J = jacobian(R, pg['params'])
            J[0] = J[0]
            J[1] = J[1]

            self.last = self.loss = self.loss if hasattr(self, 'loss') \
                                    else self.model.loss(input, target)
            # torch.cuda.nvtx.range_push("JTJc")
            J0wp = torchbsr2wp(J[0])
            J0twp = sparse.bsr_transposed(J0wp)
            U = sparse.bsr_mm(J0twp, J0wp)
            # torch.cuda.nvtx.range_pop()
            # J0D = J[0].to_dense()
            # UD = U.to_dense()
            # torch.testing.assert_close(UD, J0D.mT @ J0D)
            # del J0D
            # del UD
            # torch.cuda.nvtx.range_push("JTJp")
            J1wp = torchbsr2wp(J[1])
            J1twp = sparse.bsr_transposed(J1wp)
            V = sparse.bsr_mm(J1twp, J1wp)
            # torch.cuda.nvtx.range_pop()
            # J1D = J[1].to_dense()
            # VD = V.to_dense()
            # torch.testing.assert_close(VD, J1D.mT @ J1D)
            # del J1D
            # del VD
            
            # torch.cuda.nvtx.range_push("Clamp")
            Upt = wp2torchbsr(U)
            Vpt = wp2torchbsr(V)
            diagonal_op_(Upt, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
            diagonal_op_(Vpt, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
            # torch.cuda.nvtx.range_pop()

            while self.last <= self.loss:
                damping = pg['damping']
                R = R.reshape(-1)
                
                # torch.cuda.nvtx.range_push("Damp")
                # damp = lambda x: x.pow(2) * damping + x
                damp = partial(torch.mul, other=1+damping)
                diagonal_op_(Upt, op=damp)
                diagonal_op_(Vpt, op=damp)
                # sparse.bsr_set_diag(U, sparse.bsr_get_diag(U) * (1+pg['damping']))
                # sparse.bsr_set_diag(V, sparse.bsr_get_diag(V) * (1+pg['damping']))
                # torch.cuda.nvtx.range_pop()

                # torch.cuda.nvtx.range_push("W")
                W = J0twp @ J1wp
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_push("Ic")
                Rwp = format_vec_for_bsr(R, J0twp.block_shape)
                Ic = sparse.bsr_mv(J0twp, Rwp, alpha=-1.0)
                Ip = sparse.bsr_mv(J1twp, Rwp, alpha=-1.0)
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_push("Inv")
                V_i = torchbsr2wp(inv_op(Vpt))
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_push("WVi")
                WV_i = W @ V_i
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_push("rhs1")
                rhs = sparse.bsr_mv(WV_i, Ip, y=Ic, alpha=-1.0, beta=1.0)
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_push("lhs1")
                Wt = W.transpose()
                lhs = sparse.bsr_axpy(U, WV_i @ Wt, alpha=1.0, beta=-1.0)  # this matrix is NOT symetric
                # torch.cuda.nvtx.range_pop()
                D_c = wp.zeros_like(rhs)
                # torch.cuda.nvtx.range_push("Solve C")
                solver_tol = getattr(self.solver, "tol", None)
                solver_maxiter = getattr(self.solver, "maxiter", 0) or 0
                results = linear.cg(
                    A=lhs,
                    b=rhs,
                    x=D_c,
                    tol=solver_tol,
                    maxiter=solver_maxiter,
                    M=linear.preconditioner(lhs),
                )

                # torch.cuda.nvtx.range_pop()
                
                # torch.cuda.nvtx.range_push("rhs2")
                
                rhs = sparse.bsr_mv(Wt, D_c, alpha=-1.0, beta=1.0, y=Ip)  # rhs = Ip - Wt @ D_c
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_push("solve2")
                lhs = V
                D_p = wp.zeros_like(rhs)
                results = linear.cg(
                    A=lhs,
                    b=rhs,
                    x=D_p,
                    tol=solver_tol,
                    maxiter=solver_maxiter,
                    M=linear.preconditioner(lhs),
                )
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_push("Update")
                D_c = wp.to_torch(D_c).flatten()
                D_p = wp.to_torch(D_p).flatten()
                D = torch.cat([D_c, D_p])
                self.update_parameter(pg['params'], D)
                # torch.cuda.nvtx.range_pop()
                self.loss = self.model.loss(input, target)
                print("Loss:", self.loss, "Last Loss:", self.last, "Reject Count:", self.reject_count, "Damping:", pg['damping'])
                # torch.cuda.nvtx.range_push("Strategy")
                # self.strategy.update(pg, last=self.last, loss=self.loss, J=J, D=D, R=R.view(-1, 1))
                # Pass Warp-format Jacobians as well so strategies can do bsrmv without
                # hitting PyTorch's CUDA BSR matvec limitation for rectangular blocks.
                self.strategy.update(
                    pg,
                    last=self.last,
                    loss=self.loss,
                    J=J,
                    Jwp=[J0wp, J1wp],
                    D=[D_c, D_p],
                    R=R.view(-1, 1),
                )
                # torch.cuda.nvtx.range_pop()
                if self.last < self.loss and self.reject_count < self.reject: # reject step
                    self.update_parameter(params = pg['params'], step = -D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break
        return self.loss
