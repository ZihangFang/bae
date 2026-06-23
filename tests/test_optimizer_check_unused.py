import pytest
import pypose as pp
import torch
from torch import nn
from pypose.autograd.function import psjac

from bae.optim import LM
from bae.optim.optimizer import Schur
from bae.optim.strategy import TrustRegion
from bae.utils.pysolvers import PCG


class DummyMM:
    next_instance_id = 0

    def __init__(self):
        self.instance_id = DummyMM.next_instance_id
        DummyMM.next_instance_id += 1

    def __call__(self, J_t: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        return J_t @ J


class DummySolver(nn.Module):
    next_instance_id = 0

    def __init__(self):
        super().__init__()
        self.instance_id = DummySolver.next_instance_id
        DummySolver.next_instance_id += 1

    def forward(self, A, b):
        if hasattr(A, "to_dense"):
            dense = A.to_dense()
        else:
            eye = torch.eye(A.shape[1], device=b.device, dtype=b.dtype)
            dense = A @ eye
        return torch.linalg.solve(dense, b)


class VariablePatternResidual(nn.Module):
    def __init__(self, a0: torch.Tensor, b0: torch.Tensor):
        super().__init__()
        self.A = pp.Parameter(a0, sjac=True)
        self.B = pp.Parameter(b0, sjac=True)

    def forward(self, obs: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self.A[idx] - obs


@psjac
def project(points, camera_params):
    projection = pp.SE3(camera_params[..., :7]).Act(points)
    projection = -projection[..., :2] / projection[..., [2]]
    f = camera_params[..., [-3]]
    k1 = camera_params[..., [-2]]
    k2 = camera_params[..., [-1]]
    n = torch.sum(projection**2, axis=-1, keepdim=True)
    r = 1 + k1 * n + k2 * n**2
    return projection * r * f


class BAResidual(nn.Module):
    def __init__(self, camera_params: torch.Tensor, points: torch.Tensor):
        super().__init__()
        self.pose = pp.Parameter(camera_params, sjac=True)
        self.points = pp.Parameter(points, sjac=True)
        self.pose.trim_SE3_grad = True

    def forward(self, observes: torch.Tensor, cidx: torch.Tensor, pidx: torch.Tensor) -> torch.Tensor:
        return project(self.points[pidx], self.pose[cidx]) - observes


def make_ba_problem(device: torch.device, dtype: torch.dtype):
    camera_se3 = pp.identity_SE3(2, dtype=dtype, device=device).tensor().clone()
    camera_se3[1, 0] = 0.25
    intrinsics = torch.tensor([[200.0, 0.0, 0.0], [180.0, 0.0, 0.0]], dtype=dtype, device=device)
    camera_params = torch.cat([camera_se3, intrinsics], dim=-1)

    points = torch.tensor(
        [[0.0, 0.0, 2.0], [0.3, -0.2, 2.5], [0.2, 0.2, 3.0]],
        dtype=dtype,
        device=device,
    )
    cidx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
    pidx = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device)
    observes = project(points[pidx], camera_params[cidx]).detach()
    observes = observes + torch.tensor(
        [[0.10, -0.05], [-0.08, 0.04], [0.06, -0.03], [-0.05, 0.02]],
        dtype=dtype,
        device=device,
    )
    return camera_params, points, observes, cidx, pidx


def test_lm_check_unused_updates_only_active_blocks_and_resets_caches():
    torch.manual_seed(0)
    dtype = torch.float64

    a0 = torch.tensor([[2.0], [5.0], [7.0]], dtype=dtype)
    b0 = torch.tensor([[11.0], [13.0]], dtype=dtype)
    model = VariablePatternResidual(a0.clone(), b0.clone())

    optimizer = LM(
        model,
        solver=DummySolver(),
        strategy=pp.optim.strategy.TrustRegion(),
        reject=0,
        matrix_free_normal=True,
        check_unused=True,
    )
    optimizer.mm = DummyMM()

    first_input = {
        "obs": torch.tensor([[0.0]], dtype=dtype),
        "idx": torch.tensor([0], dtype=torch.int32),
    }
    second_input = {
        "obs": torch.tensor([[0.0], [0.0]], dtype=dtype),
        "idx": torch.tensor([0, 1], dtype=torch.int32),
    }

    optimizer.step(first_input)

    assert model.A[0].item() != a0[0].item()
    torch.testing.assert_close(model.A[1:], a0[1:])
    torch.testing.assert_close(model.B, b0)

    optimizer.step(second_input)

    assert not isinstance(optimizer.mm, DummyMM)
    assert model.A[1].item() != a0[1].item()
    torch.testing.assert_close(model.B, b0)


def test_lm_check_unused_keeps_unused_point_block_fixed():
    device = torch.device("cpu")
    dtype = torch.float64
    camera_params, points, observes, cidx, pidx = make_ba_problem(device, dtype)
    model = BAResidual(camera_params.clone(), points.clone())

    optimizer = LM(
        model,
        solver=PCG(tol=1e-4, maxiter=100),
        strategy=TrustRegion(up=2.0, down=0.5**4),
        reject=0,
        matrix_free_normal=True,
        check_unused=True,
    )

    optimizer.step({"observes": observes, "cidx": cidx, "pidx": pidx})

    assert not torch.equal(model.points[:2], points[:2])
    torch.testing.assert_close(model.points[2:], points[2:])


def test_schur_check_unused_keeps_unused_point_block_fixed():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Schur check_unused regression")

    dtype = torch.float64
    device = torch.device("cuda")
    camera_params, points, observes, cidx, pidx = make_ba_problem(device, dtype)
    model = BAResidual(camera_params.clone(), points.clone())

    optimizer = Schur(
        model,
        solver=PCG(tol=1e-4, maxiter=100),
        strategy=TrustRegion(up=2.0, down=0.5**4),
        reject=0,
        matrix_free_normal=True,
        check_unused=True,
    )

    optimizer.step({"observes": observes, "cidx": cidx, "pidx": pidx})

    assert not torch.equal(model.points[:2], points[:2])
    torch.testing.assert_close(model.points[2:], points[2:])
