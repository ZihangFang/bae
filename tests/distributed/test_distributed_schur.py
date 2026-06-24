from __future__ import annotations

import multiprocessing as mp
import os
import socket
from contextlib import closing

import pypose as pp
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as tmp
import torch.nn as nn
from pypose.autograd.function import psjac

from bae.distributed import DistributedConfig, DistributedSchur
from bae.optim.strategy import TrustRegion
from bae.utils.pysolvers import PCG


pytestmark = [
    pytest.mark.filterwarnings(r"ignore:CUDA initialization.*:UserWarning"),
    pytest.mark.filterwarnings(r"ignore:Sparse BSR tensor support is in beta state.*:UserWarning"),
]


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@psjac
def _project(points, camera_params):
    projection = pp.SE3(camera_params[..., :7]).Act(points)
    projection = -projection[..., :2] / projection[..., [2]]
    focal = camera_params[..., [-3]]
    k1 = camera_params[..., [-2]]
    k2 = camera_params[..., [-1]]
    radius2 = torch.sum(projection.square(), dim=-1, keepdim=True)
    radial = 1 + k1 * radius2 + k2 * radius2.square()
    return projection * radial * focal


class _Residual(nn.Module):
    def __init__(self, camera_params, points):
        super().__init__()
        self.pose = pp.Parameter(camera_params, sjac=True)
        self.points = pp.Parameter(points, sjac=True)
        self.pose.trim_SE3_grad = True

    def forward(self, observes, cidx, pidx):
        return _project(self.points[pidx], self.pose[cidx]) - observes


def _make_problem(dtype=torch.float64) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    num_cameras, num_points = 4, 6
    true_camera = torch.randn(num_cameras, 10, dtype=dtype)
    true_camera[:, :3].mul_(0.02)
    true_camera[:, 3:6].zero_()
    true_camera[:, 6] = 1.0
    true_camera[:, 7] = 900.0
    true_camera[:, 8:] = 0.0
    true_points = torch.randn(num_points, 3, dtype=dtype)
    true_points[:, 2].add_(6.0)

    cidx = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3], dtype=torch.long)
    pidx = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5], dtype=torch.long)
    with torch.no_grad():
        observes = _project(true_points[pidx], true_camera[cidx])

    return {
        "camera_params": (true_camera + 0.01 * torch.randn_like(true_camera)).contiguous(),
        "points_3d": (true_points + 0.01 * torch.randn_like(true_points)).contiguous(),
        "observes": observes.contiguous(),
        "cidx": cidx,
        "pidx": pidx,
    }


def _initial_loss(problem: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    model = _Residual(problem["camera_params"].to(device), problem["points_3d"].to(device)).to(device)
    residual = model(
        observes=problem["observes"].to(device),
        cidx=problem["cidx"].to(device),
        pidx=problem["pidx"].to(device),
    )
    return residual.square().sum().detach().cpu()


def _distributed_worker(rank: int, world_size: int, port: int, problem: dict[str, torch.Tensor], solve_mode: str, queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    model = _Residual(problem["camera_params"].to(device), problem["points_3d"].to(device)).to(device)
    optimizer = DistributedSchur(
        model,
        {
            "observes": problem["observes"].to(device),
            "cidx": problem["cidx"].to(device),
            "pidx": problem["pidx"].to(device),
        },
        config=DistributedConfig(device=device),
        strategy=TrustRegion(up=2.0, down=0.5**4),
        solver=PCG(tol=1e-6, maxiter=100),
        reject=5,
        solve_mode=solve_mode,
    )
    loss = optimizer.step().detach().cpu()

    gathered_pose = [None for _ in range(world_size)]
    gathered_points = [None for _ in range(world_size)]
    gathered_ghosts = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_pose, optimizer.pose_owned.detach().cpu())
    dist.all_gather_object(gathered_points, optimizer.points_owned.detach().cpu())
    ghost_count = int(optimizer.plan.camera_eval_ids.numel() - optimizer.plan.camera_owned_eval_local_ids.numel())
    ghost_count += int(optimizer.plan.point_eval_ids.numel() - optimizer.plan.point_owned_eval_local_ids.numel())
    dist.all_gather_object(gathered_ghosts, ghost_count)

    if rank == 0:
        queue.put(
            {
                "loss": float(loss.item()),
                "pose": torch.cat(gathered_pose, dim=0).tolist(),
                "points": torch.cat(gathered_points, dim=0).tolist(),
                "ghosts": sum(int(count) for count in gathered_ghosts),
            }
        )
    dist.destroy_process_group()


def _run_distributed(world_size: int, problem: dict[str, torch.Tensor], solve_mode: str) -> dict:
    port = _free_port()
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    tmp.spawn(
        _distributed_worker,
        args=(world_size, port, problem, solve_mode, queue),
        nprocs=world_size,
        join=True,
    )
    result = queue.get()
    return {
        "loss": torch.tensor(result["loss"], dtype=problem["camera_params"].dtype),
        "pose": torch.tensor(result["pose"], dtype=problem["camera_params"].dtype),
        "points": torch.tensor(result["points"], dtype=problem["points_3d"].dtype),
        "ghosts": result["ghosts"],
    }


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="At least two CUDA devices are required.")
@pytest.mark.parametrize("solve_mode", ["cg", "exact"])
def test_distributed_schur_matches_single_rank_and_reduces_loss(solve_mode):
    problem = _make_problem()
    initial_loss = _initial_loss(problem, torch.device("cuda", 0))

    distributed_single = _run_distributed(1, problem, solve_mode)
    distributed_two = _run_distributed(2, problem, solve_mode)

    tolerance = 1e-10 if solve_mode == "exact" else 1e-3
    torch.testing.assert_close(distributed_two["loss"], distributed_single["loss"], rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(distributed_two["pose"], distributed_single["pose"], rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(distributed_two["points"], distributed_single["points"], rtol=tolerance, atol=tolerance)
    assert distributed_single["loss"] < initial_loss
    assert distributed_two["loss"] < initial_loss
    assert distributed_two["ghosts"] > 0
