from __future__ import annotations

import os
import tempfile

import pypose as pp
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pypose.autograd.function import psjac
from torch import nn
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, distribute_tensor

from bae.autograd.graph import jacobian_components
from bae.distributed.context import DistributedIndexContext


@psjac
def _factor(
    camera: torch.Tensor,
    point: torch.Tensor,
    observation: torch.Tensor,
) -> torch.Tensor:
    return camera * point - observation


class _SpikeModel(nn.Module):
    def __init__(self, cameras: DTensor, points: DTensor):
        super().__init__()
        self.cameras = pp.Parameter(cameras, sjac=True)
        self.points = pp.Parameter(points, sjac=True)

    def forward(self, observations, camera_indices, point_indices):
        return _factor(
            self.cameras[camera_indices],
            self.points[point_indices],
            observations,
        )


def _spike_worker(
    rank: int,
    world_size: int,
    init_file: str,
    compile_backend: str,
    device_type: str,
) -> None:
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    process_backend = "nccl" if device_type == "cuda" else "gloo"
    dist.init_process_group(
        process_backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
        mesh = DeviceMesh(device_type, torch.arange(world_size))
        cameras_global = (
            torch.arange(15, dtype=torch.float64, device=device).reshape(5, 3) + 1
        )
        points_global = (
            torch.arange(21, dtype=torch.float64, device=device).reshape(7, 3) + 2
        )
        cameras = distribute_tensor(cameras_global, mesh, [Shard(0)])
        points = distribute_tensor(points_global, mesh, [Shard(0)])
        model = _SpikeModel(cameras, points)

        assert isinstance(model.cameras, DTensor)
        assert isinstance(model.points, DTensor)
        assert model.cameras.placements == (Shard(0),)
        assert model.points.placements == (Shard(0),)

        camera_indices = (
            torch.tensor([0, 2, 3, 3], dtype=torch.long, device=device)
            if rank == 0
            else torch.tensor([1, 4, 0, 4], dtype=torch.long, device=device)
        )
        point_indices = (
            torch.tensor([6, 1, 4, 4], dtype=torch.long, device=device)
            if rank == 0
            else torch.tensor([0, 5, 3, 5], dtype=torch.long, device=device)
        )
        observations = torch.randn(
            camera_indices.numel(), 3, dtype=torch.float64, device=device
        )

        def residual_and_components(observations, camera_indices, point_indices):
            residual = model(observations, camera_indices, point_indices)
            components = jacobian_components(
                residual, (model.cameras, model.points)
            )
            return residual, components

        with DistributedIndexContext(
            (model.cameras, model.points)
        ) as index_context:
            compiled = torch.compile(
                residual_and_components,
                backend=compile_backend,
                fullgraph=True,
            )
            residual, components = compiled(
                observations, camera_indices, point_indices
            )
            local_camera_indices = (
                torch.tensor([0, 2], dtype=torch.long, device=device)
                if rank == 0
                else torch.tensor([3, 4], dtype=torch.long, device=device)
            )
            local_point_indices = (
                torch.tensor([1, 3], dtype=torch.long, device=device)
                if rank == 0
                else torch.tensor([4, 6], dtype=torch.long, device=device)
            )
            local_observations = torch.randn(
                2, 3, dtype=torch.float64, device=device
            )
            local_residual, _ = compiled(
                local_observations,
                local_camera_indices,
                local_point_indices,
            )

        expected = (
            cameras_global[camera_indices]
            * points_global[point_indices]
            - observations
        )
        torch.testing.assert_close(residual, expected)
        assert index_context.mode.seen_distributed_index

        camera_component = components[0][0]
        point_component = components[1][0]
        torch.testing.assert_close(
            camera_component.values,
            torch.diag_embed(points_global[point_indices]),
        )
        torch.testing.assert_close(
            point_component.values,
            torch.diag_embed(cameras_global[camera_indices]),
        )
        assert torch.equal(camera_component.col_indices, camera_indices)
        assert torch.equal(point_component.col_indices, point_indices)

        expected_local = (
            cameras_global[local_camera_indices]
            * points_global[local_point_indices]
            - local_observations
        )
        torch.testing.assert_close(local_residual, expected_local)
        from bae.distributed.ops import cached_gather_plan

        camera_plan = cached_gather_plan(model.cameras)
        point_plan = cached_gather_plan(model.points)
        other_rank = 1 - rank
        assert camera_plan.receive_splits[other_rank] == 0
        assert point_plan.receive_splits[other_rank] == 0
    finally:
        dist.destroy_process_group()


def _run_spike(backend: str, device_type: str = "cpu") -> None:
    world_size = 2
    handle, init_file = tempfile.mkstemp(prefix="bae-dtensor-")
    os.close(handle)
    try:
        mp.spawn(
            _spike_worker,
            args=(world_size, init_file, backend, device_type),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def test_dtensor_parameter_dispatch_psjac_fullgraph_eager() -> None:
    _run_spike("eager")


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="two CUDA devices are required"
)
def test_dtensor_parameter_dispatch_psjac_fullgraph_inductor() -> None:
    _run_spike("inductor", "cuda")
