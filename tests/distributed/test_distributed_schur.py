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
from torch.distributed.tensor import DeviceMesh, Shard, distribute_tensor

from bae.distributed.schur import inverse_blocks
from bae.optim.optimizer import Schur
from bae.utils.pysolvers import PCG


@psjac
def _additive_factor(camera, point, observation):
    return camera + point - observation


class _AdditiveBA(nn.Module):
    def __init__(self, cameras, points):
        super().__init__()
        self.cameras = pp.Parameter(cameras, sjac=True)
        self.points = pp.Parameter(points, sjac=True)

    def forward(self, observations, camera_indices, point_indices):
        return _additive_factor(
            self.cameras[camera_indices],
            self.points[point_indices],
            observations,
        )


class _RejectingPCG(PCG):
    def forward(self, A, b, x=None, M=None):
        if hasattr(A, "scalar_inner"):
            return -100.0 * b
        return super().forward(A, b, x=x, M=M)


class _CountingPCG(PCG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operator_calls = 0
        self.owner_local_inner_calls = 0

    def forward(self, A, b, x=None, M=None):
        if hasattr(A, "scalar_inner"):
            self.operator_calls += 1
            self.owner_local_inner_calls += int(
                getattr(A, "owner_local_inner", False)
            )
        return super().forward(A, b, x=x, M=M)


@psjac
def _se3_factor(camera, point, observation):
    return pp.SE3(camera).Act(point) - observation


class _SE3BA(nn.Module):
    def __init__(self, cameras, points):
        super().__init__()
        self.cameras = pp.Parameter(cameras, sjac=True)
        self.points = pp.Parameter(points, sjac=True)
        self.cameras.trim_SE3_grad = True

    def forward(self, observations, camera_indices, point_indices):
        return _se3_factor(
            self.cameras[camera_indices],
            self.points[point_indices],
            observations,
        )


def _problem():
    dtype = torch.float64
    camera_indices = torch.tensor(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 2, 4],
        dtype=torch.long,
    )
    point_indices = torch.tensor(
        [0, 3, 1, 4, 2, 5, 0, 6, 1, 3, 5, 6, 2],
        dtype=torch.long,
    )
    true_cameras = torch.tensor(
        [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5], [0.1, -0.4], [0.3, 0.2]],
        dtype=dtype,
    )
    true_points = torch.tensor(
        [
            [0.7, 0.1],
            [-0.3, 0.2],
            [0.2, -0.6],
            [0.5, 0.4],
            [-0.1, -0.2],
            [0.6, -0.3],
            [-0.4, 0.7],
        ],
        dtype=dtype,
    )
    observations = (
        true_cameras[camera_indices] + true_points[point_indices]
    )
    return {
        "cameras": torch.zeros_like(true_cameras),
        "points": torch.zeros_like(true_points),
        "observations": observations,
        "camera_indices": camera_indices,
        "point_indices": point_indices,
    }


def test_inverse_blocks_uses_off_diagonal_entries():
    blocks = torch.tensor(
        [[[4.0, 1.0], [1.0, 3.0]]], dtype=torch.float64
    )
    actual = inverse_blocks(blocks)
    torch.testing.assert_close(actual, torch.linalg.inv(blocks))
    assert actual[0, 0, 1] != 0


def test_schur_default_damping_and_explicit_strategy():
    model = _AdditiveBA(
        torch.zeros(2, 2, dtype=torch.float64),
        torch.zeros(2, 2, dtype=torch.float64),
    )
    optimizer = Schur(model, solver=PCG(maxiter=2))
    assert optimizer.param_groups[0]["damping"] == pytest.approx(1e-3)

    custom = pp.optim.strategy.TrustRegion(radius=1e5)
    custom_optimizer = Schur(
        model, solver=PCG(maxiter=2), strategy=custom
    )
    assert custom_optimizer.param_groups[0]["damping"] == pytest.approx(1e-5)


def _schur_worker(rank, world_size, init_file, problem, queue):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = DeviceMesh("cpu", torch.arange(world_size))
        cameras = distribute_tensor(problem["cameras"], mesh, [Shard(0)])
        points = distribute_tensor(problem["points"], mesh, [Shard(0)])
        sharded_input = {
            "observations": distribute_tensor(
                problem["observations"], mesh, [Shard(0)]
            ),
            "camera_indices": distribute_tensor(
                problem["camera_indices"], mesh, [Shard(0)]
            ),
            "point_indices": distribute_tensor(
                problem["point_indices"], mesh, [Shard(0)]
            ),
        }
        model = _AdditiveBA(cameras, points)
        solver = _CountingPCG(tol=1e-10, maxiter=100)
        optimizer = Schur(
            model,
            solver=solver,
            matrix_free_normal=True,
        )

        initial_loss = torch.sum(problem["observations"].square())
        collective_counts = {
            "all_to_all_single": 0,
            "reduce_scatter_tensor": 0,
            "all_gather": 0,
            "vector_all_reduce": 0,
        }
        original_all_to_all = dist.all_to_all_single
        original_reduce_scatter = dist.reduce_scatter_tensor
        original_all_gather = dist.all_gather
        original_all_reduce = dist.all_reduce

        def counted_all_to_all(*args, **kwargs):
            collective_counts["all_to_all_single"] += 1
            return original_all_to_all(*args, **kwargs)

        def counted_reduce_scatter(*args, **kwargs):
            collective_counts["reduce_scatter_tensor"] += 1
            return original_reduce_scatter(*args, **kwargs)

        def counted_all_gather(*args, **kwargs):
            collective_counts["all_gather"] += 1
            return original_all_gather(*args, **kwargs)

        def counted_all_reduce(tensor, *args, **kwargs):
            if tensor.numel() != 1:
                collective_counts["vector_all_reduce"] += 1
            return original_all_reduce(tensor, *args, **kwargs)

        dist.all_to_all_single = counted_all_to_all
        dist.reduce_scatter_tensor = counted_reduce_scatter
        dist.all_gather = counted_all_gather
        dist.all_reduce = counted_all_reduce
        try:
            loss = optimizer.step(sharded_input)
        finally:
            dist.all_to_all_single = original_all_to_all
            dist.reduce_scatter_tensor = original_reduce_scatter
            dist.all_gather = original_all_gather
            dist.all_reduce = original_all_reduce

        if world_size > 1:
            assert collective_counts["all_to_all_single"] > 0
            assert collective_counts["reduce_scatter_tensor"] > 0
        assert collective_counts["all_gather"] == 0
        assert collective_counts["vector_all_reduce"] == 0
        assert isinstance(model.cameras, type(cameras))
        assert isinstance(model.points, type(points))
        assert model.cameras.placements == (Shard(0),)
        assert model.points.placements == (Shard(0),)
        assert loss < initial_loss
        assert solver.operator_calls == 2
        assert solver.owner_local_inner_calls == 1

        from bae.distributed.ops import cached_gather_plan

        camera_plan = cached_gather_plan(model.cameras)
        point_plan = cached_gather_plan(model.points)
        second_loss = optimizer.step(sharded_input)
        assert second_loss <= loss
        assert solver.operator_calls == 4
        assert solver.owner_local_inner_calls == 2
        loss = second_loss
        assert cached_gather_plan(model.cameras) is camera_plan
        assert cached_gather_plan(model.points) is point_plan

        if world_size > 1:
            changed_input = {
                key: value.to_local().clone()
                for key, value in sharded_input.items()
            }
            changed_input["camera_indices"] = torch.flip(
                changed_input["camera_indices"], dims=(0,)
            )
            changed_input["point_indices"] = torch.flip(
                changed_input["point_indices"], dims=(0,)
            )
            optimizer._distributed_evaluate(
                changed_input,
                None,
                (model.cameras, model.points),
            )
            assert cached_gather_plan(model.cameras) is not camera_plan
            assert cached_gather_plan(model.points) is not point_plan

        camera_locals = [None] * world_size
        point_locals = [None] * world_size
        dist.all_gather_object(camera_locals, model.cameras.to_local())
        dist.all_gather_object(point_locals, model.points.to_local())
        if rank == 0:
            queue.put(
                (
                    float(loss),
                    torch.cat(camera_locals).tolist(),
                    torch.cat(point_locals).tolist(),
                )
            )
    finally:
        dist.destroy_process_group()


def _run(world_size):
    problem = _problem()
    handle, init_file = tempfile.mkstemp(prefix="bae-schur-")
    os.close(handle)
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    try:
        mp.spawn(
            _schur_worker,
            args=(world_size, init_file, problem, queue),
            nprocs=world_size,
            join=True,
        )
        loss, cameras, points = queue.get()
        return (
            torch.tensor(loss, dtype=torch.float64),
            torch.tensor(cameras, dtype=torch.float64),
            torch.tensor(points, dtype=torch.float64),
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def test_two_rank_schur_matches_one_rank():
    one_rank = _run(1)
    two_rank = _run(2)
    for actual, expected in zip(two_rank, one_rank):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-8)


def _se3_worker(rank, world_size, init_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = DeviceMesh("cpu", torch.arange(world_size))
        dtype = torch.float64
        initial_camera = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=dtype
        )
        true_camera = torch.tensor(
            [[0.1, -0.2, 0.05, 0.0, 0.0, 0.0, 1.0]], dtype=dtype
        )
        points = torch.tensor(
            [[0.2, 0.1, 2.0], [-0.4, 0.3, 2.5], [0.5, -0.2, 3.0]],
            dtype=dtype,
        )
        camera_indices = torch.zeros(6, dtype=torch.long)
        point_indices = torch.tensor([0, 1, 2, 0, 2, 1], dtype=torch.long)
        observations = pp.SE3(true_camera).Act(points[point_indices])

        model = _SE3BA(
            distribute_tensor(initial_camera, mesh, [Shard(0)]),
            distribute_tensor(points.clone(), mesh, [Shard(0)]),
        )
        sharded_input = {
            "observations": distribute_tensor(
                observations, mesh, [Shard(0)]
            ),
            "camera_indices": distribute_tensor(
                camera_indices, mesh, [Shard(0)]
            ),
            "point_indices": distribute_tensor(
                point_indices, mesh, [Shard(0)]
            ),
        }
        initial_loss = torch.sum(
            (points[point_indices] - observations).square()
        )
        optimizer = Schur(
            model,
            solver=PCG(tol=1e-8, maxiter=100),
            matrix_free_normal=True,
        )
        loss = optimizer.step(sharded_input)
        assert loss < initial_loss
        assert model.cameras.placements == (Shard(0),)
        assert model.cameras.to_local().shape[0] == (1 if rank == 0 else 0)
        assert model.cameras.shape[-1] == 7
    finally:
        dist.destroy_process_group()


def test_se3_trim_and_rank_with_no_owned_camera_blocks():
    handle, init_file = tempfile.mkstemp(prefix="bae-se3-")
    os.close(handle)
    try:
        mp.spawn(
            _se3_worker,
            args=(2, init_file),
            nprocs=2,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def _cuda_schur_worker(rank, world_size, init_file):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cuda", rank)
        mesh = DeviceMesh("cuda", torch.arange(world_size))
        problem = {
            key: value.to(device) for key, value in _problem().items()
        }
        model = _AdditiveBA(
            distribute_tensor(problem["cameras"], mesh, [Shard(0)]),
            distribute_tensor(problem["points"], mesh, [Shard(0)]),
        )
        sharded_input = {
            "observations": distribute_tensor(
                problem["observations"], mesh, [Shard(0)]
            ),
            "camera_indices": distribute_tensor(
                problem["camera_indices"], mesh, [Shard(0)]
            ),
            "point_indices": distribute_tensor(
                problem["point_indices"], mesh, [Shard(0)]
            ),
        }
        optimizer = Schur(
            model,
            solver=PCG(tol=1e-8, maxiter=100),
            matrix_free_normal=True,
        )
        loss = optimizer.step(sharded_input)
        assert loss < problem["observations"].square().sum()
        assert model.cameras.placements == (Shard(0),)
        assert model.points.placements == (Shard(0),)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="two CUDA devices are required"
)
def test_cuda_nccl_inductor_distributed_schur():
    handle, init_file = tempfile.mkstemp(prefix="bae-cuda-schur-")
    os.close(handle)
    try:
        mp.spawn(
            _cuda_schur_worker,
            args=(2, init_file),
            nprocs=2,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def _rollback_worker(rank, world_size, init_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        problem = _problem()
        mesh = DeviceMesh("cpu", torch.arange(world_size))
        model = _AdditiveBA(
            distribute_tensor(problem["cameras"], mesh, [Shard(0)]),
            distribute_tensor(problem["points"], mesh, [Shard(0)]),
        )
        sharded_input = {
            "observations": distribute_tensor(
                problem["observations"], mesh, [Shard(0)]
            ),
            "camera_indices": distribute_tensor(
                problem["camera_indices"], mesh, [Shard(0)]
            ),
            "point_indices": distribute_tensor(
                problem["point_indices"], mesh, [Shard(0)]
            ),
        }
        camera_before = model.cameras.to_local().clone()
        point_before = model.points.to_local().clone()
        optimizer = Schur(
            model,
            solver=_RejectingPCG(maxiter=1),
            matrix_free_normal=True,
            reject=0,
        )
        loss = optimizer.step(sharded_input)
        expected_loss = problem["observations"].square().sum()
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(model.cameras.to_local(), camera_before)
        torch.testing.assert_close(model.points.to_local(), point_before)
        assert optimizer.reject_count == 1
    finally:
        dist.destroy_process_group()


def test_rejected_step_rolls_back_owned_shards():
    handle, init_file = tempfile.mkstemp(prefix="bae-rollback-")
    os.close(handle)
    try:
        mp.spawn(
            _rollback_worker,
            args=(2, init_file),
            nprocs=2,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)
