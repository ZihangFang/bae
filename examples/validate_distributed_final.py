"""Two-GPU scale validation for the largest BAL ``final`` problem.

The script loads the text dataset once into shared CPU tensors, then launches
fresh NCCL workers for the original problem and/or an interleaved disjoint 2x
copy.  DTensors are built directly from owner-local CUDA shards so the global
dataset is never staged on either GPU.
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import shutil
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _ensure_text_path(path: Path) -> Path:
    if path.suffix != ".bz2":
        return path
    text_path = path.with_suffix("")
    if text_path.exists() and text_path.stat().st_size:
        return text_path
    temporary = text_path.with_suffix(text_path.suffix + ".tmp")
    with bz2.open(path, "rb") as source, temporary.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    os.replace(temporary, text_path)
    return text_path


def _rotvec_to_quaternion(camera_parameters: torch.Tensor) -> torch.Tensor:
    rotation = camera_parameters[:, :3]
    theta = torch.linalg.vector_norm(rotation, dim=-1, keepdim=True)
    half_theta = 0.5 * theta
    scale = torch.where(
        theta > 1e-12,
        torch.sin(half_theta) / theta,
        0.5 - theta.square() / 48.0,
    )
    quaternion = torch.cat(
        [rotation * scale, torch.cos(half_theta)], dim=-1
    )
    return torch.cat(
        [
            camera_parameters[:, 3:6],
            quaternion,
            camera_parameters[:, 6:],
        ],
        dim=-1,
    )


def load_bal_fast(path: Path) -> dict[str, torch.Tensor]:
    path = _ensure_text_path(path)
    with path.open("r") as stream:
        camera_count, point_count, observation_count = map(
            int, stream.readline().split()
        )
        values = np.fromfile(stream, dtype=np.float64, sep=" ")

    observation_values = 4 * observation_count
    camera_values = 9 * camera_count
    expected = observation_values + camera_values + 3 * point_count
    if values.size != expected:
        raise RuntimeError(
            f"Expected {expected:,} numeric values after the header, "
            f"read {values.size:,}."
        )

    observations_raw = values[:observation_values].reshape(
        observation_count, 4
    )
    cameras_start = observation_values
    points_start = cameras_start + camera_values
    result = {
        "camera_indices": torch.from_numpy(
            observations_raw[:, 0].astype(np.int64, copy=True)
        ),
        "point_indices": torch.from_numpy(
            observations_raw[:, 1].astype(np.int64, copy=True)
        ),
        "observations": torch.from_numpy(
            observations_raw[:, 2:4].copy()
        ),
        "cameras": _rotvec_to_quaternion(
            torch.from_numpy(
                values[cameras_start:points_start]
                .reshape(camera_count, 9)
                .copy()
            )
        ),
        "points": torch.from_numpy(
            values[points_start:].reshape(point_count, 3).copy()
        ),
    }
    return result


def _contiguous_stride(shape: torch.Size) -> tuple[int, ...]:
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= int(size)
    return tuple(reversed(stride))


def _shard_bounds(size: int, world_size: int, rank: int) -> tuple[int, int]:
    chunk = (size + world_size - 1) // world_size
    start = min(rank * chunk, size)
    return start, min(start + chunk, size)


def _local_problem(
    source: dict[str, torch.Tensor],
    mode: str,
    rank: int,
    world_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Size]]:
    camera_count = source["cameras"].shape[0]
    point_count = source["points"].shape[0]
    observation_count = source["observations"].shape[0]

    multiplier = 2 if mode == "2xfinal" else 1
    shapes = {
        "cameras": torch.Size((multiplier * camera_count, 10)),
        "points": torch.Size((multiplier * point_count, 3)),
        "observations": torch.Size((multiplier * observation_count, 2)),
        "camera_indices": torch.Size((multiplier * observation_count,)),
        "point_indices": torch.Size((multiplier * observation_count,)),
    }

    camera_start, camera_end = _shard_bounds(
        shapes["cameras"][0], world_size, rank
    )
    point_start, point_end = _shard_bounds(
        shapes["points"][0], world_size, rank
    )
    observation_start, observation_end = _shard_bounds(
        shapes["observations"][0], world_size, rank
    )

    if mode == "final":
        return (
            {
                "cameras": source["cameras"][camera_start:camera_end],
                "points": source["points"][point_start:point_end],
                "observations": source["observations"][
                    observation_start:observation_end
                ],
                "camera_indices": source["camera_indices"][
                    observation_start:observation_end
                ],
                "point_indices": source["point_indices"][
                    observation_start:observation_end
                ],
            },
            shapes,
        )

    # The virtual problem is a disjoint duplicate. Observations from its two
    # copies are interleaved so each contiguous row shard references both
    # parameter owners and continues to exercise selective ghost exchange.
    local_observation_count = observation_end - observation_start
    virtual_rows = torch.arange(
        observation_start, observation_end, dtype=torch.int64
    )
    source_rows = torch.div(virtual_rows, 2, rounding_mode="floor")
    copy_ids = torch.remainder(virtual_rows, 2)
    local = {
        "cameras": source["cameras"][
            camera_start % camera_count : (camera_start % camera_count)
            + (camera_end - camera_start)
        ],
        "points": source["points"][
            point_start % point_count : (point_start % point_count)
            + (point_end - point_start)
        ],
        "observations": source["observations"].index_select(0, source_rows),
        "camera_indices": source["camera_indices"].index_select(
            0, source_rows
        )
        + copy_ids * camera_count,
        "point_indices": source["point_indices"].index_select(0, source_rows)
        + copy_ids * point_count,
    }
    if local["observations"].shape[0] != local_observation_count:
        raise RuntimeError("Virtual observation shard construction failed.")
    return local, shapes


def _make_model(cameras, points):
    import pypose as pp
    from pypose.autograd.function import psjac
    from torch import nn

    @psjac
    def project(points, camera_parameters):
        projection = pp.SE3(camera_parameters[..., :7]).Act(points)
        projection = -projection[..., :2] / projection[..., [2]]
        focal = camera_parameters[..., [-3]]
        k1 = camera_parameters[..., [-2]]
        k2 = camera_parameters[..., [-1]]
        radius = torch.sum(projection.square(), dim=-1, keepdim=True)
        return projection * (1 + k1 * radius + k2 * radius.square()) * focal

    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.cameras = pp.Parameter(cameras, sjac=True)
            self.points = pp.Parameter(points, sjac=True)
            self.cameras.trim_SE3_grad = True

        def forward(self, observations, camera_indices, point_indices):
            return (
                project(
                    self.points[point_indices],
                    self.cameras[camera_indices],
                )
                - observations
            )

    return Residual()


def _worker(
    rank: int,
    world_size: int,
    init_file: str,
    source: dict[str, torch.Tensor],
    mode: str,
    cg_iterations: int,
    point_inner: str,
    queue,
) -> None:
    block_operator_type = None
    original_block_scalar_inner = None
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from torch.distributed.tensor import DTensor, DeviceMesh, Shard

        from bae.distributed.ops import cached_gather_plan
        from bae.distributed.schur import (
            DistributedBlockDiagonalOperator,
            _DistributedOperator,
        )
        from bae.optim.optimizer import Schur
        from bae.utils.pysolvers import PCG

        if point_inner == "global":
            block_operator_type = DistributedBlockDiagonalOperator
            original_block_scalar_inner = block_operator_type.scalar_inner
            block_operator_type.scalar_inner = _DistributedOperator.scalar_inner

        mesh = DeviceMesh("cuda", torch.arange(world_size))
        local_cpu, global_shapes = _local_problem(
            source, mode, rank, world_size
        )
        local_cuda = {
            name: tensor.contiguous().to(torch.device("cuda", rank))
            for name, tensor in local_cpu.items()
        }
        del local_cpu

        def make_dtensor(name: str) -> DTensor:
            shape = global_shapes[name]
            return DTensor.from_local(
                local_cuda[name],
                mesh,
                [Shard(0)],
                run_check=False,
                shape=shape,
                stride=_contiguous_stride(shape),
            )

        model = _make_model(
            make_dtensor("cameras"),
            make_dtensor("points"),
        )
        sharded_input = {
            "observations": make_dtensor("observations"),
            "camera_indices": make_dtensor("camera_indices"),
            "point_indices": make_dtensor("point_indices"),
        }
        optimizer = Schur(
            model,
            solver=PCG(tol=1e-4, maxiter=cg_iterations),
            matrix_free_normal=True,
            reject=0,
        )

        counts = {
            "all_to_all_single": 0,
            "reduce_scatter_tensor": 0,
            "all_gather": 0,
            "scalar_all_reduce": 0,
            "vector_all_reduce": 0,
        }
        original_a2a = dist.all_to_all_single
        original_reduce_scatter = dist.reduce_scatter_tensor
        original_all_gather = dist.all_gather
        original_all_reduce = dist.all_reduce

        def counted_a2a(*args, **kwargs):
            counts["all_to_all_single"] += 1
            return original_a2a(*args, **kwargs)

        def counted_reduce_scatter(*args, **kwargs):
            counts["reduce_scatter_tensor"] += 1
            return original_reduce_scatter(*args, **kwargs)

        def counted_all_gather(*args, **kwargs):
            counts["all_gather"] += 1
            return original_all_gather(*args, **kwargs)

        def counted_all_reduce(tensor, *args, **kwargs):
            if tensor.numel() == 1:
                counts["scalar_all_reduce"] += 1
            else:
                counts["vector_all_reduce"] += 1
            return original_all_reduce(tensor, *args, **kwargs)

        dist.all_to_all_single = counted_a2a
        dist.reduce_scatter_tensor = counted_reduce_scatter
        dist.all_gather = counted_all_gather
        dist.all_reduce = counted_all_reduce
        torch.cuda.reset_peak_memory_stats(rank)
        torch.cuda.synchronize(rank)
        started = perf_counter()
        try:
            loss = optimizer.step(sharded_input)
            torch.cuda.synchronize(rank)
        finally:
            elapsed = perf_counter() - started
            dist.all_to_all_single = original_a2a
            dist.reduce_scatter_tensor = original_reduce_scatter
            dist.all_gather = original_all_gather
            dist.all_reduce = original_all_reduce

        camera_plan = cached_gather_plan(model.cameras)
        point_plan = cached_gather_plan(model.points)
        camera_ownership = camera_plan.ownership
        point_ownership = point_plan.ownership
        camera_ghosts = int(
            (
                (camera_plan.evaluation_ids < camera_ownership.local_start)
                | (
                    camera_plan.evaluation_ids
                    >= camera_ownership.local_start
                    + camera_ownership.local_count
                )
            )
            .sum()
            .item()
        )
        point_ghosts = int(
            (
                (point_plan.evaluation_ids < point_ownership.local_start)
                | (
                    point_plan.evaluation_ids
                    >= point_ownership.local_start
                    + point_ownership.local_count
                )
            )
            .sum()
            .item()
        )
        result = {
            "rank": rank,
            "mode": mode,
            "point_inner": point_inner,
            "global_shapes": {
                name: list(shape) for name, shape in global_shapes.items()
            },
            "local_shapes": {
                name: list(tensor.shape) for name, tensor in local_cuda.items()
            },
            "initial_loss": float(optimizer.last),
            "final_loss": float(loss),
            "rejected": int(optimizer.reject_count),
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(rank),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(rank),
            "camera_evaluation_blocks": int(
                camera_plan.evaluation_ids.numel()
            ),
            "point_evaluation_blocks": int(
                point_plan.evaluation_ids.numel()
            ),
            "camera_ghost_blocks": camera_ghosts,
            "point_ghost_blocks": point_ghosts,
            "collectives": counts,
        }
        queue.put(result)
    except Exception as error:
        queue.put(
            {
                "rank": rank,
                "mode": mode,
                "point_inner": point_inner,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        raise
    finally:
        if (
            block_operator_type is not None
            and original_block_scalar_inner is not None
        ):
            block_operator_type.scalar_inner = original_block_scalar_inner
        dist.destroy_process_group()


def run_mode(
    source: dict[str, torch.Tensor],
    mode: str,
    cg_iterations: int,
    point_inner: str,
) -> list[dict]:
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    handle, init_file = tempfile.mkstemp(prefix=f"bae-{mode}-")
    os.close(handle)
    try:
        mp.spawn(
            _worker,
            args=(
                2,
                init_file,
                source,
                mode,
                cg_iterations,
                point_inner,
                queue,
            ),
            nprocs=2,
            join=True,
        )
        return sorted([queue.get(), queue.get()], key=lambda item: item["rank"])
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("bal_data/problem-13682-4456117-pre.txt.bz2"),
    )
    parser.add_argument(
        "--mode",
        choices=("final", "2xfinal", "both"),
        default="both",
    )
    parser.add_argument("--cg-iterations", type=int, default=10)
    parser.add_argument(
        "--point-inner",
        choices=("owner-local", "global", "both"),
        default="owner-local",
        help="Use the optimized owner-local point CG inner product, the "
        "pre-optimization global baseline, or benchmark both.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("distributed_final_validation.json"),
    )
    args = parser.parse_args()

    load_started = perf_counter()
    source = load_bal_fast(args.dataset)
    for tensor in source.values():
        tensor.share_memory_()
    load_seconds = perf_counter() - load_started

    modes = ("final", "2xfinal") if args.mode == "both" else (args.mode,)
    point_inners = (
        ("global", "owner-local")
        if args.point_inner == "both"
        else (args.point_inner,)
    )
    report = {
        "dataset": str(args.dataset),
        "load_seconds": load_seconds,
        "cg_iterations": args.cg_iterations,
        "point_inner": args.point_inner,
        "source_shapes": {
            name: list(tensor.shape) for name, tensor in source.items()
        },
        "runs": {},
    }
    for mode in modes:
        for point_inner in point_inners:
            key = (
                mode
                if len(point_inners) == 1
                else f"{mode}:{point_inner}"
            )
            results = run_mode(
                source, mode, args.cg_iterations, point_inner
            )
            report["runs"][key] = results
            print(json.dumps({key: results}, indent=2), flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")

    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
