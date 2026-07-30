"""Benchmark component-based matrix-free LM on a BAL problem."""

from __future__ import annotations

import argparse
import bz2
import json
import os
import shutil
from pathlib import Path
from time import perf_counter
from collections import defaultdict

import numpy as np
import torch
import pypose as pp
from pypose.autograd.function import psjac
from pypose.optim.strategy import TrustRegion
from torch import nn

import bae.optim.optimizer as optimizer_module
from bae.optim.optimizer import LM
from bae.utils.linear_operator import (
    ComponentBlockDiagonalPreconditioner,
    ComponentJacobianOperator,
    ComponentNormalMatVec,
)
from bae.utils.pysolvers import PCG


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
    return {
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


def _make_model(cameras, points):
    @psjac
    def project(points, camera_parameters):
        projection = pp.SE3(camera_parameters[..., :7]).Act(points)
        projection = -projection[..., :2] / projection[..., [2]]
        focal = camera_parameters[..., [-3]]
        k1 = camera_parameters[..., [-2]]
        k2 = camera_parameters[..., [-1]]
        radius = torch.sum(projection.square(), dim=-1, keepdim=True)
        return projection * (
            1 + k1 * radius + k2 * radius.square()
        ) * focal

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("bal_data/problem-13682-4456117-pre.txt"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pcg-iterations", type=int, default=10)
    parser.add_argument("--pcg-tolerance", type=float, default=1e-4)
    parser.add_argument("--damping", type=float, default=0.003)
    parser.add_argument(
        "--evaluation-chunk-size",
        type=int,
        default=250_000,
        help="Observation rows per residual/Jacobian evaluation; 0 disables.",
    )
    parser.add_argument(
        "--compile-evaluation",
        action="store_true",
        help="Compile residual/Jacobian chunk evaluation with full-graph Inductor.",
    )
    parser.add_argument(
        "--warmup-compiled-evaluation",
        action="store_true",
        help="Evaluate all chunks once before timing to separate compilation cost.",
    )
    parser.add_argument(
        "--compiled-warmup-repetitions",
        type=int,
        default=2,
        help="Number of full evaluator passes used by compiled warmup.",
    )
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument(
        "--memory-snapshot",
        type=Path,
        help="Optional CUDA allocator-history snapshot output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lm_component_final_benchmark.json"),
    )
    args = parser.parse_args()
    if args.pcg_iterations <= 0:
        parser.error("--pcg-iterations must be positive")
    if args.damping <= 0:
        parser.error("--damping must be positive")
    if args.evaluation_chunk_size < 0:
        parser.error("--evaluation-chunk-size must be non-negative")
    if args.compiled_warmup_repetitions <= 0:
        parser.error("--compiled-warmup-repetitions must be positive")

    load_started = perf_counter()
    source = load_bal_fast(args.dataset)
    load_seconds = perf_counter() - load_started
    device = torch.device(args.device)
    tensors = {
        name: tensor.to(device) for name, tensor in source.items()
    }
    model = _make_model(
        tensors["cameras"].clone(),
        tensors["points"].clone(),
    ).to(device)
    inputs = {
        "observations": tensors["observations"],
        "camera_indices": tensors["camera_indices"],
        "point_indices": tensors["point_indices"],
    }
    optimizer = LM(
        model,
        solver=PCG(
            tol=args.pcg_tolerance,
            maxiter=args.pcg_iterations,
        ),
        strategy=TrustRegion(radius=1.0 / args.damping),
        matrix_free_normal=True,
        evaluation_chunk_size=args.evaluation_chunk_size or None,
        compile_evaluation=args.compile_evaluation,
        reject=0,
    )
    initial_loss = float(optimizer.model.loss(inputs, None))
    compile_warmup_seconds = None
    compile_warmup_repetition_seconds = []
    compile_warmup_peak_allocated_bytes = None
    compile_warmup_peak_reserved_bytes = None
    if args.warmup_compiled_evaluation:
        if not args.compile_evaluation:
            parser.error(
                "--warmup-compiled-evaluation requires --compile-evaluation"
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        active_params = tuple(
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        )
        warmup_started = perf_counter()
        for _ in range(args.compiled_warmup_repetitions):
            repetition_started = perf_counter()
            with torch.no_grad():
                warmup_result = optimizer._matrix_free_evaluate(
                    inputs, None, active_params
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            compile_warmup_repetition_seconds.append(
                perf_counter() - repetition_started
            )
            del warmup_result
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            compile_warmup_peak_allocated_bytes = (
                torch.cuda.max_memory_allocated(device)
            )
            compile_warmup_peak_reserved_bytes = (
                torch.cuda.max_memory_reserved(device)
            )
        compile_warmup_seconds = perf_counter() - warmup_started
    memory_events = []
    call_counts = defaultdict(int)
    restorations = []

    def record_memory(label: str) -> None:
        if not args.profile_memory or device.type != "cuda":
            return
        memory_events.append(
            {
                "label": label,
                "allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                    device
                ),
            }
        )

    def wrap_function(owner, name: str, label: str) -> None:
        original = getattr(owner, name)
        restorations.append((owner, name, original))

        def wrapped(*function_args, **function_kwargs):
            call_counts[label] += 1
            call = call_counts[label]
            record_memory(f"{label}:{call}:entry")
            result = original(*function_args, **function_kwargs)
            record_memory(f"{label}:{call}:exit")
            return result

        setattr(owner, name, wrapped)

    if args.profile_memory:
        wrap_function(
            optimizer_module,
            "jacobian_components",
            "jacobian_components",
        )
        wrap_function(
            ComponentJacobianOperator,
            "__init__",
            "component_jacobian:init",
        )
        wrap_function(
            ComponentJacobianOperator,
            "diagonal",
            "component_jacobian:diagonal",
        )
        wrap_function(
            ComponentJacobianOperator,
            "block_diagonal",
            "component_jacobian:block_diagonal",
        )
        wrap_function(
            ComponentJacobianOperator,
            "matvec",
            "component_jacobian:matvec",
        )
        wrap_function(
            ComponentJacobianOperator,
            "rmatvec",
            "component_jacobian:rmatvec",
        )
        wrap_function(
            ComponentNormalMatVec,
            "matvec",
            "component_normal:matvec",
        )
        wrap_function(
            ComponentBlockDiagonalPreconditioner,
            "__init__",
            "block_preconditioner:init",
        )

    if device.type == "cuda":
        if args.memory_snapshot is not None:
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="python",
                max_entries=200000,
                device=device,
            )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    record_memory("step:baseline")
    started = perf_counter()
    try:
        final_loss = optimizer.step(inputs)
    finally:
        for owner, name, original in reversed(restorations):
            setattr(owner, name, original)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        if args.memory_snapshot is not None:
            torch.cuda.memory._dump_snapshot(str(args.memory_snapshot))
            torch.cuda.memory._record_memory_history(enabled=None)
    record_memory("step:return")
    elapsed = perf_counter() - started

    result = {
        "dataset": str(args.dataset),
        "device": str(device),
        "load_seconds": load_seconds,
        "pcg_iterations": args.pcg_iterations,
        "pcg_tolerance": args.pcg_tolerance,
        "damping": args.damping,
        "evaluation_chunk_size": args.evaluation_chunk_size or None,
        "compile_evaluation": args.compile_evaluation,
        "compile_warmup_seconds": compile_warmup_seconds,
        "compile_warmup_repetition_seconds": (
            compile_warmup_repetition_seconds
        ),
        "compile_warmup_peak_allocated_bytes": (
            compile_warmup_peak_allocated_bytes
        ),
        "compile_warmup_peak_reserved_bytes": (
            compile_warmup_peak_reserved_bytes
        ),
        "initial_loss": initial_loss,
        "final_loss": float(final_loss),
        "elapsed_seconds": elapsed,
        "memory_events": memory_events,
    }
    if device.type == "cuda":
        result.update(
            {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                    device
                ),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                    device
                ),
            }
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
