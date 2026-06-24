from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Sequence

import pypose as pp
import torch
import torch.distributed as dist
from pypose.optim.optimizer import RobustModel

from bae.autograd.function import TrackingTensor
from bae.autograd.graph import jacobian

try:
    from torch.distributed.tensor import DeviceMesh
except Exception:  # pragma: no cover - old torch fallback
    DeviceMesh = None


def _balanced_partition_bounds(n: int, parts: int) -> list[tuple[int, int]]:
    base, rem = divmod(int(n), int(parts))
    bounds = []
    start = 0
    for rank in range(parts):
        size = base + (1 if rank < rem else 0)
        end = start + size
        bounds.append((start, end))
        start = end
    return bounds


def _owner_lookup(n: int, bounds: Sequence[tuple[int, int]], device: torch.device) -> torch.Tensor:
    owners = torch.empty(int(n), device=device, dtype=torch.long)
    for rank, (start, end) in enumerate(bounds):
        if end > start:
            owners[start:end] = rank
    return owners


def _sorted_unique(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return torch.empty(0, device=values.device, dtype=torch.long)
    return torch.unique(values.to(torch.long), sorted=True)


def _build_position_index(ids: torch.Tensor) -> dict[int, int]:
    return {int(v): i for i, v in enumerate(ids.detach().cpu().tolist())}


def _map_ids_to_positions(ids: torch.Tensor, position: dict[int, int], device: torch.device) -> torch.Tensor:
    if ids.numel() == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    mapped = [position[int(v)] for v in ids.detach().cpu().tolist()]
    return torch.tensor(mapped, device=device, dtype=torch.long)


def _cat_or_empty(chunks: Sequence[torch.Tensor], feature_shape: Sequence[int], *, dtype, device) -> torch.Tensor:
    nonempty = [chunk for chunk in chunks if chunk.numel() > 0]
    if nonempty:
        return torch.cat(nonempty, dim=0)
    return torch.empty((0, *feature_shape), dtype=dtype, device=device)


def _all_to_all_first_dim(
    tensor: torch.Tensor,
    send_splits: Sequence[int],
    recv_splits: Sequence[int],
    *,
    group=None,
) -> torch.Tensor:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized.")
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor.clone()

    send_splits = [int(v) for v in send_splits]
    recv_splits = [int(v) for v in recv_splits]
    out_shape = (sum(recv_splits), *tensor.shape[1:])
    out = torch.empty(out_shape, dtype=tensor.dtype, device=tensor.device)
    dist.all_to_all_single(
        out,
        tensor.contiguous(),
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
        group=group,
    )
    return out


def _padded_total(total_count: int, world_size: int) -> int:
    shard = (int(total_count) + int(world_size) - 1) // int(world_size)
    return shard * int(world_size)


def _apply_block_matrix(blocks: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    if blocks.numel() == 0:
        return torch.empty_like(vectors)
    return torch.einsum("oij,oj->oi", blocks, vectors)


def _jacobi_inverse_blocks(blocks: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(blocks)
    if blocks.numel() == 0:
        return out
    diag = torch.diagonal(blocks, dim1=-2, dim2=-1)
    inv_diag = torch.where(diag != 0.0, 1.0 / diag, torch.ones_like(diag))
    torch.diagonal(out, dim1=-2, dim2=-1).copy_(inv_diag)
    return out


def _damped_blocks(blocks: torch.Tensor, damping: float, min_value: float, max_value: float) -> torch.Tensor:
    out = blocks.clone()
    if out.numel() == 0:
        return out
    diag = torch.diagonal(out, dim1=-2, dim2=-1)
    diag.copy_(diag.clamp(min=min_value, max=max_value) * (1.0 + float(damping)))
    return out


def _loss_from_residual(residual: torch.Tensor) -> torch.Tensor:
    return residual.square().sum()


@dataclass
class DistributedConfig:
    device: torch.device
    process_group: Optional[dist.ProcessGroup] = None
    device_mesh: Optional[object] = None

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self.process_group) if dist.is_initialized() else 1

    @property
    def rank(self) -> int:
        return dist.get_rank(self.process_group) if dist.is_initialized() else 0

    @classmethod
    def default(cls) -> "DistributedConfig":
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
        mesh = None
        if DeviceMesh is not None and dist.is_initialized() and device.type == "cuda":
            mesh = DeviceMesh("cuda", torch.arange(dist.get_world_size(), device=device), _init_backend=False)
        return cls(device=device, device_mesh=mesh)


@dataclass
class _GatherPlan:
    eval_ids: torch.Tensor
    owned_eval_local_ids: torch.Tensor
    send_local_ids: list[torch.Tensor]
    send_splits: list[int]
    recv_splits: list[int]
    recv_positions: torch.Tensor


@dataclass
class _ReducePlan:
    global_ids: torch.Tensor
    padded_total: int
    local_start: int
    local_count: int


@dataclass
class BALShardPlan:
    rank: int
    world_size: int
    device: torch.device
    num_cameras: int
    num_points: int
    obs_start: int
    obs_end: int
    camera_start: int
    camera_end: int
    point_start: int
    point_end: int
    local_observes: torch.Tensor
    local_cidx_global: torch.Tensor
    local_pidx_global: torch.Tensor
    camera_local_obs_index: torch.Tensor
    point_local_obs_index: torch.Tensor
    camera_gather: _GatherPlan
    point_gather: _GatherPlan
    camera_reduce: _ReducePlan
    point_reduce: _ReducePlan

    @property
    def num_owned_cameras(self) -> int:
        return int(self.camera_end - self.camera_start)

    @property
    def num_owned_points(self) -> int:
        return int(self.point_end - self.point_start)

    @property
    def camera_eval_ids(self) -> torch.Tensor:
        return self.camera_gather.eval_ids

    @property
    def point_eval_ids(self) -> torch.Tensor:
        return self.point_gather.eval_ids

    @property
    def camera_owned_eval_local_ids(self) -> torch.Tensor:
        return self.camera_gather.owned_eval_local_ids

    @property
    def point_owned_eval_local_ids(self) -> torch.Tensor:
        return self.point_gather.owned_eval_local_ids

    @property
    def camera_ghost_recv_splits(self) -> list[int]:
        return self.camera_gather.recv_splits

    @property
    def point_ghost_recv_splits(self) -> list[int]:
        return self.point_gather.recv_splits

    @property
    def camera_ghost_send_local_ids(self) -> list[torch.Tensor]:
        return self.camera_gather.send_local_ids

    @property
    def point_ghost_send_local_ids(self) -> list[torch.Tensor]:
        return self.point_gather.send_local_ids

    @property
    def camera_contrib_send_row_ids(self) -> list[torch.Tensor]:
        return []

    @property
    def point_contrib_send_row_ids(self) -> list[torch.Tensor]:
        return []

    @property
    def camera_contrib_recv_splits(self) -> list[int]:
        return []

    @property
    def point_contrib_recv_splits(self) -> list[int]:
        return []

    @classmethod
    def from_global_input(
        cls,
        global_input: dict[str, torch.Tensor],
        num_cameras: int,
        num_points: int,
        config: DistributedConfig,
    ) -> "BALShardPlan":
        rank = config.rank
        world_size = config.world_size
        device = config.device

        observes = global_input["observes"].to(device)
        cidx = global_input["cidx"].to(device=device, dtype=torch.long)
        pidx = global_input["pidx"].to(device=device, dtype=torch.long)

        obs_bounds = _balanced_partition_bounds(observes.shape[0], world_size)
        cam_bounds = _balanced_partition_bounds(num_cameras, world_size)
        pt_bounds = _balanced_partition_bounds(num_points, world_size)
        obs_start, obs_end = obs_bounds[rank]
        camera_start, camera_end = cam_bounds[rank]
        point_start, point_end = pt_bounds[rank]

        local_observes = observes[obs_start:obs_end].contiguous()
        local_cidx_global = cidx[obs_start:obs_end].contiguous()
        local_pidx_global = pidx[obs_start:obs_end].contiguous()

        camera_eval_ids = _sorted_unique(local_cidx_global)
        point_eval_ids = _sorted_unique(local_pidx_global)
        camera_pos = _build_position_index(camera_eval_ids)
        point_pos = _build_position_index(point_eval_ids)
        camera_local_obs_index = _map_ids_to_positions(local_cidx_global, camera_pos, device)
        point_local_obs_index = _map_ids_to_positions(local_pidx_global, point_pos, device)

        camera_owners = _owner_lookup(num_cameras, cam_bounds, device)
        point_owners = _owner_lookup(num_points, pt_bounds, device)

        camera_gather = _make_gather_plan(camera_eval_ids, camera_owners, camera_start, rank, world_size, config)
        point_gather = _make_gather_plan(point_eval_ids, point_owners, point_start, rank, world_size, config)
        camera_reduce = _make_reduce_plan(local_cidx_global, num_cameras, camera_start, camera_end, world_size)
        point_reduce = _make_reduce_plan(local_pidx_global, num_points, point_start, point_end, world_size)

        return cls(
            rank=rank,
            world_size=world_size,
            device=device,
            num_cameras=int(num_cameras),
            num_points=int(num_points),
            obs_start=int(obs_start),
            obs_end=int(obs_end),
            camera_start=int(camera_start),
            camera_end=int(camera_end),
            point_start=int(point_start),
            point_end=int(point_end),
            local_observes=local_observes,
            local_cidx_global=local_cidx_global,
            local_pidx_global=local_pidx_global,
            camera_local_obs_index=camera_local_obs_index,
            point_local_obs_index=point_local_obs_index,
            camera_gather=camera_gather,
            point_gather=point_gather,
            camera_reduce=camera_reduce,
            point_reduce=point_reduce,
        )


def _all_gather_object(value, group=None) -> list:
    world_size = dist.get_world_size(group) if dist.is_initialized() else 1
    gathered = [None for _ in range(world_size)]
    if world_size == 1:
        gathered[0] = value
    else:
        dist.all_gather_object(gathered, value, group=group)
    return gathered


def _make_gather_plan(
    eval_ids: torch.Tensor,
    owners: torch.Tensor,
    owned_start: int,
    rank: int,
    world_size: int,
    config: DistributedConfig,
) -> _GatherPlan:
    request_by_owner: list[list[int]] = [[] for _ in range(world_size)]
    positions_by_owner: list[list[int]] = [[] for _ in range(world_size)]
    eval_cpu = eval_ids.detach().cpu().tolist()
    for pos, gid in enumerate(eval_cpu):
        owner = int(owners[int(gid)].item())
        request_by_owner[owner].append(int(gid))
        positions_by_owner[owner].append(pos)

    gathered_requests = _all_gather_object(request_by_owner, group=config.process_group)
    send_local_ids = []
    for requester in range(world_size):
        ids = gathered_requests[requester][rank]
        local = [int(gid) - int(owned_start) for gid in ids]
        send_local_ids.append(torch.tensor(local, dtype=torch.long, device=config.device))

    recv_positions = [pos for owner_pos in positions_by_owner for pos in owner_pos]
    owned_eval_local = [
        pos for pos, gid in enumerate(eval_cpu)
        if int(owners[int(gid)].item()) == rank
    ]
    return _GatherPlan(
        eval_ids=eval_ids,
        owned_eval_local_ids=torch.tensor(owned_eval_local, dtype=torch.long, device=config.device),
        send_local_ids=send_local_ids,
        send_splits=[int(t.numel()) for t in send_local_ids],
        recv_splits=[len(request_by_owner[src]) for src in range(world_size)],
        recv_positions=torch.tensor(recv_positions, dtype=torch.long, device=config.device),
    )


def _make_reduce_plan(
    global_ids: torch.Tensor,
    total_count: int,
    local_start: int,
    local_end: int,
    world_size: int,
) -> _ReducePlan:
    return _ReducePlan(
        global_ids=global_ids,
        padded_total=_padded_total(total_count, world_size),
        local_start=int(local_start),
        local_count=int(local_end - local_start),
    )


class _DistributedCG:
    def __init__(self, tol: float = 1e-5, maxiter: int = 250):
        self.tol = float(tol)
        self.maxiter = int(maxiter)

    def solve(self, operator, b: torch.Tensor, M: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        bnorm = torch.sqrt(operator.scalar_inner(b, b)).clamp_min(1e-12)
        rho_prev = None
        p = None
        if torch.sqrt(operator.scalar_inner(r, r)).item() <= self.tol * bnorm.item():
            return x
        for _ in range(self.maxiter):
            if torch.sqrt(operator.scalar_inner(r, r)).item() <= self.tol * bnorm.item():
                break
            z = _apply_block_matrix(M, r) if M is not None else r
            rho_cur = operator.scalar_inner(r, z)
            if p is None:
                p = z.clone()
            else:
                if rho_prev is None or rho_prev.abs().item() < 1e-30:
                    break
                p = z + (rho_cur / rho_prev) * p
            q = operator(p)
            denom = operator.scalar_inner(p, q)
            if denom.abs().item() < 1e-30:
                break
            alpha = rho_cur / denom
            x = x + alpha * p
            r = r - alpha * q
            rho_prev = rho_cur
        return x


class _DistributedSchurCameraOperator:
    def __init__(
        self,
        optimizer: "DistributedSchur",
        Jc_values: torch.Tensor,
        Jp_values: torch.Tensor,
        U_blocks: torch.Tensor,
        V_inv_blocks: torch.Tensor,
    ):
        self.optimizer = optimizer
        self.Jc_values = Jc_values
        self.Jp_values = Jp_values
        self.U_blocks = U_blocks
        self.V_inv_blocks = V_inv_blocks

    def scalar_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        value = torch.sum(x * y)
        if self.optimizer.config.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=self.optimizer.config.process_group)
        return value

    def __call__(self, x_owned: torch.Tensor) -> torch.Tensor:
        opt = self.optimizer
        plan = opt.plan
        x_eval = opt._gather_eval_blocks(x_owned, plan.camera_gather)
        x_obs = x_eval[plan.camera_local_obs_index]
        obs = torch.einsum("ori,oi->or", self.Jc_values, x_obs)
        point_rhs = opt._reduce_obs_contrib_to_points(torch.einsum("ori,or->oi", self.Jp_values, obs))
        point_scaled = _apply_block_matrix(self.V_inv_blocks, point_rhs)
        point_eval = opt._gather_eval_blocks(point_scaled, plan.point_gather)
        point_obs = point_eval[plan.point_local_obs_index]
        obs2 = torch.einsum("ori,oi->or", self.Jp_values, point_obs)
        camera_term = opt._reduce_obs_contrib_to_cameras(torch.einsum("ori,or->oi", self.Jc_values, obs2))
        return _apply_block_matrix(self.U_blocks, x_owned) - camera_term

    def __matmul__(self, x_owned: torch.Tensor) -> torch.Tensor:
        return self(x_owned)


class _DistributedBlockDiagonalOperator:
    def __init__(self, optimizer: "DistributedSchur", blocks: torch.Tensor):
        self.optimizer = optimizer
        self.blocks = blocks

    def scalar_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        value = torch.sum(x * y)
        if self.optimizer.config.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=self.optimizer.config.process_group)
        return value

    def __call__(self, x_owned: torch.Tensor) -> torch.Tensor:
        return _apply_block_matrix(self.blocks, x_owned)

    def __matmul__(self, x_owned: torch.Tensor) -> torch.Tensor:
        return self(x_owned)


class DistributedSchur:
    def __init__(
        self,
        model,
        global_input: dict[str, torch.Tensor],
        *,
        config: Optional[DistributedConfig] = None,
        strategy=None,
        solver: Optional[object] = None,
        reject: int = 30,
        min: float = 1e-6,
        max: float = 1e32,
        pose_attr: str = "pose",
        point_attr: str = "points",
        matrix_free_normal: bool = True,
        kernel=None,
        solve_mode: str = "cg",
    ):
        if not matrix_free_normal:
            raise ValueError("DistributedSchur currently supports matrix_free_normal=True only.")
        if not dist.is_initialized():
            raise RuntimeError("DistributedSchur requires torch.distributed to be initialized.")

        self.config = config if config is not None else DistributedConfig.default()
        self.raw_model = model.model if isinstance(model, RobustModel) else model
        self.model = model if isinstance(model, RobustModel) else RobustModel(model, kernel)
        self.raw_model.to(self.config.device)
        self.pose_attr = pose_attr
        self.point_attr = point_attr
        self.reject = int(reject)
        self.reject_count = 0
        self.strategy = pp.optim.strategy.TrustRegion() if strategy is None else strategy
        self.pg = {**{"min": min, "max": max}, **self.strategy.defaults}
        self.solve_mode = solve_mode
        if solve_mode not in ("cg", "exact"):
            raise ValueError(f"Unsupported solve_mode {solve_mode!r}; expected 'cg' or 'exact'.")

        pose = getattr(self.raw_model, self.pose_attr).detach().to(self.config.device)
        points = getattr(self.raw_model, self.point_attr).detach().to(self.config.device)
        self.plan = BALShardPlan.from_global_input(global_input, pose.shape[0], points.shape[0], self.config)
        self.pose_owned = pose[self.plan.camera_start:self.plan.camera_end].clone()
        self.points_owned = points[self.plan.point_start:self.plan.point_end].clone()
        if getattr(getattr(self.raw_model, self.pose_attr), "trim_SE3_grad", False):
            self.pose_owned.trim_SE3_grad = True

        tol = float(getattr(solver, "tol", 1e-5)) if solver is not None else 1e-5
        maxiter = int(getattr(solver, "maxiter", 250) or 250) if solver is not None else 250
        self.solver = _DistributedCG(tol=tol, maxiter=maxiter)

    @contextmanager
    def _local_eval_params(self, pose_eval: torch.Tensor, point_eval: torch.Tensor):
        old_pose = getattr(self.raw_model, self.pose_attr)
        old_points = getattr(self.raw_model, self.point_attr)
        pose_param = pp.Parameter(pose_eval, sjac=True)
        point_param = pp.Parameter(point_eval, sjac=True)
        if getattr(self.pose_owned, "trim_SE3_grad", False) or getattr(old_pose, "trim_SE3_grad", False):
            pose_param.trim_SE3_grad = True
        setattr(self.raw_model, self.pose_attr, pose_param)
        setattr(self.raw_model, self.point_attr, point_param)
        try:
            yield pose_param, point_param
        finally:
            setattr(self.raw_model, self.pose_attr, old_pose)
            setattr(self.raw_model, self.point_attr, old_points)

    def _gather_eval_blocks(self, owned_blocks: torch.Tensor, gather: _GatherPlan) -> torch.Tensor:
        feature_shape = owned_blocks.shape[1:]
        send_chunks = [owned_blocks.index_select(0, ids) if ids.numel() else owned_blocks.new_empty((0, *feature_shape))
                       for ids in gather.send_local_ids]
        send = _cat_or_empty(send_chunks, feature_shape, dtype=owned_blocks.dtype, device=owned_blocks.device)
        received = _all_to_all_first_dim(send, gather.send_splits, gather.recv_splits, group=self.config.process_group)
        eval_blocks = owned_blocks.new_empty((gather.eval_ids.numel(), *feature_shape))
        if received.numel() > 0:
            eval_blocks.index_copy_(0, gather.recv_positions, received)
        return eval_blocks

    def _reduce_obs_contrib(self, contrib: torch.Tensor, reduce: _ReducePlan, num_owned: int) -> torch.Tensor:
        feature_shape = contrib.shape[1:]
        shard_rows = reduce.padded_total // self.config.world_size
        dense = torch.zeros(
            (reduce.padded_total, *feature_shape),
            dtype=contrib.dtype,
            device=contrib.device,
        )
        if contrib.numel() > 0:
            dense.index_add_(0, reduce.global_ids.to(torch.long), contrib)

        if self.config.world_size == 1:
            reduced = dense[:shard_rows]
        else:
            reduced = torch.empty((shard_rows, *feature_shape), dtype=contrib.dtype, device=contrib.device)
            dist.reduce_scatter_tensor(
                reduced,
                dense.contiguous(),
                op=dist.ReduceOp.SUM,
                group=self.config.process_group,
            )

        local_offset = reduce.local_start - self.config.rank * shard_rows
        return reduced[local_offset:local_offset + num_owned].contiguous()

    def _reduce_obs_contrib_to_cameras(self, contrib: torch.Tensor) -> torch.Tensor:
        return self._reduce_obs_contrib(contrib, self.plan.camera_reduce, self.plan.num_owned_cameras)

    def _reduce_obs_contrib_to_points(self, contrib: torch.Tensor) -> torch.Tensor:
        return self._reduce_obs_contrib(contrib, self.plan.point_reduce, self.plan.num_owned_points)

    def _local_eval_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        pose_eval = self._gather_eval_blocks(self.pose_owned, self.plan.camera_gather)
        point_eval = self._gather_eval_blocks(self.points_owned, self.plan.point_gather)
        return pose_eval, point_eval

    def _local_input(self) -> dict[str, torch.Tensor]:
        return {
            "observes": self.plan.local_observes,
            "cidx": self.plan.camera_local_obs_index,
            "pidx": self.plan.point_local_obs_index,
        }

    def _local_residual_and_jacobians(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pose_eval, point_eval = self._local_eval_state()
        with self._local_eval_params(pose_eval, point_eval) as (pose_param, point_param):
            residual = self.model(self._local_input())[0]
            Jc, Jp = jacobian(residual, [pose_param, point_param])
        if isinstance(residual, TrackingTensor):
            residual = residual.tensor()
        if Jc.values().shape[0] != residual.shape[0] or Jp.values().shape[0] != residual.shape[0]:
            raise RuntimeError("DistributedSchur expects one sparse Jacobian block per observation row.")
        return residual, Jc.values().contiguous(), Jp.values().contiguous()

    def _global_loss(self, residual: torch.Tensor) -> torch.Tensor:
        loss = _loss_from_residual(residual)
        if self.config.world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.config.process_group)
        return loss

    def _apply_pose_step(self, step_blocks: torch.Tensor) -> None:
        if self.pose_owned.numel() == 0:
            return
        if getattr(self.pose_owned, "trim_SE3_grad", False):
            self.pose_owned[..., :7] = pp.SE3(self.pose_owned[..., :7]).add_(pp.se3(step_blocks[..., :6]))
            if self.pose_owned.shape[-1] > 7:
                self.pose_owned[..., 7:] += step_blocks[..., 6:]
        else:
            self.pose_owned.add_(step_blocks.view_as(self.pose_owned))

    def _apply_point_step(self, step_blocks: torch.Tensor) -> None:
        if self.points_owned.numel() > 0:
            self.points_owned.add_(step_blocks.view_as(self.points_owned))

    def _update_owned_params(self, camera_step: torch.Tensor, point_step: torch.Tensor, sign: float = 1.0) -> None:
        self._apply_pose_step(sign * camera_step)
        self._apply_point_step(sign * point_step)

    def _rhs_camera(
        self,
        Jc_values: torch.Tensor,
        Jp_values: torch.Tensor,
        V_inv: torch.Tensor,
        Ic: torch.Tensor,
        Ip: torch.Tensor,
    ) -> torch.Tensor:
        point_scaled = _apply_block_matrix(V_inv, Ip)
        point_eval = self._gather_eval_blocks(point_scaled, self.plan.point_gather)
        point_obs = point_eval[self.plan.point_local_obs_index]
        obs = torch.einsum("ori,oi->or", Jp_values, point_obs)
        camera_term = self._reduce_obs_contrib_to_cameras(torch.einsum("ori,or->oi", Jc_values, obs))
        return Ic - camera_term

    def _rhs_point(
        self,
        Jc_values: torch.Tensor,
        Jp_values: torch.Tensor,
        D_c: torch.Tensor,
        Ip: torch.Tensor,
    ) -> torch.Tensor:
        camera_eval = self._gather_eval_blocks(D_c, self.plan.camera_gather)
        camera_obs = camera_eval[self.plan.camera_local_obs_index]
        obs = torch.einsum("ori,oi->or", Jc_values, camera_obs)
        point_term = self._reduce_obs_contrib_to_points(torch.einsum("ori,or->oi", Jp_values, obs))
        return Ip - point_term

    def _predicted_quality(
        self,
        residual: torch.Tensor,
        Jc_values: torch.Tensor,
        Jp_values: torch.Tensor,
        D_c: torch.Tensor,
        D_p: torch.Tensor,
    ) -> torch.Tensor:
        camera_eval = self._gather_eval_blocks(D_c, self.plan.camera_gather)
        point_eval = self._gather_eval_blocks(D_p, self.plan.point_gather)
        jd = torch.einsum("ori,oi->or", Jc_values, camera_eval[self.plan.camera_local_obs_index])
        jd = jd + torch.einsum("ori,oi->or", Jp_values, point_eval[self.plan.point_local_obs_index])
        denom = -torch.sum(jd * (2.0 * residual + jd))
        if self.config.world_size > 1:
            dist.all_reduce(denom, op=dist.ReduceOp.SUM, group=self.config.process_group)
        return denom

    def _all_gather_object(self, value) -> list:
        return _all_gather_object(value, group=self.config.process_group)

    def _dense_block_diag_from_owned(self, gathered_blocks: Sequence[torch.Tensor]) -> torch.Tensor:
        blocks = [b.to(self.config.device) for b in gathered_blocks if b.numel() > 0]
        if not blocks:
            return torch.empty((0, 0), dtype=self.pose_owned.dtype, device=self.config.device)
        blocks = torch.cat(blocks, dim=0)
        return torch.block_diag(*list(blocks))

    def _gather_exact_w(self, Jc_values: torch.Tensor, Jp_values: torch.Tensor) -> torch.Tensor:
        local = {
            "c": self.plan.local_cidx_global.detach().cpu(),
            "p": self.plan.local_pidx_global.detach().cpu(),
            "w": torch.einsum("ori,orj->oij", Jc_values, Jp_values).detach().cpu(),
        }
        gathered = self._all_gather_object(local)
        dc = Jc_values.shape[-1]
        dp = Jp_values.shape[-1]
        W = torch.zeros(
            (self.plan.num_cameras * dc, self.plan.num_points * dp),
            dtype=Jc_values.dtype,
            device=self.config.device,
        )
        for item in gathered:
            cids = item["c"].to(torch.long)
            pids = item["p"].to(torch.long)
            values = item["w"].to(self.config.device)
            for i in range(values.shape[0]):
                cr = int(cids[i]) * dc
                pc = int(pids[i]) * dp
                W[cr:cr + dc, pc:pc + dp] += values[i]
        return W

    def _solve_exact_dense(
        self,
        Jc_values: torch.Tensor,
        Jp_values: torch.Tensor,
        U_eff: torch.Tensor,
        V_eff: torch.Tensor,
        Ic: torch.Tensor,
        Ip: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        U_all = self._dense_block_diag_from_owned(self._all_gather_object(U_eff.detach().cpu()))
        V_all = self._dense_block_diag_from_owned(self._all_gather_object(V_eff.detach().cpu()))
        Ic_all = torch.cat([x.reshape(-1).to(self.config.device) for x in self._all_gather_object(Ic.detach().cpu())])
        Ip_all = torch.cat([x.reshape(-1).to(self.config.device) for x in self._all_gather_object(Ip.detach().cpu())])
        W = self._gather_exact_w(Jc_values, Jp_values)
        V_inv = torch.linalg.inv(V_all)
        rhs_c = Ic_all - W @ (V_inv @ Ip_all)
        S = U_all - W @ V_inv @ W.mT
        D_c_all = torch.linalg.solve(S, rhs_c).reshape(self.plan.num_cameras, -1)
        rhs_p = Ip_all - W.mT @ D_c_all.reshape(-1)
        D_p_all = torch.linalg.solve(V_all, rhs_p).reshape(self.plan.num_points, -1)
        return (
            D_c_all[self.plan.camera_start:self.plan.camera_end].contiguous(),
            D_p_all[self.plan.point_start:self.plan.point_end].contiguous(),
        )

    def _update_strategy(self, last_loss: torch.Tensor, loss: torch.Tensor, denom: torch.Tensor) -> None:
        if loss >= last_loss or denom <= 0:
            quality = -1.0
        else:
            quality = float(((last_loss - loss) / denom).item())
        self.pg["radius"] = 1.0 / self.pg["damping"]
        if quality > self.pg["high"]:
            self.pg["radius"] = self.pg["up"] * self.pg["radius"]
            self.pg["down"] = self.strategy.down
        elif quality > self.pg["low"]:
            self.pg["radius"] = self.pg["radius"]
            self.pg["down"] = self.strategy.down
        else:
            self.pg["radius"] = self.pg["radius"] * self.pg["down"]
            self.pg["down"] = self.pg["down"] * self.pg["factor"]
        self.pg["down"] = max(self.strategy.min, min(self.pg["down"], self.strategy.max))
        self.pg["radius"] = max(self.pg["min"], min(self.pg["radius"], self.pg["max"]))
        self.pg["damping"] = 1.0 / self.pg["radius"]

    @torch.no_grad()
    def step(self) -> torch.Tensor:
        residual, Jc_values, Jp_values = self._local_residual_and_jacobians()
        self.reject_count = 0
        self.last = self.loss if hasattr(self, "loss") else self._global_loss(residual)
        self.loss = self.last

        U_blocks = self._reduce_obs_contrib_to_cameras(torch.einsum("ori,orj->oij", Jc_values, Jc_values))
        V_blocks = self._reduce_obs_contrib_to_points(torch.einsum("ori,orj->oij", Jp_values, Jp_values))
        Ic = self._reduce_obs_contrib_to_cameras(-torch.einsum("ori,or->oi", Jc_values, residual))
        Ip = self._reduce_obs_contrib_to_points(-torch.einsum("ori,or->oi", Jp_values, residual))

        while self.last <= self.loss:
            U_eff = _damped_blocks(U_blocks, self.pg["damping"], self.pg["min"], self.pg["max"])
            V_eff = _damped_blocks(V_blocks, self.pg["damping"], self.pg["min"], self.pg["max"])
            V_inv = torch.linalg.inv(V_eff) if V_eff.numel() else V_eff

            if self.solve_mode == "exact":
                D_c, D_p = self._solve_exact_dense(Jc_values, Jp_values, U_eff, V_eff, Ic, Ip)
            else:
                rhs_c = self._rhs_camera(Jc_values, Jp_values, V_inv, Ic, Ip)
                op = _DistributedSchurCameraOperator(self, Jc_values, Jp_values, U_eff, V_inv)
                M = _jacobi_inverse_blocks(U_eff)
                D_c = self.solver.solve(op, rhs_c, M=M)
                rhs_p = self._rhs_point(Jc_values, Jp_values, D_c, Ip)
                point_op = _DistributedBlockDiagonalOperator(self, V_eff)
                point_M = _jacobi_inverse_blocks(V_eff)
                D_p = self.solver.solve(point_op, rhs_p, M=point_M)

            self._update_owned_params(D_c, D_p)
            new_residual, _, _ = self._local_residual_and_jacobians()
            self.loss = self._global_loss(new_residual)
            denom = self._predicted_quality(residual, Jc_values, Jp_values, D_c, D_p)
            self._update_strategy(self.last, self.loss, denom)

            if self.last < self.loss and self.reject_count < self.reject:
                self._update_owned_params(D_c, D_p, sign=-1.0)
                self.loss = self.last
                self.reject_count += 1
            else:
                break

        return self.loss
