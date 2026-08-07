"""Functional distributed indexing and communication operations."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from .context import parameter_metadata, parameter_metadata_by_key
from .plan import GatherPlan, Ownership, build_gather_plan


_GATHER_PLAN_CACHE: dict[int, tuple[torch.Tensor, GatherPlan]] = {}


def ghost_gather(owned_blocks: torch.Tensor, plan: GatherPlan) -> torch.Tensor:
    feature_shape = owned_blocks.shape[1:]
    packed = owned_blocks.index_select(0, plan.send_local_ids)
    if plan.ownership.world_size == 1:
        received = packed.clone()
    else:
        received = torch.empty(
            (sum(plan.receive_splits), *feature_shape),
            device=owned_blocks.device,
            dtype=owned_blocks.dtype,
        )
        dist.all_to_all_single(
            received,
            packed.contiguous(),
            output_split_sizes=list(plan.receive_splits),
            input_split_sizes=list(plan.send_splits),
            group=plan.ownership.process_group,
        )

    evaluation = owned_blocks.new_empty((plan.evaluation_ids.numel(), *feature_shape))
    if received.numel():
        evaluation.index_copy_(0, plan.receive_positions, received)
    return evaluation


def _get_or_build_plan(parameter_key: int, indices: torch.Tensor) -> GatherPlan:
    metadata = parameter_metadata_by_key(parameter_key)
    cached = _GATHER_PLAN_CACHE.get(parameter_key)
    local_changed = cached is None or not torch.equal(cached[0], indices)

    # Plan construction is collective.  If one observation shard changes, all
    # ranks must rebuild even when another rank's local index values did not.
    changed = torch.tensor(int(local_changed), device=indices.device, dtype=torch.int32)
    if metadata.mesh.size() > 1:
        sent_changes = changed.expand(metadata.mesh.size()).contiguous()
        received_changes = torch.empty_like(sent_changes)
        dist.all_to_all_single(
            received_changes,
            sent_changes,
            group=metadata.process_group,
        )
        changed = received_changes.max()
    if bool(changed.item()):
        plan = build_gather_plan(indices, Ownership.from_parameter(metadata))
        _GATHER_PLAN_CACHE[parameter_key] = (indices.detach().clone(), plan)
        return plan
    assert cached is not None
    return cached[1]


@torch.library.custom_op("bae::distributed_index", mutates_args=())
def _distributed_index_op(
    owned_blocks: torch.Tensor,
    global_indices: torch.Tensor,
    parameter_key: int,
) -> torch.Tensor:
    plan = _get_or_build_plan(parameter_key, global_indices)
    evaluation = ghost_gather(owned_blocks, plan)
    result = evaluation.index_select(0, plan.observation_positions)
    return result.reshape(*global_indices.shape, *owned_blocks.shape[1:])


@_distributed_index_op.register_fake
def _distributed_index_fake(
    owned_blocks: torch.Tensor,
    global_indices: torch.Tensor,
    parameter_key: int,
) -> torch.Tensor:
    return owned_blocks.new_empty((*global_indices.shape, *owned_blocks.shape[1:]))


def _contiguous_stride(shape: torch.Size) -> tuple[int, ...]:
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def distributed_index(parameter, global_indices: torch.Tensor) -> torch.Tensor:
    """Gather globally indexed parameter rows into the local observation order."""

    metadata = parameter_metadata(parameter)
    indices_are_distributed = isinstance(global_indices, DTensor)
    if indices_are_distributed and (
        global_indices.device_mesh != metadata.mesh
        or global_indices.placements != (Shard(0),)
    ):
        raise ValueError(
            "DTensor indices must use the parameter mesh and placements=[Shard(0)]."
        )
    local_indices = (
        global_indices.to_local() if indices_are_distributed else global_indices
    )
    local_result = _distributed_index_op(
        parameter.to_local(), local_indices, metadata.key
    )
    if not indices_are_distributed:
        return local_result

    global_shape = torch.Size((*global_indices.shape, *metadata.global_shape[1:]))
    return DTensor.from_local(
        local_result,
        metadata.mesh,
        [Shard(0)],
        run_check=False,
        shape=global_shape,
        stride=_contiguous_stride(global_shape),
    )


def cached_gather_plan(parameter) -> GatherPlan:
    metadata = parameter_metadata(parameter)
    try:
        return _GATHER_PLAN_CACHE[metadata.key][1]
    except KeyError as error:
        raise RuntimeError(
            "No ghost-gather plan exists yet. Evaluate the distributed indexed "
            "model once before constructing the Schur operator."
        ) from error


def owner_reduce_scatter(
    observation_contributions: torch.Tensor,
    global_block_ids: torch.Tensor,
    ownership: Ownership,
) -> torch.Tensor:
    feature_shape = observation_contributions.shape[1:]
    shard_count = ownership.padded_total // ownership.world_size
    padded = observation_contributions.new_zeros(
        (ownership.padded_total, *feature_shape)
    )
    if observation_contributions.numel():
        padded.index_add_(
            0, global_block_ids.reshape(-1).to(torch.long), observation_contributions
        )

    if ownership.world_size == 1:
        reduced = padded[:shard_count]
    else:
        reduced = observation_contributions.new_empty((shard_count, *feature_shape))
        dist.reduce_scatter_tensor(
            reduced,
            padded.contiguous(),
            op=dist.ReduceOp.SUM,
            group=ownership.process_group,
        )
    offset = ownership.local_start - ownership.rank * shard_count
    return reduced[offset : offset + ownership.local_count].contiguous()
