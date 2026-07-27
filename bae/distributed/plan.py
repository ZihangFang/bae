"""Reusable selective ghost communication plans."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .context import DistributedParameterMetadata


@dataclass
class Ownership:
    total_count: int
    local_start: int
    local_count: int
    world_size: int
    rank: int
    process_group: object = None

    @classmethod
    def from_parameter(cls, metadata: DistributedParameterMetadata) -> "Ownership":
        mesh = metadata.mesh
        return cls(
            total_count=int(metadata.global_shape[0]),
            local_start=metadata.local_start,
            local_count=metadata.local_count,
            world_size=mesh.size(),
            rank=mesh.get_local_rank(),
            process_group=metadata.process_group,
        )

    @property
    def padded_total(self) -> int:
        shard = (self.total_count + self.world_size - 1) // self.world_size
        return shard * self.world_size

    def bounds(self) -> tuple[tuple[int, int], ...]:
        result = []
        chunk = (self.total_count + self.world_size - 1) // self.world_size
        for rank in range(self.world_size):
            start = min(rank * chunk, self.total_count)
            result.append((start, min(start + chunk, self.total_count)))
        return tuple(result)


@dataclass
class GatherPlan:
    ownership: Ownership
    evaluation_ids: torch.Tensor
    observation_positions: torch.Tensor
    send_local_ids: torch.Tensor
    send_splits: tuple[int, ...]
    receive_splits: tuple[int, ...]
    receive_positions: torch.Tensor


def _owners_for_ids(ids: torch.Tensor, ownership: Ownership) -> torch.Tensor:
    boundaries = torch.tensor(
        [end for _, end in ownership.bounds()],
        device=ids.device,
        dtype=ids.dtype,
    )
    return torch.searchsorted(boundaries, ids, right=True)


def build_gather_plan(global_indices: torch.Tensor, ownership: Ownership) -> GatherPlan:
    flat_indices = global_indices.reshape(-1).to(torch.long)
    if flat_indices.numel() and (
        torch.any(flat_indices < 0)
        or torch.any(flat_indices >= ownership.total_count)
    ):
        raise IndexError(
            "Distributed parameter index is outside the global dimension-0 range."
        )
    evaluation_ids, observation_positions = torch.unique(
        flat_indices, sorted=True, return_inverse=True
    )
    owners = _owners_for_ids(evaluation_ids, ownership)

    request_chunks = []
    position_chunks = []
    send_splits = []
    for owner in range(ownership.world_size):
        mask = owners == owner
        request_chunks.append(evaluation_ids[mask])
        position_chunks.append(torch.nonzero(mask, as_tuple=False).flatten())
        send_splits.append(int(mask.sum().item()))

    requests = (
        torch.cat(request_chunks)
        if request_chunks
        else evaluation_ids.new_empty((0,))
    )
    receive_positions = (
        torch.cat(position_chunks)
        if position_chunks
        else evaluation_ids.new_empty((0,))
    )

    if ownership.world_size == 1:
        receive_splits = send_splits
        received_requests = requests
    else:
        send_counts = torch.tensor(
            send_splits, device=global_indices.device, dtype=torch.int64
        )
        receive_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(
            receive_counts,
            send_counts,
            group=ownership.process_group,
        )
        receive_splits = [int(value) for value in receive_counts.cpu().tolist()]
        received_requests = torch.empty(
            sum(receive_splits),
            device=global_indices.device,
            dtype=torch.long,
        )
        dist.all_to_all_single(
            received_requests,
            requests.contiguous(),
            output_split_sizes=receive_splits,
            input_split_sizes=send_splits,
            group=ownership.process_group,
        )

    send_local_ids = received_requests - ownership.local_start
    if send_local_ids.numel() and (
        torch.any(send_local_ids < 0)
        or torch.any(send_local_ids >= ownership.local_count)
    ):
        raise RuntimeError("Received a ghost request for a block not owned by this rank.")

    return GatherPlan(
        ownership=ownership,
        evaluation_ids=evaluation_ids,
        observation_positions=observation_positions,
        send_local_ids=send_local_ids,
        send_splits=tuple(receive_splits),
        receive_splits=tuple(send_splits),
        receive_positions=receive_positions,
    )
