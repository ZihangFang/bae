"""Owner-local operators for the distributed Schur complement."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .ops import ghost_gather, owner_reduce_scatter
from .plan import GatherPlan, Ownership


def apply_block_matrix(blocks: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    if vectors.numel() == 0:
        return torch.empty_like(vectors)
    return torch.einsum("bij,bj->bi", blocks, vectors)


def inverse_diagonal_blocks(blocks: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(blocks)
    if blocks.numel() == 0:
        return result
    diagonal = torch.diagonal(blocks, dim1=-2, dim2=-1)
    inverse = torch.where(
        diagonal.abs() > 1e-12,
        diagonal.reciprocal(),
        torch.ones_like(diagonal),
    )
    torch.diagonal(result, dim1=-2, dim2=-1).copy_(inverse)
    return result


class _DistributedOperator:
    def __init__(self, process_group=None):
        self.process_group = process_group

    def scalar_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        value = torch.sum(x * y)
        if dist.is_initialized() and dist.get_world_size(self.process_group) > 1:
            dist.all_reduce(
                value, op=dist.ReduceOp.SUM, group=self.process_group
            )
        return value


class DistributedSchurCameraOperator(_DistributedOperator):
    def __init__(
        self,
        *,
        camera_jacobians: torch.Tensor,
        point_jacobians: torch.Tensor,
        camera_global_ids: torch.Tensor,
        point_global_ids: torch.Tensor,
        owned_u_blocks: torch.Tensor,
        owned_v_inverse_blocks: torch.Tensor,
        camera_plan: GatherPlan,
        point_plan: GatherPlan,
        camera_ownership: Ownership,
        point_ownership: Ownership,
    ):
        super().__init__(camera_ownership.process_group)
        self.camera_jacobians = camera_jacobians
        self.point_jacobians = point_jacobians
        self.camera_global_ids = camera_global_ids
        self.point_global_ids = point_global_ids
        self.owned_u_blocks = owned_u_blocks
        self.owned_v_inverse_blocks = owned_v_inverse_blocks
        self.camera_plan = camera_plan
        self.point_plan = point_plan
        self.camera_ownership = camera_ownership
        self.point_ownership = point_ownership

    def __matmul__(self, x_owned: torch.Tensor) -> torch.Tensor:
        return self.matvec(x_owned)

    def matvec(self, x_owned: torch.Tensor) -> torch.Tensor:
        camera_eval = ghost_gather(x_owned, self.camera_plan)
        camera_obs = camera_eval.index_select(
            0, self.camera_plan.observation_positions
        )
        observation_values = torch.einsum(
            "ori,oi->or", self.camera_jacobians, camera_obs
        )

        point_contributions = torch.einsum(
            "ori,or->oi", self.point_jacobians, observation_values
        )
        point_owned = owner_reduce_scatter(
            point_contributions,
            self.point_global_ids,
            self.point_ownership,
        )
        point_scaled_owned = apply_block_matrix(
            self.owned_v_inverse_blocks, point_owned
        )

        point_eval = ghost_gather(point_scaled_owned, self.point_plan)
        point_obs = point_eval.index_select(
            0, self.point_plan.observation_positions
        )
        corrected_observations = torch.einsum(
            "ori,oi->or", self.point_jacobians, point_obs
        )

        camera_contributions = torch.einsum(
            "ori,or->oi", self.camera_jacobians, corrected_observations
        )
        camera_owned = owner_reduce_scatter(
            camera_contributions,
            self.camera_global_ids,
            self.camera_ownership,
        )
        return (
            apply_block_matrix(self.owned_u_blocks, x_owned) - camera_owned
        )


class DistributedBlockDiagonalOperator(_DistributedOperator):
    def __init__(self, blocks: torch.Tensor, process_group=None):
        super().__init__(process_group)
        self.blocks = blocks

    def __matmul__(self, x_owned: torch.Tensor) -> torch.Tensor:
        return self.matvec(x_owned)

    def matvec(self, x_owned: torch.Tensor) -> torch.Tensor:
        return apply_block_matrix(self.blocks, x_owned)
