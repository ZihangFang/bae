from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from ..autograd.function import TrackingTensor
from ..autograd.graph import BSRJacobianData, jacobian_components
from ..utils.linear_operator import (
    ComponentBlockDiagonalPreconditioner,
    ComponentDiagonalPreconditioner,
    ComponentJacobianOperator,
    ComponentNormalMatVec,
)
from ..utils.parameter import parameter_update_shape


def _plain_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.tensor() if isinstance(value, TrackingTensor) else value


class FactorPartition(Protocol):
    """Describe the independent residual-factor dimension of a model call."""

    def factor_count(self, input: Any, target: Any) -> int | None: ...

    def slice(
        self,
        input: Any,
        target: Any,
        start: int,
        end: int,
        factor_count: int,
    ) -> tuple[Any, Any]: ...

    def column_sources(
        self, input: Any, factor_count: int
    ) -> tuple[torch.Tensor, ...]: ...


def _flatten_with_none(value: Any):
    return tree_flatten(value, is_leaf=lambda leaf: leaf is None)


def _normalize_dims(value: Any, dims: Any, name: str):
    value_leaves, value_spec = _flatten_with_none(value)
    if isinstance(dims, (int, type(None))):
        return value_leaves, value_spec, [dims] * len(value_leaves)

    dim_leaves, dim_spec = _flatten_with_none(dims)
    if dim_spec != value_spec:
        raise ValueError(f"{name} must have the same pytree structure as its value")
    return value_leaves, value_spec, dim_leaves


def _slice_along_dim(
    value: torch.Tensor, dim: int, start: int, end: int
) -> torch.Tensor:
    dim = dim if dim >= 0 else value.ndim + dim
    if dim < 0 or dim >= value.ndim:
        raise ValueError(f"Cannot slice dimension {dim} of a {value.ndim}-D tensor")
    index = [slice(None)] * value.ndim
    index[dim] = slice(start, end)
    return value[tuple(index)]


@dataclass(frozen=True)
class PytreePartition:
    """Partition explicitly marked Tensor leaves along factor dimensions.

    ``input_dims`` and ``target_dims`` follow ``torch.vmap``-style pytree
    semantics: an integer marks a factor dimension and ``None`` broadcasts a
    leaf unchanged to every chunk.
    """

    input_dims: Any
    target_dims: Any = None

    def _aligned_leaves(self, input: Any, target: Any):
        input_leaves, input_spec, input_dims = _normalize_dims(
            input, self.input_dims, "input_dims"
        )
        target_leaves, target_spec, target_dims = _normalize_dims(
            target, self.target_dims, "target_dims"
        )
        return (
            input_leaves,
            input_spec,
            input_dims,
            target_leaves,
            target_spec,
            target_dims,
        )

    def factor_count(self, input: Any, target: Any) -> int | None:
        aligned = self._aligned_leaves(input, target)
        leaves = (*aligned[0], *aligned[3])
        dims = (*aligned[2], *aligned[5])
        counts = []
        for leaf, dim in zip(leaves, dims):
            if dim is None:
                continue
            if not isinstance(dim, int):
                raise TypeError("Factor dimensions must be integers or None")
            if not isinstance(leaf, torch.Tensor):
                raise TypeError("Only Tensor leaves may have a factor dimension")
            normalized = dim if dim >= 0 else leaf.ndim + dim
            if normalized < 0 or normalized >= leaf.ndim:
                raise ValueError(
                    f"Factor dimension {dim} is invalid for shape {tuple(leaf.shape)}"
                )
            counts.append(int(leaf.shape[normalized]))

        if not counts:
            return None
        count = counts[0]
        if any(candidate != count for candidate in counts[1:]):
            raise ValueError("All factor-aligned Tensor leaves must have the same size")
        return count

    def slice(
        self,
        input: Any,
        target: Any,
        start: int,
        end: int,
        factor_count: int,
    ) -> tuple[Any, Any]:
        (
            input_leaves,
            input_spec,
            input_dims,
            target_leaves,
            target_spec,
            target_dims,
        ) = self._aligned_leaves(input, target)

        def slice_leaves(leaves, dims):
            return [
                _slice_along_dim(leaf, dim, start, end) if dim is not None else leaf
                for leaf, dim in zip(leaves, dims)
            ]

        return (
            tree_unflatten(slice_leaves(input_leaves, input_dims), input_spec),
            tree_unflatten(slice_leaves(target_leaves, target_dims), target_spec),
        )

    def column_sources(self, input: Any, factor_count: int) -> tuple[torch.Tensor, ...]:
        leaves, _, dims = _normalize_dims(input, self.input_dims, "input_dims")
        return tuple(
            leaf
            for leaf, dim in zip(leaves, dims)
            if (
                dim == 0
                and isinstance(leaf, torch.Tensor)
                and leaf.ndim == 1
                and leaf.shape[0] == factor_count
                and leaf.dtype in (torch.int32, torch.int64)
            )
        )


class LeadingDimensionPartition:
    """Compatibility partition using the previous shared-leading-size rule."""

    @staticmethod
    def factor_count(input: Any, target: Any) -> int | None:
        leaves, _ = tree_flatten((input, target))
        sizes = [
            int(leaf.shape[0])
            for leaf in leaves
            if isinstance(leaf, torch.Tensor) and leaf.ndim > 0
        ]
        return max(sizes) if sizes else None

    @staticmethod
    def slice(
        input: Any,
        target: Any,
        start: int,
        end: int,
        factor_count: int,
    ) -> tuple[Any, Any]:
        def slice_value(value):
            leaves, spec = tree_flatten(value)
            return tree_unflatten(
                [
                    leaf[start:end]
                    if (
                        isinstance(leaf, torch.Tensor)
                        and leaf.ndim > 0
                        and leaf.shape[0] == factor_count
                    )
                    else leaf
                    for leaf in leaves
                ],
                spec,
            )

        return slice_value(input), slice_value(target)

    @staticmethod
    def column_sources(input: Any, factor_count: int) -> tuple[torch.Tensor, ...]:
        leaves, _ = tree_flatten(input)
        return tuple(
            leaf
            for leaf in leaves
            if (
                isinstance(leaf, torch.Tensor)
                and leaf.ndim == 1
                and leaf.shape[0] == factor_count
                and leaf.dtype in (torch.int32, torch.int64)
            )
        )


class _ChunkedComponentAccumulator:
    """Assemble component rows while retaining a fixed-layout fast path."""

    def __init__(
        self,
        residual: torch.Tensor,
        components,
        factor_count: int,
        chunk_factors: int,
        column_sources=(),
    ):
        residual = _plain_tensor(residual)
        if residual.ndim == 0 or residual.shape[0] != chunk_factors:
            raise ValueError(
                "Chunked linearization requires one leading residual block per factor"
            )
        self.factor_count = factor_count
        self.residual = residual.new_empty((factor_count, *residual.shape[1:]))
        self._buffers = []

        for group in components:
            group_buffers = []
            for component in group:
                if component.crow_indices.numel() != chunk_factors + 1:
                    raise ValueError(
                        "A Jacobian component row layout does not match its "
                        "residual chunk"
                    )
                counts = component.crow_indices[1:] - component.crow_indices[:-1]
                blocks_per_factor = int(counts[0].item()) if counts.numel() else 0
                fixed = bool(torch.all(counts == blocks_per_factor).item())
                expected = int(counts.sum().item())
                if (
                    component.col_indices.numel() != expected
                    or component.values.shape[0] != expected
                ):
                    raise ValueError("Jacobian component block counts are inconsistent")

                columns = None
                if fixed and blocks_per_factor == 1:
                    for candidate in column_sources:
                        candidate_chunk = candidate[:chunk_factors]
                        if (
                            candidate_chunk.shape == component.col_indices.shape
                            and candidate_chunk.stride()
                            == component.col_indices.stride()
                            and candidate_chunk.data_ptr()
                            == component.col_indices.data_ptr()
                        ):
                            columns = candidate
                            break
                buffer = {
                    "column_size": component.size[1],
                    "row_block_size": component.values.shape[-2],
                    "value_shape": component.values.shape[1:],
                    "blocks_per_factor": blocks_per_factor,
                    "mode": "fixed" if fixed else "dynamic",
                    "counts": [],
                    "column_chunks": [],
                    "value_chunks": [],
                }
                if fixed:
                    total_blocks = factor_count * blocks_per_factor
                    buffer["owns_columns"] = columns is None
                    buffer["columns"] = (
                        component.col_indices.new_empty(total_blocks)
                        if columns is None
                        else columns
                    )
                    buffer["values"] = component.values.new_empty(
                        (total_blocks, *component.values.shape[1:])
                    )
                group_buffers.append(buffer)
            self._buffers.append(group_buffers)

    @staticmethod
    def _switch_to_dynamic(buffer, completed_factors: int) -> None:
        blocks_per_factor = buffer["blocks_per_factor"]
        completed_blocks = completed_factors * blocks_per_factor
        columns = buffer.pop("columns")
        values = buffer.pop("values")
        buffer.pop("owns_columns")
        if completed_factors:
            buffer["counts"].append(
                columns.new_full((completed_factors,), blocks_per_factor)
            )
            buffer["column_chunks"].append(columns[:completed_blocks].clone())
            buffer["value_chunks"].append(values[:completed_blocks].clone())
        buffer["mode"] = "dynamic"

    def append(self, start: int, end: int, residual: torch.Tensor, components) -> None:
        residual = _plain_tensor(residual)
        chunk_factors = end - start
        if (
            residual.shape[0] != chunk_factors
            or residual.shape[1:] != self.residual.shape[1:]
        ):
            raise ValueError("Residual shape changed between factor chunks")
        self.residual[start:end].copy_(residual)
        if len(components) != len(self._buffers):
            raise ValueError("Jacobian component groups changed between chunks")

        for group, buffers in zip(components, self._buffers):
            if len(group) != len(buffers):
                raise ValueError("Jacobian component count changed between chunks")
            for component, buffer in zip(group, buffers):
                counts = component.crow_indices[1:] - component.crow_indices[:-1]
                expected_blocks = int(counts.sum().item())
                if (
                    component.crow_indices.numel() != chunk_factors + 1
                    or component.size[1] != buffer["column_size"]
                    or component.values.shape[1:] != buffer["value_shape"]
                    or component.col_indices.numel() != expected_blocks
                    or component.values.shape[0] != expected_blocks
                ):
                    raise ValueError("Jacobian component layout changed between chunks")
                blocks_per_factor = buffer["blocks_per_factor"]
                still_fixed = bool(torch.all(counts == blocks_per_factor).item())
                if buffer["mode"] == "fixed" and not still_fixed:
                    self._switch_to_dynamic(buffer, start)

                if buffer["mode"] == "dynamic":
                    buffer["counts"].append(counts)
                    buffer["column_chunks"].append(component.col_indices)
                    buffer["value_chunks"].append(component.values)
                    continue

                block_start = start * blocks_per_factor
                block_end = end * blocks_per_factor
                columns = buffer["columns"]
                values = buffer["values"]
                if buffer["owns_columns"]:
                    columns[block_start:block_end].copy_(component.col_indices)
                else:
                    expected_columns = columns[start:end]
                    if (
                        expected_columns.shape != component.col_indices.shape
                        or expected_columns.stride() != component.col_indices.stride()
                        or expected_columns.data_ptr()
                        != component.col_indices.data_ptr()
                    ):
                        self._switch_to_dynamic(buffer, start)
                        buffer["counts"].append(counts)
                        buffer["column_chunks"].append(component.col_indices)
                        buffer["value_chunks"].append(component.values)
                        continue
                values[block_start:block_end].copy_(component.values)

    def finish(self):
        groups = []
        for buffers in self._buffers:
            group = []
            for buffer in buffers:
                if buffer["mode"] == "fixed":
                    columns = buffer["columns"]
                    values = buffer["values"]
                    crow = torch.arange(
                        self.factor_count + 1,
                        device=columns.device,
                        dtype=columns.dtype,
                    )
                    blocks_per_factor = buffer["blocks_per_factor"]
                    if blocks_per_factor != 1:
                        crow.mul_(blocks_per_factor)
                else:
                    counts = torch.cat(buffer["counts"])
                    if counts.numel() != self.factor_count:
                        raise ValueError(
                            "Jacobian component rows do not cover all factors"
                        )
                    crow = counts.new_zeros(self.factor_count + 1)
                    torch.cumsum(counts, dim=0, out=crow[1:])
                    columns = torch.cat(buffer["column_chunks"])
                    values = torch.cat(buffer["value_chunks"])
                group.append(
                    BSRJacobianData(
                        crow,
                        columns,
                        values,
                        (
                            self.factor_count * buffer["row_block_size"],
                            buffer["column_size"],
                        ),
                    )
                )
            groups.append(tuple(group))
        return self.residual, tuple(groups)


class ComponentLinearization:
    """Local least-squares quadratic model consumed by LM."""

    def __init__(
        self,
        residual: torch.Tensor,
        jacobian: ComponentJacobianOperator,
        parameter_sizes: Sequence[int],
        loss: torch.Tensor,
        diagonal_min: float,
        diagonal_max: float,
    ):
        self.residual = _plain_tensor(residual)
        self.loss = loss
        self.parameter_sizes = tuple(parameter_sizes)
        self.jacobian = jacobian
        self.damping_diagonal = self.jacobian.diagonal().clamp(
            min=diagonal_min, max=diagonal_max
        )
        self.normal = ComponentNormalMatVec(
            self.jacobian,
            damping=0.0,
            diag=self.damping_diagonal,
        )
        self._gradient = None
        self._block_diagonal = None

    @property
    def gradient(self) -> torch.Tensor:
        if self._gradient is None:
            self._gradient = self.jacobian.rmatvec(self.residual.reshape(-1, 1))
        return self._gradient

    def rhs(self) -> torch.Tensor:
        """Return ``-J.T @ residual`` without retaining a gradient copy."""
        return -self.jacobian.rmatvec(self.residual.reshape(-1, 1))

    def predicted_reduction(self, step: torch.Tensor) -> torch.Tensor:
        product = self.jacobian.matvec(step.reshape(-1, 1))
        residual = self.residual.reshape_as(product)
        return -(product.mT @ (2 * residual + product)).squeeze()

    def make_preconditioner(self):
        if float(self.normal.damping) < 1e-4:
            return ComponentDiagonalPreconditioner(self.normal.diagonal())
        if self._block_diagonal is None:
            self._block_diagonal = self.jacobian.block_diagonal()
        return ComponentBlockDiagonalPreconditioner(
            self._block_diagonal,
            self.parameter_sizes,
            self.damping_diagonal,
            self.normal.damping,
        )


class ComponentLinearizer:
    """Build component-Jacobian linearizations over residual-factor chunks."""

    def __init__(
        self,
        chunk_size: int | None = 250_000,
        *,
        partition: FactorPartition | None = None,
        compile: bool = False,
    ):
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive or None")
        self.chunk_size = chunk_size
        self.partition = LeadingDimensionPartition() if partition is None else partition
        self.compile = compile
        self._compiled_evaluator = None
        self._compiled_key = None

    def _evaluator(self, model, params):
        def residual_and_components(input, target):
            residuals = model(input, target)
            if len(residuals) != 1:
                raise ValueError(
                    "ComponentLinearizer currently requires one residual Tensor"
                )
            residual = residuals[0]
            return residual, jacobian_components(residual, params)

        if not self.compile:
            return residual_and_components

        key = (id(model), tuple(id(param) for param in params))
        if self._compiled_evaluator is None:
            from ..utils.pypose_ambient_grad import (
                install_pypose_ambient_grad_monkeypatch,
            )

            install_pypose_ambient_grad_monkeypatch()
            first_parameter = params[0]
            backend = "inductor" if first_parameter.is_cuda else "eager"
            self._compiled_evaluator = torch.compile(
                residual_and_components,
                backend=backend,
                fullgraph=True,
            )
            self._compiled_key = key
        elif self._compiled_key != key:
            raise RuntimeError(
                "A compiled ComponentLinearizer cannot be shared between "
                "different models or parameter sets"
            )
        return self._compiled_evaluator

    def evaluate(self, model, input, target, params):
        factor_count = self.partition.factor_count(input, target)
        evaluate = self._evaluator(model, params)
        if (
            self.chunk_size is None
            or factor_count is None
            or factor_count <= self.chunk_size
        ):
            residual, components = evaluate(input, target)
            return _plain_tensor(residual), components

        accumulator = None
        column_sources = self.partition.column_sources(input, factor_count)
        for start in range(0, factor_count, self.chunk_size):
            end = min(start + self.chunk_size, factor_count)
            chunk_input, chunk_target = self.partition.slice(
                input, target, start, end, factor_count
            )
            residual, components = evaluate(chunk_input, chunk_target)
            if accumulator is None:
                accumulator = _ChunkedComponentAccumulator(
                    residual,
                    components,
                    factor_count,
                    end - start,
                    column_sources,
                )
            accumulator.append(start, end, residual, components)

        if accumulator is None:
            raise ValueError("Chunked linearization received no residual factors")
        return accumulator.finish()

    def linearize(
        self,
        model,
        input,
        target,
        params,
        *,
        diagonal_min: float,
        diagonal_max: float,
    ) -> ComponentLinearization:
        residual, components = self.evaluate(model, input, target, params)
        loss = model.kernel[0](residual.square().sum(-1)).sum()
        parameter_sizes = tuple(
            torch.Size(parameter_update_shape(param)).numel() for param in params
        )
        jacobian = ComponentJacobianOperator(
            components,
            parameter_sizes,
            residual.numel(),
        )
        # Component operators retain value/column views, not BSR row
        # pointers. Drop the wrappers before allocating diagonal workspaces.
        del components
        return ComponentLinearization(
            residual,
            jacobian,
            parameter_sizes,
            loss,
            diagonal_min,
            diagonal_max,
        )

    def loss(self, model, input, target):
        factor_count = self.partition.factor_count(input, target)
        if (
            self.chunk_size is None
            or factor_count is None
            or factor_count <= self.chunk_size
        ):
            return model.loss(input, target)

        loss = None
        for start in range(0, factor_count, self.chunk_size):
            end = min(start + self.chunk_size, factor_count)
            chunk_input, chunk_target = self.partition.slice(
                input, target, start, end, factor_count
            )
            chunk_loss = model.loss(chunk_input, chunk_target)
            loss = chunk_loss if loss is None else loss + chunk_loss
        if loss is None:
            raise ValueError("Chunked loss received no residual factors")
        return loss
