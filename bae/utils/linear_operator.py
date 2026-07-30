from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Any, Tuple, Sequence

import torch
from torch import Tensor


@dataclass
class _ComponentTerm:
    values: Tensor
    columns: Tensor
    rows: Optional[Tensor]
    row_count: int
    row_block_size: int
    column_count: int
    column_block_size: int

    @classmethod
    def from_data(cls, component) -> "_ComponentTerm":
        values = component.values
        if values.ndim != 3:
            raise ValueError(
                "Jacobian component values must have shape "
                "(nnz_blocks, row_block_size, column_block_size)."
            )
        row_block_size, column_block_size = values.shape[-2:]
        row_count = component.size[0] // row_block_size
        column_count = component.size[1] // column_block_size
        crow = component.crow_indices
        if crow.numel() != row_count + 1:
            raise ValueError("Jacobian component crow length does not match its size.")

        counts = crow[1:] - crow[:-1]
        implicit_rows = (
            values.shape[0] == row_count
            and bool(torch.all(counts == 1).item())
        )
        rows = None
        if not implicit_rows:
            row_ids = torch.arange(
                row_count, device=crow.device, dtype=torch.int64
            )
            rows = torch.repeat_interleave(
                row_ids, counts.to(torch.int64)
            )

        return cls(
            values=values,
            columns=component.col_indices.to(torch.int64),
            rows=rows,
            row_count=row_count,
            row_block_size=row_block_size,
            column_count=column_count,
            column_block_size=column_block_size,
        )

    def add_forward(
        self,
        blocks: Tensor,
        result: Tensor,
        chunk_size: int = 1_000_000,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        for start in range(0, self.values.shape[0], chunk_size):
            end = min(start + chunk_size, self.values.shape[0])
            selected = blocks.index_select(
                0, self.columns[start:end]
            )
            contribution = torch.bmm(
                self.values[start:end], selected.unsqueeze(-1)
            ).squeeze(-1)
            if self.rows is None:
                result[start:end].add_(contribution)
            else:
                result.index_add_(
                    0, self.rows[start:end], contribution
                )

    def forward(
        self, blocks: Tensor, chunk_size: int = 1_000_000
    ) -> Tensor:
        result = self.values.new_zeros(
            (self.row_count, self.row_block_size)
        )
        self.add_forward(blocks, result, chunk_size)
        return result

    def add_adjoint(
        self,
        output_blocks: Tensor,
        result: Tensor,
        chunk_size: int = 1_000_000,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        for start in range(0, self.values.shape[0], chunk_size):
            end = min(start + chunk_size, self.values.shape[0])
            selected = (
                output_blocks[start:end]
                if self.rows is None
                else output_blocks.index_select(
                    0, self.rows[start:end]
                )
            )
            contribution = torch.bmm(
                self.values[start:end].transpose(-2, -1),
                selected.unsqueeze(-1),
            ).squeeze(-1)
            result.index_add_(
                0, self.columns[start:end], contribution
            )

    def adjoint(
        self, output_blocks: Tensor, chunk_size: int = 1_000_000
    ) -> Tensor:
        result = self.values.new_zeros(
            (self.column_count, self.column_block_size)
        )
        self.add_adjoint(output_blocks, result, chunk_size)
        return result

    def column_square_sum(self, chunk_size: int = 65536) -> Tensor:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        result = self.values.new_zeros(
            (self.column_count, self.column_block_size)
        )
        for start in range(0, self.values.shape[0], chunk_size):
            end = min(start + chunk_size, self.values.shape[0])
            contribution = self.values[start:end].square().sum(dim=-2)
            result.index_add_(
                0, self.columns[start:end], contribution
            )
        return result


class ComponentJacobianOperator:
    r"""A block-component Jacobian without a concatenated scalar sparse matrix.

    ``components`` is the nested result of
    :func:`bae.autograd.graph.jacobian_components`. Parameter vectors are
    packed in the same order as the outer component sequence.
    """

    def __init__(
        self,
        components: Sequence[Sequence],
        parameter_sizes: Sequence[int],
        output_size: int,
    ):
        if len(components) != len(parameter_sizes):
            raise ValueError(
                "components and parameter_sizes must have the same length"
            )
        self.parameter_sizes = tuple(int(size) for size in parameter_sizes)
        self.output_size = int(output_size)
        self.groups = tuple(
            tuple(_ComponentTerm.from_data(component) for component in group)
            for group in components
        )

        first = next(
            (
                term
                for group in self.groups
                for term in group
            ),
            None,
        )
        if first is None:
            raise ValueError("At least one Jacobian component is required")
        self.device = first.values.device
        self.dtype = first.values.dtype
        self.row_block_size = first.row_block_size
        if self.output_size % self.row_block_size:
            raise ValueError("output_size is not divisible by the row block size")
        self.row_count = self.output_size // self.row_block_size
        for parameter_size, group in zip(self.parameter_sizes, self.groups):
            for term in group:
                if (
                    term.row_count != self.row_count
                    or term.row_block_size != self.row_block_size
                ):
                    raise ValueError(
                        "All Jacobian components must share one output block layout"
                    )
                if term.column_count * term.column_block_size != parameter_size:
                    raise ValueError(
                        "A Jacobian component does not match its packed parameter size"
                    )

        self.shape = (self.output_size, sum(self.parameter_sizes))
        self.ndim = 2
        self._diagonal: Optional[Tensor] = None
        self._block_diagonal: Optional[tuple[Tensor, ...]] = None

    @staticmethod
    def _as_vector(
        value: Tensor, expected: int, name: str
    ) -> tuple[Tensor, bool]:
        was_column = value.ndim == 2 and value.shape[-1] == 1
        if was_column:
            value = value.squeeze(-1)
        if value.ndim != 1 or value.numel() != expected:
            raise ValueError(f"{name} must contain {expected} scalar values")
        return value, was_column

    def matvec(self, x: Tensor) -> Tensor:
        x, was_column = self._as_vector(x, self.shape[1], "x")
        output = x.new_zeros((self.row_count, self.row_block_size))
        offset = 0
        for parameter_size, group in zip(self.parameter_sizes, self.groups):
            parameter = x[offset : offset + parameter_size]
            offset += parameter_size
            for term in group:
                blocks = parameter.view(
                    term.column_count, term.column_block_size
                )
                term.add_forward(blocks, output)
        output = output.reshape(-1)
        return output.unsqueeze(-1) if was_column else output

    def rmatvec(self, y: Tensor) -> Tensor:
        y, was_column = self._as_vector(y, self.shape[0], "y")
        output_blocks = y.view(self.row_count, self.row_block_size)
        chunks = []
        for parameter_size, group in zip(self.parameter_sizes, self.groups):
            if not group:
                chunks.append(y.new_zeros(parameter_size))
                continue
            result = y.new_zeros(parameter_size)
            for term in group:
                term.add_adjoint(
                    output_blocks,
                    result.view(
                        term.column_count, term.column_block_size
                    ),
                )
            chunks.append(result)
        result = torch.cat(chunks)
        return result.unsqueeze(-1) if was_column else result

    def diagonal(self, chunk_size: int = 65536) -> Tensor:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self._diagonal is None:
            chunks = []
            for parameter_size, group in zip(
                self.parameter_sizes, self.groups
            ):
                if not group:
                    chunks.append(
                        torch.zeros(
                            parameter_size,
                            device=self.device,
                            dtype=self.dtype,
                        )
                    )
                    continue
                result = None
                for term in group:
                    contribution = term.column_square_sum(
                        chunk_size
                    ).reshape(-1)
                    result = (
                        contribution
                        if result is None
                        else result + contribution
                    )
                chunks.append(result)
            self._diagonal = torch.cat(chunks)
        return self._diagonal

    def block_diagonal(
        self, chunk_size: int = 65536
    ) -> tuple[Tensor, ...]:
        """Return parameter-block diagonal terms of ``J.T @ J``.

        Chunking bounds the temporary outer-product storage. Multiple
        components for one parameter may contain cross terms, so those groups
        conservatively use scalar diagonal blocks.
        """

        if self._block_diagonal is not None:
            return self._block_diagonal
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        scalar_diagonal = self.diagonal()
        diagonal_offset = 0
        result = []
        for parameter_size, group in zip(
            self.parameter_sizes, self.groups
        ):
            parameter_diagonal = scalar_diagonal[
                diagonal_offset : diagonal_offset + parameter_size
            ]
            diagonal_offset += parameter_size
            if len(group) != 1:
                result.append(parameter_diagonal.view(-1, 1, 1))
                continue

            term = group[0]
            blocks = term.values.new_zeros(
                (
                    term.column_count,
                    term.column_block_size,
                    term.column_block_size,
                )
            )
            for start in range(0, term.values.shape[0], chunk_size):
                end = min(start + chunk_size, term.values.shape[0])
                values = term.values[start:end]
                gram = torch.bmm(values.transpose(-2, -1), values)
                blocks.index_add_(0, term.columns[start:end], gram)
            result.append(blocks)
        self._block_diagonal = tuple(result)
        return self._block_diagonal

    def __call__(self, x: Tensor) -> Tensor:
        return self.matvec(x)

    def __matmul__(self, x: Tensor) -> Tensor:
        return self.matvec(x)


class ComponentNormalMatVec:
    """Matrix-free ``J.T @ J`` over :class:`ComponentJacobianOperator`."""

    def __init__(
        self,
        jacobian: ComponentJacobianOperator,
        damping: Union[float, Tensor] = 0.0,
        diag: Optional[Tensor] = None,
    ):
        self.jacobian = jacobian
        self.device = jacobian.device
        self.dtype = jacobian.dtype
        self.shape = (jacobian.shape[1], jacobian.shape[1])
        self.ndim = 2
        self._diag = jacobian.diagonal() if diag is None else diag
        if self._diag.ndim != 1 or self._diag.numel() != self.shape[0]:
            raise ValueError("diag must match the packed parameter vector")
        self.set_damping(damping)

    def set_damping(self, damping: Union[float, Tensor]) -> None:
        self.damping = damping

    def diagonal(self) -> Tensor:
        return self._diag * (1.0 + self.damping)

    def matvec(self, x: Tensor) -> Tensor:
        result = self.jacobian.rmatvec(self.jacobian.matvec(x))
        if isinstance(self.damping, Tensor):
            has_damping = bool(self.damping.ne(0).item())
        else:
            has_damping = self.damping != 0
        if has_damping:
            diagonal = (
                self._diag.unsqueeze(-1)
                if x.ndim == 2
                else self._diag
            )
            result = result + self.damping * diagonal * x
        return result

    def scalar_inner(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.sum(x * y)

    def __call__(self, x: Tensor) -> Tensor:
        return self.matvec(x)

    def __matmul__(self, x: Tensor) -> Tensor:
        return self.matvec(x)


class ComponentDiagonalPreconditioner:
    """Owner/local diagonal inverse used by the operator-hook PCG route."""

    def __init__(self, diagonal: Tensor):
        safe = diagonal.clone()
        safe[safe.abs() < 1e-6] = 1e-6
        self.inverse = safe.reciprocal()
        self.shape = (safe.numel(), safe.numel())

    def matvec(self, x: Tensor) -> Tensor:
        inverse = (
            self.inverse.unsqueeze(-1)
            if x.ndim == 2
            else self.inverse
        )
        return inverse * x

    def __matmul__(self, x: Tensor) -> Tensor:
        return self.matvec(x)


class ComponentBlockDiagonalPreconditioner:
    """Dense parameter-block Jacobi preconditioner."""

    def __init__(
        self,
        block_diagonal: Sequence[Tensor],
        parameter_sizes: Sequence[int],
        damping_diagonal: Tensor,
        damping: Union[float, Tensor],
    ):
        if len(block_diagonal) != len(parameter_sizes):
            raise ValueError(
                "block_diagonal and parameter_sizes must have the same length"
            )
        self.parameter_sizes = tuple(int(size) for size in parameter_sizes)
        self.inverse_blocks = []
        offset = 0
        for blocks, parameter_size in zip(
            block_diagonal, self.parameter_sizes
        ):
            block_size = blocks.shape[-1]
            if (
                blocks.ndim != 3
                or blocks.shape[-2] != block_size
                or blocks.shape[0] * block_size != parameter_size
            ):
                raise ValueError(
                    "A block diagonal group does not match its parameter size"
                )
            effective = blocks.clone()
            diagonal = damping_diagonal[
                offset : offset + parameter_size
            ].view(blocks.shape[0], block_size)
            torch.diagonal(
                effective, dim1=-2, dim2=-1
            ).add_(damping * diagonal)
            self.inverse_blocks.append(torch.linalg.inv(effective))
            offset += parameter_size
        self.inverse_blocks = tuple(self.inverse_blocks)
        self.shape = (
            sum(self.parameter_sizes),
            sum(self.parameter_sizes),
        )

    def matvec(self, x: Tensor) -> Tensor:
        was_column = x.ndim == 2 and x.shape[-1] == 1
        vector = x.squeeze(-1) if was_column else x
        if vector.ndim != 1 or vector.numel() != self.shape[0]:
            raise ValueError("x does not match the preconditioner shape")
        chunks = []
        offset = 0
        for inverse, parameter_size in zip(
            self.inverse_blocks, self.parameter_sizes
        ):
            blocks = vector[
                offset : offset + parameter_size
            ].view(inverse.shape[0], inverse.shape[-1])
            chunks.append(
                torch.bmm(inverse, blocks.unsqueeze(-1))
                .squeeze(-1)
                .reshape(-1)
            )
            offset += parameter_size
        result = torch.cat(chunks)
        return result.unsqueeze(-1) if was_column else result

    def __matmul__(self, x: Tensor) -> Tensor:
        return self.matvec(x)


class NormalMatVec:
    r"""Matrix-free normal-equation linear operator.

    Given a Jacobian J (dense, CSR/COO, or BSR), this operator represents:

        A x = J^T (J x) + damping * diag(J^T J) * x

    where diag(J^T J) is computed as the column-wise sum of squares of J.
    """

    def __init__(
        self,
        J: Tensor,
        damping: Union[float, Tensor] = 0.0,
        diag: Optional[Tensor] = None,
    ):
        if not torch.is_tensor(J):
            raise TypeError("J must be a torch.Tensor")
        if J.ndim != 2:
            raise ValueError("J must be 2-D")

        self.J: Tensor = J
        self._Jt: Optional[Tensor] = None

        self.device = J.device
        self.dtype = J.dtype
        self.layout = J.layout if J.layout != torch.sparse_coo else torch.sparse_csr

        self.shape: Tuple[int, int] = (J.shape[1], J.shape[1])
        self.ndim: int = 2

        self._diag: Tensor = diag if diag is not None else self._compute_diag(J)
        if self._diag.ndim != 1 or self._diag.numel() != J.shape[1]:
            raise ValueError("diag must be 1-D with length equal to J.shape[1]")

        self.set_damping(damping)

    def set_damping(self, damping: Union[float, Tensor]) -> None:
        if isinstance(damping, Tensor):
            if damping.numel() != 1:
                raise ValueError("damping tensor must be scalar")
            self.damping = damping.to(device=self.device, dtype=self.dtype)
        else:
            self.damping = float(damping)

    def diagonal(self) -> Tensor:
        damp = self._damping_value()
        if damp is None:
            return self._diag
        return self._diag * (1.0 + damp)

    def matvec(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x2d = x.unsqueeze(-1)
        elif x.ndim == 2:
            x2d = x
        else:
            raise ValueError("x must be 1-D or 2-D")

        if x2d.device != self.device or x2d.dtype != self.dtype:
            x2d = x2d.to(device=self.device, dtype=self.dtype)

        y = self.J @ x2d
        Jt = self._get_Jt()
        z = Jt @ y

        damp = self._damping_value()
        if damp is not None:
            z = z + damp * self._diag.unsqueeze(-1) * x2d

        return z.squeeze(-1) if x.ndim == 1 else z

    def __call__(self, x: Tensor) -> Tensor:
        return self.matvec(x)

    def __matmul__(self, x: Tensor) -> Tensor:
        return self.matvec(x)

    @classmethod
    def __torch_function__(  # type: ignore[override]
        cls, func: Any, types: Any, args: Tuple[Any, ...] = (), kwargs: Optional[dict] = None
    ):
        if kwargs is None:
            kwargs = {}
        if func is torch.matmul and len(args) >= 2:
            A, B = args[0], args[1]
            if isinstance(A, cls):
                out = kwargs.get("out", None)
                result = A.matvec(B)
                if out is not None:
                    out_tensor = out[0] if isinstance(out, tuple) else out
                    out_tensor.copy_(result)
                    return out_tensor
                return result
        return NotImplemented

    def _get_Jt(self) -> Tensor:
        if self._Jt is None:
            Jt = self.J.mT
            if Jt.layout == torch.sparse_csc:
                Jt = Jt.to_sparse_csr()
            elif Jt.layout == torch.sparse_bsc:
                bs = Jt.values().shape[-2:]
                Jt = Jt.to_sparse_bsr(blocksize=bs)
            self._Jt = Jt
        return self._Jt

    def _damping_value(self) -> Optional[Tensor]:
        if isinstance(self.damping, Tensor):
            if self.damping.item() == 0.0:
                return None
            return self.damping
        if self.damping == 0.0:
            return None
        return torch.tensor(self.damping, device=self.device, dtype=self.dtype)

    @staticmethod
    def _compute_diag(J: Tensor) -> Tensor:
        if J.layout == torch.strided:
            return J.square().sum(dim=0)

        if J.layout == torch.sparse_bsr:
            values = J.values()
            dm, dn = values.shape[-2], values.shape[-1]
            col_blocks = J.col_indices()
            contrib = values.square().sum(dim=-2)  # (nnz_blocks, dn)
            offsets = torch.arange(dn, device=contrib.device, dtype=col_blocks.dtype)
            cols = (col_blocks[:, None] * dn + offsets[None, :]).reshape(-1).to(torch.int64)
            contrib_flat = contrib.reshape(-1)
            diag = torch.zeros(J.shape[1], device=contrib.device, dtype=contrib.dtype)
            diag.scatter_add_(0, cols, contrib_flat)
            return diag

        if J.layout == torch.sparse_csr:
            values = J.values()
            col = J.col_indices().to(torch.int64)
            v2 = values.square()
            if v2.ndim > 1:
                v2 = v2.reshape(v2.shape[0], -1).sum(dim=-1)
            diag = torch.zeros(J.shape[1], device=values.device, dtype=v2.dtype)
            diag.scatter_add_(0, col, v2)
            return diag

        if J.layout == torch.sparse_coo:
            Jc = J.coalesce()
            col = Jc.indices()[1].to(torch.int64)
            v2 = Jc.values().square()
            if v2.ndim > 1:
                v2 = v2.reshape(v2.shape[0], -1).sum(dim=-1)
            diag = torch.zeros(J.shape[1], device=Jc.device, dtype=v2.dtype)
            diag.scatter_add_(0, col, v2)
            return diag

        raise NotImplementedError(f"Unsupported J layout: {J.layout}")
