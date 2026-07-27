"""DTensor sparse-Jacobian tracing context."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from itertools import count
from typing import Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ..autograd.trace import has_trace, set_sidecar_trace

try:
    from torch.distributed.tensor import DTensor, Shard
except ImportError:  # pragma: no cover - supported PyTorch versions provide DTensor
    DTensor = ()
    Shard = ()


@dataclass(frozen=True)
class DistributedParameterMetadata:
    key: int
    parameter: DTensor
    mesh: object
    placement: object
    global_shape: torch.Size
    local_start: int
    local_count: int
    process_group: object
    ltype: object = None
    trim_se3_grad: bool = False


_NEXT_PARAMETER_KEY = count(1)
_PARAMETERS_BY_ID: dict[int, DistributedParameterMetadata] = {}
_PARAMETERS_BY_KEY: dict[int, DistributedParameterMetadata] = {}
_PENDING_FINGERPRINTS: dict[tuple, dict[str, object]] = {}
_ACTIVE_CONTEXT: ContextVar[Optional["DistributedTraceContext"]] = ContextVar(
    "bae_distributed_trace_context", default=None
)


def _local_storage_identity(tensor: DTensor) -> tuple:
    local = tensor.to_local()
    try:
        pointer = local.untyped_storage().data_ptr()
    except RuntimeError:
        pointer = id(local)
    return (
        pointer,
        tuple(tensor.shape),
        tuple(str(placement) for placement in tensor.placements),
    )


def register_pending_parameter(tensor: DTensor) -> None:
    """Remember a DTensor passed through ``pp.Parameter(..., sjac=True)``.

    ``nn.Parameter`` detaches custom tensor subclasses, so the final DTensor is
    a distinct Python object which shares the same local storage.
    """

    if not isinstance(tensor, DTensor):
        return
    _PENDING_FINGERPRINTS[_local_storage_identity(tensor)] = {
        "ltype": getattr(tensor, "ltype", None),
        "trim_se3_grad": bool(getattr(tensor, "trim_SE3_grad", False)),
    }


def _was_pending_parameter(tensor: object) -> bool:
    return (
        isinstance(tensor, DTensor)
        and _local_storage_identity(tensor) in _PENDING_FINGERPRINTS
    )


def _shard_bounds(parameter: DTensor) -> tuple[int, int]:
    if parameter.device_mesh.ndim != 1:
        raise ValueError("Distributed Schur requires a one-dimensional DeviceMesh.")
    if len(parameter.placements) != 1 or not isinstance(parameter.placements[0], Shard):
        raise ValueError("Distributed Schur parameters must use placements=[Shard(0)].")
    placement = parameter.placements[0]
    if placement.dim != 0:
        raise ValueError("Distributed Schur parameters must be sharded on dimension 0.")

    mesh = parameter.device_mesh
    rank = mesh.get_local_rank()
    world_size = mesh.size()
    local_count, local_start = placement._local_shard_size_and_offset(
        parameter.shape[0], world_size, rank
    )
    return int(local_start or 0), int(local_count)


def register_parameter(parameter: DTensor) -> DistributedParameterMetadata:
    if not isinstance(parameter, DTensor):
        raise TypeError(f"Expected a DTensor parameter, got {type(parameter)!r}")
    existing = _PARAMETERS_BY_ID.get(id(parameter))
    if existing is not None and existing.parameter is parameter:
        return existing

    local_start, local_count = _shard_bounds(parameter)
    fingerprint = _local_storage_identity(parameter)
    pending = _PENDING_FINGERPRINTS.pop(fingerprint, {})
    metadata = DistributedParameterMetadata(
        key=next(_NEXT_PARAMETER_KEY),
        parameter=parameter,
        mesh=parameter.device_mesh,
        placement=parameter.placements[0],
        global_shape=parameter.shape,
        local_start=local_start,
        local_count=local_count,
        process_group=parameter.device_mesh.get_group(),
        ltype=getattr(parameter, "ltype", pending.get("ltype")),
        trim_se3_grad=bool(
            getattr(
                parameter,
                "trim_SE3_grad",
                pending.get("trim_se3_grad", False),
            )
        ),
    )
    _PARAMETERS_BY_ID[id(parameter)] = metadata
    _PARAMETERS_BY_KEY[metadata.key] = metadata
    return metadata


def is_registered_parameter(tensor: object) -> bool:
    entry = _PARAMETERS_BY_ID.get(id(tensor))
    if entry is not None and entry.parameter is tensor:
        return True
    return _was_pending_parameter(tensor)


def parameter_metadata(parameter: DTensor) -> DistributedParameterMetadata:
    entry = _PARAMETERS_BY_ID.get(id(parameter))
    if entry is not None and entry.parameter is parameter:
        return entry
    if _was_pending_parameter(parameter):
        return register_parameter(parameter)
    raise RuntimeError(
        "DTensor is not registered as a distributed sparse-Jacobian parameter. "
        "Construct it with pp.Parameter(..., sjac=True) and enter "
        "DistributedTraceContext."
    )


def parameter_metadata_by_key(key: int) -> DistributedParameterMetadata:
    try:
        return _PARAMETERS_BY_KEY[int(key)]
    except KeyError as error:
        raise RuntimeError(f"Unknown distributed parameter key {key}") from error


class DistributedTraceContext:
    """Registers DTensor leaves and owns their dispatch mode."""

    def __init__(self, parameters=()):
        self.parameters = tuple(parameters)
        self.metadata = tuple(
            register_parameter(parameter)
            for parameter in self.parameters
            if isinstance(parameter, DTensor)
        )
        self.mode = DistributedTraceMode(self)
        self._token = None

    def contains(self, tensor: object) -> bool:
        entry = _PARAMETERS_BY_ID.get(id(tensor))
        return (
            entry is not None
            and entry.parameter is tensor
            and any(metadata is entry for metadata in self.metadata)
        )

    def __enter__(self):
        self._token = _ACTIVE_CONTEXT.set(self)
        self.mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self.mode.__exit__(exc_type, exc_value, traceback)
        finally:
            if self._token is not None:
                _ACTIVE_CONTEXT.reset(self._token)
                self._token = None


class DistributedTraceMode(TorchDispatchMode):
    """Replaces global indexing of registered sharded parameters."""

    def __init__(self, context: DistributedTraceContext):
        super().__init__()
        self.context = context
        self.seen_distributed_index = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (
            func is torch.ops.aten.index.Tensor
            and args
            and isinstance(args[0], DTensor)
            and self.context.contains(args[0])
        ):
            parameter, tensor_indices = args[:2]
            if (
                len(tensor_indices) != 1
                or not isinstance(tensor_indices[0], torch.Tensor)
            ):
                raise NotImplementedError(
                    "Distributed sparse-Jacobian parameters support tensor indexing "
                    "on dimension 0 only."
                )
            from .ops import distributed_index

            index = tensor_indices[0]
            result = distributed_index(parameter, index)
            set_sidecar_trace(result, ("index", index, parameter))
            self.seen_distributed_index = True
            return result

        result = func(*args, **kwargs)
        if func is torch.ops.aten.index.Tensor and args and has_trace(args[0]):
            tensor_indices = args[1]
            if len(tensor_indices) == 1 and isinstance(
                tensor_indices[0], torch.Tensor
            ):
                set_sidecar_trace(
                    result, ("index", tensor_indices[0], args[0])
                )
            return result

        map_functions = {
            torch.ops.aten.add.Tensor,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.div.Tensor,
        }
        if func in map_functions and any(
            isinstance(arg, torch.Tensor) and has_trace(arg)
            for arg in args
        ):
            from ..autograd.function import _compact_map_arg

            compact_args = tuple(_compact_map_arg(arg) for arg in args)
            set_sidecar_trace(result, ("map", func, compact_args))
            return result

        if func is torch.ops.aten.cat.default:
            tensors = args[0]
            if any(has_trace(tensor) for tensor in tensors):
                dimension = kwargs.get("dim", args[1] if len(args) > 1 else 0)
                if dimension != 0:
                    raise NotImplementedError(
                        "Distributed tracing supports torch.cat(..., dim=0) only."
                    )
                tracked = []
                offset = 0
                for tensor in tensors:
                    end = offset + tensor.shape[0]
                    if has_trace(tensor) or (
                        isinstance(tensor, DTensor)
                        and is_registered_parameter(tensor)
                    ):
                        tracked.append((offset, end, tensor))
                    offset = end
                set_sidecar_trace(result, ("cat", 0, tuple(tracked)))
        return result


def active_trace_context() -> Optional[DistributedTraceContext]:
    return _ACTIVE_CONTEXT.get()
