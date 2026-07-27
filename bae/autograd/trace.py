"""Shared sparse-Jacobian trace access.

Ordinary ``TrackingTensor`` instances keep their trace as a Python attribute,
as they historically have.  Tensor subclasses which must retain their own
dispatch semantics (notably DTensor) use the identity sidecar below instead.
"""

from __future__ import annotations

import weakref
from typing import Any

import torch


_TRACE_SIDECAR: dict[int, tuple[weakref.ReferenceType[torch.Tensor], Any]] = {}


def _remove_sidecar_entry(tensor_id: int, reference) -> None:
    current = _TRACE_SIDECAR.get(tensor_id)
    if current is not None and current[0] is reference:
        _TRACE_SIDECAR.pop(tensor_id, None)


def _sidecar_trace(tensor: torch.Tensor):
    entry = _TRACE_SIDECAR.get(id(tensor))
    if entry is None or entry[0]() is not tensor:
        return None
    return entry[1]


def has_trace(tensor: object) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    if hasattr(tensor, "optrace"):
        return True
    if not getattr(tensor, "_bae_has_sidecar_trace", False):
        return False
    return _sidecar_trace(tensor) is not None


def get_trace(tensor: torch.Tensor):
    if hasattr(tensor, "optrace"):
        return tensor.optrace
    if not getattr(tensor, "_bae_has_sidecar_trace", False):
        raise AttributeError(f"{type(tensor).__name__} has no sparse-Jacobian trace")
    trace = _sidecar_trace(tensor)
    if trace is None:
        raise AttributeError(f"{type(tensor).__name__} has no sparse-Jacobian trace")
    return trace


def set_trace(tensor: torch.Tensor, trace):
    # Keep the established attribute path for ordinary tensor subclasses.
    # DTensor and plain outputs produced by distributed dispatch use a sidecar.
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()
    if (
        not isinstance(tensor, DTensor)
        and not getattr(tensor, "_bae_distributed_trace", False)
    ):
        try:
            tensor.optrace = trace
            return tensor
        except (AttributeError, RuntimeError):
            pass

    return set_sidecar_trace(tensor, trace)


def set_sidecar_trace(tensor: torch.Tensor, trace):
    tensor._bae_distributed_trace = True
    # Dynamo can model Tensor attributes but intentionally rejects ``id`` for
    # sourceless graph tensors. The trace is compile-time metadata in this
    # case; real eager distributed tensors continue to use the sidecar.
    if torch.compiler.is_compiling():
        tensor.optrace = trace
        return tensor
    tensor_id = id(tensor)
    tensor._bae_has_sidecar_trace = True
    reference = weakref.ref(
        tensor,
        lambda ref, tensor_id=tensor_id: _remove_sidecar_entry(tensor_id, ref),
    )
    _TRACE_SIDECAR[tensor_id] = (reference, trace)
    return tensor


def is_traced_parameter(tensor: object) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()

    # Import the registry lazily and only for an actual DTensor. Calling
    # ``id`` on sourceless graph tensors is unsupported by Dynamo.
    if isinstance(tensor, DTensor):
        from bae.distributed.context import is_registered_parameter

        if is_registered_parameter(tensor):
            return True

    return isinstance(tensor, torch.nn.Parameter)
