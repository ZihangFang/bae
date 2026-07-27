import torch
import numpy as np
import pypose as pp
from functools import wraps
from torch.utils._pytree import tree_map

from .trace import get_trace, has_trace, is_traced_parameter, set_trace

try:
    from torch.distributed.tensor import DTensor as _DTensor
except ImportError:
    _DTensor = ()

WHITELISTED_MAPS = tuple(
    func for func in (
        torch._C.TensorBase.__add__,
        torch._C.TensorBase.__sub__,
        torch._C.TensorBase.__mul__,
        getattr(torch._C.TensorBase, "__div__", None),
        torch._C.TensorBase.add,
        torch._C.TensorBase.sub,
        torch._C.TensorBase.mul,
    ) if func is not None
)

_LTYPE_PRESERVING_FUNCS = {
    torch._C.TensorBase.__getitem__,
    torch.cat,
    *WHITELISTED_MAPS,
    torch._C.TensorBase.clone,
    torch._C.TensorBase.to,
}

_INDEXED_TRACE_TAG = "bae.indexed"


def _iter_tracked_tensors(values):
    if isinstance(values, torch.Tensor):
        yield values
    elif isinstance(values, dict):
        for value in values.values():
            yield from _iter_tracked_tensors(value)
    elif isinstance(values, (tuple, list)):
        for value in values:
            yield from _iter_tracked_tensors(value)


def _attach_index_trace(result, index, tensor):
    if isinstance(result, TrackingTensor):
        result.optrace = ("index", index, tensor)
        return result
    return set_trace(result, ("index", index, tensor))


def _attach_cat_trace(result, tensors, dim):
    tracked = []
    offset = 0
    for t in tensors:
        n = t.shape[0]
        if (
            has_trace(t)
            or isinstance(t, TrackingTensor)
            or is_traced_parameter(t)
        ):
            tracked.append((offset, offset + n, t))
        offset += n
    trace = ("cat", dim, tuple(tracked))
    if isinstance(result, TrackingTensor):
        result.optrace = trace
        return result
    return set_trace(result, trace)


def _attach_map_trace(result, func, args):
    compact_args = tuple(_compact_map_arg(arg) for arg in args)
    trace = ("map", func, compact_args)
    if isinstance(result, TrackingTensor):
        result.optrace = trace
        return result
    return set_trace(result, trace)


def _compact_map_arg(arg):
    if isinstance(arg, torch.Tensor) and has_trace(arg):
        trace = get_trace(arg)
        if trace[0] == "index":
            return (_INDEXED_TRACE_TAG, trace[2], trace[1])
    return arg


def _find_tracking_source(values, cls):
    for value in _iter_tracked_tensors(values):
        distributed_leaf = (
            isinstance(value, _DTensor) and is_traced_parameter(value)
        )
        if isinstance(value, cls) or has_trace(value) or distributed_leaf:
            return value
    return None


def _unwrap_tracking_tensor(value):
    if not isinstance(value, TrackingTensor):
        return value
    base = torch.Tensor(value)
    if isinstance(value, pp.LieTensor):
        unwrapped = base.as_subclass(pp.LieTensor)
        unwrapped.ltype = value.ltype
        return unwrapped
    return base


def _unwrap_tracking_tensors(values):
    return tree_map(_unwrap_tracking_tensor, values)


def _rewrap_tracking_tensor(result, tracking_source):
    if tracking_source is None or not isinstance(result, torch.Tensor) or isinstance(result, TrackingTensor):
        return result
    return TrackingTensor(result)


def _retain_ltype(result, tracking_source, cls, func):
    if tracking_source is None or not issubclass(cls, pp.LieTensor):
        return result
    if func not in _LTYPE_PRESERVING_FUNCS:
        return result
    if not isinstance(result, torch.Tensor) or isinstance(result, cls):
        return result
    if result.shape[-1:] != tracking_source.ltype.dimension:
        return result
    wrapped = torch.Tensor(result).as_subclass(cls)
    wrapped.ltype = tracking_source.ltype
    return wrapped

# =============================================================================
# Class: IndexTrackingTensor
# A custom subclass of torch.Tensor that tracks the indices used for slicing.
# When an instance is sliced via __getitem__, it records the provided index.
# =============================================================================
class TrackingTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        if isinstance(data, _DTensor):
            from bae.distributed.context import register_pending_parameter

            register_pending_parameter(data)
            return data

        if cls is TrackingTensor and isinstance(data, pp.LieTensor):
            return _TrackingLieTensor(data, *args, **kwargs)

        if isinstance(data, torch.Tensor):
            return torch.Tensor._make_subclass(cls, data, *args, **kwargs)
        return torch.Tensor._make_subclass(cls, torch.as_tensor(data), *args, **kwargs)


    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = super(TrackingTensor, cls).__torch_function__(func, types, args=args, kwargs=kwargs)
        trace_inputs = (args, kwargs)
        tracking_source = _find_tracking_source(trace_inputs, cls)
        result = _retain_ltype(result, tracking_source, cls, func)

        if isinstance(result, torch.Tensor):
            if (func == torch._C.TensorBase.__getitem__) and isinstance(args[1], torch.Tensor):
                result = _attach_index_trace(result, args[1], args[0])
            elif func == torch.cat:
                tensors = args[0]
                dim = kwargs.get("dim", 0)
                if len(args) > 1:
                    dim = args[1]
                if dim != 0:
                    raise NotImplementedError("TrackingTensor only supports torch.cat(..., dim=0)")
                result = _attach_cat_trace(result, tensors, dim)
            elif func in WHITELISTED_MAPS:
                result = _attach_map_trace(result, func, args)
        return result
    
    def __getitem__(self, index):
        # if isinstance(index, (slice, list, np.ndarray)):
        #     index = self._convert_to_index_tensor(index)
        # elif isinstance(index, tuple):
        #     if index[0] == slice(None, None, None):
        #         ...
        #         # this belongs to the mapping type
        #     else:
        #         index = (self._convert_to_index_tensor(index[0]), *index[1:])
        result = super().__getitem__(index)
        return result

    def _convert_to_index_tensor(self, index):
        if isinstance(index, int):
            return index
        elif isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = index.stop  # assume stop is provided
            step = 1 if index.step is None else index.step
            return torch.arange(start, stop, step)
        elif isinstance(index, list):
            return torch.tensor(index)
        elif isinstance(index, np.ndarray):
            return torch.from_numpy(index)

    def __format__(self, format_spec):
        if self.numel() == 1:
            return format(self.item(), format_spec)
        return format(str(self), format_spec)
        
    def tensor(self) -> torch.Tensor:
        return torch.Tensor(self)


class _TrackingLieTensor(TrackingTensor, pp.LieTensor):
    def __init__(self, data=None, *args, **kwargs):
        if isinstance(data, pp.LieTensor):
            self.ltype = data.ltype

    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        if not isinstance(data, pp.LieTensor):
            raise TypeError(f"_TrackingLieTensor expects a LieTensor input, got {type(data)!r}")
        instance = torch.Tensor.as_subclass(data, cls)
        instance.ltype = data.ltype
        return instance

    def detach(self):
        detached = torch.Tensor(self).detach().as_subclass(type(self))
        detached.ltype = self.ltype
        return detached
# Each tracked tensor stores its incoming edge directly as
# ``(edge_type, edge_metadata, inputs)`` in ``tensor.optrace``. A tensor owns
# exactly one incoming edge, so an additional dictionary keyed by ``id(tensor)``
# would be redundant.


# =============================================================================
# Decorator: index_transform
# A decorator that wraps a function to transform indices.
# It prints a message, calls the function, and attaches metadata recording
# the tracked indices from any IndexTrackingTensor arguments.
# =============================================================================
def index_transform(tensor, index):
    result = tensor[index]
    return _attach_index_trace(result, index, tensor)


# =============================================================================
# Decorator: map_transform
# A decorator that wraps a function to apply a map transformation.
# It runs the function on unwrapped tensors so inner operations stay opaque,
# then reattaches a single map edge to the result.
# =============================================================================
def map_transform(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracking_source = _find_tracking_source((args, kwargs), TrackingTensor)
        result = func(*_unwrap_tracking_tensors(args), **_unwrap_tracking_tensors(kwargs))
        # map edge (edge_type, func, [input_args])
        # ensure final result is an IndexTrackingTensor
        result = _rewrap_tracking_tensor(result, tracking_source)
        result = _attach_map_trace(result, func, args)
        return result
    return wrapper

    # map_transform(vmap(func))
    # this is wrong: vmap(map_transform(func))
