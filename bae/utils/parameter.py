import pypose as pp
import torch

from .retraction_jacobian import so3_retraction_jacobian, se3_retraction_jacobian
from .pypose_ambient_grad import pypose_ambient_grad_enabled


def parameter_update_shape(param: torch.Tensor) -> torch.Size:
    if param.ndim == 0:
        return param.shape
    trim_se3_grad = getattr(param, 'trim_SE3_grad', False)
    ltype = getattr(param, "ltype", None)
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()
    if isinstance(param, DTensor):
        from bae.distributed.context import parameter_metadata

        metadata = parameter_metadata(param)
        trim_se3_grad = trim_se3_grad or metadata.trim_se3_grad
        ltype = ltype or metadata.ltype
    if trim_se3_grad:
        return torch.Size((*param.shape[:-1], param.shape[-1] - 1))
    if isinstance(param, pp.LieTensor) or ltype is not None:
        return torch.Size((*param.shape[:-1], ltype.manifold[0]))
    return param.shape


def trim_parameter_jacobian_values(
    param: torch.Tensor,
    values: torch.Tensor,
    block_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    # Jacobian blocks are terminal numeric outputs of graph traversal.  A
    # TrackingTensor subclass can survive functorch when a distributed trace
    # passes through a ``psjac`` function; retaining that subclass here would
    # incorrectly interpret shape-only trimming operations as model tracing.
    from bae.autograd.function import TrackingTensor

    if isinstance(values, TrackingTensor):
        values = torch.Tensor(values)
    if param.ndim == 0 or values.shape[-1] != param.shape[-1]:
        return values
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()
    distributed_metadata = None
    if isinstance(param, DTensor):
        from bae.distributed.context import parameter_metadata

        distributed_metadata = parameter_metadata(param)
    trim_se3_grad = getattr(param, 'trim_SE3_grad', False) or bool(
        distributed_metadata is not None
        and distributed_metadata.trim_se3_grad
    )
    if trim_se3_grad:
        if not pypose_ambient_grad_enabled():
            # Avoid a non-leading-dimension ``cat`` here. During Dynamo tracing,
            # functorch can retain the TrackingTensor subclass on Jacobian
            # blocks even though these values are terminal numeric outputs.
            # ``index_select`` expresses the same removal of the quaternion's
            # redundant scalar coordinate without creating a trace edge.
            indices = torch.arange(
                values.shape[-1] - 1,
                device=values.device,
                dtype=torch.long,
            )
            indices = indices + (indices >= 6)
            return torch.index_select(values, -1, indices)
        if distributed_metadata is not None:
            from bae.distributed.ops import distributed_index

            if block_indices is None:
                raise RuntimeError(
                    "DTensor SE3 Jacobian trimming requires global block indices."
                )
            pose = distributed_index(param, block_indices.to(torch.long))[
                ..., :7
            ].detach()
        else:
            pose = torch.Tensor(param)[..., :7].detach()
        if block_indices is not None and distributed_metadata is None:
            pose = pose[block_indices.to(torch.long)]
        pose_values = values[..., :7] @ se3_retraction_jacobian(pose)
        if param.shape[-1] == 7:
            return pose_values
        return torch.cat([pose_values, values[..., 7:]], dim=-1)
    ltype = getattr(param, "ltype", None)
    if ltype is None and distributed_metadata is not None:
        ltype = distributed_metadata.ltype
    if isinstance(param, pp.LieTensor) or ltype is not None:
        if pypose_ambient_grad_enabled():
            if distributed_metadata is not None:
                from bae.distributed.ops import distributed_index

                if block_indices is None:
                    raise RuntimeError(
                        "DTensor Lie Jacobian trimming requires global block indices."
                    )
                lie_param = distributed_index(
                    param, block_indices.to(torch.long)
                ).detach()
            else:
                lie_param = torch.Tensor(param).detach()
            if block_indices is not None and distributed_metadata is None:
                lie_param = lie_param[block_indices.to(torch.long)]
            if ltype == pp.SO3_type:
                return values @ so3_retraction_jacobian(lie_param)
            if ltype == pp.SE3_type:
                return values @ se3_retraction_jacobian(lie_param)
        step_dim = int(ltype.manifold[0])
        if step_dim != param.shape[-1]:
            return values[..., :step_dim]
    return values
