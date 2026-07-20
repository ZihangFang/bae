from __future__ import annotations

import pypose as pp
import pytest
import torch

from bae.utils.pypose_ambient_grad import (
    _so3_exp_forward,
    _so3_Jl,
    _so3_Jl_inv,
    install_pypose_ambient_grad_monkeypatch,
)
from bae.utils.pypose_compile import install_pypose_torch_compile_monkeypatch
from bae.optim.optimizer import LM


@torch.no_grad()
def _zero_and_small_angles(dtype: torch.dtype) -> torch.Tensor:
    inputs = torch.zeros(2, 3, dtype=dtype)
    inputs[1, 0] = torch.finfo(dtype).eps / 2.0
    return inputs


@pytest.mark.parametrize("function", (_so3_exp_forward, _so3_Jl, _so3_Jl_inv))
def test_so3_small_angle_forward_and_gradients_are_finite(function):
    def weighted_output(input):
        output = function(input)
        weights = torch.arange(
            1, output.numel() + 1, device=output.device, dtype=output.dtype
        ).reshape(output.shape)
        return output, (output * weights).sum()

    eager_input = _zero_and_small_angles(torch.float64).requires_grad_()
    eager_output, eager_loss = weighted_output(eager_input)
    (eager_gradient,) = torch.autograd.grad(eager_loss, eager_input)

    compiled = torch.compile(weighted_output, backend="eager", fullgraph=True)
    compiled_input = eager_input.detach().clone().requires_grad_()
    compiled_output, compiled_loss = compiled(compiled_input)
    (compiled_gradient,) = torch.autograd.grad(compiled_loss, compiled_input)

    assert torch.isfinite(eager_output).all()
    assert torch.isfinite(eager_gradient).all()
    torch.testing.assert_close(compiled_output, eager_output)
    torch.testing.assert_close(compiled_gradient, eager_gradient)


def test_lietensor_tensor_alias_is_fullgraph_traceable_and_differentiable():
    install_pypose_torch_compile_monkeypatch()
    data = pp.randn_SE3(4, sigma=0.2, dtype=torch.float64).tensor().detach().requires_grad_()
    pose = pp.SE3(data)

    compiled = torch.compile(
        lambda value: value.tensor().square().sum(), backend="eager", fullgraph=True
    )
    loss = compiled(pose)
    (gradient,) = torch.autograd.grad(loss, data)

    torch.testing.assert_close(gradient, 2.0 * data)

    compiled_inverse = torch.compile(
        lambda value: value.Inv(), backend="eager", fullgraph=True
    )
    inverse = compiled_inverse(pose)
    assert isinstance(inverse, pp.LieTensor)
    assert inverse.ltype is pp.SE3_type


@pytest.mark.parametrize(
    "operation",
    (
        lambda value: torch.clone(input=value),
        lambda value: torch.cat(tensors=(value, value), dim=0),
        lambda value: torch.reshape(input=value, shape=(1, 2, 4)),
    ),
    ids=("clone", "cat", "reshape"),
)
def test_lietensor_keyword_inputs_preserve_ltype_in_fullgraph(operation):
    value = pp.randn_SO3(2, dtype=torch.float64)

    expected = operation(value)
    compiled = torch.compile(operation, backend="eager", fullgraph=True)
    actual = compiled(value)

    assert isinstance(expected, pp.LieTensor)
    assert isinstance(actual, pp.LieTensor)
    assert expected.ltype is pp.SO3_type
    assert actual.ltype is pp.SO3_type
    torch.testing.assert_close(actual.tensor(), expected.tensor())


def test_ambient_se3_pipeline_is_fullgraph_traceable_with_local_gradients():
    install_pypose_ambient_grad_monkeypatch()
    dtype = torch.float64
    lhs = pp.randn_SE3(8, sigma=0.2, dtype=dtype).tensor().detach().requires_grad_()
    rhs = pp.randn_SE3(8, sigma=0.2, dtype=dtype).tensor().detach().requires_grad_()
    points = torch.randn(8, 3, dtype=dtype, requires_grad=True)

    def residual(a, b, p):
        pose_a = pp.SE3(a)
        pose_b = pp.SE3(b)
        relative = (pose_a.Inv() @ pose_b).Log().tensor()
        return torch.cat((relative, pose_a.Act(p)), dim=-1)

    eager = residual(lhs, rhs, points)
    eager_gradients = torch.autograd.grad(
        eager.square().sum(), (lhs, rhs, points), retain_graph=True
    )

    compiled = torch.compile(residual, backend="eager", fullgraph=True)
    actual = compiled(lhs, rhs, points)
    compiled_gradients = torch.autograd.grad(actual.square().sum(), (lhs, rhs, points))

    torch.testing.assert_close(actual, eager)
    for actual_gradient, eager_gradient in zip(compiled_gradients, eager_gradients):
        torch.testing.assert_close(actual_gradient, eager_gradient)


def test_ambient_se3_jacrev_is_fullgraph_traceable():
    install_pypose_ambient_grad_monkeypatch()
    pose = pp.randn_SE3(3, sigma=0.2, dtype=torch.float64).tensor()
    points = torch.randn(3, 3, dtype=torch.float64)

    jacobian = torch.func.jacrev(lambda value: pp.SE3(value).Act(points))
    expected = jacobian(pose)
    compiled_jacobian = torch.compile(jacobian, backend="eager", fullgraph=True)

    torch.testing.assert_close(compiled_jacobian(pose), expected)

    algebra = 0.2 * torch.randn(3, 6, dtype=torch.float64)
    exp_jacobian = torch.func.jacrev(lambda value: pp.se3(value).Exp().tensor())
    expected_exp = exp_jacobian(algebra)
    compiled_exp = torch.compile(exp_jacobian, backend="eager", fullgraph=True)

    torch.testing.assert_close(compiled_exp(algebra), expected_exp)


def test_lm_trimmed_se3_update_composes_with_compile_patch():
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    data = torch.zeros(3, 10, device=device, dtype=dtype)
    data[:, 6] = 1.0
    data[:, 7:] = torch.randn(3, 3, device=device, dtype=dtype)
    parameter = pp.Parameter(data.clone(), sjac=True)
    parameter.trim_SE3_grad = True
    update = 0.01 * torch.randn(3, 9, device=device, dtype=dtype)

    expected_pose = (
        pp.se3(update[:, :6]).Exp() * pp.SE3(data[:, :7])
    ).tensor()
    expected_intrinsics = data[:, 7:] + update[:, 6:]

    # update_parameter does not inspect optimizer state, so exercise it as an
    # unbound method without constructing a linear solver.
    LM.update_parameter(None, [parameter], update.flatten())

    torch.testing.assert_close(parameter.tensor()[:, :7], expected_pose)
    torch.testing.assert_close(parameter.tensor()[:, 7:], expected_intrinsics)
