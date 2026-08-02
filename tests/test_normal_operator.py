import pytest
import pypose as pp
import torch
from pypose.autograd.function import psjac
from torch import nn

from bae.autograd.graph import BSRJacobianData
from bae.optim import ComponentLinearizer, LM, PytreePartition
from bae.utils.linear_operator import (
    ComponentBlockDiagonalPreconditioner,
    ComponentJacobianOperator,
    ComponentNormalMatVec,
    NormalMatVec,
)
from bae.utils.pysolvers import PCG


@psjac
def _indexed_difference(value, observation):
    return value - observation


class _IndexedLeastSquares(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = pp.Parameter(value, sjac=True)

    def forward(self, observations, indices):
        return _indexed_difference(self.value[indices], observations)


@psjac
def _pose_graph_residual(measurement, node1, node2, information):
    estimate = node1.Inv() @ node2
    translation = estimate.translation() - measurement.translation()
    rotation = measurement.rotation() @ estimate.rotation().Inv()
    residual = torch.cat((translation, 2.0 * rotation.tensor()[..., :3]), dim=-1)
    return (information @ residual[..., None])[..., 0]


class _PoseGraphFixedFirst(nn.Module):
    def __init__(self, nodes_rest):
        super().__init__()
        self.nodes_rest = pp.Parameter(nodes_rest, sjac=True)

    def forward(self, edges, poses, infos, node_fixed):
        nodes = torch.cat((node_fixed, self.nodes_rest), dim=0)
        return _pose_graph_residual(
            poses,
            nodes[edges[..., 0]],
            nodes[edges[..., 1]],
            infos,
        )


def _component_problem():
    torch.manual_seed(10)
    row_count, row_block_size = 4, 2
    camera_count, camera_block_size = 3, 2
    point_count, point_block_size = 4, 1
    camera_columns = torch.tensor([0, 2, 1, 0])
    point_columns = torch.tensor([3, 1, 2, 0])
    crow = torch.arange(row_count + 1)
    camera_values = torch.randn(
        row_count, row_block_size, camera_block_size, dtype=torch.float64
    )
    point_values = torch.randn(
        row_count, row_block_size, point_block_size, dtype=torch.float64
    )
    components = (
        (
            BSRJacobianData(
                crow,
                camera_columns,
                camera_values,
                (
                    row_count * row_block_size,
                    camera_count * camera_block_size,
                ),
            ),
        ),
        (
            BSRJacobianData(
                crow,
                point_columns,
                point_values,
                (
                    row_count * row_block_size,
                    point_count * point_block_size,
                ),
            ),
        ),
    )
    dense = torch.zeros(
        row_count * row_block_size,
        camera_count * camera_block_size + point_count * point_block_size,
        dtype=torch.float64,
    )
    for row in range(row_count):
        row_slice = slice(row * row_block_size, (row + 1) * row_block_size)
        camera_start = int(camera_columns[row]) * camera_block_size
        point_start = (
            camera_count * camera_block_size
            + int(point_columns[row]) * point_block_size
        )
        dense[
            row_slice,
            camera_start : camera_start + camera_block_size,
        ] += camera_values[row]
        dense[
            row_slice,
            point_start : point_start + point_block_size,
        ] += point_values[row]
    operator = ComponentJacobianOperator(
        components,
        (
            camera_count * camera_block_size,
            point_count * point_block_size,
        ),
        row_count * row_block_size,
    )
    return operator, dense


def test_component_jacobian_forward_adjoint_and_diagonal():
    operator, dense = _component_problem()
    x = torch.randn(dense.shape[1], dtype=dense.dtype)
    y = torch.randn(dense.shape[0], dtype=dense.dtype)

    torch.testing.assert_close(operator @ x, dense @ x)
    torch.testing.assert_close(operator.rmatvec(y), dense.mT @ y)
    torch.testing.assert_close(operator.diagonal(), dense.square().sum(0))


def test_component_jacobian_supports_nonimplicit_rows():
    values = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
        ],
        dtype=torch.float64,
    )
    component = BSRJacobianData(
        torch.tensor([0, 2, 2, 3]),
        torch.tensor([0, 1, 0]),
        values,
        (3, 4),
    )
    operator = ComponentJacobianOperator(((component,),), (4,), 3)
    dense = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
            [5.0, 6.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    x = torch.randn(4, dtype=torch.float64)
    y = torch.randn(3, dtype=torch.float64)
    torch.testing.assert_close(operator @ x, dense @ x)
    torch.testing.assert_close(operator.rmatvec(y), dense.mT @ y)


def test_component_normal_operator_and_pcg_match_dense():
    jacobian, dense = _component_problem()
    damping = 0.2
    diagonal = jacobian.diagonal()
    operator = ComponentNormalMatVec(jacobian, damping=damping, diag=diagonal)
    x = torch.randn(dense.shape[1], dtype=dense.dtype)
    expected_matrix = dense.mT @ dense + damping * torch.diag(diagonal)
    torch.testing.assert_close(operator @ x, expected_matrix @ x)

    rhs = torch.randn(dense.shape[1], dtype=dense.dtype)
    block_preconditioner = ComponentBlockDiagonalPreconditioner(
        jacobian.block_diagonal(),
        jacobian.parameter_sizes,
        diagonal,
        damping,
    )
    camera_blocks = []
    for start in range(0, 6, 2):
        camera_blocks.append(expected_matrix[start : start + 2, start : start + 2])
    point_blocks = [
        expected_matrix[index : index + 1, index : index + 1]
        for index in range(6, expected_matrix.shape[0])
    ]
    expected_preconditioned = (
        torch.block_diag(*(camera_blocks + point_blocks)).inverse() @ rhs
    )
    torch.testing.assert_close(block_preconditioner @ rhs, expected_preconditioned)

    actual = PCG(tol=1e-10, maxiter=500)(operator, rhs, M=block_preconditioner)
    expected = torch.linalg.solve(expected_matrix, rhs)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)


def test_matrix_free_lm_uses_component_operator_and_decreases_loss():
    expected = torch.tensor(
        [[1.0, 2.0], [-1.0, 3.0], [2.0, -2.0]],
        dtype=torch.float64,
    )
    indices = torch.tensor([0, 1, 2, 0, 2])
    observations = expected[indices]
    model = _IndexedLeastSquares(torch.zeros_like(expected))
    optimizer = LM(
        model,
        solver=PCG(tol=1e-10, maxiter=50),
        linearizer=ComponentLinearizer(),
        reject=0,
    )

    loss = optimizer.step({"observations": observations, "indices": indices})

    assert loss < 1e-8
    torch.testing.assert_close(
        torch.Tensor(model.value), expected, rtol=1e-5, atol=1e-5
    )


def test_existing_matrix_free_normal_path_remains_available():
    expected = torch.tensor(
        [[1.0, 2.0], [-1.0, 3.0], [2.0, -2.0]],
        dtype=torch.float64,
    )
    indices = torch.tensor([0, 1, 2, 0, 2])
    model = _IndexedLeastSquares(torch.zeros_like(expected))
    optimizer = LM(
        model,
        solver=PCG(tol=1e-10, maxiter=50),
        matrix_free_normal=True,
        reject=0,
    )

    loss = optimizer.step({"observations": expected[indices], "indices": indices})

    assert loss < 1e-8


def test_chunked_matrix_free_evaluation_matches_unchunked_components():
    torch.manual_seed(12)
    values = torch.randn(4, 2, dtype=torch.float64)
    indices = torch.tensor([3, 0, 3, 1, 2, 0, 1])
    observations = torch.randn(7, 2, dtype=torch.float64)
    inputs = {
        "observations": observations,
        "indices": indices,
    }
    model = _IndexedLeastSquares(values.clone())
    optimizer = LM(
        model,
        solver=PCG(tol=1e-10, maxiter=10),
        linearizer=ComponentLinearizer(3),
        reject=0,
    )
    params = tuple(
        parameter
        for parameter in optimizer.param_groups[0]["params"]
        if parameter.requires_grad
    )

    chunked_residual, chunked_components = optimizer.linearizer.evaluate(
        optimizer.model, inputs, None, params
    )
    full_residual, full_components = ComponentLinearizer(None).evaluate(
        optimizer.model, inputs, None, params
    )

    torch.testing.assert_close(chunked_residual, full_residual)
    assert len(chunked_components) == len(full_components)
    for chunked_group, full_group in zip(chunked_components, full_components):
        assert len(chunked_group) == len(full_group)
        for chunked, full in zip(chunked_group, full_group):
            assert chunked.size == full.size
            torch.testing.assert_close(chunked.crow_indices, full.crow_indices)
            torch.testing.assert_close(chunked.col_indices, full.col_indices)
            torch.testing.assert_close(chunked.values, full.values)
    assert chunked_components[0][0].col_indices.data_ptr() == indices.data_ptr()


def test_chunked_matrix_free_lm_matches_unchunked_step():
    expected = torch.tensor(
        [[1.0, 2.0], [-1.0, 3.0], [2.0, -2.0]],
        dtype=torch.float64,
    )
    indices = torch.tensor([0, 1, 2, 0, 2])
    inputs = {
        "observations": expected[indices],
        "indices": indices,
    }
    chunked_model = _IndexedLeastSquares(torch.zeros_like(expected))
    full_model = _IndexedLeastSquares(torch.zeros_like(expected))
    common = {
        "solver": PCG(tol=1e-10, maxiter=50),
        "reject": 0,
    }
    chunked = LM(
        chunked_model,
        linearizer=ComponentLinearizer(2),
        **common,
    )
    full = LM(
        full_model,
        linearizer=ComponentLinearizer(None),
        **common,
    )

    chunked_loss = chunked.step(inputs)
    full_loss = full.step(inputs)

    torch.testing.assert_close(chunked_loss, full_loss)
    torch.testing.assert_close(
        torch.Tensor(chunked_model.value),
        torch.Tensor(full_model.value),
    )


def test_pytree_partition_chunks_marked_axes_and_broadcasts_constants():
    partition = PytreePartition(
        input_dims={"factors": 1, "constant": None},
        target_dims=(0, None),
    )
    inputs = {
        "factors": torch.arange(15).reshape(3, 5),
        "constant": torch.ones(2),
    }
    target = (torch.arange(10).reshape(5, 2), "metadata")

    assert partition.factor_count(inputs, target) == 5
    chunk_input, chunk_target = partition.slice(inputs, target, 1, 4, 5)

    torch.testing.assert_close(chunk_input["factors"], inputs["factors"][:, 1:4])
    assert chunk_input["constant"] is inputs["constant"]
    torch.testing.assert_close(chunk_target[0], target[0][1:4])
    assert chunk_target[1] == "metadata"


def test_pgo_linearization_chunks_edges_and_matches_unchunked():
    dtype = torch.float64
    nodes = pp.SE3(
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [2, 0.2, 0, 0, 0, 0, 1],
                [3, 0.1, 0, 0, 0, 0, 1],
            ],
            dtype=dtype,
        )
    )
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]])
    poses = nodes[edges[:, 0]].Inv() @ nodes[edges[:, 1]]
    infos = torch.eye(6, dtype=dtype).expand(5, -1, -1).clone()
    node_fixed = nodes[:1].clone()
    nodes_rest = nodes[1:].clone()
    torch.Tensor(nodes_rest)[:, 0].add_(0.1)
    inputs = {
        "edges": edges,
        "poses": poses,
        "infos": infos,
        "node_fixed": node_fixed,
    }
    partition = PytreePartition(
        input_dims={
            "edges": 0,
            "poses": 0,
            "infos": 0,
            "node_fixed": None,
        }
    )
    model = _PoseGraphFixedFirst(nodes_rest)
    optimizer = LM(
        model,
        solver=PCG(tol=1e-10, maxiter=20),
        linearizer=ComponentLinearizer(2, partition=partition),
        reject=0,
    )
    params = tuple(
        parameter
        for parameter in optimizer.param_groups[0]["params"]
        if parameter.requires_grad
    )
    arguments = {
        "diagonal_min": 1e-6,
        "diagonal_max": 1e6,
    }

    chunked = optimizer.linearizer.linearize(
        optimizer.model, inputs, None, params, **arguments
    )
    full = ComponentLinearizer(None, partition=partition).linearize(
        optimizer.model, inputs, None, params, **arguments
    )
    step = torch.randn_like(chunked.gradient)

    torch.testing.assert_close(chunked.residual, full.residual)
    torch.testing.assert_close(chunked.loss, full.loss)
    torch.testing.assert_close(chunked.gradient, full.gradient)
    torch.testing.assert_close(chunked.normal @ step, full.normal @ step)
    torch.testing.assert_close(
        chunked.predicted_reduction(step),
        full.predicted_reduction(step),
    )


def test_normal_matvec_dense_matches_explicit():
    torch.manual_seed(0)
    m, n = 8, 5
    J = torch.randn(m, n, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)

    op = NormalMatVec(J)
    y_op = op.matvec(x)
    y_ex = J.mT @ (J @ x)

    torch.testing.assert_close(y_op, y_ex, rtol=1e-10, atol=1e-10)


def test_normal_matvec_dense_damping_matches_explicit():
    torch.manual_seed(1)
    m, n = 7, 4
    J = torch.randn(m, n, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)
    damping = 0.3

    diag = J.square().sum(dim=0)
    op = NormalMatVec(J, damping=damping)
    y_op = op @ x
    y_ex = J.mT @ (J @ x) + damping * diag * x

    torch.testing.assert_close(y_op, y_ex, rtol=1e-10, atol=1e-10)


def test_normal_matvec_sparse_csr_matches_explicit_and_cached_diag():
    torch.manual_seed(2)
    m, n = 10, 6
    dense = torch.randn(m, n, dtype=torch.float64)
    mask = torch.rand_like(dense) < 0.5
    dense = dense * mask
    J = dense.to_sparse_csr()
    x = torch.randn(n, dtype=torch.float64)

    diag = dense.square().sum(dim=0)
    op = NormalMatVec(J, diag=diag)
    y_op = op.matvec(x)
    y_ex = dense.mT @ (dense @ x)

    torch.testing.assert_close(y_op, y_ex, rtol=1e-10, atol=1e-10)


def test_normal_matvec_sparse_bsr_matches_explicit():
    torch.manual_seed(3)
    m, n = 6, 4
    dense = torch.randn(m, n, dtype=torch.float64).contiguous()
    x = torch.randn(n, dtype=torch.float64)

    try:
        J_bsr = dense.to_sparse_bsr(blocksize=(2, 2))
    except Exception:
        pytest.skip("BSR conversion not supported on this device/build.")

    op = NormalMatVec(J_bsr)
    y_op = op @ x
    y_ex = dense.mT @ (dense @ x)

    torch.testing.assert_close(y_op, y_ex, rtol=1e-10, atol=1e-10)


def test_pcg_smoke_with_normal_operator():
    torch.manual_seed(4)
    m, n = 12, 5
    J = torch.randn(m, n, dtype=torch.float64)
    damping = 1e-3
    op = NormalMatVec(J, damping=damping)

    b = torch.randn(n, dtype=torch.float64)
    solver = PCG(tol=1e-10, maxiter=200)
    x_pcg = solver(op, b)

    diag = J.square().sum(dim=0)
    A_dense = J.mT @ J + damping * torch.diag(diag)
    x_ex = torch.linalg.solve(A_dense, b)

    torch.testing.assert_close(x_pcg, x_ex, rtol=1e-6, atol=1e-6)
