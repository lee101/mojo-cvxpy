"""Numerical and structural parity with CVXPY's canonicalization utilities."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.cvxcore.python import canonInterface as upstream
from cvxpy.reductions.utilities import ReducedMat as UpstreamReducedMat

import mojocvxpy as mcp


def assert_sparse_equal(got, expected):
    assert got.shape == expected.shape
    delta = (got - expected).tocoo()
    assert delta.nnz == 0 or np.allclose(delta.data, 0)
    assert np.allclose(got.toarray(), expected.toarray())


def tensor(var_length=7, constraints=31, params=9, seed=0):
    rng = np.random.default_rng(seed)
    shape = (constraints * (var_length + 1), params + 1)
    rows = rng.integers(0, shape[0], 700)
    cols = rng.integers(0, shape[1], 700)
    data = rng.normal(size=700)
    data[::41] = 0.0
    return sp.csc_array((data, (rows, cols)), shape=shape)


@pytest.mark.parametrize("quad_form", [False, True])
def test_reduce_problem_data_tensor_matches_upstream(quad_form):
    var_length = 7
    A = tensor(var_length=var_length)
    if quad_form:
        A = A[: 31 * var_length]
    expected = upstream.reduce_problem_data_tensor(
        A.copy(), var_length, quad_form=quad_form
    )
    got = mcp.reduce_problem_data_tensor(A.copy(), var_length, quad_form=quad_form)
    assert_sparse_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    assert np.array_equal(got[2], expected[2])
    assert got[3] == expected[3]


def test_reduce_preserves_duplicate_sums_and_drops_explicit_zeros():
    rows = np.array([0, 0, 5, 9, 9])
    cols = np.array([1, 1, 2, 0, 3])
    data = np.array([2.0, -0.5, 0.0, 4.0, 8.0])
    A = sp.coo_array((data, (rows, cols)), shape=(12, 4))
    expected = upstream.reduce_problem_data_tensor(A.copy(), 3)
    got = mcp.reduce_problem_data_tensor(A.copy(), 3)
    assert_sparse_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    assert np.array_equal(got[2], expected[2])


def test_reduce_int64_csc_indices_matches_upstream():
    A = tensor(var_length=3, constraints=12, params=4, seed=11)
    A.indices = A.indices.astype(np.int64)
    A.indptr = A.indptr.astype(np.int64)
    expected = upstream.reduce_problem_data_tensor(A.copy(), 3)
    got = mcp.reduce_problem_data_tensor(A.copy(), 3)
    assert_sparse_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    assert np.array_equal(got[2], expected[2])


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_reduce_csr_indices_matches_upstream(index_dtype):
    A = tensor(var_length=3, constraints=12, params=4, seed=12).tocsr()
    A.indices = A.indices.astype(index_dtype)
    A.indptr = A.indptr.astype(index_dtype)
    expected = upstream.reduce_problem_data_tensor(A.copy(), 3)
    got = mcp.reduce_problem_data_tensor(A.copy(), 3)
    assert_sparse_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    assert np.array_equal(got[2], expected[2])


def test_reduce_empty_csc_tensor_matches_upstream():
    A = sp.csc_array((24, 5), dtype=np.float64)
    expected = upstream.reduce_problem_data_tensor(A.copy(), 3)
    got = mcp.reduce_problem_data_tensor(A.copy(), 3)
    assert_sparse_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    assert np.array_equal(got[2], expected[2])


@pytest.mark.parametrize("var_length", [-1, 2.5])
def test_reduce_rejects_invalid_var_length_before_ffi(var_length):
    A = sp.csc_array(np.eye(4))
    with pytest.raises((TypeError, ValueError)):
        mcp.reduce_problem_data_tensor(A, var_length)


def test_reduce_normalizes_real_float32_and_rejects_complex():
    real = tensor(var_length=3, constraints=4, params=2).astype(np.float32)
    reduced, *_ = mcp.reduce_problem_data_tensor(real, 3)
    assert reduced.dtype == np.float64

    complex_tensor = real.astype(np.complex128)
    complex_tensor.data += 1j
    with pytest.raises((TypeError, ValueError), match="[Cc]omplex"):
        mcp.reduce_problem_data_tensor(complex_tensor, 3)


@pytest.mark.parametrize("with_offset", [False, True])
def test_apply_reduced_tensor_matches_upstream(with_offset):
    var_length = 7
    A = tensor(var_length=var_length, seed=3)
    reduced, indices, indptr, shape = upstream.reduce_problem_data_tensor(
        A.copy(), var_length
    )
    rng = np.random.default_rng(4)
    param = rng.normal(size=A.shape[1])
    index = (indices, indptr, shape)
    expected = upstream.get_matrix_from_tensor(
        reduced, param, var_length, with_offset=with_offset, problem_data_index=index
    )
    got = mcp.get_matrix_from_tensor(
        reduced, param, var_length, with_offset=with_offset, problem_data_index=index
    )
    assert_sparse_equal(got[0], expected[0])
    if with_offset:
        assert np.allclose(got[1], expected[1])
    else:
        assert got[1] is expected[1] is None


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_apply_reduced_tensor_simd_tail_matches_upstream(index_dtype):
    columns = 23
    indptr = np.array([0, 19, 23], dtype=index_dtype)
    indices = np.array(
        list(range(19)) + [1, 5, 13, 21], dtype=index_dtype
    )
    data = np.linspace(-2.0, 3.0, indices.size)
    reduced = sp.csr_array(
        (data, indices, indptr), shape=(2, columns)
    )
    param = np.linspace(0.25, 1.75, columns)
    problem_data_index = (
        np.array([0, 1], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
        (2, 1),
    )
    expected = upstream.get_matrix_from_tensor(
        reduced,
        param,
        1,
        with_offset=False,
        problem_data_index=problem_data_index,
    )
    got = mcp.get_matrix_from_tensor(
        reduced,
        param,
        1,
        with_offset=False,
        problem_data_index=problem_data_index,
    )
    assert_sparse_equal(got[0], expected[0])


@pytest.mark.parametrize("row_width", range(1, 34))
def test_apply_reduced_tensor_all_simd_tail_lengths(row_width):
    columns = 40
    indptr = np.array([0, row_width], dtype=np.int32)
    indices = np.arange(row_width, dtype=np.int32)
    data = np.linspace(-3.0, 2.0, row_width)
    reduced = sp.csr_array((data, indices, indptr), shape=(1, columns))
    param = np.linspace(0.25, 1.75, columns)
    index = (
        np.array([0], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        (1, 1),
    )
    expected = upstream.get_matrix_from_tensor(
        reduced, param, 1, with_offset=False, problem_data_index=index
    )
    got = mcp.get_matrix_from_tensor(
        reduced, param, 1, with_offset=False, problem_data_index=index
    )
    assert_sparse_equal(got[0], expected[0])


def test_apply_rejects_complex_parameter_without_silent_narrowing():
    reduced = sp.csr_array(np.eye(2))
    index = (
        np.array([0, 1], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
        (2, 1),
    )
    with pytest.raises(TypeError, match="complex"):
        mcp.get_matrix_from_tensor(
            reduced,
            np.array([1 + 2j, 3 + 4j]),
            1,
            with_offset=False,
            problem_data_index=index,
        )


def test_apply_accepts_noncontiguous_parameter_buffer():
    reduced = sp.csr_array(np.eye(3))
    source = np.arange(6.0)
    param = source[::2]
    assert not param.flags.c_contiguous
    index = (
        np.arange(3, dtype=np.int64),
        np.array([0, 3], dtype=np.int64),
        (3, 1),
    )
    got, _ = mcp.get_matrix_from_tensor(
        reduced, param, 1, with_offset=False, problem_data_index=index
    )
    assert np.array_equal(got.toarray().ravel(), param)


def test_apply_reduced_tensor_parallel_threshold_matches_upstream():
    rows = 70_000
    entries_per_row = 5
    nonzeros = rows * entries_per_row
    indptr = np.arange(
        0, nonzeros + 1, entries_per_row, dtype=np.int32
    )
    indices = np.tile(np.arange(entries_per_row, dtype=np.int32), rows)
    data = np.linspace(-1.0, 1.0, nonzeros)
    reduced = sp.csr_array(
        (data, indices, indptr), shape=(rows, 97)
    )
    param = np.linspace(0.5, 1.5, 97)
    problem_data_index = (
        np.arange(rows, dtype=np.int64),
        np.array([0, rows], dtype=np.int64),
        (rows, 1),
    )
    expected = upstream.get_matrix_from_tensor(
        reduced,
        param,
        1,
        with_offset=False,
        problem_data_index=problem_data_index,
    )
    got = mcp.get_matrix_from_tensor(
        reduced,
        param,
        1,
        with_offset=False,
        problem_data_index=problem_data_index,
    )
    assert_sparse_equal(got[0], expected[0])


def test_apply_reuses_noncanonical_csr_buffers_without_changing_result():
    indices = np.array([3, 1, 3, 0, 2], dtype=np.int32)
    indptr = np.array([0, 3, 5], dtype=np.int32)
    data = np.array([2.0, -1.0, 0.5, 4.0, -3.0])
    reduced = sp.csr_array((data, indices, indptr), shape=(2, 4))
    param = np.array([1.5, -2.0, 0.25, 3.0])
    problem_data_index = (
        np.array([0, 1], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
        (2, 1),
    )
    expected = upstream.get_matrix_from_tensor(
        reduced,
        param,
        1,
        with_offset=False,
        problem_data_index=problem_data_index,
    )
    got = mcp.get_matrix_from_tensor(
        reduced,
        param,
        1,
        with_offset=False,
        problem_data_index=problem_data_index,
    )
    assert_sparse_equal(got[0], expected[0])


def test_apply_constant_only_tensor_matches_upstream():
    var_length = 4
    A = tensor(var_length=var_length, constraints=12, params=0)
    reduced, indices, indptr, shape = upstream.reduce_problem_data_tensor(
        A.copy(), var_length
    )
    index = (indices, indptr, shape)
    expected = upstream.get_matrix_from_tensor(
        reduced, None, var_length, problem_data_index=index
    )
    got = mcp.get_matrix_from_tensor(
        reduced, None, var_length, problem_data_index=index
    )
    assert_sparse_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])


def test_unreduced_tensor_path_matches_upstream():
    var_length = 3
    A = tensor(var_length=var_length, constraints=10, params=4)
    param = np.arange(A.shape[1], dtype=float) + 0.5
    expected = upstream.get_matrix_from_tensor(A, param, var_length)
    got = mcp.get_matrix_from_tensor(A, param, var_length)
    assert_sparse_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])


def test_parameter_vector_fortran_order_and_zero_offset():
    values = {11: np.arange(6).reshape(2, 3), 12: np.array([7.0, 8.0])}
    cols = {11: 0, 12: 6, -1: 8}
    sizes = {11: 6, 12: 2}
    fn = values.__getitem__
    expected = upstream.get_parameter_vector(8, cols, sizes, fn)
    got = mcp.get_parameter_vector(8, cols, sizes, fn)
    assert np.array_equal(got, expected)
    zeroed = mcp.get_parameter_vector(8, cols, sizes, fn, zero_offset=True)
    assert zeroed[-1] == 0
    assert np.array_equal(zeroed[:-1], expected[:-1])


def test_nonzero_csc_includes_explicit_zero():
    A = sp.csc_array(
        (np.array([1.0, 0.0, 2.0]), np.array([2, 0, 1]), np.array([0, 2, 3])),
        shape=(3, 2),
    )
    expected = upstream.nonzero_csc_array(A.copy())
    got = mcp.nonzero_csc_array(A.copy())
    assert np.array_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])


def test_mapping_nonzero_rows_matches_upstream():
    A = tensor(var_length=5, constraints=17, params=6, seed=7)
    assert np.array_equal(
        mcp.A_mapping_nonzero_rows(A, 5),
        upstream.A_mapping_nonzero_rows(A, 5),
    )


def test_keep_zeros_structure_matches_upstream():
    var_length = 5
    A = tensor(var_length=var_length, constraints=17, params=6, seed=9)
    expected_reduced = UpstreamReducedMat(A.copy(), var_length)
    expected_reduced.cache(keep_zeros=True)
    got_reduced = mcp.ReducedMat(A.copy(), var_length)
    got_reduced.cache(keep_zeros=True)
    param = np.zeros(A.shape[1])
    param[-1] = 1.0
    expected = expected_reduced.get_matrix_from_tensor(param)
    got = got_reduced.get_matrix_from_tensor(param)
    assert_sparse_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    assert set(zip(*mcp.nonzero_csc_array(got[0]))) == set(
        zip(*upstream.nonzero_csc_array(expected[0]))
    )


def test_accelerated_context_drives_real_cvxpy_problem_data():
    x = cp.Variable(5)
    p = cp.Parameter(5, value=np.linspace(0.5, 1.5, 5))
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(x - p)),
        [np.arange(1.0, 6.0) @ x <= 3, x >= -2],
    )
    expected, _, _ = problem.get_problem_data(cp.CLARABEL)
    with mcp.accelerated():
        p.value = np.linspace(-1.0, 1.0, 5)
        got, _, _ = problem.get_problem_data(cp.CLARABEL)
        assert upstream.get_matrix_from_tensor is mcp.get_matrix_from_tensor
    assert upstream.get_matrix_from_tensor is not mcp.get_matrix_from_tensor

    p.value = np.linspace(-1.0, 1.0, 5)
    reference, _, _ = problem.get_problem_data(cp.CLARABEL)
    for key in ("c", "b"):
        assert np.allclose(got[key], reference[key])
    assert_sparse_equal(got["A"], reference["A"])
    assert got["dims"] == reference["dims"]
    assert expected["A"].shape == got["A"].shape


def test_install_is_idempotent_and_uninstall_restores():
    original = upstream.reduce_problem_data_tensor
    mcp.install()
    mcp.install()
    try:
        assert upstream.reduce_problem_data_tensor is mcp.reduce_problem_data_tensor
    finally:
        mcp.uninstall()
    assert upstream.reduce_problem_data_tensor is original


def test_accelerated_context_can_be_nested():
    original = upstream.reduce_problem_data_tensor
    with mcp.accelerated():
        with mcp.accelerated():
            assert upstream.reduce_problem_data_tensor is mcp.reduce_problem_data_tensor
        assert upstream.reduce_problem_data_tensor is mcp.reduce_problem_data_tensor
    assert upstream.reduce_problem_data_tensor is original
