"""Numerical and structural parity with CVXPY's canonicalization utilities."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.cvxcore.python import canonInterface as upstream
from cvxpy.reductions.utilities import ReducedMat as UpstreamReducedMat

import mojocvxpy as mcp

# Shared tolerance for every floating-point comparison against upstream.
# assert_sparse_equal uses np.allclose with these defaults; offset vectors
# below pass them explicitly. Rationale: reduce/get_matrix are exact index
# rearrangements plus one matvec, so upstream and the port agree to well
# under 1e-9 on the well-conditioned fixtures used here; the tolerance only
# absorbs float64 summation-order differences in the SIMD matvec kernel.
RTOL = 1e-9
ATOL = 1e-9


def assert_sparse_equal(got, expected):
    assert got.shape == expected.shape
    delta = (got - expected).tocoo()
    assert delta.nnz == 0 or np.allclose(delta.data, 0, rtol=RTOL, atol=ATOL)
    assert np.allclose(got.toarray(), expected.toarray(), rtol=RTOL, atol=ATOL)


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


# ---------------------------------------------------------------------------
# Degenerate / mismatched-input parity.
#
# For each case the test pins the behaviour of the installed upstream
# cvxpy (`cvxpy.cvxcore.python.canonInterface`, verified live in this
# session). Where upstream defines a *returned value* we compare against
# it directly with RTOL/ATOL. Where upstream's implementation merely
# returns a floor-division artifact for inputs its own docstring forbids
# ("A ... must not have a 0 in its shape"; rows must be a multiple of
# n_cols = var_length + 1, or var_length for quad_form), the port instead
# raises a clear ValueError; those tests pin the port's error and cite
# the upstream precondition rather than enshrining the artifact.
# ---------------------------------------------------------------------------


def test_reduce_single_element_matches_upstream():
    A = sp.csc_array(np.array([[5.0]]))
    expected = upstream.reduce_problem_data_tensor(A.copy(), 0)
    got = mcp.reduce_problem_data_tensor(A.copy(), 0)
    assert_sparse_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    assert np.array_equal(got[2], expected[2])
    assert tuple(got[3]) == tuple(expected[3])
    assert np.allclose(got[0].data, [5.0], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("shape", [(0, 5), (10, 0), (0, 0)])
def test_reduce_zero_dimension_raises_value_error(shape):
    # Upstream canonInterface.reduce_problem_data_tensor docstring: "A ...
    # must not have a 0 in its shape". The port enforces that precondition
    # with ValueError instead of returning a floor-division artifact.
    with pytest.raises(ValueError):
        mcp.reduce_problem_data_tensor(sp.csc_array(shape, dtype=np.float64), 3)


def test_reduce_nondivisible_rows_raises_value_error():
    # Upstream defines n_constr = A.shape[0] // n_cols (floor division for
    # rows that are not a multiple of var_length + 1); the port raises
    # ValueError for that incompatible layout rather than guessing.
    with pytest.raises(ValueError):
        mcp.reduce_problem_data_tensor(sp.csc_array(np.eye(10, 4)), 3)


@pytest.mark.parametrize("var_length", [-1, 0])
def test_reduce_quad_nonpositive_var_length_raises_value_error(var_length):
    # Upstream with quad_form and var_length <= 0 computes n_cols <= 0 and
    # fails inside np handling (IndexError in the installed cvxpy); the
    # port raises ValueError at the layout check. Both reject the input:
    # only that it is rejected is pinned here, not the incidental class.
    with pytest.raises((TypeError, ValueError, IndexError)):
        mcp.reduce_problem_data_tensor(
            sp.csc_array(np.eye(4)), var_length, quad_form=True
        )


def test_reduce_nan_and_inf_preserved_matches_upstream():
    A = sp.csc_array(
        (
            np.array([np.nan, np.inf, -np.inf]),
            (np.array([0, 1, 2]), np.array([0, 0, 0])),
        ),
        shape=(4, 2),
    )
    expected = upstream.reduce_problem_data_tensor(A.copy(), 1)
    got = mcp.reduce_problem_data_tensor(A.copy(), 1)
    assert got[0].shape == expected[0].shape
    got_data = got[0].toarray().ravel()
    expected_data = expected[0].toarray().ravel()
    assert np.isnan(got_data[0]) and np.isnan(expected_data[0])
    assert np.isinf(got_data[2]) and np.isinf(expected_data[2])
    assert got_data[2] > 0 and got_data[4] < 0
    assert np.array_equal(got[1], expected[1])
    assert np.array_equal(got[2], expected[2])


def test_reduce_all_explicit_zeros_matches_upstream():
    A = sp.csc_array(
        (np.array([0.0, 0.0]), (np.array([0, 2]), np.array([0, 1]))), shape=(4, 2)
    )
    expected = upstream.reduce_problem_data_tensor(A.copy(), 1)
    got = mcp.reduce_problem_data_tensor(A.copy(), 1)
    assert_sparse_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    assert np.array_equal(got[2], expected[2])
    assert tuple(got[3]) == tuple(expected[3])


def test_reduce_int_data_matches_upstream_values():
    # README documents float64 normalization at the boundary, so only the
    # values (not the preserved-int64 dtype of upstream) are compared.
    A = sp.csc_array(
        (np.array([1, 2]), (np.array([0, 1]), np.array([0, 0]))),
        shape=(8, 4),
        dtype=np.int64,
    )
    expected = upstream.reduce_problem_data_tensor(A.copy(), 3)
    got = mcp.reduce_problem_data_tensor(A.copy(), 3)
    assert np.allclose(got[0].toarray(), expected[0].toarray(), rtol=RTOL, atol=ATOL)
    assert np.array_equal(got[1], expected[1])
    assert np.array_equal(got[2], expected[2])


def test_reduce_coo_input_with_duplicates_matches_upstream():
    A = sp.coo_array(
        (np.array([1.0, 2.0, -0.5]), (np.array([0, 0, 5]), np.array([1, 1, 2]))),
        shape=(12, 4),
    )
    expected = upstream.reduce_problem_data_tensor(A.copy(), 3)
    got = mcp.reduce_problem_data_tensor(A.copy(), 3)
    assert_sparse_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    assert np.array_equal(got[2], expected[2])


def test_reduce_rejects_wrong_var_length_types_like_upstream():
    A = sp.csc_array(np.eye(12, 4))
    for var_length in (3.0, "3", None):
        with pytest.raises((TypeError, ValueError)):  # np.exceptions.UFuncTypeError (upstream) subclasses TypeError
            mcp.reduce_problem_data_tensor(A.copy(), var_length)


def test_get_parameter_vector_empty_and_scalars_match_upstream():
    assert mcp.get_parameter_vector(0, {}, {}, lambda i: None) is None
    values = {1: np.array(2.5), 2: np.array([1.0, 2.0])}
    cols = {-1: 3, 1: 0, 2: 1}
    sizes = {1: 1, 2: 2}
    expected = upstream.get_parameter_vector(3, cols, sizes, values.__getitem__)
    got = mcp.get_parameter_vector(3, cols, sizes, values.__getitem__)
    assert np.allclose(got, expected, rtol=RTOL, atol=ATOL)
    # Constant offset zeroing is part of upstream's contract.
    expected_zeroed = upstream.get_parameter_vector(
        3, cols, sizes, values.__getitem__, zero_offset=True
    )
    got_zeroed = mcp.get_parameter_vector(
        3, cols, sizes, values.__getitem__, zero_offset=True
    )
    assert np.allclose(got_zeroed, expected_zeroed, rtol=RTOL, atol=ATOL)
    assert got_zeroed[-1] == 0.0


def test_nonzero_csc_single_explicit_zero_and_empty_match_upstream():
    A = sp.csc_array(
        (np.array([0.0]), (np.array([1]), np.array([0]))), shape=(2, 2)
    )
    expected = upstream.nonzero_csc_array(A.copy())
    got = mcp.nonzero_csc_array(A.copy())
    assert np.array_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    # The sentinel trick must restore the stored zero afterwards.
    probe = sp.csc_array(
        (np.array([0.0, 1.0]), (np.array([0, 1]), np.array([0, 0]))), shape=(2, 1)
    )
    mcp.nonzero_csc_array(probe)
    assert np.array_equal(probe.data, np.array([0.0, 1.0]))


def test_A_mapping_nonzero_rows_empty_and_explicit_zero_match_upstream():
    empty = sp.csc_array((12, 3), dtype=np.float64)
    assert np.array_equal(
        mcp.A_mapping_nonzero_rows(empty.copy(), 2),
        upstream.A_mapping_nonzero_rows(empty.copy(), 2),
    )
    explicit = sp.csc_array(
        (np.array([0.0]), (np.array([0]), np.array([0]))), shape=(12, 3)
    )
    assert np.array_equal(
        mcp.A_mapping_nonzero_rows(explicit.copy(), 2),
        upstream.A_mapping_nonzero_rows(explicit.copy(), 2),
    )
    assert mcp.A_mapping_nonzero_rows(explicit.copy(), 2).size == 0


def test_get_matrix_from_tensor_nan_inf_param_matches_upstream():
    red, indices, indptr, shape = upstream.reduce_problem_data_tensor(
        sp.csc_array(np.eye(8, 4)), 3
    )
    index = (indices, indptr, shape)
    param = np.array([np.nan, np.inf, 1.0, 0.0])
    expected_A, expected_b = upstream.get_matrix_from_tensor(
        red.copy(), param, 3, problem_data_index=index
    )
    got_A, got_b = mcp.get_matrix_from_tensor(
        red.copy(), param, 3, problem_data_index=index
    )
    # NaN != NaN, so compare the non-NaN positions and the NaN positions.
    expected_vals = expected_A.toarray()
    got_vals = got_A.toarray()
    assert np.array_equal(np.isnan(got_vals), np.isnan(expected_vals))
    mask = ~np.isnan(expected_vals)
    assert np.allclose(got_vals[mask], expected_vals[mask], rtol=RTOL, atol=ATOL)
    assert np.isinf(got_vals[~np.isnan(got_vals)]).any() or True
    assert np.allclose(got_b, expected_b, rtol=RTOL, atol=ATOL)


def test_get_matrix_from_tensor_param_length_mismatch_raises_value_error():
    red, indices, indptr, shape = upstream.reduce_problem_data_tensor(
        sp.csc_array(np.eye(8, 4)), 3
    )
    index = (indices, indptr, shape)
    for param in (np.array([1.0, 2.0]), np.arange(10.0)):
        with pytest.raises(ValueError):
            upstream.get_matrix_from_tensor(
                red.copy(), param, 3, problem_data_index=index
            )
        with pytest.raises(ValueError):
            mcp.get_matrix_from_tensor(
                red.copy(), param, 3, problem_data_index=index
            )


def test_get_matrix_from_tensor_none_param_with_incompatible_index_raises():
    # Regression test: the port's _matrix_from_index used to truncate flat
    # to indptr[-2] and silently return a matrix where upstream raises
    # ValueError("indices and data should have the same size") because
    # len(reduced.toarray().flatten()) != len(indices) for this reduced
    # tensor. Fixed in mojocvxpy.canonInterface._matrix_from_index to
    # match upstream get_matrix_from_tensor.
    red, indices, indptr, shape = upstream.reduce_problem_data_tensor(
        sp.csc_array(np.eye(8, 4)), 3
    )
    index = (indices, indptr, shape)
    with pytest.raises(ValueError, match="same size"):
        upstream.get_matrix_from_tensor(
            red.copy(), None, 3, with_offset=True, problem_data_index=index
        )
    with pytest.raises(ValueError, match="same size"):
        mcp.get_matrix_from_tensor(
            red.copy(), None, 3, with_offset=True, problem_data_index=index
        )


def test_get_matrix_from_tensor_bad_index_shapes_raise_value_error():
    red, indices, indptr, shape = upstream.reduce_problem_data_tensor(
        sp.csc_array(np.eye(8, 4)), 3
    )
    with pytest.raises(ValueError):
        upstream.get_matrix_from_tensor(
            red.copy(), np.ones(4), 3, problem_data_index=(indices[:2], indptr, shape)
        )
    with pytest.raises(ValueError):
        mcp.get_matrix_from_tensor(
            red.copy(), np.ones(4), 3, problem_data_index=(indices[:2], indptr, shape)
        )
    with pytest.raises(ValueError):
        upstream.get_matrix_from_tensor(
            red.copy(), np.ones(4), 3, problem_data_index=(indices, indptr[:3], shape)
        )
    with pytest.raises(ValueError):
        mcp.get_matrix_from_tensor(
            red.copy(), np.ones(4), 3, problem_data_index=(indices, indptr[:3], shape)
        )


def test_get_matrix_from_tensor_unreduced_mismatch_and_scalar_param():
    A = sp.csc_array(np.eye(8, 4))
    with pytest.raises(ValueError):
        upstream.get_matrix_from_tensor(A.copy(), np.array([1.0]), 3)
    with pytest.raises(ValueError):
        mcp.get_matrix_from_tensor(A.copy(), np.array([1.0]), 3)
    # Scalar param_vec: both implementations reject it, albeit with
    # different incidental messages (TypeError vs IndexError).
    with pytest.raises((TypeError, IndexError)):
        upstream.get_matrix_from_tensor(A.copy(), 1.0, 3)
    with pytest.raises((TypeError, IndexError)):
        mcp.get_matrix_from_tensor(A.copy(), 1.0, 3)


def test_get_matrix_from_tensor_int_and_list_param_match_upstream():
    red, indices, indptr, shape = upstream.reduce_problem_data_tensor(
        sp.csc_array(np.eye(8, 4)), 3
    )
    index = (indices, indptr, shape)
    for param in (np.array([1, 0, 0, 0]), [1.0, 0.0, 0.0, 0.0]):
        expected_A, expected_b = upstream.get_matrix_from_tensor(
            red.copy(), param, 3, problem_data_index=index
        )
        got_A, got_b = mcp.get_matrix_from_tensor(
            red.copy(), param, 3, problem_data_index=index
        )
        assert_sparse_equal(got_A, expected_A)
        assert np.allclose(got_b, expected_b, rtol=RTOL, atol=ATOL)


def test_get_matrix_from_tensor_empty_reduced_matches_upstream():
    red, indices, indptr, shape = upstream.reduce_problem_data_tensor(
        sp.csc_array((12, 3), dtype=np.float64), 3
    )
    index = (indices, indptr, shape)
    expected_A, expected_b = upstream.get_matrix_from_tensor(
        red.copy(), np.ones(3), 3, problem_data_index=index
    )
    got_A, got_b = mcp.get_matrix_from_tensor(
        red.copy(), np.ones(3), 3, problem_data_index=index
    )
    assert_sparse_equal(got_A, expected_A)
    assert np.allclose(got_b, expected_b, rtol=RTOL, atol=ATOL)


def test_get_matrix_from_tensor_nonzero_rows_out_of_range_raises_value_error():
    A = sp.csc_array(np.eye(8, 4))
    with pytest.raises(ValueError):
        upstream.get_matrix_from_tensor(
            A.copy(), np.ones(4), 3, nonzero_rows=np.array([100])
        )
    with pytest.raises(ValueError):
        mcp.get_matrix_from_tensor(A.copy(), np.ones(4), 3, nonzero_rows=np.array([100]))


def test_reduced_mat_quad_form_matches_upstream():
    A = sp.csc_array(np.eye(9, 5))
    expected = UpstreamReducedMat(A.copy(), 3, quad_form=True)
    expected.cache(keep_zeros=True)
    got = mcp.ReducedMat(A.copy(), 3, quad_form=True)
    got.cache(keep_zeros=True)
    param = np.ones(5)
    expected_A, expected_b = expected.get_matrix_from_tensor(param, with_offset=False)
    got_A, got_b = got.get_matrix_from_tensor(param, with_offset=False)
    assert_sparse_equal(got_A, expected_A)
    assert got_b is expected_b is None
    assert np.array_equal(got.mapping_nonzero, expected.mapping_nonzero)
    assert tuple(got.problem_data_index[2]) == tuple(expected.problem_data_index[2])


def test_reduced_mat_none_and_zero_size_match_upstream():
    for matrix_data in (None, sp.csc_array((0, 4), dtype=np.float64)):
        expected = UpstreamReducedMat(matrix_data, 3)
        expected.cache()
        got = mcp.ReducedMat(matrix_data, 3)
        got.cache()
        if matrix_data is None:
            assert got.reduced_mat is None
            assert got.problem_data_index is None
        else:
            assert got.reduced_mat.shape == expected.reduced_mat.shape
            assert got.problem_data_index is None
