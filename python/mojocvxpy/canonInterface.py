"""Drop-in implementations of CVXPY's problem-data tensor utilities.

These functions preserve the signatures and sparse-matrix contracts in
``cvxpy.cvxcore.python.canonInterface``. The covered path is the condensation
and repeated application of canonicalized parameter tensors.
"""

from __future__ import annotations

from contextlib import contextmanager
import operator
from typing import Callable, Iterator

import numpy as np
import scipy.sparse as sp

from ._lib import addr, f64, lib

try:
    from cvxpy.lin_ops.lin_op import CONSTANT_ID
except ImportError:
    CONSTANT_ID = -1


def get_parameter_vector(
    param_size,
    param_id_to_col,
    param_id_to_size,
    param_id_to_value_fn,
    zero_offset: bool = False,
):
    """Return CVXPY's Fortran-flattened parameter vector."""
    if param_size == 0:
        return None
    param_vec = np.zeros(param_size + 1)
    for param_id, col in param_id_to_col.items():
        if param_id == CONSTANT_ID:
            if not zero_offset:
                param_vec[col] = 1
        else:
            value = np.asarray(param_id_to_value_fn(param_id)).flatten(order="F")
            size = param_id_to_size[param_id]
            param_vec[col : col + size] = value
    return param_vec


def _csr(matrix) -> sp.csr_array:
    if np.issubdtype(np.dtype(matrix.dtype), np.complexfloating):
        raise TypeError("complex tensors are not supported")
    csr = sp.csr_array(matrix, dtype=np.float64)
    csr.sum_duplicates()
    csr.sort_indices()
    csr.check_format(full_check=True)
    index_dtype = (
        csr.indices.dtype
        if csr.indices.dtype in (np.dtype(np.int32), np.dtype(np.int64))
        and csr.indices.dtype == csr.indptr.dtype
        else np.dtype(np.int64)
    )
    csr = sp.csr_array(
        (
            np.ascontiguousarray(csr.data, dtype=np.float64),
            np.ascontiguousarray(csr.indices, dtype=index_dtype),
            np.ascontiguousarray(csr.indptr, dtype=index_dtype),
        ),
        shape=csr.shape,
        copy=False,
    )
    return csr


def _csc(matrix) -> sp.csc_array:
    if np.issubdtype(np.dtype(matrix.dtype), np.complexfloating):
        raise TypeError("complex tensors are not supported")
    csc = sp.csc_array(matrix, dtype=np.float64)
    csc.sum_duplicates()
    csc.sort_indices()
    csc.check_format(full_check=True)
    index_dtype = (
        csc.indices.dtype
        if csc.indices.dtype in (np.dtype(np.int32), np.dtype(np.int64))
        and csc.indices.dtype == csc.indptr.dtype
        else np.dtype(np.int64)
    )
    csc = sp.csc_array(
        (
            np.ascontiguousarray(csc.data, dtype=np.float64),
            np.ascontiguousarray(csc.indices, dtype=index_dtype),
            np.ascontiguousarray(csc.indptr, dtype=index_dtype),
        ),
        shape=csc.shape,
        copy=False,
    )
    return csc


def _csr_matvec(matrix, vector) -> np.ndarray:
    csr = _csr(matrix)
    vector = f64(vector)
    if vector.ndim != 1 or vector.size != csr.shape[1]:
        raise ValueError("dimension mismatch")
    result = np.empty(csr.shape[0], dtype=np.float64)
    if csr.nnz == 0:
        result.fill(0)
    elif csr.shape[0]:
        fn = (
            lib().mcvx_csr_matvec_i32
            if csr.indices.dtype == np.int32
            else lib().mcvx_csr_matvec
        )
        fn(
            addr(csr.data),
            addr(csr.indices),
            addr(csr.indptr),
            addr(vector),
            addr(result),
            csr.shape[0],
            csr.nnz,
        )
    return result


def reduce_problem_data_tensor(A, var_length, quad_form: bool = False):
    """Condense a CVXPY problem-data tensor to its structurally nonzero rows."""
    if 0 in A.shape:
        raise ValueError("problem data tensor must not have a zero dimension")
    var_length = operator.index(var_length)
    n_cols = var_length + (0 if quad_form else 1)
    if n_cols <= 0:
        raise ValueError("var_length is incompatible with the tensor layout")
    n_constr, remainder = divmod(A.shape[0], n_cols)
    if remainder:
        raise ValueError("tensor row count is incompatible with var_length")
    if getattr(A, "format", None) == "csr":
        csr = _csr(A)
        csr.eliminate_zeros()
        rows = csr.shape[0]
        old_rows = np.empty(rows, dtype=np.int64)
        reduced_indptr = np.empty(rows + 1, dtype=csr.indptr.dtype)
        fn = (
            lib().mcvx_csr_nonempty_rows_i32
            if csr.indptr.dtype == np.int32
            else lib().mcvx_csr_nonempty_rows
        )
        count = fn(
            addr(csr.indptr), rows, addr(old_rows), addr(reduced_indptr)
        )
        old_rows = old_rows[:count]
        reduced_indptr = reduced_indptr[: count + 1]
        reduced_A = sp.csr_array(
            (csr.data, csr.indices, reduced_indptr),
            shape=(count, csr.shape[1]),
            copy=False,
        )
    else:
        csc = _csc(A)
        csc.eliminate_zeros()
        rows, cols = csc.shape
        old_rows = np.empty(rows, dtype=np.int64)
        scratch = np.zeros(rows, dtype=csc.indptr.dtype)
        reduced_data = np.empty(csc.nnz, dtype=np.float64)
        reduced_indices = np.empty(csc.nnz, dtype=csc.indices.dtype)
        reduced_indptr = np.empty(rows + 1, dtype=csc.indptr.dtype)
        if csc.nnz:
            fn = (
                lib().mcvx_csc_compact_i32
                if csc.indices.dtype == np.int32
                else lib().mcvx_csc_compact
            )
            count = fn(
                addr(csc.data),
                addr(csc.indices),
                addr(csc.indptr),
                rows,
                cols,
                csc.nnz,
                addr(scratch),
                addr(reduced_data),
                addr(reduced_indices),
                addr(reduced_indptr),
                addr(old_rows),
            )
        else:
            count = 0
            reduced_indptr[0] = 0
        old_rows = old_rows[:count]
        reduced_indptr = reduced_indptr[: count + 1]
        reduced_A = sp.csr_array(
            (reduced_data, reduced_indices, reduced_indptr),
            shape=(count, cols),
            copy=False,
        )

    indices = old_rows % n_constr
    boundaries = np.arange(n_cols + 1, dtype=np.int64) * n_constr
    indptr = np.searchsorted(old_rows, boundaries).astype(np.int64, copy=False)
    return reduced_A, indices, indptr, (n_constr, n_cols)


def nonzero_csc_array(A):
    """Return stored CSC coordinates, including explicitly stored zeros."""
    zero_indices = A.data == 0
    try:
        A.data[zero_indices] = np.nan
        rows, cols = A.nonzero()
        order = np.argsort(cols, kind="mergesort")
        rows, cols = rows[order], cols[order]
    finally:
        A.data[zero_indices] = 0
    return rows, cols


def A_mapping_nonzero_rows(problem_data_tensor, var_length):
    """Return coefficient-map rows affected by non-constant parameters."""
    tensor = problem_data_tensor.tocsc()
    nrows = tensor.shape[0] // (var_length + 1)
    mapping = tensor[: nrows * var_length, :-1]
    rows, _ = mapping.nonzero()
    return np.unique(rows)


def _matrix_from_index(flat, problem_data_index, with_offset):
    indices, indptr, shape = problem_data_index
    indices = np.asarray(indices)
    indptr = np.asarray(indptr)
    flat = np.asarray(flat, dtype=np.float64)
    if with_offset:
        split = int(indptr[-2])
        A = sp.csc_array(
            (flat[:split], indices[:split], indptr[:-1]),
            shape=(shape[0], shape[1] - 1),
            copy=False,
        )
        b = np.zeros(shape[0], dtype=np.float64)
        b[indices[split : int(indptr[-1])]] = flat[split : int(indptr[-1])]
        return A, np.squeeze(b)
    return (
        sp.csc_array((flat, indices, indptr), shape=shape, copy=False),
        None,
    )


def get_matrix_from_tensor(
    problem_data_tensor,
    param_vec,
    var_length,
    nonzero_rows=None,
    with_offset=True,
    problem_data_index=None,
):
    """Apply a parameter vector and reconstruct CVXPY's CSC matrix and offset."""
    if problem_data_index is not None:
        if param_vec is None:
            flat = np.asarray(problem_data_tensor.toarray()).flatten()
        else:
            flat = _csr_matvec(problem_data_tensor, param_vec)
        A, b = _matrix_from_index(flat, problem_data_index, with_offset)
    else:
        if param_vec is None:
            flat_problem_data = problem_data_tensor
        else:
            sparse_vec = sp.csc_array(np.asarray(param_vec)[:, None])
            flat_problem_data = problem_data_tensor @ sparse_vec
        n_cols = var_length + (1 if with_offset else 0)
        M = flat_problem_data.reshape((-1, n_cols), order="F").tocsc()
        if with_offset:
            A = M[:, :-1].tocsc()
            b = np.squeeze(M[:, [-1]].toarray().flatten())
        else:
            A, b = M.tocsc(), None

    if nonzero_rows is not None and np.asarray(nonzero_rows).size:
        nrows = A.shape[0]
        A_rows, A_cols = nonzero_csc_array(A)
        extra = np.asarray(nonzero_rows, dtype=np.int64)
        values = np.append(A.data, np.zeros(extra.size))
        rows = np.append(A_rows, extra % nrows)
        cols = np.append(A_cols, extra // nrows)
        A = sp.csc_array((values, (rows, cols)), shape=A.shape)
    return A, b


class ReducedMat:
    """API-compatible condensed parameter tensor used by CVXPY reductions."""

    def __init__(self, matrix_data, var_len: int, quad_form: bool = False) -> None:
        self.matrix_data = matrix_data
        self.var_len = var_len
        self.quad_form = quad_form
        self.reduced_mat = None
        self.problem_data_index = None
        self.mapping_nonzero = None

    def cache(self, keep_zeros: bool = False) -> None:
        if self.matrix_data is None:
            return
        if self.reduced_mat is None:
            if np.prod(self.matrix_data.shape) != 0:
                reduced, indices, indptr, shape = reduce_problem_data_tensor(
                    self.matrix_data, self.var_len, self.quad_form
                )
                self.reduced_mat = reduced
                self.problem_data_index = (indices, indptr, shape)
            else:
                self.reduced_mat = self.matrix_data
                self.problem_data_index = None
        if keep_zeros and self.mapping_nonzero is None:
            self.mapping_nonzero = A_mapping_nonzero_rows(
                self.matrix_data, self.var_len
            )

    def get_matrix_from_tensor(self, param_vec, with_offset: bool = True):
        return get_matrix_from_tensor(
            self.reduced_mat,
            param_vec,
            self.var_len,
            nonzero_rows=self.mapping_nonzero,
            with_offset=with_offset,
            problem_data_index=self.problem_data_index,
        )


_UPSTREAM_NAMES = (
    "get_parameter_vector",
    "reduce_problem_data_tensor",
    "nonzero_csc_array",
    "A_mapping_nonzero_rows",
    "get_matrix_from_tensor",
)
_originals: dict[str, Callable] | None = None


def install() -> None:
    """Route CVXPY's covered canonicalization utilities through Mojo."""
    global _originals
    if _originals is not None:
        return
    from cvxpy.cvxcore.python import canonInterface as upstream

    _originals = {name: getattr(upstream, name) for name in _UPSTREAM_NAMES}
    for name in _UPSTREAM_NAMES:
        setattr(upstream, name, globals()[name])


def uninstall() -> None:
    """Restore CVXPY's original utility functions."""
    global _originals
    if _originals is None:
        return
    from cvxpy.cvxcore.python import canonInterface as upstream

    for name, fn in _originals.items():
        setattr(upstream, name, fn)
    _originals = None


@contextmanager
def accelerated() -> Iterator[None]:
    """Temporarily enable Mojo canonicalization inside CVXPY."""
    owns_install = _originals is None
    if owns_install:
        install()
    try:
        yield
    finally:
        if owns_install:
            uninstall()
