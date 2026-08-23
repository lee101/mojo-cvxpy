# mojo-cvxpy

Mojo acceleration for the sparse problem-data tensor stage of
[CVXPY](https://www.cvxpy.org/) canonicalization. It is an add-on to CVXPY,
not a modeling-language fork: existing `Variable`, `Parameter`, `Problem`, and
solver APIs remain unchanged.

The useful boundary covered here is CVXPY's `ReducedMat` path. Canonicalization
first builds a sparse tensor mapping parameters to solver matrices. That tensor
is condensed once and multiplied by a new parameter vector every time a
parameterized problem is reapplied. mojo-cvxpy ports the row condensation,
sparse matrix-vector product, and direct CSC reconstruction used by that path.

## Covered subset

The Python module mirrors these names and signatures from
`cvxpy.cvxcore.python.canonInterface`:

| API | coverage |
| --- | --- |
| `get_parameter_vector` | CVXPY-compatible Fortran-order parameter packing |
| `reduce_problem_data_tensor` | affine and quadratic tensor condensation |
| `get_matrix_from_tensor` | reduced and unreduced tensor application, offsets, explicit zeros |
| `nonzero_csc_array` | stored CSC coordinates, including explicit zeros |
| `A_mapping_nonzero_rows` | parameter-affected coefficient rows |
| `ReducedMat` | compatible cache and application helper |
| `install`, `uninstall`, `accelerated` | opt-in routing of real CVXPY reductions through Mojo |

The parity suite covers duplicate sparse entries, explicit zeros, constant-only
tensors, affine and quadratic layouts, offset extraction, retained structural
zeros, and a real parameterized CVXPY cone program.

Not covered are expression-tree construction (`get_problem_matrix`), DCP/DGP
rewrites, atom canonicalizers, cone selection, solver interfaces, or numerical
solvers. Those continue to run in upstream CVXPY. The Mojo kernels target
CVXPY's normal real problem-data path: real inputs are normalized to `float64`
at the boundary, and complex tensors or parameter vectors are not supported.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi install` provides the pinned Mojo nightly, Python, CVXPY, NumPy, SciPy,
and the test tools. `pixi run build` writes
`dist/libmojo-cvxpy.so`.

## Usage

The context manager is the smallest opt-in and restores upstream CVXPY
afterward:

```python
import cvxpy as cp
import numpy as np
import mojocvxpy as mcp

x = cp.Variable(4)
target = cp.Parameter(4, value=np.array([1.0, -2.0, 0.5, 3.0]))
problem = cp.Problem(cp.Minimize(cp.sum_squares(x - target)), [x >= 0])

with mcp.accelerated():
    value = problem.solve(solver=cp.CLARABEL)

print(round(value, 2))  # 4.0
```

For a process that always wants acceleration, call `mcp.install()` once and
`mcp.uninstall()` before restoring CVXPY's original functions. The covered
functions can also be called directly with their upstream signatures.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
CVXPY 1.9.2, and Python 3.13.14. Times are the best of three warm runs on a
500,000-triplet parameter tensor.

| case | mojo-cvxpy | CVXPY | result |
| --- | ---: | ---: | ---: |
| reduce problem-data tensor (500k triplets) | 63.99 ms | 74.23 ms | 1.16x faster |
| apply parameter vector and rebuild CSC | 0.38 ms | 4.52 ms | 11.84x faster |
| 20 cached parameter applications | 27.91 ms | 174.95 ms | 6.27x faster |

The one-time condensation preserves SciPy's native 32- or 64-bit index buffers
across the ABI. For CSC input, Mojo counts source rows and scatters directly
into the compact CSR buffers instead of first materializing a CSR row pointer
for every empty source row. It then uses the sorted nonempty row IDs directly
to build CSC column boundaries. Repeated parameter application is the intended
workload: Mojo evaluates CSR rows and the Python layer reuses the result
buffers directly as CSC data instead of constructing and slicing an
intermediate sparse matrix.

No GPU path is provided. The covered operations are sparse conversions,
index scans, and memory-bound CSR matrix-vector products, for which device
transfer and launch overhead would work against the intended workload.

## How it works

`src/capi.mojo` is one compilation unit and exports a small C ABI. Python calls
it with `ctypes`; NumPy-owned buffers cross as integer addresses because Mojo
exports cannot be parametric. The wrapper reconstructs mutable
`UnsafePointer[..., AnyOrigin[mut=True]]` values only inside the exported
functions.

Sparse input uses SciPy's CSR layout: contiguous `float64` data with native
`int32` or `int64` column indices and row pointers. Eligible CSR buffers pass
directly across the FFI boundary without normalization or copying. Long rows
use native-width SIMD loads, an indexed gather, and a scalar remainder loop;
short rows skip SIMD setup. Large matvecs are split across a bounded CPU worker
pool; smaller inputs stay serial. Condensed tensor values are already
ordered as the eventual solver matrix's CSC data, so the wrapper attaches the
original row-index and column-pointer arrays without another transpose or
sort. All allocations and lifetimes remain owned by NumPy/SciPy; Mojo neither
allocates nor retains Python memory.

## License

MIT
