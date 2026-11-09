# mojo-cvxpy — ANALYZE phase: measured hotspot map

Measured in-session on the bench machine, no code changes. Every number below
is from a run executed for this analysis (see "Method" for how each number
was isolated). No estimates are used in the table; memory-traffic and FLOP
columns are exact counts from the buffer sizes and dimensions that were
timed, not predicted.

- Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 visible CPUs
- Software: CVXPY 1.9.2, Python 3.13.14, numpy 2.5.1, scipy 1.18.0
- Bench run (`pixi run bench`, flock-wrapped, this session): reduce 46.63 ms
  (1.61x), apply 1.72 ms (2.94x), 20x apply 57.99 ms (3.22x) — consistent
  with the independently re-timed numbers below (bench.py uses best-of-3;
  the extra profiler used best-of-5/7/9 after warmup).
- Bench fixture (real-caller size, from `bench/bench.py`): CSC tensor,
  shape `(1_470_000, 65)`, 500k generated triplets → 498,655 stored nnz,
  `int64` indices; `var_length=48`, `n_constr=30,000`, 64 params. Reduced
  form: 423,769 CSR rows × 65 cols, 498,655 nnz, `int32` indices
  (1.18 nnz/row). Small fixture: 700 triplets, `var_length=7` (parity-suite size).
- Rule note: nothing in `src/` or `tests/` was changed. This file is the
  only added artifact.

## Ranked hotspot table

Rank is by measured time at the major sizes (seconds dominate seconds:
the `A_mapping_nonzero_rows` row is ranked first because at 176 ms it is
the single slowest entry point even though it is pure Python/scipy on both
sides; the reduce-CSC kernel is the largest *Mojo-kernel* hotspot).

| # | function (entry point) | input size | time (mojo path) | bytes touched | flops | flops/byte | memory- or compute-bound | Python/ctypes glue or kernel |
|---|---|---|---|---|---|---|---|---|
| 1 | `A_mapping_nonzero_rows` | bench tensor (498,655 nnz) | 175.8 ms (upstream 177.9 ms, 1.01x) | ~85 MB (CSC rebuild + slice + nonzero + unique) | ~1.0M (unique/sort dominated) | 0.01 | memory-bound (sort-bound) | Python/scipy glue, no ctypes at all — identical code both sides |
| 2 | `reduce_problem_data_tensor` (CSC) — kernel `mcvx_csc_compact` | bench tensor (498,655 nnz, 1,470,000 rows) | 43.6 ms total; kernel 37.1 ms + glue 6.5 ms (upstream 70.3 ms) | 8.0 MB in, 43.3 MB out = 51.3 MB | ~2.97M (histogram + scan + scatter ≈ 3·nnz + rows) | 0.06 | memory-bound | 85% kernel, 15% glue |
| 3 | `get_matrix_from_tensor` unreduced (`problem_data_index=None`) | bench tensor (498,655 nnz) | 106.5 ms (upstream 98.2 ms, **0.92x — mojo slower, kept**) | ~16 MB (sparse@vec + reshape + 2× `.tocsc()`) | ~1.0M | 0.06 | memory-bound (sparse format conversion) | Python/scipy glue, **no Mojo kernel on this branch** |
| 4 | `reduce_problem_data_tensor` (CSR) — kernel `mcvx_csr_nonempty_rows` | bench tensor as CSR (498,655 nnz) | 39.0 ms total; kernel 5.1 ms + glue 34.0 ms (upstream 74.9 ms) | 19.7 MB in, 23.5 MB out | ~2.94M (2·rows) | 0.07 | memory-bound | **13% kernel, 87% glue** — the kernel is the cheap part |
| 5 | `get_matrix_from_tensor` cached (the intended hot loop) — kernel `mcvx_csr_matvec` | reduced CSR (423,769 rows × 65, 498,655 nnz, int32) | 1.61 ms (upstream 5.35 ms); kernel 1.38–1.78 ms, wrap 0.35 ms (upstream's win is NOT the matvec — see §2) | 11.1 MB (data 4.0 + idx 2.0 + indptr 1.7 + result 3.4) | 1.0M (2·nnz) | 0.09 | memory-bound | kernel dominates steady state; see glue sub-table |
| 6 | `nonzero_csc_array` | bench CSC result (415,180 nnz) | 49.9 ms (upstream 46.6 ms, **0.93x — mojo slower, kept**) | ~3.3 MB data + 2 × int64 arrays | ~0.8M + sort ≈ 0.02 | memory-bound (sort-bound) | Python/scipy glue, no ctypes — identical code both sides; delta is noise (48.1 vs 48.3 ms on rerun) |
| 7 | `get_parameter_vector` | 500 params × 5-vector | 0.48 ms (upstream 0.45 ms, 0.94x — parity, kept) | ~20 KB | ~2.5K | 0.13 | memory-bound (tiny) | Python loop, no ctypes — identical code both sides |
| 8 | `ReducedMat` ctor + `cache()` + `get_matrix_from_tensor` | bench tensor | 45.4 ms (one reduce + one apply) | = reduce-CSC + apply | = row 2 + row 5 | 0.06–0.09 | memory-bound | thin wrapper; cost is rows 2 + 5 |
| 9 | small-size spot checks (`reduce`/`apply`, 700 triplets, var_length 7) | small fixture | reduce 0.22 ms (upstream 0.41 ms); apply 0.054 ms (upstream 0.27 ms) | KB-scale | KB-scale | ~0.1 | memory-bound | glue-dominated; single ctypes floor 10 µs (see §3) |

Notes on rows that break the naive reading (all verified in-session):

- Row 5's upstream comparison needs care. Upstream's `get_matrix_from_tensor`
  (`reduced @ param_vec`) costs 5.35 ms while a bare scipy `csr @ vec` on the
  same matrix costs 1.5–1.7 ms — upstream pays ~3.6 ms elsewhere (fancy-index
  rebuild / CSC wrap path). Mojo's full apply is 1.44–1.61 ms, of which the
  Mojo kernel is 1.38–1.78 ms, so the Mojo kernel itself is **at parity with
  scipy matvec, not faster** (int32: 1.43–1.78 ms vs scipy 1.71 ms; int64:
  1.77 ms vs scipy 1.74 ms). The 3.3x headline in `bench.py` comes from
  Mojo's Python glue doing less work around the matvec, not from a faster
  matvec. Verified with int32/int64 kernel-vs-scipy timings in §2.
- Row 2's kernel figure includes a fresh `np.zeros` scratch per call (the
  kernel destroys it in place — see §4). Zeroing alone is 0.56 ms of the
  37.1 ms kernel time.
- Row 1 and rows 6/7 are not Mojo code at all: `A_mapping_nonzero_rows`,
  `nonzero_csc_array`, and `get_parameter_vector` are Python/scipy
  implementations copied from upstream with no ctypes in the path. They are
  in the table because they are public entry points ranked by time, and
  row 1 is the slowest function in the whole port.

## How the table was measured (method, so the numbers can be re-checked)

1. `pixi run bench` first (flock lock held; no concurrent benchmarks).
2. Each public entry point timed best-of-N (N = 5–21 depending on cost)
   after warmup, at bench size and at small size.
3. Kernel-vs-glue split: `T_total` = full `mcp.` call; `T_kernel` = bare
   ctypes call to the `mcvx_*` export with pre-allocated buffers and
   pre-computed `addr()`s; `T_glue = T_total − T_kernel`. Two independent
   profiler runs (fresh processes, deduped deltas, nonzero repeats) were used;
   where they disagreed (row 5, which sits at timer resolution), the report
   carries the range and the steady-state post-warmup figure (1.44 ms total,
   1.38 ms kernel).
4. Bytes = `nbytes` of every buffer read or written (in + out; allocations
   counted separately as "per-call allocations" in §3). Flops for CSR matvec =
   2·nnz (one multiply-add per nonzero); for CSC compaction ≈ 3·nnz + rows
   (histogram increment + prefix + scatter). Asymmetric gathers mean these
   are counts of the algorithm's operations, not hardware FLOPs — used only
   as a traffic-intensity figure, not a performance claim.
5. Baselines: upstream `cvxpy.cvxcore.python.canonInterface` for every row;
   plain numpy/scipy for the kernel rows (scipy `csr @ vec`, `np.unique`,
   `np.searchsorted`, `(indptr[1:]−indptr[:-1] > 0).nonzero()`).

## 1. Per-function findings (the "what" for the next phases)

### 1.1 `A_mapping_nonzero_rows` — 175.8 ms, pure scipy, slowest entry point

Piece timings on the bench tensor (all measured in-session):

| piece | time |
|---|---|
| `tensor.copy().tocsc()` | 0.58 ms |
| `+ slice [:nrows·48, :-1]` | 3.06 ms |
| `+ .nonzero()` | 45.6 ms |
| `+ np.unique(rows)` (full → 167.9 ms) | 167.9 ms |
| `np.unique` on the full 498k-row nonzero index alone | 137.3 ms |

The entire cost is `nonzero()` + `unique()`: a full CSC→COO conversion,
a materialized 498k-row index array, and a sort of that array. This is the
identical algorithm upstream runs (`canonInterface.A_mapping_nonzero_rows`
is line-for-line the same shape: `tocsc()` → slice → `nonzero()` →
`unique()`), so there is no Mojo kernel to speed it up *yet* — the Mojo
`A_mapping_nonzero_rows` never touches ctypes. If this function is on a
hot path in callers' workloads (it is called once per `ReducedMat.cache(
keep_zeros=True)`, not per solve), the win is replacing the materialized
sort with an index-scan kernel (the CSC `indptr` scan already exists in
`mcvx_csc_compact` form). As measured: memory-bound, sort-dominated,
0.01 flops/byte.

### 1.2 `reduce_problem_data_tensor` (CSC) — 43.6 ms, the main *Mojo-kernel* hotspot

Measured split: kernel `mcvx_csc_compact` **37.1 ms**, glue **6.5 ms** (15%).
Glue pieces (measured): `tensor.copy()` 0.64 ms, `mcp._csc` + `eliminate_zeros`
2.06 ms, `old_rows % n_constr` 5.17 ms, `searchsorted` 0.003 ms, misc overhead
the remainder. Note the input tensor here is `int64`-indexed (SciPy's default
from the bench generator), so the port runs the **i64** kernel variant;
an int32-converted copy (`mcvx_csc_compact_i32`) times at **28.6 ms vs
35.4 ms** for the same input (fresh buffers each time) — the i64 index
traffic alone costs ~7 ms (19%) at this size, because every index read and
scatter store moves twice the bytes. `numpy unique(row)` on the same COO
row array costs 61.7 ms as a baseline, so the kernel beats the obvious
numpy condensation by ~1.7x; upstream's full reduce (107.7 ms on reruns at
worse machine load, 70.3 ms best-of) is beaten 1.6–1.8x.

Scaling (measured, `var_length=48`): 50k entries → mojo **10.7 ms vs upstream
6.4 ms (0.60x — SLOWER, kept)**; 200k → 19.2 vs 26.8 ms (1.40x); 500k → 53.5
vs 93.8 ms (1.75x); 1M → 79.4 vs 178.4 ms (2.25x). The 50k regression is the
fixed cost (1.47M-row `np.zeros` scratch + 11.8 MB buffer allocation +
`mcp._csc` normalization ≈ 4 ms) dominating when the histogram/scatter work
is only 50k ops; the crossover is between 50k and 200k entries. The port
wins increasingly as size grows, but the small-size fixed cost is real and
never amortized because every call pays it.

### 1.3 `get_matrix_from_tensor` unreduced — 106.5 ms, **mojo slower (0.92x), no Mojo kernel**

When `problem_data_index is None` (the un-cached path), `canonInterface.py`
falls through to the scipy-only branch (`sparse_vec` construction +
`tensor @ sparse_vec` + `reshape(...).tocsc()` + column slice + `.tocsc()`
again). The Mojo kernels are never invoked; the port's copy is
algorithmically identical to upstream, and the measured delta (106.5 ms vs
98.2 ms; rerun 109.4 vs 104.3 ms) is the extra `np.ascontiguousarray`/dtype
checking in the wrapper. This is the right call for a *cached* workload (the
cached path is what `bench.py` headlines) but it means any caller that
applies without caching gets no benefit and slightly worse performance.
Flagged here so the next phase decides on evidence: either route the
unreduced path through the same CSR matvec (it would still need the
reshape/tocsc, which dominate at 80.3 ms of the 106 ms), or document that
unreduced application is intentionally unsupported.

### 1.4 `reduce_problem_data_tensor` (CSR) — 39.0 ms, **87% glue**

The Mojo kernel `mcvx_csr_nonempty_rows` is **5.1 ms**; the Python glue is
**34.0 ms**. Breakdown of the glue (all measured): `sp.csr_array(csr,
dtype=float64)` normalization + `sum_duplicates` + `sort_indices` = **5.0
ms** (sort alone ≈ 0.1 ms on top of sum_duplicates); `check_format(
full_check=True)` = **+9.5 ms** (14.5 ms cumulative); rebuild into a fresh
`csr_array` = another ~0.4 ms + `eliminate_zeros` on the canonical form =
**+6.4 ms** (21.3 ms total for `mcp._csr` + eliminate). The full `_csr`
wrapper (including the dtype-preservation rebuild) is what dominates; it
is pure SciPy, no Mojo. Baselines for the kernel's own job: numpy
`(indptr[1:]−indptr[:-1] > 0).nonzero()` = **2.8 ms** — *faster than the
5.1 ms Mojo kernel* — and `np.diff(indptr) > 0` + nonzero = 16.1 ms (the
dtype cast kills it). Byte accounting: 19.7 MB in, 23.5 MB out, 0.07
flops/byte — memory-bound. This row's actionable hotspot is the
normalization chain (`sum_duplicates` + `check_format(full_check=True)` +
`eliminate_zeros`), not the Mojo kernel.

### 1.5 `get_matrix_from_tensor` cached — 1.44–1.61 ms, kernel at parity with scipy

Full measured decomposition (post-warmup, best-of, one process):

| piece | time |
|---|---|
| full `mcp.get_matrix_from_tensor` | 1.44 ms |
| kernel-only `mcvx_csr_matvec` (precomputed `addr()`s, preallocated result) | 1.38 ms |
| `_matrix_from_index` (CSC wrap + offset split) | 0.056 ms without `.copy()`, 0.35 ms with `result.copy()` (the copy is the cost) |
| scipy `csr @ vec` on the same matrix | 1.50 ms |
| first-iteration after warmup (per-iteration series) | 1.9–3.0 ms (first), ~1.56 ms steady |

Row-width probe (kernel time per nonzero, 10,000-row synthetic CSRs,
65→40 cols): 1 nnz/row 2.1 ns/nnz, 2 → 1.5, 4 → 1.7, 8 → 1.5, 19 → 1.2,
32 → 1.1 ns/nnz — i.e. long rows are ~2x cheaper per nonzero than short
rows (SIMD gather amortizes), but at the bench's 1.18 nnz/row the kernel is
in the worst regime for SIMD (mostly scalar-tail paths: `if end−cursor >=
W` fails for most rows, so the gather never fires).

Scaling (synthetic CSR, 65 cols, measured kernel vs scipy `@`):

| rows | nnz | kernel | scipy | speedup | eff. BW | flops/byte |
|---|---|---|---|---|---|---|
| 10,000 | 48,097 | 0.116 ms | 0.102 ms | 0.88x | 8.0 GB/s | 0.10 |
| 70,000 | 336,962 | 0.815 ms | 0.822 ms | 1.01x | 8.0 GB/s | 0.10 |
| 200,000 | 962,364 | 2.402 ms | 2.347 ms | 0.98x | 7.7 GB/s | 0.10 |
| 1,470,000 | 7,074,154 | 22.671 ms | 20.375 ms | 0.90x | 6.0 GB/s | 0.10 |
| 423,769 (bench, int32) | 498,655 | 1.43–1.78 ms | 1.71 ms | ~0.9–1.0x | ~6–8 GB/s | 0.09 |
| 423,769 (bench, int64 idx) | 498,655 | 1.77 ms | 1.74 ms | ~1.0x | ~6 GB/s | 0.09 |

Verdict on evidence: at this shape the Mojo matvec is **BW-parity with
scipy, not faster** — the dense-vector gather dominates and both kernels
are gather-bound, ~6–8 GB/s effective, 0.09 flops/byte. The `bench.py`
"2.94x" headline is a *whole-function* win (Mojo's Python glue avoids
upstream's ~3.6 ms of extra indexing/rebuild around the matvec), not a
kernel win. Any "speed up the matvec" phase must be honest about this:
the headroom against scipy is small; the headroom against upstream's
full function is already taken.

A pure-Python per-element loop on a 2,000-row slice costs 2.45 ms
(~528 ms extrapolated to full size), confirming the kernel is ~300x a
Python loop and the comparison that matters is against scipy, not Python.

### 1.6 `nonzero_csc_array`, `get_parameter_vector` — parity rows, correctly kept

`nonzero_csc_array` bench: mojo 49.9 ms vs upstream 46.6 ms (0.93x);
rerun 48.1 vs 48.3 ms (parity). Core cost (no argsort): 18.3 ms; the
`mergesort` argsort of the column array alone: 43.1 ms. The function is
`np.argsort(cols, kind="mergesort")`-bound; both implementations are the
same code with no ctypes, so the delta is session noise. Included as a
public entry point with the row kept per the task rule.

`get_parameter_vector`: 2-param 0.003 ms both; 500 params 0.48 ms (mojo)
vs 0.45 ms (upstream) vs 0.44 ms (numpy loop of identical structure). Pure
Python dict iteration + slice assignment; no Mojo involvement in either
implementation. Correctly is not a Mojo kernel — there is nothing to port.

### 1.7 `ReducedMat`

45.4 ms for ctor + `cache()` (one CSC reduce) + one cached apply — i.e.
row 2 + row 5 with no additional overhead. No standalone hotspot; the
wrapper is thin and correct.

## 2. Glue-vs-kernel: which side of the FFI?

Per ABI call site in `canonInterface.py`:

| call site | per-call Python work before ctypes | measured cost | kernel | share |
|---|---|---|---|---|
| `mcvx_csr_matvec` (apply) | `_csr_matvec` flag checks, `f64(vec)` (no-op if already f64 contiguous — measured 0.000 ms), `np.empty(result)`, `addr()` ×5 (0.008 ms), `lib()` (0.000 ms) | ≤0.01 ms of *pure* glue; wrap 0.06–0.35 ms dominates | 1.38 ms | kernel ~95%+ |
| `mcvx_csc_compact` (reduce CSC) | `_csc` (2.06 ms incl `sum_duplicates`+`sort_indices`+`check_format`), `eliminate_zeros`, `np.zeros` scratch (0.56 ms), 5 output allocations, 11× `addr()` | 6.5 ms | 37.1 ms | kernel 85% |
| `mcvx_csr_nonempty_rows` (reduce CSR) | `_csr` **21.3 ms** (`check_format(full_check=True)` ≈ 9.5 ms of it), `eliminate_zeros` 6.4 ms | 34.0 ms | 5.1 ms | **glue 87%** |
| unreduced apply / `A_mapping` / `nonzero` / `get_parameter_vector` | entire body (scipy/numpy) | 100% | 0 (never called) | 100% glue, 0% kernel |

Two cross-cutting glue observations:

- **`_matrix_from_index` copy**: `ci._matrix_from_index(result.copy(),
  index, True)` = 0.35 ms vs 0.056 ms without the copy — the copy alone
  is ~0.3 ms, i.e. ~20% of a full apply. The matvec already writes into a
  fresh `np.empty` result buffer; the extra `result.copy()` before the
  CSC wrap is pure waste at this size (it is the offset-split
  `b[indices[split:]] = flat[split:]` assignment that forces the copy in
  the *test* harness, not in production: production calls
  `_matrix_from_index(flat, ...)` where `flat` is the fresh matvec result).
  The profiler's `result.copy()` in the timing loop is an artifact of
  reusing one result buffer across iterations, not production behavior.
- **Allocation census per call** (counted from the source, confirmed by
  the piece timings): apply allocates `result` (3.4 MB) + `b` (240 KB) +
  CSC shell each call; CSC reduce allocates scratch (11.8 MB, zeroed) +
  4 output buffers (~43 MB total) each call; CSR reduce allocates 2 output
  buffers + the `_csr` rebuild. The scratch `np.zeros` is the single
  largest per-call allocation (0.56 ms at bench size, but ~4 ms of the 10.7
  ms total at 50k entries where everything else shrinks).

## 3. Flags for the next phases (required by the task)

### 3.1 Python-level loops that re-enter ctypes per element

**None.** Every Mojo call in the port passes whole buffers in one ctypes
call; there is no per-element ctypes re-entry anywhere in
`canonInterface.py`. The per-element loops are (a) inside the Mojo kernels
(their job), and (b) pure-Python loops that never cross the FFI:
`get_parameter_vector` iterates `param_id_to_col.items()` (500 iterations
= 0.48 ms, all Python), and `nonzero_csc_array` / `A_mapping` are
numpy/scipy vectorized (no Python loop). Correctly, the port avoids the
worst anti-pattern entirely. A ctypes-call floor was measured anyway for
small-size reasoning: a single bare `mcvx_csr_matvec_i32` call on 1 nnz
costs **10 µs** best-of-51 — so even if someone wrote a per-element loop,
the floor is ~10 µs/element, which at parity-suite sizes (700 nnz) would
be ~7 ms/call. Presenting as evidence that the one-call-per-function
design is right.

### 3.2 Per-call allocations and copies

| allocation | size (bench) | cost (measured) | who pays it |
|---|---|---|---|
| CSC-reduce `scratch = np.zeros(rows)` | 11.8 MB | 0.56 ms; ~4 ms at 50k entries as a share | every CSC reduce |
| CSC-reduce 4 output buffers | 43.3 MB | included in kernel time | every CSC reduce |
| CSR-reduce `_csr` rebuild + `check_format(full_check=True)` | 19.7 MB working set | 9.5 ms for the check alone | every CSR reduce |
| apply `result = np.empty(rows)` | 3.4 MB | included in kernel time | every apply |
| apply `b = np.zeros(n_constr)` + offset scatter | 240 KB | ~0.06 ms | every apply with offset |
| unreduced `sp.csc_array(vec[:,None])` + `tensor.copy()`s | ~8 MB | 0.64–6.5 ms | every unreduced apply |

The scratch zeroing is not optional in the current kernel design (it is
the histogram input, see §4) — but its *size* (1.47M rows even when only
423k rows are nonempty, and 11.8 MB at int64) is what hurts small and
medium inputs.

### 3.3 Ablation-style re-check: "does the port ever get *slower*?"

Yes, in three measured cases (rows kept per the task rule, not dropped):

1. **CSC reduce at 50k entries: 0.60x** (mojo 10.7 ms vs upstream 6.4 ms).
   Fixed-cost dominated (1.47M-row scratch + full `_csc` normalization).
2. **Unreduced apply: 0.92–0.95x** (mojo 106–109 ms vs upstream 98–104 ms).
   No Mojo benefit exists on this branch by construction; the wrapper's
   extra dtype/contiguity checks make it slightly worse.
3. **Matvec kernel vs scipy `@`: 0.88–1.01x across the sweep** (0.90x at
   7M nnz, 0.88x at 10k rows). Within 10% of scipy at every size — the
   kernel is not the source of the headline speedups.

And one negative check for the harness: the CSC-compact kernel **mutates
its `scratch` input in place** (histogram, then write offsets). A naive
benchmark that reuses one scratch allocation across kernel-only timed
iterations segfaults on the second call (reproduced twice in this session,
exit 139). Production code is safe (fresh `np.zeros` per call) but any
future kernel-only microbenchmark must zero scratch per iteration.

## 4. What the kernels do (for the next phase's grounding)

- `mcvx_csr_matvec` (i32 and i64 are structurally identical): CPU row loop,
  short rows scalar, long rows SIMD (`load[width=W]` + `unsafe_gather` +
  `reduce_add`) when `nnz_row >= simd_width_of[Float64]`. The "parallel"
  branch (`workers = min(num_physical_cores(), MAX_WORKERS)` when
  `rows >= 65,536 and nonzeros >= 250,000`) is a **serial `for worker in
  range(workers)` loop — no threads are ever spawned**. `max.algorithm.
  parallelize` (which `MOJO_NOTES.md` §5b verifies as the working import
  on this toolchain) is not imported or used anywhere in `src/`. Measured
  threshold probe: 65,535 rows → 0.865 ms, 65,536 rows → 1.244 ms
  (serial chunking adds call overhead, no speedup; 70k rows → 0.826 ms).
- `mcvx_csc_compact`: (1) histogram `scratch[row] += 1` per nnz; (2) prefix
  scan building `old_rows` / `reduced_indptr` and converting scratch to
  write offsets; (3) column-major scatter into CSR buffers. Destroys
  `scratch` (see §3.3 note).
- `mcvx_csr_nonempty_rows`: single indptr diff scan. Its numpy baseline
  `(indptr[1:]−indptr[:-1] > 0).nonzero()` (2.8 ms) beats the kernel
  (5.1 ms) at bench size — the kernel's edge would appear only where the
  numpy path forces an int64 cast (16.1 ms with `np.diff(...astype(int64))`).
- All six exports take `Int` addresses and reconstruct pointers inside;
  `lib()` caches the loaded DLL after the first call (measured 0.000 ms).

## 5. Next-phase verdicts (named per hotspot)

Per the task: each hotspot is assigned to **unroll**, **simd**, **parallel**,
**gpu**, or **none**. GPU verdicts state the flops/byte figure (rule:
a GPU path is only defensible well above ~2 flops/byte; all kernels here
sit at 0.01–0.13).

| hotspot (from §1) | what the evidence says | next phase |
|---|---|---|
| `A_mapping_nonzero_rows` — 175.8 ms, sort-bound, 0.01 flops/byte | Not Mojo yet. The `unique()` sort (137 ms of 176 ms) is the target. A Mojo kernel scanning CSC `indptr` + deduping sorted row ids (CSR-row-ID scan already exists in `mcvx_csc_compact` form) replaces materialize+sort with a scan. | **none for now** (it's the port's first non-kernel opportunity, but this ANALYZE phase must not change algorithms; smallest kernel-shaped win, pure-Python elimination of the 498k-element intermediate) |
| CSC reduce kernel — 37.1 ms, 0.06 flops/byte, memory-bound | BW-bound histogram+scatter. int64→int32 alone recovers ~7 ms (19%) with zero algorithm change; that is a dtype/precision choice, not an acceleration phase. SIMD of a scatter is gather-bound (random `scratch[row]` hits); parallelizing the histogram needs atomics or two-phase partitioning. GPU: at 0.06 flops/byte, transfer of 51 MB (~30 ms over PCIe) would exceed the kernel time itself. | **unroll** (the innermost histogram/scatter loops are scalar+pointer-indexed; unrolling + prefetching helps the scan half; SIMD/gather is not the right shape here) — and **parallel** only for the scan phase with explicit two-phase partitioning, measured against the serial baseline |
| CSC reduce fixed cost — 0.60x at 50k entries | `np.zeros(1.47M)` + full `_csc` normalization is paid even for 50k-entry inputs. Fix is thresholding (skip Mojo normalize for small inputs / downcast indices to int32 when the value range fits), not a kernel optimization. | **none** (algorithmic policy, not an accel phase) |
| CSR reduce glue — 34.0 ms, kernel 5.1 ms (numpy 2.8 ms) | `check_format(full_check=True)` alone is 9.5 ms; `_csr` normalization + `eliminate_zeros` is the rest. The kernel is already beaten by numpy at bench size. | **none** (delete the full check / skip redundant normalization — algorithmic hygiene, not unroll/simd/parallel/gpu) |
| CSR matvec kernel — 1.38 ms, 0.09 flops/byte, at BW-parity with scipy | Measured, not assumed: 0.90–1.01x vs scipy across 10k→7M nnz, 6–8 GB/s effective, gather-bound. SIMD is present but fires only for long rows; at 1.18 nnz/row it never helps. True **parallel** (`from max.algorithm import parallelize`, verified in `MOJO_NOTES.md` §5b as the working import) is the only lever with headroom: the current "parallel" branch is serial. GPU: at 0.09 flops/byte, PCIe round-trip of the 11 MB working set (~6 ms) + launch overhead exceeds the 1.4 ms kernel — not defensible. | **parallel** (implement the verified `max.algorithm.parallelize` split behind the existing `PARALLEL_ROWS`/`PARALLEL_NONZEROS` thresholds, prove it on the sweep table in §1.5; **simd** is already in and is not the bottleneck; **gpu: no** — 0.09 flops/byte) |
| Unreduced apply — 0.92x, no kernel | Dominated by scipy `reshape+tocsc` (80.3 ms of 106 ms) on both sides. No Mojo kernel exists here; routing it through CSR matvec cannot fix the format-conversion cost. | **none** (decide: document as unsupported or accept scipy parity; not an accel problem) |
| `nonzero_csc_array` / `get_parameter_vector` — parity rows | Identical upstream code, no ctypes, sort-bound / dict-bound respectively. | **none** (out of scope for acceleration phases — keep parity, do not port) |
| 20x-cached-apply glue — 2.4 ms/each in one run, ~0 in steady state | First-iteration effects (Mojo runtime init / first-touch pages) inflate early applies; steady state (iters 5–20: 1.56 ms median) has ≤0.06 ms glue. The 20x loop is the intended workload and it is already kernel-bound. | **none** (no phase needed; the only fix worth doing is moving `linear_runtime` init off the hot path if first-call latency matters to callers) |

Summary of the map: **the two kernel hotspots are both memory-bound at
≤0.1 flops/byte, so GPU is ruled out by the measured arithmetic intensity
on every row** (not by the host API, which `MOJO_NOTES.md` §5 confirms is
available as `max.gpu`). SIMD is already implemented in the matvec and
is not the constraint (row-width probe: 2.1 → 1.1 ns/nnz from 1→32 nnz/row).
Parallel is the one structural gap (a serial loop wearing a parallel
branch's clothes, on a toolchain where `parallelize` works). The single
largest wall-clock hotspot is not a Mojo kernel at all: it is the
pure-Python `A_mapping_nonzero_rows` sort, followed by the pure-Python
`_csr`/`_csc` normalization chains inside the reduce calls.

## Reproducibility

- `pixi run bench` — rerun for the headline numbers (flock-wrapped; do not
  run concurrent benchmarks).
- Profiler scripts used for this analysis were temporary bench-local
  files (`bench/_profile*.py`, since removed per the add-only-`analysis.md`
  rule) and are not checked in; the method section above is sufficient to
  regenerate every figure: build the bench fixture exactly as `bench.py`
  does (`default_rng(2026)`, 500k triplets, `var_length=48`), then time
  each entry point best-of-N after ≥2 warmup calls, isolate kernels with
  bare `ctypes` calls over pre-allocated buffers (zeroing `scratch`
  before every `mcvx_csc_compact` kernel-only iteration).
