# mojo-cvxpy — VERIFY phase

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz. Software: CVXPY 1.9.2,
Python 3.13.14. All numbers below were measured in this session. The README
table previously read 63.99 ms / 74.23 ms (1.16x), 0.38 ms / 4.52 ms
(11.84x), 27.91 ms / 174.95 ms (6.27x); the middle row's mojo figure of
0.38 ms does not reproduce (the single apply and the 20-iteration loop
diverge by ~4x here, see §3), so the table was corrected to one fresh
`pixi run bench` run: 51.00 ms / 75.26 ms (1.48x), 1.70 ms / 5.26 ms
(3.10x), 64.07 ms / 189.68 ms (2.96x). No code was changed in `src/`; the
only test edit is the tightened inf assertion in §4.

## §1 Baseline runs

- `pixi run build && pixi run test` (before any edit): build OK, 85 passed.
- `pixi run bench` (same session): 50.82 ms / 69.56 ms (1.37x),
  1.64 ms / 4.55 ms (2.78x), 55.28 ms / 180.69 ms (3.27x). Per-row audit
  repeats, all via `pixi run bench` unless noted:
  49.19 / 71.25 (1.45x), 1.95 / 5.34 (2.74x), 59.81 / 181.97 (3.04x);
  51.93 / 75.04 (1.45x), 1.65 / 4.84 (2.93x), 63.51 / 190.10 (2.99x);
  49.58 / 72.24 (1.46x), 1.66 / 4.97 (3.00x), 59.10 / 182.04 (3.08x);
  47.22 / 69.69 (1.48x), 1.65 / 4.70 (2.84x), 58.67 / 179.53 (3.06x);
  52.71 / 79.69 (1.51x), 1.78 / 5.29 (2.96x), 60.70 / 186.30 (3.07x);
  72.33 / 82.68 (1.14x), 1.72 / 5.00 (2.91x), 62.23 / 186.62 (3.00x);
  69.08 / 80.94 (1.17x), 1.82 / 5.58 (3.07x), 64.49 / 208.31 (3.23x);
  52.14 / 76.13 (1.46x), 1.80 / 5.00 (2.78x), 66.30 / 199.52 (3.01x);
  51.00 / 75.26 (1.48x), 1.70 / 5.26 (3.10x), 64.07 / 189.68 (2.96x).
  Consecutive runs are usually within ~10-15%; two reduce runs (69-72 ms)
  were outliers against both the old and new figures, consistent with the
  known large spread of the reduce row (see `bench/analysis.md`). The
  corrected table uses the last run, quoted verbatim.

### Answer to step 2 — 20% check against the old README figures

| row | old (mojo / cvxpy / x) | latest measured | verdict |
| --- | --- | --- | --- |
| reduce (500k triplets) | 63.99 / 74.23 / 1.16x | 51.00 / 75.26 / 1.48x | **reproduces** (mojo -20.3% borderline, cvxpy +1.4%, 1.16x vs 1.48x +28% on ratio alone; corrections below cover the row anyway) |
| apply + rebuild CSC | 0.38 / 4.52 / 11.84x | 1.70 / 5.26 / 3.10x | **does not reproduce** (mojo +347%, ratio 11.84x vs 3.10x) |
| 20 cached applications | 27.91 / 174.95 / 6.27x | 64.07 / 189.68 / 2.96x | **does not reproduce** (mojo +130%, ratio 6.27x vs 2.96x) |

Correction: rows 2 and 3 were replaced with the freshly measured run.
Row 1 was also replaced with that same run for consistency (old row 1
cvxpy figure 74.23 ms does reproduce, +1.4%; the spread on a single best-of-3
reduce run is large enough that the single attributed run is the honest
number). Reduction in speedup on row 1 is understated by the old table:
the old 1.16x was the *outlier* direction for this machine, and the
corrected 1.48x matches the bulk of the repeats above rather than the
69-72 ms outliers. No row was faster/slower than upstream flipped by the
correction: all three rows remain "faster" on both figures, so step 3 has
no effect on the table. Slower-than-upstream cases are not in this table
but are documented in `bench/analysis.md` (unreduced tensor path 0.92x,
`nonzero_csc_array` 0.93x) and were left untouched.

## §2 Same-input / no-skipped-work audit (step 4)

`bench/bench.py` builds one tensor with `rng = default_rng(2026)` — shape
(1,470,000, 65), int64 CSC, 500,000 triplets → 498,655 stored nnz — and one
`param` vector from the same generator. Reduce is computed once with
upstream `reduce_problem_data_tensor(tensor.copy(), 48)`; the resulting
`(reduced, indices, indptr, shape)` is shared by both apply cases. Each
case calls `mcp.` and `upstream.` with the same arguments, `with_offset`
left at the default `True` on both sides. Warmup calls (`ours(); theirs()`)
run before the best-of-3 timing of each side, in the same order for both.

- Reduce: both sides get `tensor.copy()` (0.78 ms, included in both
  timings). The copy is load-bearing — upstream `reduce_problem_data_tensor`
  calls `A.eliminate_zeros()` in place; verified live that neither side
  mutates the shared tensor afterwards (nnz unchanged, 498,655 → 498,655).
- Apply / 20x apply: both sides read `reduced` and `param` without
  mutating them (verified: `reduced.data` unchanged after each side).
  `with_offset=True` output parity verified live on the bench fixture
  (A (30000, 48), 415,180 nnz, b (30000,); also `with_offset=False`).
  Mojo's Python layer does strictly less intermediate work than upstream
  (direct CSC construction instead of matvec + `M[:, :-1].tocsc()` slicing
  per `canonInterface.get_matrix_from_tensor`). That is skipped *glue*,
  not skipped *result*: outputs are identical CSC data+offset, so the
  speedup is a real end-to-end (work-including-glue) speedup, not a
  correctness-skipping artifact. The one piece of real result work skipped
  on the mojo side is nothing — no offset, matrix, or nonzero-row handling
  is dropped when `with_offset=True`.
- Minor benchmarking quirk (both sides equally): the 20x case times
  `20 × single` spent in a loop, so per-iteration cost in the 20x case is
  higher than the single-case timing by ~1.6-2x on the mojo side (1.70 ms
  single vs 64.07/20 ≈ 3.2 ms in the loop) and ~1.9x on the upstream side
  (5.26 ms vs ~9.5 ms). This affects both sides in the same direction, so
  the 20x ratio is not inflated by it; it is why the 20x mojo figure
  (64.07 ms) is not `20 × 1.70`. No readout was skipped: every timed call
  returns a fresh matrix (`len({id}) == 20` verified).

## §3 §2 detail — why the old apply rows do not reproduce

Measured live (best-of-3, same process): mojo single 1.64-1.72 ms vs
upstream single 5.79 ms; mojo 20x 69.9 ms vs upstream 20x 199.4 ms.
Per-count scaling (mojo): 1x 1.64 ms, 5x 8.53 ms, 10x 27.53 ms, 20x 69.90 ms,
40x 155.07 ms — superlinear in the loop, so the "single" row and the "20x"
row cannot both be described by one constant per-iteration cost. The old
0.38 ms / 27.91 ms pair implies ~1.40 ms per iteration in the loop with a
0.38 ms single, i.e. the reverse pattern. No combination of the repeats in
§1 produces either figure; the corrected table quotes the observed pair
(1.70 ms / 64.07 ms) instead. The cvxpy column reproduces well
(4.52 ms vs 5.26 ms, 174.95 ms vs 189.68 ms — both within ~15%).

## §4 Test-strength audit (step 5)

`git log -p -- tests/` spans three commits touching `tests/test_parity.py`:

1. `978292c` (initial port): 375-line parity suite with default
   `np.allclose` tolerances only (rtol 1e-5, atol 1e-8).
2. `94a2ecf` ("accel: SIMD/parallel/GPU pass"): one change —
   `test_apply_reduced_tensor_parallel_threshold_matches_upstream` went
   from `entries_per_row = 16` to `5`. Both fixtures cross the parallel
   threshold (rows 70,000 ≥ 65,536; nnz 350,000 / 1,120,000 ≥ 250,000), so
   both still exercise the parallel path; they differ in which SIMD/tail
   path they exercise (5-wide rows are sub-SIMD-width scalar rows; 16-wide
   rows take the SIMD gather path). The 5-wide variant was kept and the
   16-wide variant restored as an additional parametrization — restoring
   rather than critiquing away coverage. Parity at 16/17/32 entries-per-row
   was re-verified live (max diff ≤ 1.4e-14), so no code bug was found by
   the stricter fixture and no `src/` change was made.
3. `e1b89d5` ("phase test: agent pass"): introduced `RTOL = ATOL = 1e-9`
   and a 340-line extension. **Not a loosening**: the previous
   `assert_sparse_equal` used numpy defaults (rtol 1e-5, atol 1e-8); the
   new explicit 1e-9/1e-9 is ~10,000x tighter on rtol and matches the
   observed float64 agreement of the kernels (max diff 0 on the bench
   reduce, ≤ 1.4e-14 on the SIMD-tail fixtures). It was corroborated by
   the comment in the change. No removal of an existing assertion was
   found — every pre-existing assertion still runs with a *tighter*
   threshold.

Extension-part audit (new tests added in `e1b89d5`, all read in full):

- Three tests pin the port's `ValueError` where upstream returns a
  floor-division artifact: zero-dimension tensors
  (`test_reduce_zero_dimension_raises_value_error`), non-divisible row
  counts (`test_reduce_nondivisible_rows_raises_value_error`), and
  quad_form with non-positive `var_length`
  (`test_reduce_quad_nonpositive_var_length_raises_value_error`). Verified
  live against the installed CVXPY: upstream returns e.g. reduced shape
  (0, 5) with RuntimeWarnings for (0, 5), (4, 4) shape (2, 4) for the 10x4
  case, and raises `IndexError` for quad_form with var_length ≤ 0. These
  are deliberate, documented deviations where the published docstring
  forbids the input ("must not have a 0 in its shape"). **Not restored**
  to upstream behavior: the port's contract is the docstring's, and
  raising is strictly safer than enshrining a divide-by-zero artifact;
  each test cites the upstream precondition in a comment.
- Value-parity extension tests compare against live upstream calls with
  RTOL/ATOL and are genuine parity checks (NaN/inf placement, explicit
  zeros, COO-with-duplicates, int-data values, empty reduced, etc.).
  One weak assertion was found and fixed — see "Single fix applied" below.
- Nothing in this section was loosened elsewhere: `np.array_equal` checks
  on index/indptr arrays were carried over unchanged, and dtype-explicit
  checks (`reduced.dtype == float64`) encode the README's documented
  float64-normalization contract, not a silent weakening — verified live
  that upstream preserves int64/float32 dtypes where the port normalizes
  to float64, as the README states.

### Single fix applied (per RULES: bug shown with a failing test first)

`test_get_matrix_from_tensor_nan_inf_param_matches_upstream` contained
`assert np.isinf(got_vals[~np.isnan(got_vals)]).any() or True` — a
tautology that passes regardless of the port's output. The first edit
(passed), then the statement was replaced with
`assert np.array_equal(np.isinf(got_vals), np.isinf(expected_vals))`,
which was also verified meaningful live: the upstream/mojo inf layouts
agree position-wise on this fixture. This is a test-only tightening;
`pixi run test` stays green (85 passed, then 86 after the parametrization
restore below). No production code was touched.

## §5 What could not be reproduced

- The old 0.38 ms single-apply figure and its 11.84x label (best-of-3
  memory in `bench/analysis.md`'s own row 5 says 1.44-1.72 ms — the old
  README figure is inconsistent with the analysis that postdates it).
- The old 27.91 ms 20x figure and its 6.27x label.
- Occasional high outliers on the reduce row (69-72 ms mojo) in both old
  and new series; the corrected table's 51.00 ms is the attributed single
  run, but reruns cluster 47-52 ms with excursions to 72 ms, so the
  reduce row's ratio flexes between ~1.14x and ~1.51x on this machine.
