# mojo-cvxpy — VERIFY phase (re-verification)

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz. Software: CVXPY 1.9.2,
Python 3.13.14, numpy 2.5.1, scipy 1.18.0. All numbers below were measured in
this session; no number from a previous session is reused as a measurement.

Note on provenance of this file: a `bench/verify.md` already existed (from the
previous verify commit `29173f4`), covering the same steps. Its numbers were
**not** trusted: every figure was re-measured here and the README table was
re-checked against fresh `pixi run bench` runs. Where the old file's
methodology was sound and its claims re-verified with new evidence, the claim
is recorded below with this session's measurements. The README table at
`51.00 / 75.26 (1.48x)`, `1.70 / 5.26 (3.10x)`, `64.07 / 189.68 (2.96x)`
reproduces within 20% on every fresh run here (see §1), so no README correction
was needed in this pass. No production code was changed in this pass.

## §1 Baseline runs and per-row reproduction (steps 1–2)

- `pixi run build && pixi run test` (before any edit): build OK, **86 passed**
  (exit 0). The previous file reported 85 → 86 across its own two fixes; this
  session starts and ends at 86 — the parametrisation restore and the tightened
  inf assertion from that pass are both already in the tree (see §4).
- `pixi run bench` — first run this session: 50.06 / 74.43 (1.49x),
  1.61 / 5.06 (3.16x), 59.80 / 185.12 (3.10x). Then five consecutive repeats,
  all via `flock`-wrapped `pixi run bench` (best-of-3 per bench.py):
  - 49.31 / 74.43 (1.51x), 1.65 / 4.94 (2.99x), 66.56 / 191.68 (2.88x)
  - 43.60 / 72.93 (1.67x), 1.66 / 5.00 (3.01x), 62.26 / 188.35 (3.03x)
  - 51.97 / 74.63 (1.44x), 1.71 / 4.93 (2.89x), 55.58 / 185.28 (3.33x)
  - 47.26 / 72.58 (1.54x), 1.76 / 5.17 (2.95x), 63.16 / 171.31 (2.71x)
  - 46.29 / 71.59 (1.55x), 1.71 / 4.98 (2.90x), 61.34 / 183.30 (2.99x)

### Step 2 — 20% check against the current README figures

Deviations of each fresh run from the README table (mojo / cvxpy, max across
the six runs above):

| row | README (mojo / cvxpy / x) | fresh range (mojo / cvxpy) | max deviation | verdict |
| --- | --- | --- | --- | --- |
| reduce (500k triplets) | 51.00 / 75.26 / 1.48x | 43.60–51.97 / 71.59–74.63 | mojo −14.5%, cvxpy −4.9% | **reproduces** |
| apply + rebuild CSC | 1.70 / 5.26 / 3.10x | 1.61–1.76 / 4.93–5.17 | mojo +3.5%, cvxpy −6.3% | **reproduces** |
| 20 cached applications | 64.07 / 189.68 / 2.96x | 55.58–66.56 / 171.31–191.68 | mojo −13.3%, cvxpy −9.7% | **reproduces** |

Every row is within 20% on every run, on both columns; the speedup ratios
recompute to 1.44–1.67x, 2.89–3.16x, 2.71–3.33x respectively. **No README
correction was made** — the existing table is the measured one. The reduce
row's mojo column has the widest spread (43.6–52.0 ms, ratio flickering
1.44–1.67x) but stays inside the 20% band throughout. The old file's history
of this row is consistent: the pre-correction README figures (63.99 / 74.23 /
1.16x, 0.38 / 4.52 / 11.84x, 27.91 / 174.95 / 6.27x) are not re-litigated
here beyond noting the current table, which replaced them, reproduces.

### Step 3 — slower-than-upstream rows

No row in the README table flips direction: all three remain "faster" in every
fresh run. Slower-than-upstream cases are not in this table but remain
documented in `bench/analysis.md` and were left untouched: unreduced tensor
path 0.92x, `nonzero_csc_array` 0.93x, `get_parameter_vector` 0.94x, CSC
reduce at 50k entries 0.60x. None were removed or reworded.

## §2 Same-input / no-skipped-work audit (step 4)

`bench/bench.py` builds one tensor with `default_rng(2026)` — shape
(1,470,000, 65), int64 CSC, 500,000 triplets → 498,655 stored nnz — and one
`param` vector from the same generator; `var_length=48`. The reduced form is
computed once with upstream `reduce_problem_data_tensor(tensor.copy(), 48)`
and shared by both apply cases. Re-verified live this session:

- Reduce: `reduced` (423,769 × 65 CSR, int32, 498,655 nnz) from the mojo path
  equals the upstream path exactly — max data diff 0.0, indices/indptr/shape
  `array_equal`. Both sides receive `tensor.copy()`; the shared tensor's nnz
  and data are unchanged after either side (498,655 → 498,655), confirming the
  copy protects against upstream's in-place `eliminate_zeros()`. Copy cost is
  included in both timings equally.
- Apply / 20x apply: both sides read the same `reduced` and `param` with no
  mutation (verified: `reduced.data` and `param` bit-identical after each
  side). Output parity on this fixture: A (30000, 48), 415,180 nnz, max data
  diff 1.8e-15; offset vector max diff 8.9e-16. All 20 loop iterations return
  distinct matrix objects on both sides (`len({id}) == 20`), so no readout is
  skipped or cached across iterations.
- Skipped *glue*, not skipped *result*: mojo's Python layer builds CSC
  directly where upstream does a matvec plus `M[:, :-1].tocsc()` slicing per
  call. Outputs are identical (above), so the speedup is real end-to-end work
  reduction, not a correctness-skipping artifact. (The kernel-vs-scipy
  matvec itself is at bandwidth parity, ~0.9–1.0x — the win is the glue, as
  `bench/analysis.md` §3 already states.) `with_offset` defaults to `True` on
  both sides; no offset or nonzero-row handling is dropped.
- The 20x per-iteration cost in the loop (mojo ≈3.2 ms/iter, upstream
  ≈9.5 ms/iter) exceeds the single-apply timing (~1.7 / ~5.0 ms) on both
  sides by a similar factor, so the 20x ratio is not inflated by it; the 20x
  figure is not `20 × single` and the README does not imply it is.

## §3 Test-strength audit (step 5)

`git log -p -- tests/` spans four commits touching `tests/`: `978292c`
(initial port), `94a2ecf` (accel pass), `e1b89d5` (test pass), `29173f4`
(prior verify pass). No removed assertions or loosened tolerances were found
still outstanding; the two weakenings identified historically are already
restored in the tree:

1. `94a2ecf`: `test_apply_reduced_tensor_parallel_threshold_matches_upstream`
   went from `entries_per_row = 16` to `5`. The current file parametrises
   over `[5, 16]` (restored by `29173f4`), so both widths run. Re-verified
   live this session: parity at 5, 16, 17 and 32 entries/row, max diff
   ≤ 1.4e-14 — no code bug, nothing to fix in `src/`.
2. `e1b89d5` introduced `RTOL = ATOL = 1e-9` for `assert_sparse_equal`. The
   prior code used bare `np.allclose` (defaults rtol 1e-5, atol 1e-8), so the
   explicit 1e-9/1e-9 is ~10,000x tighter on rtol, not a loosening. Index/
   indptr checks remain exact `np.array_equal` throughout, and the float64-
   normalization contract (README-stated) still explains dtype differences
   vs upstream's preserved int64/float32. No action required.
3. The one tautology that existed — `assert ... .any() or True` in
   `test_get_matrix_from_tensor_nan_inf_param_matches_upstream` (introduced
   by `e1b89d5`) — was already replaced with
   `assert np.array_equal(np.isinf(got_vals), np.isinf(expected_vals))` by
   `29173f4`. A full grep of the current file finds no `or True`, `xfail`,
   `skipif`, or `allow_nan` patterns. Nothing left to restore.

Extension-part audit (tests added in `e1b89d5`, read in full): three tests
pin the port's `ValueError` where upstream returns a floor-division artifact
for inputs its own docstring forbids ("must not have a 0 in its shape") —
zero-dimension tensors, non-divisible row counts, quad_form with
non-positive `var_length`. Re-verified live this session against installed
CVXPY 1.9.2: upstream returns (0, 5) with RuntimeWarnings for (0, 5), (0, 4)
for the 10×4 case, and raises `IndexError` for quad_form `var_length ≤ 0`;
the port's `ValueError` is the documented, safer contract and each test cites
the upstream precondition. **Not restored** to upstream behavior — restoring
would enshrine a divide-by-zero artifact. All other extension tests compare
against live upstream calls at RTOL/ATOL and are genuine parity checks.

**No test restoration was needed in this pass**: the weakenings were already
fixed by the prior verify pass, and this session's re-check confirms the
fixed state. Therefore no test files were edited, and no `src/` change was
made (none was justified under RULES — no failing test was found).

## §4 What you changed

Nothing besides this file (which supersedes the previous `bench/verify.md`):
no production code, no test code, no README numbers. The README table
reproduces as-is; the slower rows stay as documented in `bench/analysis.md`;
the test suite is restored as-found and already green.

## §5 What could not be reproduced

- The historical README figures replaced by the prior verify pass (0.38 ms
  single-apply / 11.84x; 27.91 ms 20x / 6.27x). Per the prior file's §1/§3
  measurement record, no combination of its repeats produced them; this pass
  makes no claim about them beyond the current table reproducing.
- Occasional high outliers on the reduce row (prior file saw 69–72 ms; this
  pass's range was tighter, 43.6–52.0 ms). The README's 51.00 ms is within
  the observed spread, but the row's ratio flexes (1.44–1.67x over six runs),
  so a single best-of-3 run should not be quoted as a stable constant.
- Single-run numbers quoted anywhere else (e.g. `bench/analysis.md`'s own
  bench line) were not re-derived here and appear nowhere in the README.
