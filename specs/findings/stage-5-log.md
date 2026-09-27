# Stage 5 — log

Dated corrections and measurements for Stage 5's roadmap block. Stage 5 is
unticked; this file exists because its `RE-VALIDATED` text already carried a
stale code pin (roadmap `## Stages`, stage-block rules 1 and 2).

## 2026-09-10 — `assert_declared_provenance` caller count and line were both wrong

**Superseded reading** (roadmap Stage 5 `RE-VALIDATED`, until `1be5bbf`):

> but its only `src/` caller is `baselines/runner.py:218`.

**Why it changed.** There are two callers, and neither is at that line:
`baselines/runner.py:312` and `validate/harness.py:165` (line numbers as recorded at `1be5bbf` and left
as the historical record; both have since drifted, so find the calls by symbol — `run_baselines`
and `run_pseudo_suppression`). The second arrived with
the Stage 4 harness and the block was never re-validated against it.

**Why it mattered.** The clause was load-bearing for the argument that followed
it — that a sixth reconciliation status invented by Stage 5 would go unchecked.
With two callers, one of them inside the validation harness, the blast radius of
a new status is larger than the block claimed.

**Standing hazard.** This is the pattern `specs/logging-employment-spec.md` and
this repo's `CLAUDE.md` both warn about: a `file:line` pin is a dated witness.
Prefer naming the symbol and letting the reader grep.

## 2026-09-11 — the §9.3 settle-before trigger was attached to the wrong stage

**Superseded reading** (roadmap Stage 6 `Consumes`, until this plan):

> if any margin is wanted it must be settled BEFORE this stage consumes `deterministic_bounds`

**Why it changed.** The sentence was true of Stage 6 and also of Stage 5, and only
Stage 6 carried it. Stage 5's own `Consumes` already reads "Stage 2's
`deterministic_bounds`", so a Stage 5 planner reading the artifact they are told to
read could not see the precondition attached to it. R-S5G-7.

**Why it mattered.** The trigger guards an ORDER, and a trigger one stage late does
not guard anything: by the time Stage 6 reads it, Stage 5 has already been gated
against the identification set the measurement might change.

## 2026-09-11 — the §9.3 margins, measured: a bound exists on 252 of 409

**Measurement** (R-S5G-5; `scripts/audit/qcew_parent_margins.py`, 128/128 slices at 200,
rendered in full at `specs/findings/qcew-parent-margins.md`):

| | |
|---|---|
| private `113310` state-quarters / suppressed | 1,572 / 409 — the 2026-09-11 witness holds |
| `113` disclosed where the child is `N` | **252** (61.6%; 756 of the 1,227 months) |
| `1133`, `11331` disclosed where the child is `N` | **0** each |
| `own_code 0` rows at state x 6-digit | **0** — a measured absence, `fetched: true` beside it |
| `exact` / `upper_bound` / `none` | **0 / 252 / 157** |

**Superseded reading** (`specs/completed/stage5-gate-inputs.md` §1.2, and `D-092` as filed):

> With that single-child chain, a disclosed `1133`, `11331` or total-ownership `113310` cell
> where `113310` private is `N` is an **exact reconstruction** of the suppressed value

**Why it changed.** The inference was sound and its antecedent is false. `1133` and `11331` are
disclosed on 1,163 of 1,572 state-quarters — and on **zero** of the 409 where the child is
suppressed. A 1:1 chain is suppressed *together*: that is what makes the chain safe for BLS to
publish at all. The exact-reconstruction hazard the spec anticipated does not arise through ANY MEASURED margin
on D1. It is not ruled out: `113 - 1131 - 1132` was not measured (`D-110`), so on the 252 bounded
quarters `REQ-027`'s status is unknown rather than absent.

**What is real instead.** `113` aggregates `1131` and `1132` as well, so it survives the
suppression that kills the chain below it, and it is disclosed on 252 of the 409. By
nonnegativity of the two siblings that is `113310 <= 113` — §9.3's parent-total constraint
doing real work on **61.6%** of the cells the engine currently leaves at `[0, +inf)`.

**Why `exact = 0` is measured, not constructed.** `identification` maps a disclosed `113` to
`UPPER_BOUND` by its ladder, which would understate a `'-'` parent — a published true zero forces
the child to exactly 0. Derived from the stored extracts: 3 of the 1,278 disclosed `113` private
state-quarters carry `'-'`, and **0** of those falls on a suppressed quarter. The ladder is not
hiding an exact case.

**Why it mattered.** Stage 4's scoreboard and `runs/f03023ac9f3a` were computed against an
identification set with no finite state upper bound anywhere. A finite upper bound on 756 months
makes those cells ELIGIBLE for MILP, which has run on 0 of 4,775 rows. `_needs_milp` also requires
the LP width to fall below 25, so `D-093`'s gap binds only on the narrow subset the finding derives.

**Not measured, and named rather than assumed.** `113 - 1131 - 1132 = 1133`, and the chain below
`1133` is 1:1 — so a quarter with all three disclosed is an *exact* reconstruction. `1131` and
`1132` are outside R-S5G-5's named scope and were not fetched. `specs/completed/stage5-parent-margin.md`
owns that measurement.

## 2026-09-11 — `runs/f03023ac9f3a` validation artifacts regenerated under the stratum schema

**Command:** `uv run logging-estimates validate --config config.yaml` (11 min wall clock), after
plan 14 Task 5 appended `stratum_kind` / `stratum_value` to `VALIDATION_METRIC_SCHEMA`. The run id
did not move — `run_id` hashes config and inputs, not source — so the same directory was
overwritten. The prior four validation files were backed up first and compared by join, not by
position.

| artifact | before | after |
|---|---|---|
| `validation_scoreboard.parquet` | (270, 13) | **byte-identical** |
| `validation_scores.parquet` | — | **byte-identical** |
| `validation_metrics.parquet` | (4,530, 19) | (6,929, 21): 4,800 `overall` + 2,129 `census_division` |

On the 4,530 pre-existing rows, joined on the six-field key with `nulls_equal=True`: 0 unmatched on
either side, and exactly one column moved.

**The column that moved was stale, not regressed.** `interval_source` read
`rolling_residual_ensemble` on 1,190 rows before and `leave_one_out_residual_ensemble` on the same
1,190 after. That rename is `e17b1f0` (plan 13, R-S5P-7, 2026-09-10). **The §13.10 comparand's
metrics file had therefore been stale with respect to the code since plan 13**, carrying the label
R-S5P-7 exists to retire, and nothing noticed — the standing hazard of a run id that ignores
source. The `null` count rose 3,240 -> 3,510, i.e. +270, exactly the new
`state_share_absolute_error` rows, which carry no interval.

**Why the scoreboard matters most.** Its byte-identity is Task 5's fan-out guard holding on D1
rather than on the fixture: 2,129 stratified rows entered the metrics frame and the board did not
grow a single row. §13.10's promotion comparand is unchanged.

**Provenance caveat.** The new `validation_manifest.json` records
`code_commit = c5bb7ca…-dirty`. The dirt was one untracked file,
`scripts/audit/render_parent_margins.py`, outside `src/`; no package code differed from `c5bb7ca`.

## 2026-09-12 — regenerated again under the review's `n_scored` fix

**Command:** `uv run logging-estimates validate --config config.yaml` at `542ed37`, whose manifest
records a CLEAN `code_commit` (no `-dirty`). The four validation files were backed up first and
compared by join on the eight-field key (the six old fields plus `stratum_kind`/`stratum_value`).

| artifact | at `c5bb7ca-dirty` (entry above) | at `542ed37` |
|---|---|---|
| `validation_scoreboard.parquet` | (270, 13) | **byte-identical** |
| `validation_scores.parquet` | — | **byte-identical** |
| `validation_metrics.parquet` | (6,929, 21) | (6,932, 21): 4,800 `overall` + 2,132 `census_division` |

**0 rows lost, 3 added, one column moved.** The 3 added rows are null `coverage_0.90` rows for
divisions whose masked cells never reached a leave-one-out ensemble — the whole-branch review found
the coverage family omitting divisions the WAPE family emitted. The column that moved is `n_scored`,
on **717 of the 782** division coverage rows: those rows had inherited the estimator-wide scored
count, and `n_scored > denominator` held on 717 of them before and on **0** after.

**This supersedes the counts in the entry above.** Its 6,929 / 2,129 were the defect's shape — the
right rows with a wrong column — not a correct artifact. The byte-identical scoreboard is again the
check that matters: §13.10's promotion comparand did not move across either regeneration.

## 2026-09-12 — §13.7's residual ensemble has the wrong sign (`D-112`)

**Found by** plan 14's review-verification pass, while checking whether plan 14's new per-division
coverage oracle was independent of the code. **Not introduced by plan 14:** present since `8939026`
(2026-09-07, Stage 4).

`probabilistic_metrics` pools `estimate - truth`, and `residual_ensemble` ADDS the pool to the point
estimate, so cell i's ensemble is `estimate_i + (estimate_j - truth_j)`. The predictive distribution of
`truth_i` is `estimate_i - (estimate_j - truth_j)`. Measured with the shipped `intervals` helpers on a
synthetic 40-cell method (`default_rng(0)`,
truth ~ U(80, 120), estimate = 1.25 x truth + N(0, 3)): 90% coverage **0.00** as shipped, **0.85** with
the sign corrected. On an unbiased method the two agree to sampling noise (0.90 vs
0.85). The oracles do not pass because of symmetry — both COPY the code's expression. On the golden
fixture the residuals are one-sided (62.2% positive in the oracle group), and the corrected sign
changes `coverage_0.90` in 37 of 43 interval-bearing groups, the hand-derived oracle's from 33 to 31
of 37.

**Why it matters for Stage 5.** Plan 14 made §13.10's coverage gate evaluable; this makes its values
untrustworthy for exactly the estimators a gate exists to catch. WAPE and the (270, 13) scoreboard are
unaffected. Recorded as a Stage 5 precondition in the roadmap block rather than fixed in plan 14,
because the fix changes Stage 4's shipped probabilistic numbers.

## 2026-09-12 — `D-112` fixed: the residual ensemble subtracts the residual

**Fixed in** `19fbdec` (`fix/d112-residual-sign`). `intervals.residual_ensemble` returns `point - residual`;
the pool in `probabilistic_metrics` stays `estimate - truth`, so the orientation has one owner. The failing
tests were written first from `truth = estimate - residual` and observed red on the old sign: a one-sided
helper pool, a constant +25 bias (coverage 0.0 on the old sign, 1.0 corrected, at every level) and a
hand-derived zero-clip count (0 old, 3 corrected). **Every probabilistic value in the artifacts the entries
above describe was computed on the old sign;** those entries quote none apart from `D-112`'s own, whose
golden predictions are confirmed below.

**Golden** `2bec94b0…` → `a544559f…`, by join on the eight-field key: 0 key-only rows, and only `value`
moved, on 279 of 1,490 rows, all `probabilistic`; the fixture run's scores and scoreboard are
byte-identical. `coverage_0.90` moved in 37 of 43 interval-bearing groups and the hand-derived oracle went
33 → 31 of 37, both as `D-112` predicted; it also moved in 40 of 72 division rows.

**`runs/f03023ac9f3a`** re-run at `19fbdec` (clean tree, 12 min 08 s), its artifacts backed up first:

| artifact | before (`542ed37`) | after (`19fbdec`) |
|---|---|---|
| `validation_scores.parquet` | `4d9912fb…` | byte-identical |
| `validation_scoreboard.parquet` | `d4e1187b…`, (270, 13) | byte-identical |
| `validation_metrics.parquet` | `9f6b6a75…` | `049ecbec…` |

Apart from `code_commit` and the metrics digest in the table, the manifest is unchanged. By join on the same key: 6,932 rows and 2,132
stratified on both sides, 0 key-only rows, and only `value` moved — on 1,488 of the 2,072 `probabilistic`
rows and on no other family. Overall `coverage_0.90` moved in 154 of 170 interval-bearing groups (rose 132,
fell 22) and in 455 of 782 division rows; `crps` in all 170; `n_clipped_at_zero` in 103;
`mean_interval_width_0.90` beyond 1e-9 in 104, every one where the zero clip binds on either side (elsewhere
at most 7.4e-13). Coverage stays monotone in level in all 170 groups.

**What Stage 5's coverage gate will see.** The corrected intervals still UNDER-cover. Over the 170 groups,
overall 90% coverage moved from mean 0.35 to 0.76 and median 0.25 to 0.80, and its maximum is now exactly
0.90, so no group over-covers. Counted exactly — coverage is hits over `calibration_sample_size`, a rational
— 51 groups sit within `nominal_coverage_tolerance` (0.05) of nominal, 44 of them on the boundary at 17/20,
against 25 under the old sign (14 at 17/20, one at 19/20); strictly inside the band it is 7 against 10, and
119 groups sit more than 0.05 below nominal (145 before). **A float comparison miscounts this:**
`abs(0.85 - 0.9)` is `0.05000000000000004`, so `abs(c - 0.9) <= 0.05` drops all 44 boundary groups and
reports 7 (recorded against `D-109`, which owns the unbuilt gate). The seven interval-bearing estimators'
means land between 0.75 and 0.77, where the old sign spread them from 0.16 to 0.68. The intervals are cross-sectional leave-one-out rather than §10.7's
rolling ones (`R-S5P-7`), which this fix does not change.

## 2026-09-13 — R-PM-5 measured: the sibling path reconstructs nothing, so R-PM-4 does not bind

`scripts/audit/qcew_parent_margins.py` now walks `1131` and `1132` with the other four industries and
judges them with the same ladder (plan 15 Task 1); `specs/findings/qcew-parent-margins.md` is the
rendering. On the 252 suppressed private `113310` state-quarters a published `113` bounds, all of
`113`, `1131` and `1132` are published on **0**, and on
**0** when a missing sibling row is read as zero
establishments. The ladder's tally is exact 0, upper bound 252, none
157.

**Ruling on R-PM-4:** no measured margin, whether parent, ownership or sibling, reconstructs a suppressed
state-quarter, so `exact_reconstruction_flag` keeps no live instance and the parent margin enters the
constraint system as a bound only. `D-110` closes on this measurement. One sibling is published on
17 of the bounded quarters, which would tighten the bound to
`113 - sibling`; no row builds it, and plan 15's completion files that as its own item.

## 2026-09-13 — R-PM-7: the comparand re-run against the bounded identification set

`runs/4cf47a918dd8` (plan 15 Task 9) against `runs/f03023ac9f3a`, compared by join on each table's key,
never by position. R-PM-7's answer is that the comparand IS re-run: Task 4 re-ids every run, and both
the production runner and the harness now read bounds (Decisions 3 and 6), so every estimate a bound
binds on moves. The rows below are that movement, derived. `runs/f03023ac9f3a` stays on disk as Stage 4's
acceptance run; Stage 5's §13.10 promotion compares against `runs/4cf47a918dd8`.

| `baseline_results` | value |
|---|---|
| rows old / new | 12270 / 12270 |
| key-only rows old / new | 0 / 0 |
| estimates moved beyond 1e-09 | 10870 |
| estimator-months with a moved estimate | 850 |
| estimates now exactly at a finite upper bound | 3391 |

Moved estimates by estimator: `cbp_intensity` 1054, `constrained_regression` 1227, `equal_residual` 1227, `establishment_proportional` 1227, `share_break_adjusted` 1227, `share_exponentially_weighted` 1227, `share_last_observed` 1227, `share_rolling_median` 1227, `share_same_month_prior_year` 1227.

| `validation_scores` | value |
|---|---|
| rows old / new | 12530 / 12530 |
| key-only rows old / new | 0 / 0 |
| scored rows with a finite `selected_upper`, old / new | 0 / 9690 |
| estimates moved beyond 1e-09 | 5917 |

| `validation_scoreboard` | value |
|---|---|
| rows old / new | 270 / 270 |
| key-only rows old / new | 0 / 0 |
| WAPE moved beyond 1e-09 | 224 |
| WAPE change, min / median / max | -0.6543 / -0.0969 / 8.0117 |

| regime | preferred baseline, old | new |
|---|---|---|
| `clustered_states_within_month` | `share_last_observed` | `share_last_observed` |
| `concentration_proxy` | `share_last_observed` | `constrained_regression` |
| `long_consecutive_runs` | `cbp_intensity` | `cbp_intensity` |
| `naics_transition` | `cbp_intensity` | `cbp_intensity` |
| `regional_blocks` | `share_last_observed` | `share_same_month_prior_year` |
| `small_cell_biased` | `cbp_intensity` | `share_last_observed` |
| `structural_break` | `cbp_intensity` | `share_exponentially_weighted` |
| `whole_seasonal_blocks` | `cbp_intensity` | `cbp_intensity` |
| `whole_state_year_blocks` | `cbp_intensity` | `cbp_intensity` |

`validation_metrics`: 6932 / 6932 rows, key-only 0 / 0; `value` moved on 3662: `deterministic_bounds` 250, `point` 2392, `probabilistic` 1020.
Exactly recoverable targets rejected from scoring (§13.2 step 6): 0. `validation_manifest.json` records `code_commit` `0e473e9dcadf645f8acb5e63ff6e2323d87b7615`.

### Added by plan 15's final review (derived from the run directories)

The tables above were re-measured after the review's fixes; `validation_manifest.json` records `code_commit` `0e473e9dcadf645f8acb5e63ff6e2323d87b7615`.

| `disclosure_flags` | `runs/f03023ac9f3a` | `runs/4cf47a918dd8` |
|---|---|---|
| `narrow_feasible_interval_flag`, by cell kind | `national_size` 1 | `national_size` 1, `state_total` 23 |
| `exact_reconstruction_flag` | 0 | 0 |

The new narrow flags are parent-bounded state cells with `selected_upper` between 2 and 10. A `[0, U]` interval's relative width is 2, so only the absolute arm fires (`narrow_interval_absolute_width: 10`), and `narrow_interval_action: manual_review` routes each to §9.8 review.

**What the WAPE movement is.** Over the 5932 scored rows with an estimate in both runs, the absolute error summed 2,180,827 before and 1,315,134 after, a net 537,515 employees of estimate moved INTO scored targets, and 0 estimates exceed a finite cap. By regime (a scoreboard row counts as worse or better when its WAPE moves by more than 0.05):

| regime | worse | better | median change | max change |
|---|---|---|---|---|
| `clustered_states_within_month` | 0 | 11 | -0.0300 | +0.0067 |
| `concentration_proxy` | 0 | 26 | -0.1473 | -0.0301 |
| `long_consecutive_runs` | 19 | 2 | +0.2669 | +0.8864 |
| `naics_transition` | 0 | 23 | -0.1017 | -0.0093 |
| `regional_blocks` | 0 | 14 | -0.0808 | +0.0372 |
| `small_cell_biased` | 0 | 21 | -0.1160 | +0.0041 |
| `structural_break` | 0 | 22 | -0.0892 | -0.0097 |
| `whole_seasonal_blocks` | 0 | 6 | -0.1858 | -0.0058 |
| `whole_state_year_blocks` | 3 | 20 | -0.1399 | +8.0117 |

The largest rise, +8.0117 on `whole_state_year_blocks` seed 4096 `equal_residual`, is 12 scored targets with a summed truth of 62, 0 of them with a finite upper bound and 12 above their truth, receiving +496.7 employees of estimate. Clamping an estimate to a valid bound cannot raise the absolute error summed over every suppressed cell of a month -- a capped cell loses exactly the mass the others gain, and stays at or above its truth -- but the harness scores only masked targets. A capped REAL suppression's improvement is unscored, while the mass it sheds lands on scored ones.

**§13.10 caveat.** A masked target's score now depends on which bounds bind on the other cells of its month, so masked-target WAPE is not comparable across runs with different identification sets: read `runs/f03023ac9f3a` and `runs/4cf47a918dd8` as different comparands, not as one comparand improved.

**What the review's fixes moved** (the pre-review run at `code_commit` `bc0498aa1f96e6ec765ae9d1eefabf5fdd98b169`, kept aside, against this one):

| quantity | pre-review / final |
|---|---|
| reconcile `max_residual_drift` | 9.932e-10 / 4.547e-13 (Stage 4's comparand: 4.547e-13) |
| `baseline_results` key-only rows | 0 / 0 |
| `baseline_results` estimates moved beyond 1e-09, largest change | 0, 7.637e-10 |
| `validation_scores` estimates moved beyond 1e-09, largest change | 0, 9.068e-10 |
| `validation_scoreboard` WAPE, largest change | 5.906e-12 |
| `validation_metrics` `value` moved beyond 1e-09, by family | `probabilistic` 1 |

## 2026-09-13 — superseded Stage 5 roadmap reading (stage-block rule 1)

Plan 15 Task 10 replaced this span of the Stage 5 block, which treated `D-111` as pending:

> **The §9.3 margins are a precondition of THIS stage, not Stage 6's.** Every one of the 1,227 suppressed state cells is `[0, +inf)` in the SHIPPED `deterministic_bounds` — `selected_upper` null on all 1,227 — because no parent-industry or ownership margin has ever entered the constraint system. This stage consumes `deterministic_bounds`, so the question is settled before it, by `specs/completed/stage5-gate-inputs.md` R-S5G-5. **MEASURED 2026-09-11: a margin EXISTS and the shipped identification set is therefore incomplete.** Of the 409 suppressed private state-quarters, **252 (61.6%, 756 of the 1,227 months) carry a disclosed private `113` parent**, which bounds `113310` above by nonnegativity of `1131` and `1132`; **0 are exactly reconstructed** — `1133` and `11331` are disclosed on 0 of 409 because the single-child chain is suppressed together, `own_code 0` does not exist at state x 6-digit (a measured absence, not an unfetched one), and no disclosed `113` parent on a suppressed quarter is a `'-'` true zero. No MEASURED margin therefore gives REQ-027/§14.4 a live instance (`D-110`) — but the `113 - 1131 - 1132` path was not measured, so on the 252 bounded quarters exactness is unknown rather than absent. The BOUND is not absorbed here: it needs a `registry/sources.yaml` row per new slice, new cell kinds and a `rows.size_margin_rows`-style parent-margin builder in `constraints/`, a §13.2 step-4 rule in `validate/recover.py` for when the parent stays visible under the synthetic pattern (it is public on 252 of 409 real suppressions, so hiding it always would score methods under harder identification than production — today that moves §13.5's bound metrics and `validation_scores`' `selected_*` columns, not WAPE, coverage or the scoreboard, since baselines never read masked bounds (`D-087`); step 6 labels any exact case), and it makes `D-093`'s MILP gap reachable for the bounded cells whose LP width falls below §9.6's threshold of 25 — `_needs_milp` needs a finite `upper` AND that width, which is why MILP has run on 0 of 4,775 rows. Routed to `specs/stage5-parent-margin.md`; **Stage 5 MUST NOT consume `deterministic_bounds` as identification-complete until that spec lands.** See `specs/findings/qcew-parent-margins.md`; `specs/findings/stage-5-log.md` carries the history.

## 2026-09-26 — `D-113`: the comparand re-run under a calendar-month discount

`5667d93` makes `share_exponentially_weighted` discount each share by its age in calendar months,
counted back from the newest observation, instead of by its position in a list of disclosed months.
`runs/4cf47a918dd8` was re-run at that commit with a clean tree -- `run-baselines`, `reconcile`, then
`validate` (12 min 42 s, with the slow tests sharing the CPU) -- after the directory was copied to
`runs/_pre_d113_4cf47a918dd8`. The run id did not move, since the change adds no config key. Every
manifest the re-run wrote records
`code_commit` `5667d93`, so this also brings the comparand up to date with `e937605`'s exactly
rounded metric sums, which the 2026-09-13 run predated. Compared by join on each table's key, never
by position:

| artifact | rows, both sides | key-only | what moved |
|---|---|---|---|
| `baseline_results` | 12,270 | 0 | 501 rows, all `share_exponentially_weighted`: `raw_weight` on 120, `estimate` on 486, `estimate_integer` on 64 |
| `validation_scores` | 12,530 | 0 | 214 rows, all `share_exponentially_weighted` |
| `validation_metrics` | 6,932 | 0 | `value` on 238 exponential rows, 223 beyond rel 1e-12 (point 184, probabilistic 36, constraint 3), and on 351 rows of the other eight, by at most 1.3e-14 relative |
| `validation_scoreboard` | 270 | 0 | `wape` on 19 exponential rows, and on 52 of the other eight by at most 4.1e-16 relative |

**The attribution is clean.** No other estimator's estimate moved: every non-exponential row of
`baseline_results` and `validation_scores` is identical. The other eight's metric and scoreboard
movement is the last-ulp signature of `e937605`, which changed how the metrics sum, not what they
sum. `raw_weight` moves on exactly 120 rows, and matched by month and state they are the 120 gappy
histories. `estimate` moves on 486 because §12.2 renormalizes every cell of a month in which one own
weight moved, and plan 15's §12.3
scaling then clips the largest moves: 3.92% at most here, against 7.60% under allocation alone. The
exponential variant's `weight_basis_counts` (327 own, 900 fallback) and its declines (none) are
unchanged, as they must be, since a positively weighted mean of the same shares is positive exactly
when the old one was. `max_residual_drift` stays 4.547e-13. The three constraint rows are
`anchor_adding_up_max_abs` moving at machine precision.

**§13.10's comparand is unchanged in 9 of 9 regimes, and so is the unrestricted best.**
`structural_break` still names `share_exponentially_weighted`. Pooled across seeds, its WAPE went
0.04015 -> 0.04009 against the runner-up `share_last_observed`'s unchanged 0.04019, so the margin
that makes it the comparand widened from 0.00004 to 0.00010: a near tie either way. Its per-seed
WAPE moved in both directions across regimes, by at most 3.96% (`regional_blocks` seed 4096,
0.00580 -> 0.00557). The largest relative move anywhere is a division stratum, `naics_transition`
seed 1024 `mountain`, 0.0160 -> 0.0737, where a few cells dominate the ratio. Two coverage values
moved.

## 2026-09-26 — resume reconcile: the comparand is current, and two superseded Stage 5 readings

`derive-roadmap` §5 re-validated Stage 5 against what shipped after the roadmap's last edit
(`a0dcdb6`). Measured at `ecfa1d6`: `run_id(cfg, _input_digests(cfg))` for `config.yaml` resolves to
`4cf47a918dd8`. Its manifests record `code_commit` `5667d93` for `baseline_manifest.json`,
`reconcile_manifest.json` and `validation_manifest.json`, and `bc0498aa` for `bounds_manifest.json`
and `schema_manifest.json`. The older two are current, not stale: between `bc0498aa` and `ecfa1d6`
the only changes under `constraints/`, `disclosure/`, `harmonize/`, `build.py` and `contracts.py` are
two docstring paths re-pointed to `specs/completed/` and `constraints/CLAUDE.md`.

The spec's Stage 4 stamp still named `runs/f03023ac9f3a` as Stage 5's comparand and "CURRENT", in the
Rollout note writing-plans reads. It now carries an italic *Superseded…* qualification rather than a
rewrite, the way plan 15 qualified the Stage 2 stamp; its `COMPLETE` line is untouched.

**Superseded readings** (stage-block rule 1), replaced in the roadmap's Stage 5 block.

`Consumes`, the comparand sentence, which did not say when the directory's contents were written:

> Both the runner and the harness scale estimates into finite bounds by §12.3 (`D-087`), and the §13.10 comparand is `runs/4cf47a918dd8`, not Stage 4's acceptance run `runs/f03023ac9f3a`.

`Exit`, which contradicted `Consumes` by pointing a planner back at Stage 4's acceptance run:

> the promotion record states beat or not-beaten against Stage 4's numbers and the simpler method is selected when not beaten (§13.10 final line).

## 2026-09-26 — `/deferred`: the two carved-out spec amendments get owners, then land

`specs/completed/stage5-preconditions.md` §6 carved four spec amendments out of plan 13. The roadmap
routed two of them: Appendix B to Stage 8 as a precondition, and the §13.10/§11.1 re-gate to Stage 7.
The other two were owned by nothing once that spec retired. One `/deferred` pass filed them as `D-120`
and `D-121`, closed both as quick fixes in commits of their own (`416f0f3`, `90b30e2`), and filed
`D-122`, R-04's code clause, for this stage.

- `D-120`: §12.2 now defines `N_t` as a declared anchor: the published national total the
  establishment-closure gate admits, a `modeling_assumption` stamped `declared_national_total`, with
  `SRC-QCEW-006`'s decline standing. It lists the three allowed `anchor_basis` values and the
  retirement condition, and §15.2 lists `anchor_basis`. The retirement check has no implementation:
  `baselines/runner.py::run_baselines` skips a month with an empty missing set without testing `R_t`.
  That is unreachable on D1, where every month has a suppressed state cell, and `D-123` owns it.
- `D-121`: Appendix A carries `config.yaml`'s `baselines:` block and both narrow-interval widths, the
  widths labelled as the governance owner's policy (§21). The fence now fails only on `model:`.

Checked while filing, against the spec at `f842897`. §15.2 contained no "compatible national total"
text: it was a bare field list, and its gap was the absent field. §2.2's Reconciliation row names no
total, so it holds once §12.2 names `N_t`, and it needed no edit.

**Superseded reading** (stage-block rule 1), replaced in the roadmap's Stage 5 `Consumes`. It claimed
§15.2 defines the residual, and it named an owner that could no longer act:

> `reconcile/anchor.py::national_residual` stamps `anchor_basis = 'declared_national_total'` (Stage 3 `SHIPPED` point (5)), while spec §12.2 and §15.2 still define the residual against a "compatible national total", which SRC-QCEW-006 declined. That amendment, together with Appendix A's three missing fields, is owned only by the retired `specs/completed/stage5-preconditions.md` §6.

## 2026-09-26 — `/deferred` quick fixes: `D-122` and `D-123` land

Both were filed by the pass above and fixed test-first. `D-122`: `Anchor.__post_init__` refuses an
`anchor_basis` outside `contracts.ANCHOR_BASES`, which is the only basis check this stage's draws path
has. `D-123`: `assert_universe_closes` halts the run on a month with no suppressed state cell whose
integer residual is not 0, the check §12.2's retirement condition requires. Neither changes D1 output:
every anchor is built with `declared_national_total`, and none of the 96 months has an empty missing
set. The local suite rose from 1532 to 1538 passed: the six new tests, and nothing else moved.

**Superseded readings** (stage-block rule 1), replaced in the roadmap's Stage 5 `Consumes` as PR #33
merged it (`eddc488`):

> Open items that name this stage: `D-109`, `D-115`, `D-116`, `D-122`.

> Since `D-120`, spec §12.2 names that anchor, the establishment-closure gate that admits it and its retirement condition (whose identity check no code runs yet, `D-123`), and §15.2's release fields carry `anchor_basis`; SRC-QCEW-006's `decline` stands. §7.11's `posterior_summary`, this stage's own artifact, has no `anchor_basis` column, so on the draws path nothing checks an anchor's basis until `D-122` refuses a bad one when the `Anchor` is built.

## 2026-09-27 — plan 16 on D1: the comparand re-run byte for byte, the production fit, the promotion record

Plan 16 executed inline (executing-plans), one commit per task, on `worktree-plan16-stage5` from
`main` at `956f52c`. The plan was written at `40b2688`. Main had since merged #38 and #39, 38
hermetic tests between them, and #40, which changed only the plan's text. Before Task 1, all 22 of
the plan's diff blocks applied in plan order at `956f52c`.

One deviation was approved at execution. `h5netcdf` 1.8.1 made its HDF5 backends extras, so the
plan's bare `h5netcdf>=1.8.1` installed none, and Task 1's round-trip probe failed with "No module
named 'h5py', backend not available". The dependency is now `h5netcdf[h5py]>=1.8.1` (h5py 3.16.0).
No version already in `uv.lock` moved. The 13 tests that needed h5netcdf ran for the first time and
all passed.

**Suite counts, measured.** Base at `956f52c` with `data/` linked: 1608 passed for the whole suite,
and 1581 passed, 27 deselected for the non-slow tier. Every task's non-slow delta was the plan's, 125
in all, ending at 1706 passed, 47 deselected. At Task 14 with `data/`: 1753 passed, 0 skipped
(1608 + 145), and nothing under `data/` or `runs/` was written. Without `data/`: the bare run gave
1681 passed, 72 skipped, and `-m "not slow and not network"` gave 1661 passed, 45 skipped, 47
deselected. The plan had written 1643/72 and 1623/45/47, predicted at `40b2688`. The difference is
main's 38 exactly, so `CLAUDE.md` and `ci.yml` carry the measured numbers. Skipped stayed at 72 and
45, so every test the plan added runs without `data/`. There are seventeen `mark.slow` sites.

**The comparand, re-run under `dd7337e89047`.** Task 15 ran `build-constraints`, `solve-bounds`,
`run-baselines`, `reconcile` and `validate` at `b666506` with a clean tree. All five exited 0, and
`validate` took 692.9 s. The byte check printed "15 run files and 3 constraint tables: identical":
- every Parquet file matches `runs/4cf47a918dd8`'s by sha256;
- every JSON manifest matches except `code_commit` and `uv_lock_sha256`;
- `config.resolved.yaml` differs only by `model:` and `promotion.catastrophic_stratum_coverage_alpha`;
- `build-constraints` rewrote the main checkout's `data/constraints/` with bytes identical to the copy
  taken first.

Nothing under `runs/4cf47a918dd8`, `runs/f03023ac9f3a`, `runs/source_manifest.parquet`, `data/staged`
or `data/raw` was written.

**The production fit.** `fit-state-model` exited 0 in 139.5 s wall ("fit 4 chains; gate passed;
1227 cells reconciled"). This was its first run on the locked jax 0.11.2 and numpyro 0.22.0.
`posterior/diagnostics.json`, in full:

```json
{
  "cells_constant": 0,
  "checks": {
    "cell_ess_bulk_min": {
      "gating": true,
      "passed": true,
      "threshold": 400.0,
      "value": 1009.993799359656
    },
    "cell_ess_tail_min": {
      "gating": true,
      "passed": true,
      "threshold": 400.0,
      "value": 433.91598390921814
    },
    "cell_rhat_max": {
      "gating": true,
      "passed": true,
      "threshold": 1.01,
      "value": 1.0044927598072078
    },
    "divergences": {
      "gating": true,
      "passed": true,
      "threshold": 0.0,
      "value": 0.0
    },
    "parameter_ess_bulk_min": {
      "gating": false,
      "passed": true,
      "threshold": 400.0,
      "value": 654.6116146720314
    },
    "parameter_ess_tail_min": {
      "gating": false,
      "passed": true,
      "threshold": 400.0,
      "value": 799.877996606086
    },
    "parameter_rhat_max": {
      "gating": true,
      "passed": true,
      "threshold": 1.01,
      "value": 1.0066868251700758
    },
    "ppc_coverage_90": {
      "gating": true,
      "passed": true,
      "threshold": 0.85,
      "value": 0.9041016753321779
    },
    "reconciliation_anchor_drift_max": {
      "gating": true,
      "passed": true,
      "threshold": 1e-09,
      "value": 4.547473508864641e-13
    },
    "reconciliation_bound_violations": {
      "gating": true,
      "passed": true,
      "threshold": 0.0,
      "value": 0.0
    }
  },
  "code_commit": "b666506be2579ad57c3edcd7134ca8ca458b6b3d",
  "constraint_set_hash": "83b95de07b1b21d733e8aaae4ad18955d5f07c503d67b3b785a5216aed7f80c6",
  "failures": [],
  "passed": true,
  "run_id": "dd7337e89047",
  "scope": "production",
  "uv_lock_sha256": "f3dc9f623a90a35a92321033171b3f12c2ae30a70152ac46aafea32792a7b8cd"
}
```

The `sampler` block of `state_model_manifest.json`:

```json
{
  "backend": "numpyro",
  "chain_method": "vectorized",
  "chains": 4,
  "draws": 1000,
  "jax_version": "0.11.2",
  "max_tree_depth": 10,
  "mean_leapfrog_steps": 31.0,
  "model_id": "state_total_model",
  "model_version": "student_t_ar1.1",
  "numpyro_version": "0.22.0",
  "seed": 3645,
  "target_accept": 0.9,
  "tree_depth_saturation_share": 0.0,
  "warmup": 1000
}
```

The fit reproduces Decision 15's profile: 31 leapfrog steps a draw, no tree at the depth ceiling,
and posterior medians σ_η 0.0364 and ρ 0.838. Two margins are thinner than on the older environment.
Parameter R-hat is 1.0067 against 1.0047 there, and cell tail ESS is 434 against 1,176, where the
floor is 400. `reconcile` then re-read the store: 4,000 draws × 1,227 cells over 96 months, max
anchor drift 4.547e-13, 0 bound violations, a matching draws digest, and `state_model.passed` true.

**Cells on a bound.** Of 1,227 imputed cells, 189 have a zero-width 90% interval, and every one sits
on its §9 upper bound: VT (50) 67, NM (35) 28, NH (33) 22, CO (08) 20, UT (49) 14, WV (54) 12, WI
(55) 12, KY (21) 5, IA (19) 4, CT (09) 2, WY (56) 2, SD (46) 1. The older environment gave 192.

**The promotion record.** `validate-state-model` exited 0 after 3,826 s wall, 63.8 minutes against
the plan's estimate of about 77. It wrote `not_beaten` and selected `section_10_8_hierarchy`.
`provisional` is true until "Stage 7 re-runs §13.10 with the harvest factor and §11.13's variants",
and `disclosure_review` is `pending_stage_8`. Both `replicates_checked` are 27. The `comparand` block
names `dd7337e89047` and the sha256 of its three validation tables, which equal
`runs/4cf47a918dd8`'s. Each gate:

| gate | passed | what decided it |
|---|---|---|
| `hard_constraints` | true | 0 bound violations and anchor drift 4.5e-13, for the production draws in memory and in the store and for all 27 replicates |
| `convergence` | false | 1 replicate of 27, `whole_seasonal_blocks` seed 2048, had parameter R-hat 1.0187 against 1.01. The other 26 ran from 1.0037 to 1.0079, and all 27 had 0 divergences |
| `coverage` | false | pooled 940 of 1,253 = 0.750, against 0.90 ± 0.05. Catastrophic at α 0.001 in 7 of 9 regimes and 6 of 9 divisions. Within tolerance only in `clustered_states_within_month`, 55 of 60 |
| `improvement` | false | matched-pair WAPE beats the comparand in 4 of 9 regimes: `whole_seasonal_blocks` +72.2%, `clustered_states_within_month` +54.5%, `naics_transition` +42.3%, `structural_break` +16.3%. It loses in 5: `long_consecutive_runs` −59.5%, `regional_blocks` −68.3%, `small_cell_biased` −90.3%, `whole_state_year_blocks` −124%, `concentration_proxy` −235%. The calibrated-uncertainty alternative rescues none, because the model's own coverage is catastrophic in each |
| `stratum_degradation` | true | the model's WAPE, pooled across regimes, is below the comparand's in all 9 divisions, by 36% to 62% |
| `disclosure_review` | `pending_stage_8` | Stage 8 |

Any one of the three failures makes the verdict `not_beaten`, because `evaluate_promotion` requires
all five gates to pass (Decision 4). The model is more accurate where most of the pool lies, since
`whole_seasonal_blocks` is 69% of it. But its reconciled-draw intervals are too narrow on hidden
cells, and the simpler comparand beats it in five of the smaller regimes. Several replicate fits
would fail production scope, recorded for information: cell R-hat reached 1.0167, and cell tail ESS
fell to 228.

§19 Phase 3's acceptance, read against this run:
- "diagnostics pass" holds for the production fit and for 26 of the 27 replicate fits;
- "constraints pass on every draw" holds;
- the baselines are not beaten, so the simpler method is retained.

**`D-109`'s evidence (Task 12 Step 4).** Before its conversion, the tripwire reddened naming all four
keys:

```text
E       AssertionError: now read by {'validate/promotion.py': ['catastrophic_stratum_coverage_alpha', 'maximum_major_stratum_wape_degradation', 'minimum_wape_improvement', 'nominal_coverage_tolerance']}; update PromotionConfig's docstring
FAILED tests/unit/test_config_validation_block.py::test_the_promotion_keys_are_still_unread_and_the_docstring_still_says_so
1 failed, 6 passed
```

**Open for the Plan Completion Protocol:**
- Decision 7's four flagged readings: §11.3's exact observation, §11.12's capped persistence,
  §11.14's non-centred SHOULD set aside for trained states, and §11.1's split X.
- Decision 15's open points: the 189 imputed cells whose interval is a bound, and the unstable
  posterior means in NV and RI.
- `run-baselines`' unchecked bounds hash (Decision 9).
- The dispositions of `D-116` and `D-115` (Decision 1). `D-116` is declined again, because a new
  propensity predictor would re-draw every regime's mask. `D-115` is not taken, because the model and
  the comparand are scored on one identification set.
