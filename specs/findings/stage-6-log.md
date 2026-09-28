# Stage 6 — log

Dated corrections and measurements for Stage 6's roadmap block (roadmap
`## Stages`, stage-block rules 1 and 2). Stage 6 is unticked; this file was
created at the 2026-09-26 resume reconcile, the first time a Stage 6 sentence
was replaced rather than amended in place.

## 2026-09-26 — three clauses that plans 13 and 15 made false

`derive-roadmap` §5 re-validated Stage 6 against what shipped after the roadmap's last edit
(`a0dcdb6`), and against plans 13 and 15, which the block had absorbed only in `Consumes` (2).

**`RE-VALIDATED` (2).** Plan 15's `51b1777` made `baselines/runner.py::run_baselines` pass per-cell
bounds to `integerize` through `integer_bounds` wherever a finite bound is scoped, so the first
bounded call site already exists and is not this stage's. Superseded reading:

> (2) Plan 5 changed `reconcile.integerize`'s bound handling and names this stage as where it goes live: bounds are read per cell in `values` rather than per key of `upper`, a `lower` above an `upper` now raises `ValueError`, and the round-robin remainder orders on the floor actually used (`reconcile/integerize.py:55-63`, `:79-87`). All three are unexercised today — the only `src/` caller, `baselines/runner.py::run_baselines`, passes no bounds — so a plan written against the pre-plan-5 function would get a different allocation on bounded input with no error. A plan for this stage must locate the first bounded call site rather than assume §11.9's `[L_k,U_k]` class bands are it.

The roadmap's Stage 3 `SHIPPED` point (6) carries the same pre-plan-15 reading (`run_baselines`
"calls `integerize(allocated, total=...)` with neither `lower` nor `upper`"). That block is frozen
(rule 3) and is one of the three unmigrated oversized blocks, so it was left as written; the Stage 6
block is the live carrier.

**`Consumes` (4).** `apply_size_mask` does have a `src/` caller, `validate/recover.py::mask_and_solve_size`,
which itself has none. And since plan 15, §13.2 step 6 and §13.5 do fire on the state arm (`D-085`'s
2026-09-13 note), so "never been able to fire" held only for the size arm. Superseded reading:

> (4) the harness's size arm is defined but unwired — `mask_and_solve_size` and `apply_size_mask` have no `src/` caller, so §13.2 step 6 and §13.5 have never been able to fire;

**`Exit`, one clause removed.** INV-002's per-cell-bounds half was enforced on the production path by
plan 13 (R-S5P-3, 2026-09-10) and on the validation path by plan 15 (`D-087`, 2026-09-13), which
`Consumes` (2) already recorded as CLOSED while the `Exit` still demanded it. INV-002 is not among this
stage's `Gap closed:` rows (the gap table assigns it to Stage 3's layer and Stage 8's release check),
so the discharged clause is removed rather than re-scoped to this stage's draws. Superseded reading:

> every baseline estimate is checked against its per-cell bounds (INV-002's bounds half, unenforced since Stage 3);

## 2026-09-28 — one clause that `D-119` made false

`D-119`'s `/deferred` quick fix made `reconcile/integerize.py::integerize` raise
`InfeasibleResidualError` on all three of its infeasibility refusals, so the last sentence of
`RE-VALIDATED` (2) no longer held. Only that sentence changed. Superseded reading:

> Its three infeasibility refusals are still plain `ValueError` (`D-119`).

The Stage 3 `SHIPPED` point (6) still says a `lower` above an `upper` "raises `ValueError`". That
block is frozen (rule 3), as the 2026-09-26 entry records for the same point, so it is left as
written; this block is the live carrier.
