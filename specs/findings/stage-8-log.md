# Stage 8 — log

Dated corrections and measurements for Stage 8's roadmap block (roadmap
`## Stages`, stage-block rules 1 and 2). Stage 8 is unticked; this file was
created at the 2026-09-26 resume reconcile.

## 2026-09-26 — Appendix B's step 7 is met in part since `D-111`

`derive-roadmap` §5 re-validated the precondition in Stage 8's `Exit`. Plan 15 (`D-111`, 2026-09-13)
put the published private `113` parent into the constraint system, so 756 of the 1,227 suppressed
state cells carry a finite upper bound and 471 stay `[0, +inf)` (`specs/findings/stage-5-log.md`).
Appendix B's step 6 is unchanged: a national state-sum constraint is still refused. The line pin
`constraints/system.py:100` had drifted to `:101` and is now the symbol
`constraints/system.py::build_constraint_system`. Superseded reading:

> Its step 6 requires building "a national state-sum constraint", which SRC-QCEW-006 declined and which `constraints/system.py:100` refuses unconditionally via `assert_no_national_employment_margin`; its step 7 assumes LP bounds for suppressed states that the engine does not produce (all 1,227 are `[0, +inf)`).
