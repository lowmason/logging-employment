# Fail-Closed Named Errors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> **subagent-driven-development** (the default) — or **executing-plans** when your human partner
> chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status: COMPLETE (2026-09-28)** — executed via subagent-driven-development; nothing deferred

> Deviation (final review, `34a4f4f`): the whole-branch review (a code-reviewer and a Codex second
> opinion) found that the inventory as written collected its keys in a `set`, so a second
> `raise ValueError` added inside a function already in `KEEP` passed silently. `KEEP` now maps each
> key to `(count, reason)`, `_bare_value_error_sites` became `_bare_value_error_counts` (a
> `Counter`), and the test refuses unclassified, stale and miscounted keys; its docstring names what
> the walk cannot see. The same commit corrected docstrings this plan wrote or kept:
> `SecretInPayloadError` (it backs up `_without_credentials` only; `resolved_dict`'s output is
> never scanned), `FallbackExhaustedError`, `preferred_estimator` and its new test (reachable only
> through `run-baselines`, not from a §13 mask), `InfeasibleResidualError` (a summary true of all
> ten raise sites, with `scale_into_bounds`, `allocate` and `release_integers` named), and
> `errors.py`'s module docstring and root `CLAUDE.md`'s fail-closed bullet (`SecretInPayloadError`
> carries no value). It reflowed `bridge_frame`'s docstring and the secrets bullet. The code blocks
> below are what Tasks 1–6 landed (`79bcb6d`..`8ec3922`), not the final text.

> Deviation (environment): executed in the primary checkout, which has `data/`, so the hermetic
> tier's 45 data-bound skips ran and passed: base `1803 passed, 49 deselected` at `8e25074`, and
> `1810 passed, 49 deselected` after Task 6. The +7 holds. The per-step counts below differ only
> where a skip became a pass, and every list of failures matched the plan's.

> No spec file: this plan is named for its theme, as `/deferred` prescribes for a Plan disposition,
> and its requirements are one register item. The Plan Completion Protocol ticks that item.

**Goal:** Make root `CLAUDE.md`'s fail-closed rule say which refusals take a named `errors.py`
error, convert the ten sites `D-140` assigns one, and hold the resulting inventory of bare
`ValueError`s with a test rather than a dated count.

**Architecture:** `errors.py` gains three classes (`ClassificationContinuityError`,
`FallbackExhaustedError`, `SecretInPayloadError`) and two widened docstrings; nine functions across
`ingest/`, `harmonize/`, `baselines/`, `validate/`, `reconcile/` and `store.py` change the class they
raise and nothing else — every message is unchanged, so every existing `match=` still matches
(`D-119`'s precedent). A new test, `tests/unit/test_fail_closed_inventory.py`, walks `src/` with
`ast`, keys every `raise ValueError` by `module::function`, and refuses one that is not listed with
its reason. Its `PENDING` set names the sites this plan converts and is drained one task at a
time, so each task's first step turns that test red before the site's own test does.

**Tech Stack:** Python ≥ 3.14, uv + hatchling, Polars, pytest, ruff, interrogate. Nothing new.

**Requirements input:** `specs/deferred_items.md` `D-140`, read at `46ef072`. Its rule, its NAME,
KEEP and DECIDE lists, and its Done-when are the whole brief. Also read: root `CLAUDE.md`'s
"Fail closed with a named error" bullet, `src/logging_employment/ingest/CLAUDE.md`'s fail-closed
table, `errors.py`, and the closed `D-119`, which converted four sites the same way.

**Provenance of the code in this plan.** Every code block in Tasks 1–6 was executed before this
plan was saved, in plan order, as one commit per task, in a throwaway local clone of `46ef072`
(deleted after). Tests ran through this checkout's own `.venv` with `PYTHONPATH=<clone>/src`, so
the package under test was the clone's. Every commit passed `ruff format --check`,
`ruff check`, `interrogate` and its own tests; Task 6 ran the hermetic tier. The observed lines are
quoted where they appear, as "observed". One test was written wrong the first time and is shown
corrected, with the failure it produced, in Task 2 Step 2.

**Suite counts, measured.** No test here needs `data/`, so the counts are the hermetic tier's,
`uv run pytest -m "not slow and not network"`, which is what CI runs:
- Base, at `46ef072`, in the clone (no `data/`): `1758 passed, 45 skipped, 49 deselected in 127.23s`.
- After Task 6: `1765 passed, 45 skipped, 49 deselected` — seven tests added, none slow, none
  data-bound: Task 1 +1, Task 2 +3 (one parametrized over two archives), Task 3 +1, Task 4 +2,
  Task 5 +0.
- `rg -c 'raise ValueError' src/` summed: 37 before, 25 after. The twelve gone are the eleven
  raise statements in the nine NAME functions (`assert_113310_survives_the_window` holds four) and
  the one in `reconcile/scaling.py::Bounds` (Decision 3).

---

## Global Constraints

- `requires-python = ">=3.14"`; every command below is `uv run …` from the repo root.
- `uv run ruff format src tests` and `uv run ruff check src tests` clean after every task.
  **Never `ruff format .`** (root `CLAUDE.md`, Gotchas: it rewrites two `scripts/audit/` files
  into syntax their own PEP 723 headers cannot parse).
- `uv run interrogate src` is `fail-under = 100`: every new class has a docstring.
- **Every message is unchanged.** A site changes the class it raises and nothing inside the
  parentheses. Every existing `match=` pattern in `tests/` must still match, which is how `D-119`
  landed and how a reviewer can check the sweep changed no behaviour but the type.
- Every class in `errors.py` subclasses `LoggingEmploymentError` and carries the offending value
  (root `CLAUDE.md`). One exception, stated in its docstring: `SecretInPayloadError` carries none,
  because the value is the secret.
- Root `CLAUDE.md` is the file to edit. `AGENTS.md` is a symlink to it (`6738894`); never edit it.
- The rule, verbatim from `D-140` ("ruled 2026-09-28"): a named `errors.py` error for any refusal
  data can trigger — source bytes, fetched metadata, staged tables, or the run's own state (§18.3)
  — and `ValueError` for a caller misusing the API, where it must also stay in pydantic
  validators, which turn only `ValueError` and `AssertionError` into a `ValidationError`.
- `ingest/qcew.py::read_bulk_zip` is excluded from the sweep: `D-049`'s delete-or-fix decides it.
- Work on a branch cut from `main`, never on `main`; one commit per task; the user approves the
  PR (house convention, PR #47).

---

## Decisions this plan makes

`D-140` rules the principle and names the sites; it leaves three DECIDE sites and the class each
NAME site takes. Each decision gives its evidence so a reviewer can reject it rather than
rediscover it. Decisions 3 and 6 are the two a reviewer is most likely to want changed.

1. **Three new classes, each for one refusal that no existing class describes.**
   - `ClassificationContinuityError` for the four premises of
     `harmonize/naics.py::assert_113310_survives_the_window`. `UnsupportedReferenceYearError` is
     the neighbouring class, and its docstring is explicit that it refuses a YEAR, not the
     crosswalk; `ConceptViolationError` would fit only by stretching "concept" to cover a vendored
     file's contents. §3.1's "The ETL MUST verify the 113310 mapping mechanically" and `D-102`
     (the check runs on the build path) are a refusal of their own, and the class says so.
   - `FallbackExhaustedError` for `baselines/runner.py::preferred_estimator`. No class covers "no
     rung ran". The function's own docstring already argues for a raise over a sentinel; the class
     docstring carries that argument to where a reader of `baseline_manifest.json` would look.
   - `SecretInPayloadError` for `store.py::assert_no_secret`. Root `CLAUDE.md`'s "Secrets never
     reach an artifact" bullet lists three guards; this is the last one, and it is the only class
     here that must not carry the offending value.
   The two remaining NAME sites take `ConceptViolationError`, which their modules already raise:
   `_check_dash_rows_carry_no_establishments` crosses §2.2's "Meaning of a QCEW zero", and
   `select_targets`'s lookback refusal crosses §13.3's line between a single-month regime and a
   blackout. `validate/regimes.py` raises `ConceptViolationError` at three other sites already.

2. **`bridge_frame` takes `SchemaMismatchError`, and that class's docstring widens to say so.**
   `D-140` puts it under NAME. The only caller passes `build._BRIDGE_ROWS`, a code literal, so a
   reviewer may read it as a caller's mistake; the ruling stands because the bridge table is a
   persisted §8.6 artifact and Polars would persist the missing field as a null
   `verification_status`, the one thing a bridge row exists to state. The class docstring, "A
   fetched file's columns do not match the schema the parser declares", did not cover a declared
   row, so it now names both readers and `read_by_size_zip`.

3. **DECIDE (a): `reconcile/scaling.py::Bounds` is NAMED, as `InfeasibleResidualError`.
   FLAGGED FOR REVIEW.** `D-140`'s own text leans the other way ("`solve_bounds` refuses an
   infeasible component first, so this is probably misuse"). Two facts decide it here. The bounds
   are read off `deterministic_bounds.parquet` (`baselines/runner.py::state_total_bounds`), and a
   file is "the run's own state", which the rule names. And `D-119` gave `integerize`'s "a lower
   bound above its cap" this class, and `Bounds.__post_init__`'s docstring calls its refusal "the
   same refusal one layer earlier"; two names for one shape would be the inconsistency the rule
   exists to remove. If review prefers KEEP: Task 5 shrinks to the store site, and Task 1's `KEEP`
   gains `"reconcile/scaling.py::Bounds.__post_init__": "solve-bounds refuses the shape first"`.

4. **DECIDE (b): `classification.py`'s four raises KEEP `ValueError`.** `classification_memo` and
   `_section_31_fence` read the spec file, which is repo content, and `D-058` records that nothing
   in `src/` calls them. A refusal no run can reach is a test's, and a test is a caller. The
   inventory records the reason and names `D-058`; wiring the memo into `build_harmonized` is the
   day this entry is re-decided.

5. **DECIDE (c): `store.py::assert_no_secret` is NAMED (Decision 1), with no value in the
   message.** Task 5's test pins that the secret is absent from `str(error)`.

6. **The inventory is held by a test. FLAGGED FOR REVIEW: it is beyond `D-140`'s Done-when.**
   `D-140` closes when the rule is stated, the sites converted and the table updated, and it says
   "Re-count before starting; the inventory is dated": the count has been re-done by hand at
   `D-119` and again at `D-140`. `tests/unit/test_fail_closed_inventory.py` makes the count live.
   It parses every module under `src/` with `ast`, so a site is keyed by its enclosing function and
   survives edits above it (the pointer refreshes on `D-056` are what a line-keyed list costs).
   Its `PENDING` set is a planning device: each of Tasks 2–5 removes its entries first and sees the
   test name the still-unconverted site, so the tripwire is the first red of every task; Task 6
   deletes the empty set. To drop the tripwire entirely, delete the file, its sentence in root
   `CLAUDE.md`'s bullet, and the `PENDING` steps; nothing else depends on it.

7. **`ingest/CLAUDE.md`'s combined row is split.** Its last row listed two sites under one
   `ValueError`; they now take different classes, so each gets a row, and `read_bulk_zip`'s row
   records that `D-140` left it for `D-049`.

---

## File structure

Modified, by task:

| Task | `src/logging_employment/` | `tests/` | docs |
|---|---|---|---|
| 1 | `errors.py` | create `unit/test_fail_closed_inventory.py` | root `CLAUDE.md` (fail-closed bullet) |
| 2 | `ingest/qcew.py`, `ingest/qcew_size.py` | `unit/test_qcew_routes.py`, `unit/test_qcew_parser.py`, `unit/test_qcew_size.py` | `ingest/CLAUDE.md` (table) |
| 3 | `harmonize/naics.py`, `harmonize/bridge.py` | `unit/test_harmonize.py`, `integration/test_build_harmonized.py` | — |
| 4 | `baselines/runner.py`, `validate/regimes.py` | `unit/test_baselines_runner.py`, `unit/test_validate_regimes.py` | — |
| 5 | `reconcile/scaling.py`, `store.py` | `unit/test_scaling.py`, `unit/test_store.py` | `reconcile/CLAUDE.md`, root `CLAUDE.md` (secrets bullet) |
| 6 | — | `unit/test_fail_closed_inventory.py` | — |

Line numbers below are at `46ef072`.

---

### Task 1: The rule, the three classes, and the inventory tripwire

**Files:**
- Modify: `src/logging_employment/errors.py:57-58` (`SchemaMismatchError`), `:119-126`
  (`InfeasibleResidualError`'s docstring), append after `:240` (end of file)
- Modify: `CLAUDE.md:177-179`
- Create: `tests/unit/test_fail_closed_inventory.py`

**Interfaces:**
- Consumes: `logging_employment.errors.LoggingEmploymentError`.
- Produces: `errors.ClassificationContinuityError`, `errors.FallbackExhaustedError`,
  `errors.SecretInPayloadError`, each `(LoggingEmploymentError)` with a docstring and no body;
  `tests/unit/test_fail_closed_inventory.py` with module-level `KEEP: dict[str, str]` and
  `PENDING: dict[str, str]`, both keyed `"<module path under src/logging_employment>::<qualified
  function>"`. Tasks 2–5 delete their lines from `PENDING`; Task 6 deletes `PENDING`.

- [x] **Step 1: Write the failing test**

> Deviation (final review): this is the code `79bcb6d` landed; `34a4f4f` changed it (see the
> note under Status).

Create `tests/unit/test_fail_closed_inventory.py`:

```python
"""Root `CLAUDE.md`'s fail-closed rule, held by a test rather than by a dated count (`D-140`).

A refusal that data can trigger -- source bytes, fetched metadata, staged tables, or the run's
own state (§18.3) -- raises a named `errors.py` error. A bare `ValueError` is for a caller misusing
the API, and stays in pydantic validators, which turn only `ValueError` and `AssertionError` into a
`ValidationError`. `D-119` converted four sites and `D-140` found ten more, each time by re-counting
`rg 'raise ValueError' src/` by hand. This test is that count, made live: every `raise ValueError`
under `src/` is named here with the reason it keeps a bare `ValueError`, and a new one is refused
until its author classifies it.

Sites are keyed by module and enclosing function, never by line, so an edit above a site does not
move it (`D-056`'s pointer refreshes are what a line-keyed inventory costs).
"""

from __future__ import annotations

import ast
from pathlib import Path

import logging_employment

SRC = Path(logging_employment.__file__).parent

# Every `raise ValueError` under `src/`, and why each is a caller's mistake rather than a refusal
# data can trigger. A site's key is `<module>::<qualified function>`; several raises in one
# function share one key.
KEEP: dict[str, str] = {
    "build.py::build_harmonized": "`allow_network=True` is a caller asking for what the build forbids",
    "classification.py::_section_31_fence": (
        "reads the spec file, repo content; called from no run path (`D-058` owns wiring it)"
    ),
    "classification.py::classification_memo": "as `_section_31_fence`",
    "config.py::ModelConfig._the_fitted_multiplier_is_present": "pydantic validator",
    "config.py::ProjectConfig._is_a_month": "pydantic validator",
    "config.py::ValidationConfig._refuse_a_random_mask_only_design": "pydantic validator",
    "constraints/rows.py::_check_relation": "argument check on a row builder's own literals",
    "constraints/rows.py::constraint": "argument check on a row builder's own literals",
    "disclosure/flags.py::build_flags": "the bounds and cells passed in disagree: a caller's join",
    "fetching.py::fetch_source": "an unknown `source` string from the caller; the CLI refuses first",
    "ingest/base.py::HttpFetcher.__post_init__": "a fetcher built without a contact email",
    "ingest/qcew.py::read_bulk_zip": "dead code; `D-049`'s delete-or-fix decides it",
    "reconcile/draws.py::reconcile_draws": "a caller-supplied draw shape",
    "reconcile/projection.py::_require_indicator_margins": "a caller-supplied indicator shape",
}

# Sites the sweep this test landed with converts, one task at a time. Each task removes its entry
# first, sees this test name the site as unclassified, and then converts it. Empty when the sweep
# is complete, and deleted then.
PENDING: dict[str, str] = {
    "baselines/runner.py::preferred_estimator": "Task 4",
    "harmonize/bridge.py::bridge_frame": "Task 3",
    "harmonize/naics.py::assert_113310_survives_the_window": "Task 3",
    "ingest/qcew.py::_check_dash_rows_carry_no_establishments": "Task 2",
    "ingest/qcew.py::probe_slice_boundary": "Task 2",
    "ingest/qcew_size.py::read_by_size_zip": "Task 2",
    "reconcile/scaling.py::Bounds.__post_init__": "Task 5",
    "store.py::assert_no_secret": "Task 5",
    "validate/regimes.py::select_targets": "Task 4",
}


def _bare_value_error_sites() -> set[str]:
    """`<module>::<qualified function>` for every `raise ValueError(...)` under `src/`."""
    sites: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            raised = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if not (isinstance(raised, ast.Name) and raised.id == "ValueError"):
                continue
            scope: list[str] = []
            parent = parents.get(node)
            while parent is not None:
                if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    scope.append(parent.name)
                parent = parents.get(parent)
            sites.add(f"{path.relative_to(SRC).as_posix()}::{'.'.join(reversed(scope))}")
    return sites


def test_every_bare_value_error_is_a_callers_mistake_and_says_so() -> None:
    """A new bare `ValueError` is refused until it is classified: named in `errors.py` if data can
    trigger it, or listed in `KEEP` with the reason it cannot. A stale entry is refused too, so
    the list describes the tree."""
    found = _bare_value_error_sites()
    classified = set(KEEP) | set(PENDING)
    unclassified = sorted(found - classified)
    stale = sorted(classified - found)
    assert not unclassified, (
        f"`raise ValueError` at {unclassified} is not classified. If data can trigger it, raise "
        "a named `errors.py` error (root CLAUDE.md, fail closed); if only a caller can, add it "
        "to KEEP with the reason"
    )
    assert not stale, f"KEEP names sites that no longer raise a bare ValueError: {stale}"
```

- [x] **Step 2: Run it, and see what it refuses without `PENDING`**

> Deviation (mechanical): the red was observed with `PENDING: dict[str, str] = {}`, not with
> `PENDING` absent, because the test body reads it: `1 failed`, naming the nine functions in the
> order above and no other, then `1 passed` with `PENDING` as written.

With `PENDING` in place this test passes at `46ef072`: the tree holds exactly the 14 KEEP
functions and the 9 PENDING ones, 37 raise statements in all. Its red state is what `PENDING`
exists to defer, and it was observed before `PENDING` was added:

Run: `uv run pytest tests/unit/test_fail_closed_inventory.py -q`
Observed with `PENDING` absent: `1 failed in 0.11s`, the assertion naming, in order,
`baselines/runner.py::preferred_estimator`, `harmonize/bridge.py::bridge_frame`,
`harmonize/naics.py::assert_113310_survives_the_window`,
`ingest/qcew.py::_check_dash_rows_carry_no_establishments`, `ingest/qcew.py::probe_slice_boundary`,
`ingest/qcew_size.py::read_by_size_zip`, `reconcile/scaling.py::Bounds.__post_init__`,
`store.py::assert_no_secret`, `validate/regimes.py::select_targets` — the nine functions this
plan converts, and no other.
Observed with `PENDING` as written above: `1 passed in 0.10s`.

- [x] **Step 3: Add the three classes and widen two docstrings in `errors.py`**

> Deviation (final review): `34a4f4f` rewrote parts of these docstrings (see the note under
> Status).

Replace `SchemaMismatchError` (`:57-58`):

```python
class SchemaMismatchError(LoggingEmploymentError):
    """A file's columns, or a row bound for a persisted table, do not match the declared schema.

    The parsers raise it for a fetched file whose columns are not the ones they declare;
    `ingest/qcew_size.py::read_by_size_zip` for a by-size archive that does not hold exactly one
    CSV, so there is no file to read the columns of; and `harmonize/bridge.py::bridge_frame` for a
    §8.6 bridge row missing a declared field, which Polars would otherwise persist as a null
    `verification_status` or `uncertainty_treatment` -- the one thing §8.6 asks a bridge row to
    say. Each carries the offending value: the columns, the members, or the fields (`D-140`).
    """
```

In `InfeasibleResidualError`'s docstring (`:119-126`), replace its last sentence so the paragraph
reads:

```python
    `reconcile.integerize` raises it for §12.6's three integer refusals (`D-119`): a lower bound
    above its cap, lower bounds summing past the total, and caps summing short of it. Each leaves no
    integer allocation inside the bounds that sums to the total (`D-139`). `reconcile.scaling.Bounds`
    raises it one layer earlier, as it is built from `deterministic_bounds`, for a cell whose lower
    bound sits above its upper bound (`D-096`): the same "lower above its cap" shape, refused before
    either clipping site can settle it in the cap's favour. `solve-bounds` refuses an infeasible
    component before it writes a bound, so on a bounds file it wrote the shape is unreachable; it is
    named all the same because the bounds are read off a file, and a file is the run's own state
    (`D-140`).
    """
```

Append at the end of the file, after `StoredObjectMismatchError`:

```python


class ClassificationContinuityError(LoggingEmploymentError):
    """113310 does not survive the D1 window's two NAICS vintages unchanged (§3.1, `D-102`).

    `harmonize/naics.py::assert_113310_survives_the_window` raises it for any of its four premises:
    a window vintage missing from the vendored crosswalk, a title other than Logging, a non-empty
    structure-file change indicator, or a 2017 -> 2022 link that is not one-to-one. §3.1: "The ETL
    MUST verify the 113310 mapping mechanically", and `build.build_harmonized` runs the check
    before it writes a table, so a re-vendored crosswalk halts the build by name. Distinct from
    `UnsupportedReferenceYearError`, which refuses a YEAR outside the vintages this package
    handles; this refuses the CROSSWALK for the years it does.
    """


class FallbackExhaustedError(LoggingEmploymentError):
    """No rung of §10.8's fallback hierarchy produced an estimate, so nothing can be preferred.

    `baselines/runner.py::preferred_estimator` raises it. Unreachable on D1 -- §10.2's inputs are
    complete on every suppressed cell, so rung 4 always produces estimates -- but reachable from a
    §13 mask that empties every month's missing set, which is why it is a raise rather than a
    sentinel: a `baseline_manifest.json` recording `preferred_estimator: null` would read as a
    considered choice, and `validate/scoreboard.py` ranks over what ran.
    """


class SecretInPayloadError(LoggingEmploymentError):
    """A credential's value reached bytes bound for an artifact (§7.2, D3).

    `store.assert_no_secret` raises it before a `source_snapshot` row is recorded, scanning the
    payload for every non-empty value of `config.SECRET_ENV_VARS`. It is the last of three guards:
    `fetching._without_credentials` strips the `key` parameter before a response is recorded, and
    `config.resolved_dict` keeps only an env var's NAME. This one catches a branch that forgot
    either. Alone among the classes here it carries no offending value, because the value is the
    secret.
    """
```

- [x] **Step 4: State the rule in root `CLAUDE.md`**

> Deviation (final review): `34a4f4f` adds that `SecretInPayloadError` carries no value.

Replace the bullet at `CLAUDE.md:177-179`:

```markdown
- **Fail closed with a named error.** Everything in `errors.py` subclasses `LoggingEmploymentError`
  and carries the offending value. Some classes and guards are *deliberately unraised/uncalled* and
  say so (`NoHarvestFactorError`, both guards in `harmonize/concepts.py`): later stages own them.
```

with:

```markdown
- **Fail closed with a named error.** Everything in `errors.py` subclasses `LoggingEmploymentError`
  and carries the offending value. WHICH refusals (ruled 2026-09-28, `D-140`): any refusal data can
  trigger -- source bytes, fetched metadata, staged tables, or the run's own state (§18.3) -- raises
  a named `errors.py` error. A bare `ValueError` is for a caller misusing the API, and it stays in
  pydantic validators, which turn only `ValueError` and `AssertionError` into a `ValidationError`.
  `tests/unit/test_fail_closed_inventory.py` lists every bare `ValueError` under `src/` with the
  reason it is a caller's mistake, and refuses a new one until it is classified. Some classes and
  guards are *deliberately unraised/uncalled* and say so (`NoHarvestFactorError`, both guards in
  `harmonize/concepts.py`): later stages own them.
```

`AGENTS.md` is a symlink to this file; it needs nothing.

- [x] **Step 5: Run the gates**

Run: `uv run pytest tests/unit/test_fail_closed_inventory.py -q`
Observed: `1 passed in 0.10s`

Run: `uv run ruff format src tests && uv run ruff check src tests && uv run interrogate src`
Observed: format and check clean, `interrogate` at 100%.

- [x] **Step 6: Commit**

```bash
git add src/logging_employment/errors.py CLAUDE.md tests/unit/test_fail_closed_inventory.py
git commit -m "feat(errors): state the fail-closed rule and add the three classes it needs (D-140)"
```

---

### Task 2: The three ingest sites

**Files:**
- Modify: `src/logging_employment/ingest/qcew.py:30` (import), `:89-90` (`probe_slice_boundary`),
  `:194-195` (`_check_dash_rows_carry_no_establishments`)
- Modify: `src/logging_employment/ingest/qcew_size.py:16-20` (import), `:58-59` (`read_by_size_zip`)
- Modify: `src/logging_employment/ingest/CLAUDE.md:84-86` (the fail-closed table)
- Test: `tests/unit/test_qcew_routes.py:16-18` (import), after `:44` (new test), `:229-232`
  (a docstring); `tests/unit/test_qcew_parser.py:11`, `:94-95`; `tests/unit/test_qcew_size.py:15`,
  before `:30` (new test)
- Modify: `tests/unit/test_fail_closed_inventory.py` (three `PENDING` lines)

**Interfaces:**
- Consumes: `errors.SourceFetchError`, `errors.ConceptViolationError`, `errors.SchemaMismatchError`
  (all existing); `PENDING` from Task 1.
- Produces: nothing new. `probe_slice_boundary` raises `SourceFetchError`,
  `_check_dash_rows_carry_no_establishments` raises `ConceptViolationError`, `read_by_size_zip`
  raises `SchemaMismatchError`, messages unchanged.

- [x] **Step 1: Drain `PENDING` of this task's three sites**

Delete these three lines from `PENDING` in `tests/unit/test_fail_closed_inventory.py`:

```python
    "ingest/qcew.py::_check_dash_rows_carry_no_establishments": "Task 2",
    "ingest/qcew.py::probe_slice_boundary": "Task 2",
    "ingest/qcew_size.py::read_by_size_zip": "Task 2",
```

- [x] **Step 2: Write the failing tests**

In `tests/unit/test_qcew_routes.py`, add the import and a test after
`test_boundary_probe_returns_the_earliest_year_that_answers`:

```python
from logging_employment import constants
from logging_employment.errors import SourceFetchError
from logging_employment.ingest import qcew
```

```python
def test_a_probe_no_candidate_year_answers_fails_closed_by_name() -> None:
    """Every candidate 404s, so no boundary exists to route on. `SourceFetchError`, the class
    `fetching.fetch_source` raises for an undeclared non-200 (R-S5P-4): the route answered
    nothing `fetch` could record, and a bare `ValueError` read as a caller's mistake (`D-140`)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"")

    with pytest.raises(SourceFetchError, match="served no candidate year for industry 113310"):
        qcew.probe_slice_boundary(_fetcher(handler), "113310", range(2010, 2020))
```

and in `test_every_window_year_still_routes_to_the_slice_endpoint`'s docstring (`:231-232`)
change "and raise ValueError -- which is why" to "and raise `SourceFetchError` -- which is why".

In `tests/unit/test_qcew_parser.py`, widen the import (`:11`) and change the existing test
(`:94-95`):

```python
from logging_employment.errors import ConceptViolationError, UnknownDisclosureCodeError
```

```python
def test_a_dash_row_with_establishments_halts_rather_than_claiming_a_true_zero() -> None:
    """The true-zero reading is a §2.2 concept ("Meaning of a QCEW zero"), and a '-' row carrying
    establishments crosses it: `ConceptViolationError`, not a bare `ValueError` (`D-140`)."""
    with pytest.raises(ConceptViolationError, match="qtrly_estabs"):
```

(the `qcew.parse_qcew_monthly(_row(disclosure_code="-", qtrly_estabs="7", ...))` body below it is
unchanged).

In `tests/unit/test_qcew_size.py`, widen the import (`:15`) and add a test before
`test_the_assertion_passes_on_the_real_file` (`io`, `re` and `zipfile` are already imported there):

```python
from logging_employment.errors import (
    MissingCrossTabulationError,
    SchemaMismatchError,
    UnknownSizeCodeError,
)
```

```python
@pytest.mark.parametrize("members", [[], ["a.csv", "b.csv"]], ids=["none", "two"])
def test_an_archive_without_exactly_one_csv_fails_closed_by_name(members: list[str]) -> None:
    """There is no file to read the columns of, so the schema cannot be matched:
    `SchemaMismatchError`, naming the members found (`D-140`)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for member in members:
            archive.writestr(member, "x\n1\n")
    with pytest.raises(SchemaMismatchError, match=re.escape(f"found {members}")):
        qcew_size.read_by_size_zip(buffer.getvalue())
```

The `re.escape` is load-bearing. Written first as
`match=f"expected one CSV member, found {members}"`, the two cases passed the type check and then
failed on `Regex pattern did not match`: a list's repr contains `[` and `]`, which are a character
class to `re`, so `found []` matches nothing and `found ['a.csv', 'b.csv']` matches one character.
Observed, then corrected.

- [x] **Step 3: Run the tests to verify they fail**

> Deviation (environment): `5 failed, 43 passed`, the plan's five failures. The skip the plan
> saw is not the findings-document monitor: it is `test_qcew_routes.py:289`'s `pytest.skip` in
> `test_the_member_selector_is_unique_in_the_real_archive_not_just_the_fixture`, which runs here
> because its archive under `data/raw/audit/` exists.

Run: `uv run pytest tests/unit/test_fail_closed_inventory.py tests/unit/test_qcew_routes.py tests/unit/test_qcew_parser.py tests/unit/test_qcew_size.py -q`
Observed: `5 failed, 42 passed, 1 skipped in 0.60s` — the inventory test naming the three sites,
the probe test, the dash-row test, and both archive cases (each a `ValueError` where a named
class was expected). The skip is `test_qcew_routes.py`'s findings-document monitor, which skips
without `specs/findings/`' extract; it is not this task's.

- [x] **Step 4: Convert the three sites, and the table**

`src/logging_employment/ingest/qcew.py`, the import (`:30`):

```python
from ..errors import ConceptViolationError, SourceFetchError, UnknownDisclosureCodeError
```

`probe_slice_boundary` (`:89-90`):

```python
    if not served:
        raise SourceFetchError(f"the slice route served no candidate year for industry {industry}")
    return served[0]
```

`_check_dash_rows_carry_no_establishments` (`:194-195`):

```python
    if offending.height:
        raise ConceptViolationError(
            f"{offending.height} row(s) carry disclosure_code '-' with qtrly_estabs > 0; the "
            "true-zero rule rests on those two never co-occurring, so this run halts rather than "
            "guessing which reading is right"
        )
```

`src/logging_employment/ingest/qcew_size.py`, the import (`:16-20`) and `read_by_size_zip`
(`:58-59`):

```python
from ..errors import (
    MissingCrossTabulationError,
    SchemaMismatchError,
    UnknownDisclosureCodeError,
    UnknownSizeCodeError,
)
```

```python
        if len(members) != 1:
            raise SchemaMismatchError(f"expected one CSV member, found {members}")
```

`src/logging_employment/ingest/CLAUDE.md`, the last three rows of the fail-closed table
(`:84-86`) become four:

```markdown
| `disclosure_code == "-"` with `qtrly_estabs > 0` (breaks the true-zero premise, a §2.2 concept) | `ConceptViolationError` | `qcew.py::_check_dash_rows_carry_no_establishments` |
| by-size zip does not hold exactly one CSV | `SchemaMismatchError` | `qcew_size.py::read_by_size_zip` |
| boundary probe served no candidate year (an undeclared non-200, as `fetch`'s) | `SourceFetchError` | `qcew.py::probe_slice_boundary` |
| bulk-zip industry substring does not narrow to one member (`11331` hits `111331 Apple orchards`) | `ValueError`, dead code: `D-049`'s delete-or-fix decides it, `D-140` left it | `qcew.py::read_bulk_zip` |
```

The sentence at `ingest/CLAUDE.md:44`, "`probe_slice_boundary` would raise", stays true.

- [x] **Step 5: Run the tests to verify they pass**

> Deviation (environment): `48 passed`, for the same reason.

Run: `uv run pytest tests/unit/test_fail_closed_inventory.py tests/unit/test_qcew_routes.py tests/unit/test_qcew_parser.py tests/unit/test_qcew_size.py -q`
Observed: `47 passed, 1 skipped in 0.52s`

Run: `uv run ruff format src tests && uv run ruff check src tests && uv run interrogate src`
Observed: clean, 100%.

- [x] **Step 6: Commit**

```bash
git add src/logging_employment/ingest tests/unit/test_qcew_routes.py tests/unit/test_qcew_parser.py tests/unit/test_qcew_size.py tests/unit/test_fail_closed_inventory.py
git commit -m "fix(ingest): the probe, the dash-row check and the by-size reader raise named errors (D-140)"
```

---

### Task 3: The two harmonize sites

**Files:**
- Modify: `src/logging_employment/harmonize/naics.py:23` (import), `:136` (docstring), `:152`,
  `:157`, `:162`, `:168` (the four raises in `assert_113310_survives_the_window`)
- Modify: `src/logging_employment/harmonize/bridge.py:9` (import), `:20` (docstring), `:27`
- Test: `tests/unit/test_harmonize.py:13` (import), `:82`, `:181`, `:192`, `:198`, and one new
  test after `test_a_flagged_change_indicator_would_fail_the_window_assertion`;
  `tests/integration/test_build_harmonized.py:16` (import), `:283`
- Modify: `tests/unit/test_fail_closed_inventory.py` (two `PENDING` lines)

**Interfaces:**
- Consumes: `errors.ClassificationContinuityError` (Task 1), `errors.SchemaMismatchError`.
- Produces: `assert_113310_survives_the_window` raises `ClassificationContinuityError` on all four
  premises; `bridge_frame` raises `SchemaMismatchError`; messages unchanged.

- [x] **Step 1: Drain `PENDING` of this task's two sites**

Delete from `PENDING`:

```python
    "harmonize/bridge.py::bridge_frame": "Task 3",
    "harmonize/naics.py::assert_113310_survives_the_window": "Task 3",
```

- [x] **Step 2: Write the failing tests**

`tests/unit/test_harmonize.py`, the import (`:13`):

```python
from logging_employment.errors import (
    ClassificationContinuityError,
    ConceptViolationError,
    SchemaMismatchError,
    UnsupportedReferenceYearError,
)
```

Change the class in four existing `pytest.raises` calls, keeping every `match=`:

```python
    with pytest.raises(SchemaMismatchError, match="verification_status"):
        bridge.bridge_frame([incomplete])
```

```python
    with pytest.raises(ClassificationContinuityError, match="one-to-one"):
        naics.assert_113310_survives_the_window(frame=doctored)
```

```python
    with pytest.raises(ClassificationContinuityError, match="title is not stable"):
        naics.assert_113310_survives_the_window(frame=doctored)
```

```python
    with pytest.raises(ClassificationContinuityError, match="change_indicator"):
        naics.assert_113310_survives_the_window(frame=doctored)
```

Then add, after that last test, the premise no test reached at `46ef072`:

```python
def test_a_crosswalk_missing_a_window_vintage_would_fail_the_window_assertion() -> None:
    """The fourth premise, which no test reached: a crosswalk vendored without one of the two
    window vintages has nothing to pair, and names the vintages it found (`D-140`)."""
    doctored = naics.crosswalk_113310().filter(pl.col("vintage") == "2017")
    with pytest.raises(ClassificationContinuityError, match=r"found \['2017'\]"):
        naics.assert_113310_survives_the_window(frame=doctored)
```

`tests/integration/test_build_harmonized.py`, the import (`:16`) and the class at `:283`. This
test runs without `data/`: its `frozen_raw` fixture copies `tests/fixtures/`.

```python
from logging_employment.errors import (
    ClassificationContinuityError,
    ConceptViolationError,
    UnknownDisclosureRegimeError,
)
```

```python
    out = tmp_path / "doctored"
    with pytest.raises(ClassificationContinuityError, match="one-to-one"):
        build_harmonized(_cfg(), raw_root=frozen_raw, out_root=out)
    assert not out.exists()
```

- [x] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_fail_closed_inventory.py tests/unit/test_harmonize.py tests/integration/test_build_harmonized.py -q`
Observed: `7 failed, 38 passed in 1.38s` — the inventory test, the bridge test, the four
crosswalk tests (three changed, one new) and the build-halts test.

- [x] **Step 4: Convert the two sites**

> Deviation (final review): `34a4f4f` reflowed `bridge_frame`'s docstring; the words are the
> ones shown.

`src/logging_employment/harmonize/naics.py`, the import (`:23`):

```python
from ..errors import ClassificationContinuityError, UnsupportedReferenceYearError
```

Add one line to `assert_113310_survives_the_window`'s docstring, after its first line and a
blank line (before "Mechanical rather than string-continuity"):

```python
    """Raise unless 113310 is present, titled Logging, and unchanged across both vintages.

    Every refusal is `ClassificationContinuityError`, naming what the crosswalk holds (`D-140`).

    Mechanical rather than string-continuity: §3.1 states outright that apparent code-string
```

and its body becomes:

```python
    frame = crosswalk_113310() if frame is None else frame
    vintages = set(frame["vintage"].to_list())
    if vintages != {"2017", "2022"}:
        raise ClassificationContinuityError(
            f"expected both window vintages in the crosswalk, found {sorted(vintages)}"
        )
    titles = set(frame["title"].to_list())
    if titles != {"Logging"}:
        raise ClassificationContinuityError(
            f"113310's title is not stable across vintages: {sorted(titles)}"
        )
    changed = frame.filter(
        pl.col("change_indicator").is_not_null() & (pl.col("change_indicator") != "")
    )
    if changed.height:
        raise ClassificationContinuityError(
            f"113310 carries a non-empty change_indicator in {changed['vintage'].to_list()}; the "
            "structure file marks it as changed from the prior vintage"
        )
    link = frame.filter(pl.col("vintage") == "2017")["link_type_to_next"].to_list()
    if link != ["1:1"]:
        raise ClassificationContinuityError(
            f"the 2017->2022 concordance does not pair 113310 one-to-one: {link}"
        )
```

`src/logging_employment/harmonize/bridge.py`, the import (`:9`), the docstring sentence (`:20`)
and the raise (`:27`):

```python
from ..contracts import BRIDGE_SCHEMA
from ..errors import SchemaMismatchError
```

```python
    A row missing a declared field raises `SchemaMismatchError` (`D-140`). Polars would render
    the absent field as null, and a
    bridge carrying a null `verification_status` or `uncertainty_treatment` states nothing about
```

```python
        if missing:
            raise SchemaMismatchError(
                f"bridge row {index} is missing the declared field(s) {missing}"
            )
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_fail_closed_inventory.py tests/unit/test_harmonize.py tests/integration/test_build_harmonized.py -q`
Observed: `45 passed in 1.14s`

Run: `uv run ruff format src tests && uv run ruff check src tests && uv run interrogate src`
Observed: clean, 100%.

- [x] **Step 6: Commit**

```bash
git add src/logging_employment/harmonize tests/unit/test_harmonize.py tests/integration/test_build_harmonized.py tests/unit/test_fail_closed_inventory.py
git commit -m "fix(harmonize): the crosswalk check and the bridge builder raise named errors (D-140)"
```

---

### Task 4: The runner's preference and the regime selector

**Files:**
- Modify: `src/logging_employment/baselines/runner.py:46-51` (import), `:651-653` (docstring),
  `:663` (`preferred_estimator`)
- Modify: `src/logging_employment/validate/regimes.py:339` (`select_targets`)
- Test: `tests/unit/test_baselines_runner.py:20` (import), new test after `:95`;
  `tests/unit/test_validate_regimes.py:11` (import), new test after `:72`
- Modify: `tests/unit/test_fail_closed_inventory.py` (two `PENDING` lines)

**Interfaces:**
- Consumes: `errors.FallbackExhaustedError` (Task 1); `errors.ConceptViolationError`, already
  imported by `regimes.py`; the `make_monthly` fixture (`tests/unit/conftest.py`) and
  `appendix_a_config` (`tests/conftest.py`).
- Produces: `preferred_estimator` raises `FallbackExhaustedError`; `select_targets` raises
  `ConceptViolationError` for a single-month regime that would black out a state's history;
  messages unchanged.

- [x] **Step 1: Drain `PENDING` of this task's two sites**

Delete from `PENDING`:

```python
    "baselines/runner.py::preferred_estimator": "Task 4",
    "validate/regimes.py::select_targets": "Task 4",
```

- [x] **Step 2: Write the failing tests**

> Deviation (final review): `34a4f4f` corrected the new test's docstring: the refusal is
> reachable only through `run-baselines`, not from a §13 mask.

`tests/unit/test_baselines_runner.py`, the import (`:20`) and a test after
`test_the_preferred_estimator_follows_the_fallback_order`. `preferred_estimator` reads two
columns, so the frame carries two:

```python
from logging_employment.errors import (
    ConceptViolationError,
    FallbackExhaustedError,
    UniverseClosureError,
)
```

```python
def test_a_run_where_no_rung_produced_an_estimate_is_refused_by_name() -> None:
    """Reachable from a §13 mask that empties every month's missing set, never from D1. A raise
    and not a sentinel, because `preferred_estimator: null` in `baseline_manifest.json` would
    read as a considered choice: `FallbackExhaustedError` (`D-140`)."""
    declined = pl.DataFrame(
        {
            "estimator_id": ["cbp_intensity", "equal_allocation"],
            "reconciliation_status": ["declined", "declined"],
        }
    )
    with pytest.raises(FallbackExhaustedError, match="no estimator in §10.8's fallback hierarchy"):
        preferred_estimator(declined)
```

`tests/unit/test_validate_regimes.py`, the import (`:11`) and a test after
`test_a_single_month_regime_leaves_lookback_history_unmasked`. The module's other selector tests
are `requires_staged`; this one builds twelve observed months for one state with the
`make_monthly` factory, so it runs in a bare checkout. `small_cell_biased` draws up to
`replicates_per_regime` (20) targets from the 12 eligible cells, takes all 12, and leaves 0
months of lookback against `minimum_unmasked_lookback_months` (6, `config.yaml`):

```python
from logging_employment.contracts import HOLDOUT_REGIMES, HarmonizedData
from logging_employment.errors import ConceptViolationError
from logging_employment.validate.regimes import REGIME_SPECS, select_targets
```

```python
def test_a_single_month_regime_that_would_black_out_a_states_history_is_refused_by_name(
    make_monthly, appendix_a_config
):
    """Twelve observed months for one state and a draw of twenty (`replicates_per_regime`), so
    every month is a target and the six-month `minimum_unmasked_lookback_months` cannot hold.
    The line between a single-month regime and a blackout is a §13.3 concept, so the refusal is
    `ConceptViolationError`, not a bare `ValueError` (`D-140`). Hermetic: no `data/`."""
    monthly = make_monthly(
        *(
            {
                "reference_month": f"2023-{month:02d}",
                "employment_raw": str(100 + 7 * month),
                "employment_value": 100 + 7 * month,
            }
            for month in range(1, 13)
        )
    )
    with pytest.raises(ConceptViolationError, match="leaving fewer than 6 lookback months"):
        select_targets("small_cell_biased", monthly, seed=1024, config=appendix_a_config)
```

- [x] **Step 3: Run the tests to verify they fail**

> Deviation (environment): `3 failed, 37 passed`, the plan's three failures: with `data/`
> present the ten `requires_staged` tests run and pass.

Run: `uv run pytest tests/unit/test_fail_closed_inventory.py tests/unit/test_baselines_runner.py tests/unit/test_validate_regimes.py -q`
Observed: `3 failed, 27 passed, 10 skipped in 0.87s` — the inventory test and the two new tests.
The 10 skips are `test_validate_regimes.py`'s `requires_staged` tests, absent `data/`.

- [x] **Step 4: Convert the two sites**

> Deviation (final review): `34a4f4f` corrected `preferred_estimator`'s docstring the same way.

`src/logging_employment/baselines/runner.py`, the import block (`:46-51`), the docstring
(`:651-653`) and the raise (`:663`):

```python
from ..errors import (
    BoundViolationError,
    ConceptViolationError,
    FallbackExhaustedError,
    InfeasibleResidualError,
    WeightDomainError,
)
```

```python
def preferred_estimator(results: pl.DataFrame) -> str:
    """§10.8's ordering, applied to whichever estimators actually produced estimates.

    Raises `FallbackExhaustedError` when no rung ran (`D-140`). Unreachable on D1 -- §10.2's
    inputs are complete on every suppressed cell, so rung 4 always produces estimates -- but
    reachable from a Stage 4 mask that empties every month's missing set, which is why it raises
    rather than returning a sentinel.
    """
```

```python
    for estimator_id in FALLBACK_ORDER:
        if estimator_id in ran:
            return estimator_id
    raise FallbackExhaustedError("no estimator in §10.8's fallback hierarchy produced any estimate")
```

`src/logging_employment/validate/regimes.py` (`:339`), the module's only bare `ValueError`;
`ConceptViolationError` is already imported at `:23`:

```python
            if total_months - count < floor:
                raise ConceptViolationError(
                    f"{regime} would mask {count} of {total_months} months for state {state}, "
                    f"leaving fewer than {floor} lookback months. A single-month regime must not "
                    "black out a state's own history — that is what the blackout regimes are for."
                )
```

- [x] **Step 5: Run the tests to verify they pass**

> Deviation (environment): `40 passed`, for the same reason.

Run: `uv run pytest tests/unit/test_fail_closed_inventory.py tests/unit/test_baselines_runner.py tests/unit/test_validate_regimes.py -q`
Observed: `30 passed, 10 skipped in 0.83s`

Run: `uv run ruff format src tests && uv run ruff check src tests && uv run interrogate src`
Observed: clean, 100%.

- [x] **Step 6: Commit**

```bash
git add src/logging_employment/baselines/runner.py src/logging_employment/validate/regimes.py tests/unit/test_baselines_runner.py tests/unit/test_validate_regimes.py tests/unit/test_fail_closed_inventory.py
git commit -m "fix(baselines,validate): preferred_estimator and select_targets raise named errors (D-140)"
```

---

### Task 5: The bounds and the secret guard

**Files:**
- Modify: `src/logging_employment/reconcile/scaling.py:48-49` (docstring), `:57` (`Bounds.__post_init__`)
- Modify: `src/logging_employment/store.py:11` (import), `:28-31` (`assert_no_secret`)
- Modify: `src/logging_employment/reconcile/CLAUDE.md:137-140`
- Modify: `CLAUDE.md:180-181` (the secrets bullet)
- Test: `tests/unit/test_scaling.py:257-261`; `tests/unit/test_store.py:11` (import), `:51-53`
- Modify: `tests/unit/test_fail_closed_inventory.py` (two `PENDING` lines)

**Interfaces:**
- Consumes: `errors.InfeasibleResidualError` (already imported by `scaling.py` and its test);
  `errors.SecretInPayloadError` (Task 1).
- Produces: `Bounds(...)` raises `InfeasibleResidualError` for an inverted pair; `assert_no_secret`
  raises `SecretInPayloadError`; messages unchanged.

- [x] **Step 1: Drain `PENDING` of this task's two sites**

Delete from `PENDING`:

```python
    "reconcile/scaling.py::Bounds.__post_init__": "Task 5",
    "store.py::assert_no_secret": "Task 5",
```

- [x] **Step 2: Write the failing tests**

`tests/unit/test_scaling.py`, in `test_an_inverted_bound_pair_is_refused_when_the_bounds_are_built`
(`:257-261`): extend the docstring's last sentence and change the class, keeping the pattern:

```python
    (`scale_into_bounds`, `clipped_sum`) can receive one. The same name as `integerize`'s refusal
    of the same shape, `InfeasibleResidualError` (`D-140`)."""
    with pytest.raises(
        InfeasibleResidualError, match=r"'01' has lower bound 90\.0 above its upper bound 1\.0"
    ):
        Bounds(
            lower={"01": 90.0, "02": 0.0, "04": 0.0},
            upper={"01": 1.0, "02": 100.0, "04": 100.0},
        )
```

`tests/unit/test_store.py`, the import (`:11`) and the test (`:51-53`), which now also pins that
the secret is absent from the message:

```python
from logging_employment.errors import SecretInPayloadError, StoredObjectMismatchError
```

```python
def test_assert_no_secret_raises_on_a_leaked_value() -> None:
    """`SecretInPayloadError`, and the message carries no value: the value is the secret."""
    with pytest.raises(SecretInPayloadError, match="secret") as caught:
        assert_no_secret("url=...&key=abc123", ["abc123"])
    assert "abc123" not in str(caught.value)
```

- [x] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_fail_closed_inventory.py tests/unit/test_scaling.py tests/unit/test_store.py -q`
Observed: `3 failed, 19 passed in 0.20s` — the inventory test and the two changed tests.

- [x] **Step 4: Convert the two sites, and the two docs that describe them**

> Deviation (final review): `34a4f4f` reflowed the secrets bullet; the words are the ones shown.

`src/logging_employment/reconcile/scaling.py`, `Bounds.__post_init__`'s docstring (`:48-49`)
and raise (`:57`):

```python
        it whenever the sums stay feasible. `integerize` already refuses this shape by name; this is
        the same refusal one layer earlier, under the same name, `InfeasibleResidualError`
        (`D-140`). Equality is not an inversion: a cell pinned to a single value is legitimate, and
        the strict `>` admits it.
        """
```

```python
        if inverted:
            cell = inverted[0]
            raise InfeasibleResidualError(
                f"cell {cell!r} has lower bound {self.lower[cell]} above its upper bound "
                f"{self.upper[cell]} ({len(inverted)} inverted cell(s) in all); clamping to the cap "
                "would silently return a value below the lower bound the caller declared"
            )
```

`src/logging_employment/store.py`, the import (`:11`) and `assert_no_secret` (`:27-31`):

```python
from .errors import SecretInPayloadError, StoredObjectMismatchError
```

```python
def assert_no_secret(payload: str, secrets: Sequence[str | None]) -> None:
    """Raise `SecretInPayloadError` if any non-empty secret value appears in the payload (§7.2, D3).

    The message names no value, because the value is the secret (`D-140`).
    """
    for secret in secrets:
        if secret and secret in payload:
            raise SecretInPayloadError(
                "a secret value reached a manifest payload; refusing to write it"
            )
```

`src/logging_employment/reconcile/CLAUDE.md` (`:137-140`), extend the `integerize` bullet's last
sentence:

```markdown
  All three refusals — a lower bound above its cap, lower bounds summing past the total, caps
  summing short of it — raise `InfeasibleResidualError` (`D-119`), never a bare `ValueError`, and
  they are exactly the infeasible cases: seats past the total hand units back from cells above
  their lower bound, in the reverse of placement order (`D-139`). `scaling.Bounds` refuses the
  first shape one layer earlier, as it is built, under the same name (`D-096`, `D-140`).
```

Root `CLAUDE.md`, the secrets bullet (`:180-181` at `46ef072`; six lines lower after Task 1):

```markdown
- **Secrets never reach an artifact** (D3, §7.2): `store.assert_no_secret` refuses, with
  `SecretInPayloadError`, a snapshot row containing one of `config.SECRET_ENV_VARS`,
  `fetching._without_credentials` strips
  the `key` param before recording, and `resolved_dict` keeps only the env var's *name*.
```

- [x] **Step 5: Run the tests to verify they pass**

`tests/unit/test_fetching.py` is included because its CBP end-to-end test is the caller-side
witness that a stripped key never reaches `assert_no_secret`:

Run: `uv run pytest tests/unit/test_fail_closed_inventory.py tests/unit/test_scaling.py tests/unit/test_store.py tests/unit/test_fetching.py -q`
Observed: `59 passed in 0.54s`

Run: `uv run ruff format src tests && uv run ruff check src tests && uv run interrogate src`
Observed: clean, 100%.

- [x] **Step 6: Commit**

```bash
git add src/logging_employment/reconcile/scaling.py src/logging_employment/store.py src/logging_employment/reconcile/CLAUDE.md CLAUDE.md tests/unit/test_scaling.py tests/unit/test_store.py tests/unit/test_fail_closed_inventory.py
git commit -m "fix(reconcile,store): Bounds and assert_no_secret raise named errors (D-140)"
```

---

### Task 6: The sweep is complete

**Files:**
- Modify: `tests/unit/test_fail_closed_inventory.py` (delete `PENDING`)

**Interfaces:**
- Consumes: the empty `PENDING` left by Tasks 2–5.
- Produces: the inventory test in its final form, `KEEP` alone describing the tree.

- [x] **Step 1: Delete `PENDING`**

> Deviation (final review): `34a4f4f` made the body count raises per key and added a third,
> miscounted check (see the note under Status).

`PENDING` is now `{}`. Delete the dict and the three comment lines above it, and simplify the
test body to its Task 1 Step 2 form:

```python
    found = _bare_value_error_sites()
    unclassified = sorted(found - set(KEEP))
    stale = sorted(set(KEEP) - found)
```

- [x] **Step 2: Run the inventory test**

Run: `uv run pytest tests/unit/test_fail_closed_inventory.py -q`
Observed: `1 passed in 0.09s`

- [x] **Step 3: Run the whole hermetic tier and the gates**

> Deviation (environment): `1810 passed, 49 deselected in 149.44s`, against this checkout's
> base of `1803 passed, 49 deselected` with `data/` present; the +7 is the check. `rg` summed to 25.

Run: `uv run pytest -m "not slow and not network" -q`
Observed, in the clone without `data/`: `1765 passed, 45 skipped, 49 deselected in 124.26s
(0:02:04)`, against the base's `1758 passed, 45 skipped, 49 deselected in 127.23s`: seven added,
none skipped, none deselected. With
`data/` present the skipped count falls and the passed count rises by the same number, exactly as
the base does (root `CLAUDE.md`, Gotchas); the delta of seven is the check.

Run: `uv run ruff format src tests && uv run ruff check src tests && uv run interrogate src`
Observed: clean, 100%.

Run: `rg -c 'raise ValueError' src/ | awk -F: '{s+=$2} END{print s}'`
Observed: `25` (37 at `46ef072`).

- [x] **Step 4: Commit**

> Deviation (mechanical): committed as `1f9350f` with its trailer folded into the subject; the
> controller amended the message alone, tree `565ddfa` unchanged, to `8ec3922`.

```bash
git add tests/unit/test_fail_closed_inventory.py
git commit -m "test: the fail-closed inventory holds only callers' mistakes (D-140)"
```

- [x] **Step 5: Plan Completion Protocol**

> Deviation (by assignment): run by the controller after the final review, as the writing-plans
> Plan Completion Protocol, in the retire commit.

Tick `D-140` in `specs/deferred_items.md` as `→ done in plan 17`, per the writing-plans skill's
Plan Completion Protocol. `D-049` and `D-058` are named by the inventory's `KEEP` reasons and are
not touched. Nothing in this plan changes normative spec text, so no roadmap or stage-log sentence
needs reconciling.

---

## Self-review

**Spec coverage against `D-140`'s Done-when.**
- "root `CLAUDE.md`'s fail-closed bullet states which refusals take a named error and which keep
  `ValueError`" → Task 1 Step 4.
- "every site the rule assigns a named error raises one with its test updated" → the nine NAME
  functions: `probe_slice_boundary`, `_check_dash_rows_carry_no_establishments`,
  `read_by_size_zip` (Task 2); `assert_113310_survives_the_window`, `bridge_frame` (Task 3);
  `preferred_estimator`, `select_targets` (Task 4); plus the DECIDE sites `Bounds` and
  `assert_no_secret` (Task 5, Decisions 3 and 5) and `classification.py` kept (Decision 4). Every
  site has a test that asked for the class before the site gave it.
- "`ingest/CLAUDE.md`'s fail-closed table matches" → Task 2 Step 4.
- "Re-count before starting; the inventory is dated" → Task 1's test, which counted 37 and refuses
  drift.

**Placeholder scan.** No TBD, no "similar to", every step carries its code, every run its observed
line.

**Type consistency.** The three class names are spelled identically in Task 1's `errors.py`,
each site's import, each test's import and each docstring; checked by the clone's `ruff check`
(F821 would have caught a misspelling) and by the tests passing.

---

## Execution Handoff

**Plan complete and saved to `specs/plans/17-fail-closed-named-errors.md`.**

**Recommended: `/clear` (or open a new session) and execute against the saved plan** — a fresh
session drops this planning conversation and lets execution run on the standard model default.
Two execution options, either session:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, two-stage review between tasks
   (REQUIRED SUB-SKILL: subagent-driven-development).
2. **Inline Execution** — execute the tasks in plan order in one session (REQUIRED SUB-SKILL:
   executing-plans).

Which approach?
