"""Root `CLAUDE.md`'s fail-closed rule, held by a test rather than by a dated count (`D-140`).

A refusal that data can trigger -- source bytes, fetched metadata, staged tables, or the run's
own state (§18.3) -- raises a named `errors.py` error. A bare `ValueError` is for a caller misusing
the API, and stays in pydantic validators, which turn only `ValueError` and `AssertionError` into a
`ValidationError`. `D-119` converted four sites and `D-140` found ten more, each time by re-counting
`rg 'raise ValueError' src/` by hand. This test is that count, made live: every `raise ValueError`
under `src/` is named here with the reason it keeps a bare `ValueError`, and a new one is refused
until its author classifies it.

Sites are keyed by module and enclosing function, never by line, so an edit above a site does not
move it (`D-056`'s pointer refreshes are what a line-keyed inventory costs). Each key carries the
count of raises its function holds, so a raise added to a listed function changes that count and
is refused like a raise in a new one.

The AST walk matches the bare name `ValueError` at a `raise`, so it does not see an aliased name
(`VE = ValueError`), `builtins.ValueError`, an instance built earlier and raised by name, or a
`ValueError` subclass defined in `src/`; none is present at `8ec3922`. Nor does the count see a
raise that replaces another in the same function, which leaves it unchanged.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import logging_employment

SRC = Path(logging_employment.__file__).parent

# Every `raise ValueError` under `src/`, and why each is a caller's mistake rather than a refusal
# data can trigger. A site's key is `<module>::<qualified function>`; several raises in one
# function share one key, and its count is how many raises that function holds, so adding or
# removing one changes it.
KEEP: dict[str, tuple[int, str]] = {
    "build.py::build_harmonized": (
        1,
        "`allow_network=True` is a caller asking for what the build forbids",
    ),
    "classification.py::_section_31_fence": (
        3,
        "reads the spec file, repo content; called from no run path (`D-058` owns wiring it)",
    ),
    "classification.py::classification_memo": (1, "as `_section_31_fence`"),
    "config.py::ModelConfig._the_fitted_multiplier_is_present": (1, "pydantic validator"),
    "config.py::ProjectConfig._is_a_month": (1, "pydantic validator"),
    "config.py::ValidationConfig._refuse_a_random_mask_only_design": (1, "pydantic validator"),
    "constraints/rows.py::_check_relation": (6, "argument check on a row builder's own literals"),
    "constraints/rows.py::constraint": (5, "argument check on a row builder's own literals"),
    "disclosure/flags.py::build_flags": (
        1,
        "the bounds and cells passed in disagree: a caller's join",
    ),
    "fetching.py::fetch_source": (
        1,
        "an unknown `source` string from the caller; the CLI refuses first",
    ),
    "ingest/base.py::HttpFetcher.__post_init__": (1, "a fetcher built without a contact email"),
    "ingest/qcew.py::read_bulk_zip": (1, "dead code; `D-049`'s delete-or-fix decides it"),
    "reconcile/draws.py::reconcile_draws": (1, "a caller-supplied draw shape"),
    "reconcile/projection.py::_require_indicator_margins": (1, "a caller-supplied indicator shape"),
}


def _bare_value_error_counts() -> Counter[str]:
    """The `raise ValueError(...)` count of each `<module>::<qualified function>` under `src/`."""
    counts: Counter[str] = Counter()
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
            counts[f"{path.relative_to(SRC).as_posix()}::{'.'.join(reversed(scope))}"] += 1
    return counts


def test_every_bare_value_error_is_a_callers_mistake_and_says_so() -> None:
    """A new bare `ValueError` is refused until it is classified: named in `errors.py` if data can
    trigger it, or listed in `KEEP` with the reason it cannot. One added to a listed function is
    refused by the count it changes. A stale entry or count is refused too, so the list describes
    the tree."""
    found = _bare_value_error_counts()
    unclassified = sorted(set(found) - set(KEEP))
    stale = sorted(set(KEEP) - set(found))
    miscounted = sorted(
        f"{key}: {found[key]} found, {KEEP[key][0]} listed"
        for key in set(found) & set(KEEP)
        if found[key] != KEEP[key][0]
    )
    assert not unclassified, (
        f"`raise ValueError` at {unclassified} is not classified. If data can trigger it, raise "
        "a named `errors.py` error (root CLAUDE.md, fail closed); if only a caller can, add it "
        "to KEEP with the reason"
    )
    assert not stale, f"KEEP names sites that no longer raise a bare ValueError: {stale}"
    assert not miscounted, (
        f"`raise ValueError` counts differ from KEEP's at {miscounted}. Classify the new raise: if "
        "data can trigger it, raise a named `errors.py` error (root CLAUDE.md, fail closed); if "
        "only a caller can, or a raise was removed, correct the count and the reason in KEEP"
    )
