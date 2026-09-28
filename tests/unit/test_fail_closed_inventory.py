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
PENDING: dict[str, str] = {}


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
