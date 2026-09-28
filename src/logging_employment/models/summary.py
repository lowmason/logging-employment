"""§7.11's `posterior_summary`, computed from reconciled joint draws and nothing else (§12.7).

ONE ROW PER STATE-TOTAL CELL THE PANEL PUBLISHES A ROW FOR: 4,716 on D1. An observed or true-zero
cell carries its published value in every posterior column, with `observed_or_imputed = 'observed'`
and `reconciliation_status = 'observed'`. INV-001 requires disclosed values to be preserved exactly,
and a posterior interval around a published number would imply uncertainty the source does not
have. A suppressed cell summarises its reconciled draws, with `observed_or_imputed = 'imputed'` and
`reconciliation_status = 'anchored_and_reconciled'`.

DETERMINISTIC AND POSTERIOR INTERVALS ARE DIFFERENT COLUMNS AND NEVER ONE ANOTHER (INV-008).
`deterministic_lower` / `deterministic_upper` are §9's `selected_lower` / `selected_upper` copied
through, with a null upper where no public fact bounds the cell. The `ci*` columns are equal-tailed
quantiles of the reconciled draws. `models/reconciliation.py::DrawCheck` has already checked that
every draw lies inside the deterministic interval, so each posterior interval nests inside it, and
nothing here widens, clips or relabels either one.

THREE COLUMNS ARE NULL, EACH FOR A DECLARED REASON:

* `model_sensitivity_low` / `_high`: §11.13's sensitivity variants are Stage 7's
  (`models/suppression_sensitivity.py` in its Produces). Stage 7's roadmap block is where "later
  stages may assume `model_sensitivity_low` and `model_sensitivity_high` for every release cell".
* `probability_thresholds_json`: §7.11 names the column, and no section of the spec states a
  threshold for it. Inventing one here would be the policy §21 leaves to its owner.

`source_vintage_set` is a JSON list. For an observed cell it is the cell's own `release_vintage`.
For an imputed cell it is every `release_vintage` among the training cells, because a
hierarchical fit informs each imputed cell from the whole panel.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from ..baselines.runner import state_total_bounds
from ..contracts import POSTERIOR_SUMMARY_SCHEMA, assert_declared_provenance, validate_frame
from ..errors import ConceptViolationError
from .data import PREDICTION_STATUS, TRAINING_STATUS, state_cell_ids
from .interfaces import MODEL_ID, MODEL_VERSION
from .reconciliation import ReconciledDraws, exact_column_means

PUBLISHED_STATUSES: tuple[str, ...] = (TRAINING_STATUS, "true_zero")
# (level, column stem): equal-tailed central intervals, §7.11's four.
LEVELS: tuple[tuple[float, str], ...] = (
    (0.50, "ci50"),
    (0.80, "ci80"),
    (0.90, "ci90"),
    (0.95, "ci95"),
)


def posterior_summary(
    draws: ReconciledDraws,
    monthly: pl.DataFrame,
    bounds: pl.DataFrame,
    *,
    run_id: str,
    constraint_set_hash: str,
) -> pl.DataFrame:
    """§7.11's table for every state cell of `monthly`, validated and provenance-checked.

    `bounds` is the run's `deterministic_bounds` frame. A state cell with a published row whose
    status is neither published nor suppressed is refused, as is an imputed cell with no suppressed
    row to hang it on. Either one would drop a cell from the release without a signal.
    """
    states = monthly.filter(pl.col("area_type") == "state")
    identifiers = state_cell_ids(states)
    deterministic = state_total_bounds(bounds)
    column_of = {cell: index for index, cell in enumerate(draws.cell_ids)}
    quantiles = [0.5]
    for level, _stem in LEVELS:
        quantiles += [(1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0]
    q = np.quantile(draws.values, quantiles, axis=0) if draws.values.size else None
    means = exact_column_means(draws.values) if draws.values.size else None
    training = sorted(
        str(v)
        for v in states.filter(pl.col("observation_status") == TRAINING_STATUS)["release_vintage"]
        .unique()
        .to_list()
    )
    rows: list[dict[str, object]] = []
    used: set[str] = set()
    for identifier, row in zip(identifiers, states.iter_rows(named=True), strict=True):
        status = row["observation_status"]
        base = {
            "cell_id": identifier,
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "run_id": run_id,
            "probability_thresholds_json": None,
            "deterministic_lower": deterministic.lower.get(identifier),
            "deterministic_upper": deterministic.upper.get(identifier),
            "model_sensitivity_low": None,
            "model_sensitivity_high": None,
            "constraint_set_hash": constraint_set_hash,
        }
        if status in PUBLISHED_STATUSES:
            value = float(row["employment_value"])
            interval = {
                f"{stem}_{end}": value for _level, stem in LEVELS for end in ("low", "high")
            }
            rows.append(
                base
                | interval
                | {
                    "posterior_mean": value,
                    "posterior_median": value,
                    "observed_or_imputed": "observed",
                    "reconciliation_status": "observed",
                    "source_vintage_set": json.dumps([str(row["release_vintage"])]),
                }
            )
        elif status == PREDICTION_STATUS:
            if identifier not in column_of:
                raise ConceptViolationError(
                    f"suppressed cell {identifier} has no reconciled draws; §7.11 would release "
                    "it without a posterior"
                )
            j = column_of[identifier]
            used.add(identifier)
            interval = {}
            for position, (_level, stem) in enumerate(LEVELS):
                interval[f"{stem}_low"] = float(q[1 + 2 * position, j])
                interval[f"{stem}_high"] = float(q[2 + 2 * position, j])
            rows.append(
                base
                | interval
                | {
                    "posterior_mean": float(means[j]),
                    "posterior_median": float(q[0, j]),
                    "observed_or_imputed": "imputed",
                    "reconciliation_status": "anchored_and_reconciled",
                    "source_vintage_set": json.dumps(training),
                }
            )
        else:
            raise ConceptViolationError(
                f"state cell {identifier} has observation_status {status!r}; §7.11 has no row "
                "shape for it"
            )
    stray = sorted(set(draws.cell_ids) - used)
    if stray:
        raise ConceptViolationError(
            f"{len(stray)} reconciled cell(s) match no suppressed row, first {stray[:5]}"
        )
    frame = pl.DataFrame(rows, schema=POSTERIOR_SUMMARY_SCHEMA)
    validate_frame(frame, POSTERIOR_SUMMARY_SCHEMA, "posterior_summary")
    assert_declared_provenance(frame)
    return frame
