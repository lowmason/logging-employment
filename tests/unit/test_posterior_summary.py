"""§7.11's `posterior_summary`, from reconciled draws (§12.7) and the run's deterministic bounds."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from logging_employment.baselines.runner import state_total_bounds
from logging_employment.contracts import (
    DETERMINISTIC_BOUNDS_SCHEMA,
    POSTERIOR_SUMMARY_SCHEMA,
    assert_declared_provenance,
)
from logging_employment.errors import ConceptViolationError
from logging_employment.models.data import state_cell_ids
from logging_employment.models.reconciliation import reconcile_fit
from logging_employment.models.summary import LEVELS, posterior_summary

SEED = sum(map(ord, "tests/posterior-summary"))
POSTERIOR_COLUMNS = ["posterior_mean", "posterior_median"] + [
    f"{stem}_{end}" for _level, stem in LEVELS for end in ("low", "high")
]


def _states(monthly: pl.DataFrame) -> pl.DataFrame:
    return monthly.filter(pl.col("area_type") == "state")


def _bounds_frame(monthly: pl.DataFrame, upper: dict[str, float]) -> pl.DataFrame:
    """A `deterministic_bounds` table of every state cell: published ones pinned, others [0, U]."""
    states = _states(monthly)
    rows = []
    for identifier, row in zip(state_cell_ids(states), states.iter_rows(named=True), strict=True):
        published = row["observation_status"] != "suppressed"
        low = float(row["employment_value"]) if published else 0.0
        high = float(row["employment_value"]) if published else upper.get(identifier)
        rows.append(
            {
                "cell_id": identifier,
                "component_id": "c0",
                "rank": 0,
                "nullity": 0,
                "lp_lower": low,
                "lp_upper": high,
                "milp_lower": None,
                "milp_upper": None,
                "selected_lower": low,
                "selected_upper": high,
                "bound_status": "observed" if published else "unbounded",
                "exactly_identified": published,
                "integer_exactly_identified": published,
                "solver_status": "optimal",
                "solver_tolerance": 1e-7,
                "constraint_set_hash": "hash0",
            }
        )
    return pl.DataFrame(rows, schema=DETERMINISTIC_BOUNDS_SCHEMA)


@pytest.fixture()
def summarised(harmonized_toy, appendix_a_config, make_state_fit):
    """The toy's three suppressed cells reconciled over 40 draws, '06' in February capped at 60."""
    monthly = harmonized_toy.qcew_monthly
    suppressed = _states(monthly).filter(pl.col("observation_status") == "suppressed")
    cells = state_cell_ids(suppressed.sort("reference_month", "state_fips"))
    bounds = _bounds_frame(monthly, {cells[2]: 60.0})
    fit = make_state_fit(np.random.default_rng(SEED).gamma(2.0, 20.0, size=(40, 3)), cells)
    draws = reconcile_fit(fit, monthly, state_total_bounds(bounds), appendix_a_config)
    frame = posterior_summary(draws, monthly, bounds, run_id="run0", constraint_set_hash="hash0")
    return monthly, draws, bounds, frame


def test_one_row_per_state_cell_in_section_7_11s_field_order(summarised) -> None:
    monthly, _draws, _bounds, frame = summarised
    assert list(frame.schema.items()) == list(POSTERIOR_SUMMARY_SCHEMA.items())
    assert frame.height == _states(monthly).height == 8
    assert frame["cell_id"].n_unique() == 8


def test_a_published_cell_carries_its_value_in_every_posterior_column(summarised) -> None:
    """INV-001: a posterior interval around a published number would invent uncertainty."""
    monthly, _draws, _bounds, frame = summarised
    observed = _states(monthly).filter(pl.col("observation_status") == "observed")
    values = dict(
        zip(state_cell_ids(observed), observed["employment_value"].to_list(), strict=True)
    )
    rows = frame.filter(pl.col("observed_or_imputed") == "observed")
    assert rows.height == 5
    for row in rows.iter_rows(named=True):
        assert {row[column] for column in POSTERIOR_COLUMNS} == {float(values[row["cell_id"]])}
        assert row["reconciliation_status"] == "observed"


def test_an_imputed_cells_summaries_come_from_its_draws_and_nest(summarised) -> None:
    _monthly, draws, _bounds, frame = summarised
    for j, cell in enumerate(draws.cell_ids):
        row = frame.filter(pl.col("cell_id") == cell).row(0, named=True)
        column = draws.values[:, j]
        assert row["posterior_median"] == float(np.quantile(column, 0.5))
        assert row["posterior_mean"] == pytest.approx(float(column.mean()), rel=1e-12, abs=0.0)
        assert row["ci95_low"] <= row["ci90_low"] <= row["ci80_low"] <= row["ci50_low"]
        assert row["ci50_high"] <= row["ci80_high"] <= row["ci90_high"] <= row["ci95_high"]
        assert row["observed_or_imputed"] == "imputed"
        assert row["reconciliation_status"] == "anchored_and_reconciled"


def test_deterministic_and_posterior_intervals_are_separate_columns(summarised) -> None:
    """INV-008: §9's interval is copied through, and the posterior one sits inside it."""
    _monthly, draws, _bounds, frame = summarised
    imputed = frame.filter(pl.col("observed_or_imputed") == "imputed").sort("cell_id")
    capped = imputed.filter(pl.col("cell_id") == draws.cell_ids[2]).row(0, named=True)
    assert (capped["deterministic_lower"], capped["deterministic_upper"]) == (0.0, 60.0)
    assert capped["ci95_high"] <= 60.0
    uncapped = imputed.filter(pl.col("cell_id") != draws.cell_ids[2])
    assert uncapped["deterministic_upper"].null_count() == uncapped.height
    assert (imputed["ci95_low"] >= imputed["deterministic_lower"]).all()


def test_the_columns_later_owners_fill_are_null(summarised) -> None:
    """Sensitivity is Stage 7's (§11.13); §7.11's thresholds have no stated value anywhere."""
    frame = summarised[3]
    for column in (
        "model_sensitivity_low",
        "model_sensitivity_high",
        "probability_thresholds_json",
    ):
        assert frame[column].null_count() == frame.height


def test_the_source_vintage_set_is_a_json_list(summarised) -> None:
    frame = summarised[3]
    assert {json.loads(value)[0] for value in frame["source_vintage_set"].to_list()} == {"2024q1"}


def test_a_suppressed_cell_with_no_draws_is_refused(summarised) -> None:
    monthly, draws, bounds, _frame = summarised
    doctored = monthly.with_columns(
        pl.when((pl.col("state_fips") == "01") & (pl.col("reference_month") == "2023-01"))
        .then(pl.lit("suppressed"))
        .otherwise(pl.col("observation_status"))
        .alias("observation_status")
    )
    with pytest.raises(ConceptViolationError, match="no reconciled draws"):
        posterior_summary(draws, doctored, bounds, run_id="run0", constraint_set_hash="hash0")


def test_draws_for_a_cell_that_is_not_suppressed_are_refused(summarised) -> None:
    monthly, draws, bounds, _frame = summarised
    doctored = monthly.with_columns(
        pl.when((pl.col("state_fips") == "06") & (pl.col("reference_month") == "2023-02"))
        .then(pl.lit("observed"))
        .otherwise(pl.col("observation_status"))
        .alias("observation_status"),
        pl.when((pl.col("state_fips") == "06") & (pl.col("reference_month") == "2023-02"))
        .then(pl.lit(30))
        .otherwise(pl.col("employment_value"))
        .alias("employment_value"),
    )
    with pytest.raises(ConceptViolationError, match="match no suppressed row"):
        posterior_summary(draws, doctored, bounds, run_id="run0", constraint_set_hash="hash0")


def test_an_undeclared_observed_or_imputed_value_is_refused() -> None:
    with pytest.raises(ConceptViolationError, match="observed_or_imputed"):
        assert_declared_provenance(pl.DataFrame({"observed_or_imputed": ["guessed"]}))
