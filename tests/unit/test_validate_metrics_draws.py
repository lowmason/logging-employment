"""`metrics.draw_interval_metrics`: §13.7's rows from a model's own reconciled draws."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from logging_employment.errors import ConceptViolationError
from logging_employment.validate.intervals import crps
from logging_employment.validate.metrics import draw_interval_metrics

MODEL = "state_total_model"


def _scored(estimates: list[float | None], truths: list[float], states: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "estimator_id": [MODEL] * len(truths),
            "cell_id": [f"c{index}" for index in range(len(truths))],
            "state_fips": states,
            "estimate": estimates,
            "truth": truths,
        },
        schema={
            "estimator_id": pl.String,
            "cell_id": pl.String,
            "state_fips": pl.String,
            "estimate": pl.Float64,
            "truth": pl.Float64,
        },
    )


def _rows(frame: pl.DataFrame, name: str, kind: str = "overall") -> pl.DataFrame:
    return frame.filter((pl.col("metric_name") == name) & (pl.col("stratum_kind") == kind))


def test_each_cell_is_scored_against_its_own_draws() -> None:
    """Cell 0's draws cover its truth and cell 1's miss: coverage 1/2, and a hand-computed CRPS."""
    ensembles = {"c0": np.arange(1.0, 101.0), "c1": np.arange(1.0, 101.0)}
    frame = draw_interval_metrics(
        _scored([50.5, 50.5], [50.0, 500.0], ["01", "06"]),
        regime="regional_blocks",
        seed=1024,
        arm="state_total",
        ensembles=ensembles,
    )
    coverage = _rows(frame, "coverage_0.90").row(0, named=True)
    assert coverage["value"] == 0.5
    assert coverage["calibration_sample_size"] == 2
    assert coverage["interval_source"] == "reconciled_posterior_draws"
    expected = (crps(ensembles["c0"], 50.0) + crps(ensembles["c1"], 500.0)) / 2
    assert _rows(frame, "crps")["value"].item() == pytest.approx(expected, rel=1e-12)
    # Every draw already lies inside its bounds, so nothing is clipped at zero.
    assert _rows(frame, "n_clipped_at_zero")["value"].item() == 0.0


def test_divisions_get_their_own_coverage_rows() -> None:
    frame = draw_interval_metrics(
        _scored([50.5, 50.5], [50.0, 500.0], ["01", "06"]),
        regime="regional_blocks",
        seed=1024,
        arm="state_total",
        ensembles={"c0": np.arange(1.0, 101.0), "c1": np.arange(1.0, 101.0)},
    )
    divisions = _rows(frame, "coverage_0.90", "census_division").sort("stratum_value")
    assert divisions["stratum_value"].to_list() == ["east_south_central", "pacific"]
    assert divisions["value"].to_list() == [1.0, 0.0]
    assert divisions["calibration_sample_size"].to_list() == [1, 1]


def test_a_scored_cell_with_no_draws_is_refused() -> None:
    with pytest.raises(ConceptViolationError, match="carry no draws"):
        draw_interval_metrics(
            _scored([50.5, 50.5], [50.0, 500.0], ["01", "06"]),
            regime="regional_blocks",
            seed=1024,
            arm="state_total",
            ensembles={"c0": np.arange(1.0, 101.0)},
        )


def test_an_estimator_that_scored_nothing_writes_one_null_row() -> None:
    frame = draw_interval_metrics(
        _scored([None], [50.0], ["01"]),
        regime="regional_blocks",
        seed=1024,
        arm="state_total",
        ensembles={},
    )
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert (row["metric_name"], row["value"], row["interval_source"]) == (
        "coverage_0.90",
        None,
        "none",
    )
