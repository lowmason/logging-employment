"""`models/validation.py::model_results`: reconciled draws as `baseline_results` rows."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from logging_employment.contracts import BASELINE_RESULT_SCHEMA
from logging_employment.models.data import state_cell_ids
from logging_employment.models.reconciliation import reconcile_fit
from logging_employment.models.validation import model_results
from logging_employment.reconcile.scaling import Bounds

SEED = sum(map(ord, "tests/model-validation"))


def _results(harmonized_toy, appendix_a_config, make_state_fit) -> tuple[pl.DataFrame, np.ndarray]:
    monthly = harmonized_toy.qcew_monthly
    suppressed = monthly.filter(
        (pl.col("area_type") == "state") & (pl.col("observation_status") == "suppressed")
    ).sort("reference_month", "state_fips")
    cells = state_cell_ids(suppressed)
    bounds = Bounds(lower=dict.fromkeys(cells, 0.0), upper=dict.fromkeys(cells))
    fit = make_state_fit(np.random.default_rng(SEED).gamma(2.0, 20.0, size=(8, 3)), cells)
    draws = reconcile_fit(fit, monthly, bounds, appendix_a_config)
    return model_results(draws, bounds, appendix_a_config), draws.values


def test_the_estimate_is_the_mean_of_the_reconciled_draws(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    results, values = _results(harmonized_toy, appendix_a_config, make_state_fit)
    assert list(results.schema.items()) == list(BASELINE_RESULT_SCHEMA.items())
    np.testing.assert_allclose(
        results["estimate"].to_numpy(), values.mean(axis=0), rtol=1e-12, atol=0.0
    )


def test_the_mean_adds_up_exactly_as_every_draw_does(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    """Why the mean and not the median: `constraint_metrics` reads adding-up off these rows."""
    results, _values = _results(harmonized_toy, appendix_a_config, make_state_fit)
    for (month,), group in results.group_by("reference_month"):
        residual = group["residual"][0]
        assert math.isclose(math.fsum(group["estimate"].to_list()), residual, abs_tol=1e-9)
        assert sum(group["estimate_integer"].to_list()) == round(residual), month


def test_every_row_declares_the_models_provenance(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    results, _values = _results(harmonized_toy, appendix_a_config, make_state_fit)
    assert set(results["estimator_id"]) == {"state_total_model"}
    assert set(results["weight_basis"]) == {"own_estimator"}
    assert set(results["anchor_basis"]) == {"declared_national_total"}
    assert set(results["reconciliation_status"]) == {"anchored_and_reconciled"}
    assert results["decline_reason"].null_count() == results.height
