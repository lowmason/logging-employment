"""`models/reconciliation.py`: every draw reconciled month by month, and INV-012 measured on draws.

The toy panel's missing sets are '04' in January (residual 50) and '04' and '06' in February
(residual 80), so a January draw has one cell to hold the whole residual and a February draw two.
No sampler runs here: the fits are raw-score arrays (`make_state_fit`).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from logging_employment.errors import ConceptViolationError, WeightDomainError
from logging_employment.models.data import state_cell_ids
from logging_employment.models.reconciliation import (
    check_reconciled,
    exact_column_means,
    reconcile_fit,
)
from logging_employment.reconcile.scaling import Bounds

SEED = sum(map(ord, "tests/model-reconciliation"))
TOLERANCE = 1e-9


def _predicted(harmonized_toy) -> tuple[str, ...]:
    """Suppressed toy cells as seven-field ids, in `build_model_data`'s (month, state) order."""
    monthly = harmonized_toy.qcew_monthly
    suppressed = monthly.filter(
        (pl.col("area_type") == "state") & (pl.col("observation_status") == "suppressed")
    ).sort("reference_month", "state_fips")
    return state_cell_ids(suppressed)


def _bounds(cells: tuple[str, ...], upper: dict[str, float] | None = None) -> Bounds:
    """Nonnegativity for every cell, and a finite upper only where `upper` names one."""
    return Bounds(
        lower=dict.fromkeys(cells, 0.0),
        upper={cell: (upper or {}).get(cell) for cell in cells},
    )


def _raw(draws: int, cells: int) -> np.ndarray:
    return np.random.default_rng(SEED).gamma(2.0, 20.0, size=(draws, cells))


def test_every_draw_adds_up_and_stays_inside_its_bounds(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    cells = _predicted(harmonized_toy)
    fit = make_state_fit(_raw(6, 3), cells)
    draws = reconcile_fit(fit, harmonized_toy.qcew_monthly, _bounds(cells), appendix_a_config)
    check = check_reconciled(draws, tolerance=TOLERANCE)
    assert check.passed
    assert (check.draws_checked, check.months_checked, check.cells_checked) == (6, 2, 3)
    # January's missing set is one cell, so every draw pins it to the residual.
    np.testing.assert_allclose(draws.values[:, 0], 50.0, rtol=0.0, atol=TOLERANCE)
    for row in draws.values.tolist():
        assert math.isclose(row[1] + row[2], 80.0, rel_tol=0.0, abs_tol=TOLERANCE)


def test_a_finite_upper_bound_scales_every_draw_into_it(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    """§12.3 on each draw: '06' in February may hold at most 10, so '04' takes the rest."""
    cells = _predicted(harmonized_toy)
    fit = make_state_fit(np.tile([[5.0, 1.0, 9.0]], (4, 1)), cells)
    draws = reconcile_fit(
        fit, harmonized_toy.qcew_monthly, _bounds(cells, {cells[2]: 10.0}), appendix_a_config
    )
    assert np.all(draws.values[:, 2] <= 10.0 + TOLERANCE)
    np.testing.assert_allclose(draws.values[:, 1], 70.0, atol=TOLERANCE)
    assert check_reconciled(draws, tolerance=TOLERANCE).passed


def test_a_negative_draw_is_refused_rather_than_floored(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    """Stage 3's refusal reaches the model path unchanged (§18.3)."""
    cells = _predicted(harmonized_toy)
    raw = _raw(2, 3)
    raw[1, 2] = -1.0
    with pytest.raises(WeightDomainError, match="negative"):
        reconcile_fit(
            make_state_fit(raw, cells),
            harmonized_toy.qcew_monthly,
            _bounds(cells),
            appendix_a_config,
        )


def test_a_missing_cell_the_fit_never_scored_is_refused(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    cells = _predicted(harmonized_toy)
    fit = make_state_fit(_raw(2, 2), cells[:2])
    with pytest.raises(ConceptViolationError, match="scored no draw"):
        reconcile_fit(fit, harmonized_toy.qcew_monthly, _bounds(cells), appendix_a_config)


def test_a_scored_cell_in_no_missing_set_is_refused(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    """A score with no anchor to reconcile against is an estimate §12 never checked."""
    cells = _predicted(harmonized_toy)
    stray = cells[0].replace("|04|", "|01|")
    fit = make_state_fit(_raw(2, 4), (*cells, stray))
    with pytest.raises(ConceptViolationError, match="no month's missing set"):
        reconcile_fit(fit, harmonized_toy.qcew_monthly, _bounds(cells), appendix_a_config)


def test_the_draw_axis_and_its_chain_index_survive(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    """§12.7: nothing here summarises, so the draws keep the index §15.4's store must preserve."""
    cells = _predicted(harmonized_toy)
    raw = _raw(6, 3)
    fit = make_state_fit(raw, cells)
    draws = reconcile_fit(fit, harmonized_toy.qcew_monthly, _bounds(cells), appendix_a_config)
    assert draws.chain.tolist() == [0, 0, 0, 1, 1, 1]
    assert draws.draw.tolist() == [0, 1, 2, 0, 1, 2]
    np.testing.assert_allclose(draws.raw_mean, raw.mean(axis=0), rtol=1e-12, atol=0.0)
    assert draws.state_fips == ("04", "04", "06")
    assert draws.reference_months == ("2023-01", "2023-02", "2023-02")
    assert {month: anchor.anchor_basis for month, anchor in draws.anchors.items()} == {
        "2023-01": "declared_national_total",
        "2023-02": "declared_national_total",
    }


def test_a_column_mean_depends_on_the_values_alone() -> None:
    """Neither the draws' order nor the array's memory layout moves a persisted mean."""
    rng = np.random.default_rng(SEED)
    values = rng.gamma(2.0, 20.0, size=(4000, 7))
    reference = exact_column_means(values)
    assert np.array_equal(reference, exact_column_means(np.asfortranarray(values)))
    assert np.array_equal(reference, exact_column_means(values[rng.permutation(4000)]))
    np.testing.assert_allclose(reference, values.mean(axis=0), rtol=1e-12, atol=0.0)
    with pytest.raises(ConceptViolationError, match="zero draws"):
        exact_column_means(np.empty((0, 3)))


def test_the_check_reads_the_draws_not_a_summary(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    """One drifted draw and one escaped value are each caught, and reported apart."""
    cells = _predicted(harmonized_toy)
    draws = reconcile_fit(
        make_state_fit(_raw(4, 3), cells),
        harmonized_toy.qcew_monthly,
        _bounds(cells, {cells[2]: 1000.0}),
        appendix_a_config,
    )
    draws.values[3, 1] += 1.0
    drifted = check_reconciled(draws, tolerance=TOLERANCE)
    assert drifted.max_anchor_drift == pytest.approx(1.0)
    assert drifted.bound_violations == 0
    assert not drifted.passed
    draws.values[3, 1] -= 1.0
    draws.values[2, 2] = 1001.0
    escaped = check_reconciled(draws, tolerance=TOLERANCE)
    assert escaped.bound_violations == 1
    assert not escaped.passed
