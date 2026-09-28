"""`models/data.py::build_model_data`: which cells train, which predict, and what is refused."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from logging_employment.baselines.runner import missing_cell_ids
from logging_employment.errors import ConceptViolationError
from logging_employment.models.data import build_model_data
from logging_employment.reconcile.anchor import observed_partition


def test_observed_cells_train_and_suppressed_cells_predict(harmonized_toy) -> None:
    """The toy's months: '04' suppressed in both, '06' in February; everything else observed."""
    data = build_model_data(harmonized_toy.qcew_monthly)
    assert data.states == ("01", "02", "04", "06")
    assert data.months == ("2023-01", "2023-02")
    assert data.train_y.shape == (5,)
    assert [
        (data.states[s], data.months[t])
        for s, t in zip(data.predict_state, data.predict_month, strict=True)
    ] == [("04", "2023-01"), ("04", "2023-02"), ("06", "2023-02")]


def test_the_response_is_log_employees_per_establishment(harmonized_toy) -> None:
    data = build_model_data(harmonized_toy.qcew_monthly)
    observed = harmonized_toy.qcew_monthly.filter(
        (pl.col("area_type") == "state") & (pl.col("observation_status") == "observed")
    ).sort("reference_month", "state_fips")
    expected = np.log(
        observed["employment_value"].to_numpy() / observed["qtrly_establishments"].to_numpy()
    )
    np.testing.assert_allclose(data.train_y, expected, rtol=1e-15, atol=0)


def test_a_suppressed_cell_is_never_a_response(harmonized_toy) -> None:
    """§11.3: no pseudo-observation. The two sets share no (state, month)."""
    data = build_model_data(harmonized_toy.qcew_monthly)
    train = set(zip(data.train_state.tolist(), data.train_month.tolist(), strict=True))
    predict = set(zip(data.predict_state.tolist(), data.predict_month.tolist(), strict=True))
    assert not train & predict
    assert data.predict_exposure.shape == (len(predict),)


def test_prediction_cells_carry_the_runners_seven_field_ids(harmonized_toy) -> None:
    """The join key a fit's draws, `deterministic_bounds` and `baseline_results` all share."""
    monthly = harmonized_toy.qcew_monthly
    data = build_model_data(monthly)
    expected: list[str] = []
    for _month, partition in sorted(observed_partition(monthly).items()):
        ids = missing_cell_ids(partition)
        expected.extend(ids[state] for state in sorted(ids))
    assert list(data.predict_cell_ids) == expected


def test_a_true_zero_cell_is_in_neither_set(make_monthly) -> None:
    monthly = make_monthly(
        {"state_fips": "01", "area_fips": "01000"},
        {
            "state_fips": "02",
            "area_fips": "02000",
            "observation_status": "true_zero",
            "employment_value": 0,
            "qtrly_establishments": 0,
            "is_true_zero": True,
        },
        {
            "state_fips": "04",
            "area_fips": "04000",
            "observation_status": "suppressed",
            "employment_value": None,
        },
    )
    data = build_model_data(monthly)
    assert data.train_y.shape == (1,)
    assert [data.states[s] for s in data.predict_state] == ["04"]
    # Still a state of the grid: its latent path runs through the month it published a zero.
    assert "02" in data.states


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (
            {"observation_status": "suppressed", "employment_value": 40},
            "pseudo-observation",
        ),
        (
            {
                "observation_status": "suppressed",
                "employment_value": None,
                "qtrly_establishments": 0,
            },
            "no establishments",
        ),
        ({"release_status": "preliminary"}, "final-vintage"),
        ({"employment_value": 0}, "no positive employment"),
        ({"industry_code": "113"}, "industry_code"),
        ({"state_fips": "72", "area_fips": "72000"}, "Census division"),
    ],
)
def test_a_cell_the_model_cannot_read_is_refused_by_name(make_monthly, row, reason) -> None:
    monthly = make_monthly(
        {"state_fips": "01", "area_fips": "01000"},
        {"state_fips": "04", "area_fips": "04000", **row},
    )
    with pytest.raises(ConceptViolationError, match=reason):
        build_model_data(monthly)


def test_calendar_indexes_follow_the_month_string(make_monthly) -> None:
    monthly = make_monthly(
        {
            "reference_month": "2021-12",
            "reference_quarter": "2021Q4",
            "naics_vintage": "NAICS 2017",
        },
        {"reference_month": "2022-01", "reference_quarter": "2022Q1"},
    )
    data = build_model_data(monthly)
    assert data.months == ("2021-12", "2022-01")
    assert data.month_of_year.tolist() == [11, 0]
    assert data.years == ("2021", "2022")
    assert data.year_of_month.tolist() == [0, 1]


def test_exposure_is_standardized_over_both_sets(harmonized_toy) -> None:
    data = build_model_data(harmonized_toy.qcew_monthly)
    x = np.concatenate([data.train_x, data.predict_x])
    assert abs(float(np.mean(x))) < 1e-12
    assert abs(float(np.std(x)) - 1.0) < 1e-12


def test_row_order_does_not_move_a_single_array(harmonized_toy) -> None:
    monthly = harmonized_toy.qcew_monthly
    reversed_rows = build_model_data(monthly.reverse())
    data = build_model_data(monthly)
    for name in ("train_state", "train_month", "train_y", "predict_state", "predict_exposure"):
        np.testing.assert_array_equal(getattr(reversed_rows, name), getattr(data, name))
    assert reversed_rows.predict_cell_ids == data.predict_cell_ids
