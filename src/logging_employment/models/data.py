"""`ModelData` from the harmonized QCEW panel (§11.1, §11.3, and §11.11's QCEW rows).

It reads `observation_status` and nothing else to decide which cells may carry a response. So the
pseudo-suppression mask (`validate.mask.apply_mask`, which rewrites a hidden cell to `suppressed`
with a null value) moves a cell from training to prediction without this module knowing a mask
exists. That is how the §13 harness fits the model on exactly what a real suppression would leave
public.

ONE LATENT PATH CROSSES THE 2022-01 NAICS SEAM, deliberately. §10.3's share baselines refuse to
cross it (`BaselinesConfig.historical_may_cross_naics_vintage`), because a share is a ratio of two
classification periods. Here the seam is one month on a continuous state series.
`harmonize/naics.py::assert_113310_survives_the_window` establishes that 113310 pairs one-to-one,
under the same title, across the 2017 and 2022 vintages, so the two vintages are not incompatible
in INV-007's sense. A level shift at the seam is absorbed by the year effects. Each cell keeps its
own `naics_vintage` inside its `cell_id`, so no value is relabelled from one vintage to the other.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..constraints.cells import KIND_STATE_TOTAL, TOTAL_SIZE_CLASS, cell_id
from ..errors import ConceptViolationError
from ..validate.regimes import DIVISION_OF
from .interfaces import ModelData

TRAINING_STATUS = "observed"
PREDICTION_STATUS = "suppressed"
FINAL = "final"


def _refuse(frame: pl.DataFrame, reason: str) -> None:
    """Raise naming the first cells of `frame`, when it has any."""
    if frame.height:
        cells = frame.select("state_fips", "reference_month").rows()[:5]
        raise ConceptViolationError(f"{frame.height} cell(s) {reason}; first {cells}")


def state_cell_ids(frame: pl.DataFrame) -> tuple[str, ...]:
    """The seven-field `cell_id` of every row, read from the row's own published fields.

    Public because `models/summary.py` keys §7.11's rows the same way. The fields are read off each
    row rather than hardcoded, as `baselines/runner.py::missing_cell_ids` does, so a vintage change
    follows the data.
    """
    return tuple(
        cell_id(
            KIND_STATE_TOTAL,
            state_fips=str(row["state_fips"]),
            reference_month=str(row["reference_month"]),
            ownership_code=str(row["ownership_code"]),
            industry_code=str(row["industry_code"]),
            naics_vintage=str(row["naics_vintage"]),
            size_class=TOTAL_SIZE_CLASS,
        )
        for row in frame.iter_rows(named=True)
    )


def _indices(frame: pl.DataFrame, column: str, lookup: dict[str, int]) -> np.ndarray:
    """`frame[column]` as positions in `lookup`, the integer index the model gathers by."""
    return np.array([lookup[str(v)] for v in frame[column].to_list()], dtype=np.int64)


def build_model_data(monthly: pl.DataFrame) -> ModelData:
    """The state-total panel of `monthly`, split into training and prediction cells.

    FAIL-CLOSED REFUSALS, each naming the offending cells:

    * a training or prediction cell with `qtrly_establishments <= 0`. §11.5's score is
      `A * exp(mu)`, and at `A = 0` that is zero, which `reconcile_draws` floors to a 1e-12 weight
      rather than refusing. Measured on D1 on 2026-09-26: no such cell (observed A >= 3,
      suppressed A >= 1).
    * a training cell whose `employment_value` is null or `<= 0`, where log(E / A) is undefined.
    * a suppressed cell carrying a non-null `employment_value`. That is the pseudo-observation §11.3
      forbids, or a mask that failed to clear the value.
    * a training row whose `release_status` is not `final`. §11.11 says preliminary QCEW is a
      "separate as-of observation; do not mix with final hard controls". D1 is final throughout.
    * more than one `industry_code` or `ownership_code` among the state rows, because one fit is one
      estimand (§3).
    * a state outside `validate/regimes.py::DIVISION_OF`, which carries §17.5's region effect.

    Cells are ordered by (month, state) in both sets, so the same frame always gives the same
    arrays.
    """
    states_frame = monthly.filter(pl.col("area_type") == "state")
    for column in ("industry_code", "ownership_code"):
        values = sorted(str(v) for v in states_frame[column].unique().to_list())
        if len(values) != 1:
            raise ConceptViolationError(
                f"the state panel carries {len(values)} {column} values {values}; one fit is "
                "one estimand (§3)"
            )
    states = tuple(sorted(str(s) for s in states_frame["state_fips"].unique().to_list()))
    unmapped = [state for state in states if state not in DIVISION_OF]
    if unmapped:
        raise ConceptViolationError(
            f"state_fips {unmapped} fall in no Census division; §17.5's region effect partitions "
            "on validate/regimes.py::CENSUS_DIVISIONS"
        )
    months = tuple(sorted(str(m) for m in monthly["reference_month"].unique().to_list()))
    divisions = tuple(sorted({DIVISION_OF[state] for state in states}))
    years = tuple(sorted({month[:4] for month in months}))

    order = ["reference_month", "state_fips"]
    train = states_frame.filter(pl.col("observation_status") == TRAINING_STATUS).sort(order)
    predict = states_frame.filter(pl.col("observation_status") == PREDICTION_STATUS).sort(order)

    _refuse(
        train.filter(pl.col("release_status") != FINAL),
        "are not final-vintage, and §11.11 keeps preliminary QCEW out of a final fit",
    )
    _refuse(
        train.filter(pl.col("employment_value").is_null() | (pl.col("employment_value") <= 0)),
        "are observed with no positive employment, so log(E / A) is undefined",
    )
    _refuse(
        predict.filter(pl.col("employment_value").is_not_null()),
        "are suppressed yet carry a value, which §11.3 forbids as a pseudo-observation",
    )
    for name, frame in (("training", train), ("prediction", predict)):
        _refuse(
            frame.filter(
                pl.col("qtrly_establishments").is_null() | (pl.col("qtrly_establishments") <= 0)
            ),
            f"in the {name} set have no establishments, so §11.5's A * exp(mu) is zero there",
        )

    state_index = {state: index for index, state in enumerate(states)}
    month_index = {month: index for index, month in enumerate(months)}
    train_exposure = train["qtrly_establishments"].cast(pl.Float64).to_numpy()
    predict_exposure = predict["qtrly_establishments"].cast(pl.Float64).to_numpy()
    log_exposure = np.log(np.concatenate([train_exposure, predict_exposure]))
    centre = float(np.mean(log_exposure)) if log_exposure.size else 0.0
    spread = float(np.std(log_exposure)) if log_exposure.size else 0.0
    # A constant exposure carries no information for beta; a scale of 1.0 leaves x at zero rather
    # than dividing by zero, and beta's posterior is then its prior, which is the truth of it.
    scale = spread if spread > 0.0 else 1.0

    return ModelData(
        states=states,
        months=months,
        divisions=divisions,
        years=years,
        state_division=np.array(
            [divisions.index(DIVISION_OF[state]) for state in states], dtype=np.int64
        ),
        month_of_year=np.array([int(month[5:7]) - 1 for month in months], dtype=np.int64),
        year_of_month=np.array([years.index(month[:4]) for month in months], dtype=np.int64),
        train_state=_indices(train, "state_fips", state_index),
        train_month=_indices(train, "reference_month", month_index),
        train_y=np.log(train["employment_value"].cast(pl.Float64).to_numpy() / train_exposure),
        train_x=(np.log(train_exposure) - centre) / scale,
        predict_state=_indices(predict, "state_fips", state_index),
        predict_month=_indices(predict, "reference_month", month_index),
        predict_exposure=predict_exposure,
        predict_x=(np.log(predict_exposure) - centre) / scale,
        predict_cell_ids=state_cell_ids(predict),
        log_exposure_centre=centre,
        log_exposure_scale=scale,
    )
