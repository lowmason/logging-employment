"""§13.10 applied (`validate/promotion.py`): exact coverage, the comparand, and the verdict.

Every scenario is two regimes of twenty masked cells, one seed each: ten cells in `01` (East South
Central) and ten in `06` (Pacific), every truth 100. A method's error on a cell is the percentage
point amount its estimate misses by, so a WAPE reads straight off the scenario.
"""

from __future__ import annotations

import json
from fractions import Fraction
from math import comb

import polars as pl
import pytest

from logging_employment.config import PromotionConfig
from logging_employment.contracts import (
    VALIDATION_METRIC_SCHEMA,
    VALIDATION_SCORE_SCHEMA,
    VALIDATION_SCOREBOARD_SCHEMA,
)
from logging_employment.errors import ConceptViolationError
from logging_employment.validate.promotion import (
    FALLBACK_METHOD,
    catastrophic,
    coverage_hits,
    evaluate_promotion,
    production_failed_record,
    within_tolerance,
)

MODEL = "state_total_model"
COMPARAND = "share_last_observed"
REGIMES = ("regional_blocks", "small_cell_biased")
SEED = 1024
STATES = ("01",) * 10 + ("06",) * 10
PASSING_CHECKS = {
    "reconciliation_anchor_drift_max": {"passed": True},
    "reconciliation_bound_violations": {"passed": True},
}
PASSING_GATE = {"passed": True, "failures": [], "checks": PASSING_CHECKS}


def _tail(hits: int, n: int) -> Fraction:
    """P(X <= hits) for X ~ Binomial(n, 9/10), in exact rational arithmetic: the test's oracle."""
    return sum(
        Fraction(comb(n, k)) * Fraction(9, 10) ** k * Fraction(1, 10) ** (n - k)
        for k in range(hits + 1)
    )


def _scores(
    estimator: str, errors: list[float | None], *, masked_hash: str = "masked"
) -> pl.DataFrame:
    """One scored row per masked cell and regime; a None error is a declined cell."""
    rows = []
    for regime in REGIMES:
        for index, (state, error) in enumerate(zip(STATES, errors, strict=True)):
            estimate = None if error is None else 100.0 + error
            rows.append(
                {
                    "regime": regime,
                    "seed": SEED,
                    "replicate": 0,
                    "mask_arm": "state_total",
                    "estimator_id": estimator,
                    "cell_id": f"state_total|{state}|2023-{index + 1:02d}|5|113310|NAICS 2022|ALL",
                    "state_fips": state,
                    "reference_month": f"2023-{index + 1:02d}",
                    "suppression_type": "primary_like",
                    "truth": 100.0,
                    "estimate": estimate,
                    "estimate_integer": None if estimate is None else round(estimate),
                    "weight_basis": "none" if estimate is None else "own_estimator",
                    "decline_kind": "data_gap" if estimate is None else None,
                    "bound_status": "unbounded",
                    "selected_lower": 0.0,
                    "selected_upper": None,
                    "masked_constraint_set_hash": masked_hash,
                    "lookback_months_masked": 1,
                    "missing_set_size": 10,
                    "raw_weight": estimate,
                    "anchor_basis": "declared_national_total",
                    "reconciliation_status": "declined"
                    if estimate is None
                    else "anchored_and_reconciled",
                    "decline_reason": "no history" if estimate is None else None,
                    "residual": 1000.0,
                    "constraint_set_hash": None,
                }
            )
    return pl.DataFrame(rows, schema=VALIDATION_SCORE_SCHEMA)


def _coverage(
    estimator: str,
    overall: tuple[int, int],
    divisions: dict[str, tuple[int, int]],
    *,
    source: str,
    crps: float = 1.0,
) -> list[dict[str, object]]:
    """Each regime's `coverage_0.90` and `crps` rows, overall and per division."""
    rows: list[dict[str, object]] = []
    for regime in REGIMES:
        base = {
            "regime": regime,
            "seed": SEED,
            "mask_arm": "state_total",
            "estimator_id": estimator,
            "metric_family": "probabilistic",
            "denominator_basis": "masked_cell_rows",
            "interval_source": source,
        }
        hits, n = overall
        overall_base = {
            **base,
            "denominator": float(n),
            "n_scored": n,
            "calibration_sample_size": n,
        }
        rows.append(
            {
                **overall_base,
                "stratum_kind": "overall",
                "stratum_value": "all",
                "metric_name": "coverage_0.90",
                "value": hits / n,
            }
        )
        rows.append(
            {
                **overall_base,
                "stratum_kind": "overall",
                "stratum_value": "all",
                "metric_name": "crps",
                "value": crps,
            }
        )
        for division, (division_hits, division_n) in divisions.items():
            rows.append(
                {
                    **base,
                    "denominator": float(division_n),
                    "n_scored": division_n,
                    "calibration_sample_size": division_n,
                    "stratum_kind": "census_division",
                    "stratum_value": division,
                    "metric_name": "coverage_0.90",
                    "value": division_hits / division_n,
                }
            )
    return rows


def _board(estimators: dict[str, float]) -> pl.DataFrame:
    """A scoreboard row per regime for each estimator, at the given WAPE."""
    rows = [
        {
            "regime": regime,
            "seed": SEED,
            "mask_arm": "state_total",
            "estimator_id": estimator,
            "wape": wape,
            "denominator": 20.0,
            "denominator_basis": "masked_cell_rows",
            "n_scored": 20,
            "n_declined_by_design": 0,
            "n_declined_data_gap": 0,
            "n_declined_reconciliation_failure": 0,
            "n_own_estimator": 20,
            "n_establishment_fallback": 0,
        }
        for regime in REGIMES
        for estimator, wape in estimators.items()
    ]
    return pl.DataFrame(rows, schema=VALIDATION_SCOREBOARD_SCHEMA)


def _evaluate(
    *,
    model_errors: list[float | None],
    comparand_errors: list[float | None],
    model_coverage: tuple[tuple[int, int], dict[str, tuple[int, int]]] = (
        (18, 20),
        {"east_south_central": (9, 10), "pacific": (9, 10)},
    ),
    comparand_coverage: tuple[int, int] = (18, 20),
    model_crps: float = 1.0,
    comparand_crps: float = 2.0,
    board: pl.DataFrame | None = None,
    replicate_gate: dict[str, object] = PASSING_GATE,
    comparand_hash: str = "masked",
    model_source: str = "reconciled_posterior_draws",
) -> dict[str, object]:
    overall, divisions = model_coverage
    metrics = pl.DataFrame(
        _coverage(MODEL, overall, divisions, source=model_source, crps=model_crps),
        schema=VALIDATION_METRIC_SCHEMA,
    )
    comparand_metrics = pl.DataFrame(
        _coverage(
            COMPARAND,
            comparand_coverage,
            {},
            source="leave_one_out_residual_ensemble",
            crps=comparand_crps,
        ),
        schema=VALIDATION_METRIC_SCHEMA,
    )
    return evaluate_promotion(
        model_id=MODEL,
        model_scores=_scores(MODEL, model_errors),
        model_metrics=metrics,
        comparand_scores=_scores(COMPARAND, comparand_errors, masked_hash=comparand_hash),
        comparand_metrics=comparand_metrics,
        comparand_scoreboard=_board({COMPARAND: 0.1}) if board is None else board,
        production_gate=PASSING_GATE,
        production_store_check={"passed": True, "draws_sha256_matches": True},
        replicate_gates={
            regime: [{"seed": SEED, "gate": replicate_gate, "draws_sha256": "d"}]
            for regime in REGIMES
        },
        promotion=PromotionConfig(),
    )


def test_seventeen_of_twenty_is_within_five_points_of_ninety() -> None:
    """`D-109`'s pin: exactly 17/20 against nominal 0.90 and tolerance 0.05 counts as within.

    The float comparison is the defect the item records: 44 of `runs/f03023ac9f3a`'s 170 groups sat
    at exactly 0.85 and a float `abs(c - 0.9) <= 0.05` excluded every one.
    """
    assert within_tolerance(17, 20, tolerance=0.05)
    assert not abs(17 / 20 - 0.9) <= 0.05
    assert not within_tolerance(16, 20, tolerance=0.05)


def test_coverage_hits_recovers_the_count_or_refuses_a_ratio() -> None:
    assert coverage_hits(17 / 20, 20) == 17
    with pytest.raises(ConceptViolationError, match="not an integer count"):
        coverage_hits(0.8123, 20)


@pytest.mark.parametrize(("hits", "n"), [(12, 20), (13, 20), (72, 100), (86, 100)])
def test_catastrophic_is_the_exact_binomial_lower_tail(hits: int, n: int) -> None:
    alpha = PromotionConfig().catastrophic_stratum_coverage_alpha
    assert catastrophic(hits, n, alpha=alpha) is (_tail(hits, n) < Fraction(repr(alpha)))


def test_a_model_that_beats_the_comparand_everywhere_is_selected() -> None:
    record = _evaluate(model_errors=[5.0] * 20, comparand_errors=[10.0] * 20)
    assert (record["verdict"], record["selected_method"]) == ("beat", MODEL)
    assert record["provisional"] is True
    assert record["disclosure_review"] == "pending_stage_8"
    regime = record["gates"]["improvement"]["per_regime"]["regional_blocks"]
    assert (regime["comparand"], regime["status"]) == (COMPARAND, "wape")
    assert (regime["n_matched"], regime["n_model_only"]) == (20, 0)
    assert regime["relative_improvement"] == pytest.approx(0.5)
    json.dumps(record, allow_nan=False)


def test_a_model_that_does_not_beat_it_deploys_the_simpler_method() -> None:
    """§13.10's last line: "If these gates are not met, deploy the simpler method"."""
    record = _evaluate(model_errors=[10.0] * 20, comparand_errors=[10.0] * 20)
    assert (record["verdict"], record["selected_method"]) == ("not_beaten", FALLBACK_METHOD)
    assert record["gates"]["improvement"]["per_regime"]["regional_blocks"]["status"] == "not_beaten"


def test_calibrated_uncertainty_can_carry_a_regime_the_wape_gate_does_not() -> None:
    """The comparand's coverage is catastrophic (5 of 20) and the model's CRPS is lower."""
    record = _evaluate(
        model_errors=[10.0] * 20, comparand_errors=[10.0] * 20, comparand_coverage=(5, 20)
    )
    regime = record["gates"]["improvement"]["per_regime"]["regional_blocks"]
    assert regime["status"] == "calibrated_uncertainty"
    assert record["verdict"] == "beat"
    worse = _evaluate(
        model_errors=[10.0] * 20,
        comparand_errors=[10.0] * 20,
        comparand_coverage=(5, 20),
        model_crps=3.0,
    )
    assert worse["gates"]["improvement"]["per_regime"]["regional_blocks"]["status"] == "not_beaten"


def test_cells_the_comparand_declined_are_counted_not_compared() -> None:
    record = _evaluate(model_errors=[5.0] * 20, comparand_errors=[None, None] + [10.0] * 18)
    regime = record["gates"]["improvement"]["per_regime"]["small_cell_biased"]
    assert (regime["n_matched"], regime["n_model_only"]) == (18, 2)
    assert regime["model_wape"] == pytest.approx(0.05)


def test_a_regime_with_no_comparand_is_neither_a_pass_nor_a_failure() -> None:
    """Stage 4's SHIPPED point (3): `preferred_baseline` returning None means "no comparand"."""
    no_hierarchy = _board({"equal_residual": 0.1})
    record = _evaluate(model_errors=[5.0] * 20, comparand_errors=[10.0] * 20, board=no_hierarchy)
    statuses = {
        regime: row["status"]
        for regime, row in record["gates"]["improvement"]["per_regime"].items()
    }
    assert statuses == dict.fromkeys(REGIMES, "not_applicable")
    # With no regime to compare in, nothing shows the model beats anything.
    assert record["gates"]["improvement"]["passed"] is False
    assert record["verdict"] == "not_beaten"


def test_a_division_the_model_degrades_fails_the_gate() -> None:
    """Better in every regime, worse in the Pacific: 0.11 against 0.10 is a 10% degradation."""
    record = _evaluate(model_errors=[0.0] * 10 + [11.0] * 10, comparand_errors=[10.0] * 20)
    assert record["gates"]["improvement"]["passed"] is True
    pacific = record["gates"]["stratum_degradation"]["per_division"]["pacific"]
    assert pacific["degraded"] is True
    assert pacific["relative_degradation"] == pytest.approx(0.1)
    assert record["verdict"] == "not_beaten"


def test_a_catastrophic_division_fails_coverage_when_the_pool_passes() -> None:
    """86% pooled is within five points; one division at 72 of 100 is not calibration's doing."""
    record = _evaluate(
        model_errors=[5.0] * 20,
        comparand_errors=[10.0] * 20,
        model_coverage=((86, 100), {"east_south_central": (50, 50), "pacific": (36, 50)}),
    )
    coverage = record["gates"]["coverage"]
    assert coverage["overall"]["within_tolerance"] is True
    assert [stratum["stratum_value"] for stratum in coverage["catastrophic_strata"]] == ["pacific"]
    assert record["verdict"] == "not_beaten"


def test_a_failed_replicate_fails_convergence_and_is_named() -> None:
    failed = {"passed": False, "failures": ["parameter_rhat_max"], "checks": PASSING_CHECKS}
    record = _evaluate(model_errors=[5.0] * 20, comparand_errors=[10.0] * 20, replicate_gate=failed)
    convergence = record["gates"]["convergence"]
    assert convergence["passed"] is False
    assert convergence["replicate_failures"][0]["failures"] == ["parameter_rhat_max"]
    assert record["verdict"] == "not_beaten"


def test_runs_that_masked_under_different_systems_are_refused() -> None:
    with pytest.raises(ConceptViolationError, match="different masked"):
        _evaluate(model_errors=[5.0] * 20, comparand_errors=[10.0] * 20, comparand_hash="other")


def test_the_models_coverage_must_come_from_its_own_draws() -> None:
    with pytest.raises(ConceptViolationError, match="reconciled draws"):
        _evaluate(
            model_errors=[5.0] * 20,
            comparand_errors=[10.0] * 20,
            model_source="leave_one_out_residual_ensemble",
        )


def test_a_failed_production_fit_is_recorded_without_being_scored() -> None:
    gate = {"passed": False, "failures": ["cell_rhat_max"], "checks": PASSING_CHECKS}
    record = production_failed_record(MODEL, gate, PromotionConfig())
    assert (record["verdict"], record["selected_method"]) == ("not_beaten", FALLBACK_METHOD)
    assert record["gates"]["convergence"]["production_failures"] == ["cell_rhat_max"]
    assert record["gates"]["coverage"]["status"] == "not_evaluated_production_fit_failed"
