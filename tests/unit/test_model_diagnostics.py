"""§11.14's gate in code: what each check reads, which scope it gates, and how it fails."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from logging_employment.config import StateModelDiagnostics
from logging_employment.errors import ConceptViolationError, ModelDiagnosticsError
from logging_employment.models.diagnostics import assert_gate_passes, evaluate_gate
from logging_employment.models.reconciliation import DrawCheck, ReconciledDraws

SEED = sum(map(ord, "tests/model-diagnostics"))
CHAINS, PER_CHAIN = 4, 500
THRESHOLDS = StateModelDiagnostics()
PASSING = DrawCheck(
    draws_checked=CHAINS * PER_CHAIN,
    months_checked=1,
    cells_checked=4,
    max_anchor_drift=0.0,
    bound_violations=0,
    tolerance=1e-9,
)


def _draws(values: np.ndarray) -> ReconciledDraws:
    """Reconciled draws with only the fields the gate reads filled in meaningfully."""
    cells = values.shape[1]
    return ReconciledDraws(
        cell_ids=tuple(f"cell{j}" for j in range(cells)),
        state_fips=("01",) * cells,
        reference_months=("2023-01",) * cells,
        values=values,
        chain=np.repeat(np.arange(CHAINS), PER_CHAIN),
        draw=np.tile(np.arange(PER_CHAIN), CHAINS),
        raw_mean=values.mean(axis=0),
        lower=np.zeros(cells),
        upper=np.full(cells, math.inf),
        anchors={},
    )


def _mixed(shape: tuple[int, ...], offset_last_chain: float = 0.0) -> np.ndarray:
    values = np.random.default_rng(SEED).normal(size=shape)
    values[CHAINS - 1] += offset_last_chain
    return values


def _inputs(make_state_fit, *, stuck_parameter=0.0, stuck_cell=0.0, divergences=0, ppc=0.9):
    """A fit that mixes unless told otherwise, and its draws: three varying cells, one pinned."""
    cells = _mixed((CHAINS, PER_CHAIN, 3), stuck_cell).reshape(CHAINS * PER_CHAIN, 3) + 50.0
    values = np.column_stack([cells, np.full(CHAINS * PER_CHAIN, 20.0)])
    diverging = np.zeros((CHAINS, PER_CHAIN), dtype=bool)
    diverging.flat[:divergences] = True
    fit = make_state_fit(
        values,
        tuple(f"cell{j}" for j in range(4)),
        chains=CHAINS,
        parameters={
            "alpha": _mixed((CHAINS, PER_CHAIN), stuck_parameter),
            "u": _mixed((CHAINS, PER_CHAIN, 3)),
        },
        diverging=diverging,
        ppc_coverage_90=ppc,
    )
    return fit, _draws(values)


def test_a_well_mixed_fit_passes_the_production_gate(make_state_fit) -> None:
    fit, draws = _inputs(make_state_fit)
    report = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="production")
    assert report.passed, report.failures
    assert report.cells_constant == 1


def test_the_ess_floor_is_one_hundred_per_chain(make_state_fit) -> None:
    fit, draws = _inputs(make_state_fit)
    report = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="production")
    floors = {check.name: check.threshold for check in report.checks if "ess" in check.name}
    assert set(floors.values()) == {100.0 * CHAINS}


@pytest.mark.parametrize("scope", ["production", "replicate"])
def test_one_divergence_fails_either_scope(make_state_fit, scope: str) -> None:
    fit, draws = _inputs(make_state_fit, divergences=1)
    report = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope=scope)
    assert report.failures == ("divergences",)


@pytest.mark.parametrize("scope", ["production", "replicate"])
def test_a_stuck_chain_fails_parameter_rhat_in_either_scope(make_state_fit, scope: str) -> None:
    fit, draws = _inputs(make_state_fit, stuck_parameter=3.0)
    report = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope=scope)
    assert "parameter_rhat_max" in report.failures


def test_a_replicate_records_cell_diagnostics_without_gating_on_them(make_state_fit) -> None:
    """Decision 5: 27 replicates of ~1,400 cells would let one cell at 1.011 decide promotion."""
    fit, draws = _inputs(make_state_fit, stuck_cell=3.0)
    production = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="production")
    replicate = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="replicate")
    assert "cell_rhat_max" in production.failures
    assert replicate.passed
    cell_check = next(check for check in replicate.checks if check.name == "cell_rhat_max")
    assert (cell_check.passed, cell_check.gating) == (False, False)


def test_an_unmeasurable_check_fails_rather_than_passes(make_state_fit) -> None:
    """A NaN compares False: a predictive check that could not run is a failed check."""
    fit, draws = _inputs(make_state_fit, ppc=math.nan)
    report = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="production")
    assert report.failures == ("ppc_coverage_90",)


def test_a_reconciliation_failure_fails_production(make_state_fit) -> None:
    fit, draws = _inputs(make_state_fit)
    broken = DrawCheck(
        draws_checked=CHAINS * PER_CHAIN,
        months_checked=1,
        cells_checked=4,
        max_anchor_drift=1.0,
        bound_violations=2,
        tolerance=1e-9,
    )
    report = evaluate_gate(fit, draws, broken, THRESHOLDS, scope="production")
    assert report.failures == (
        "reconciliation_anchor_drift_max",
        "reconciliation_bound_violations",
    )


def test_the_report_is_json_without_nan(make_state_fit) -> None:
    fit, draws = _inputs(make_state_fit, ppc=math.nan)
    report = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="production")
    payload = json.loads(json.dumps(report.to_json(), allow_nan=False))
    assert payload["checks"]["ppc_coverage_90"]["value"] is None
    assert payload["passed"] is False


def test_a_failed_gate_raises_naming_each_failed_check(make_state_fit) -> None:
    fit, draws = _inputs(make_state_fit, divergences=2)
    report = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="production")
    with pytest.raises(ModelDiagnosticsError, match="divergences=2.0"):
        assert_gate_passes(report)


def test_an_unknown_scope_is_refused(make_state_fit) -> None:
    fit, draws = _inputs(make_state_fit)
    with pytest.raises(ConceptViolationError, match="scope"):
        evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="exploratory")
