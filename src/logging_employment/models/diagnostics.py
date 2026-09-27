"""§11.14's diagnostic gate, enforced in code rather than documented.

§11.14 lists what "Promotion requires": no unresolved divergent transitions, R-hat at or below
1.01 for monitored parameters, adequate ESS for all release-relevant summaries, stable posterior
summaries across independent seeds and chains, posterior predictive checks on observed cells, and
exact reconciliation checks after draw transformation. The roadmap's exit criterion is that the
gate is "enforced in code, not documented". `fit-state-model` writes this report and then raises
`ModelDiagnosticsError` on a failure, and `validate-state-model` will not score a model whose
production fit failed.

WHAT EACH CHECK READS (plan 16, Decision 5):

* `divergences`: post-warmup `diverging`, summed over chains.
* `parameter_rhat_max`: rank R-hat, max over every element of `state_total.MONITORED`.
* `cell_rhat_max`: rank R-hat over the reconciled draws of every imputed cell. This is how
  "stable posterior summaries across independent seeds/chains" is enforced. Each chain starts from
  its own key, so between-chain agreement on the released quantity is the stability §11.14 asks
  for.
* `cell_ess_bulk_min` / `cell_ess_tail_min`: min over imputed cells, against
  `min_ess_per_chain * chains`. §11.14 scopes ESS to "release-relevant summaries", and the released
  summaries are these cells' intervals. Parameter ESS is recorded and does not gate.
* `ppc_coverage_90`: `StateModelFit.ppc_coverage_90` against `min_ppc_coverage_90`. It is
  one-step-ahead coverage of the training cells (`state_total._ppc_coverage_90`), because an exact
  training cell's in-sample replicate is the cell itself.
* `reconciliation_*`: `DrawCheck`, which is INV-012 measured on the draws.

A cell whose reconciled draws never vary has no R-hat and no ESS. That happens in a month whose
missing set is one cell, pinned to R_t in every draw. It is counted in `cells_constant` and left
out, because the constraints determine it and nothing sampled it. A NaN from any other quantity
fails its check, because a comparison with NaN is false and an unverifiable check must not pass.

TWO SCOPES. `production` gates on every check above. `replicate` covers the 27 fits §13's harness
runs, which record everything but gate only on divergences and parameter R-hat. Over about 1,400
quantities and 27 fits, a max-statistic gate would let one cell at R-hat 1.011 in one replicate
decide promotion. `validate/promotion.py` reads the replicate reports for §13.10's convergence
gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import StateModelDiagnostics
from ..errors import ConceptViolationError, ModelDiagnosticsError
from .arviz_io import ess_bulk_tail, rank_rhat
from .interfaces import StateModelFit
from .reconciliation import DrawCheck, ReconciledDraws

SCOPES: tuple[str, ...] = ("production", "replicate")


def _plain(value: float) -> float | None:
    """A float JSON can carry: NaN and the infinities become None rather than a non-JSON token."""
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class GateCheck:
    """One §11.14 check: what was measured, against what, and whether it gates this scope."""

    name: str
    value: float
    threshold: float
    passed: bool
    gating: bool


@dataclass(frozen=True)
class GateReport:
    """Every §11.14 check for one fit, and the verdict of the ones that gate its scope."""

    scope: str
    checks: tuple[GateCheck, ...]
    cells_constant: int

    @property
    def passed(self) -> bool:
        """True when every gating check passed; recorded-only checks cannot fail a fit."""
        return all(check.passed for check in self.checks if check.gating)

    @property
    def failures(self) -> tuple[str, ...]:
        """The names of the gating checks that failed, in check order."""
        return tuple(check.name for check in self.checks if check.gating and not check.passed)

    def to_json(self) -> dict[str, object]:
        """The report as `diagnostics.json` and the harness manifest record it."""
        return {
            "scope": self.scope,
            "passed": self.passed,
            "failures": list(self.failures),
            "cells_constant": self.cells_constant,
            "checks": {
                check.name: {
                    "value": _plain(float(check.value)),
                    "threshold": float(check.threshold),
                    "passed": check.passed,
                    "gating": check.gating,
                }
                for check in self.checks
            },
        }


def _parameter_statistics(fit: StateModelFit) -> tuple[float, float, float]:
    """Max rank R-hat and min bulk and tail ESS over every monitored parameter element."""
    rhats, bulks, tails = [], [], []
    for values in fit.parameters.values():
        per_element = np.asarray(values, dtype=np.float64).reshape(
            values.shape[0], values.shape[1], -1
        )
        rhats.append(rank_rhat(per_element).ravel())
        bulk, tail = ess_bulk_tail(per_element)
        bulks.append(bulk.ravel())
        tails.append(tail.ravel())
    if not rhats:
        return math.nan, math.nan, math.nan
    return (
        float(np.max(np.concatenate(rhats))),
        float(np.min(np.concatenate(bulks))),
        float(np.min(np.concatenate(tails))),
    )


def evaluate_gate(
    fit: StateModelFit,
    draws: ReconciledDraws,
    check: DrawCheck,
    thresholds: StateModelDiagnostics,
    *,
    scope: str,
) -> GateReport:
    """§11.14 for one fit and its reconciled draws, gating according to `scope`."""
    if scope not in SCOPES:
        raise ConceptViolationError(f"gate scope {scope!r} is not one of {list(SCOPES)}")
    production = scope == "production"
    chains = fit.chains
    per_chain = draws.values.shape[0] // chains
    parameter_rhat, parameter_bulk, parameter_tail = _parameter_statistics(fit)
    varying = np.ptp(draws.values, axis=0) > 0.0
    if varying.any():
        cells = draws.values[:, varying].reshape(chains, per_chain, -1)
        cell_rhat = float(np.max(rank_rhat(cells)))
        bulk, tail = ess_bulk_tail(cells)
        cell_bulk, cell_tail = float(np.min(bulk)), float(np.min(tail))
    else:
        # Every imputed cell is pinned by the constraints: nothing released was sampled.
        cell_rhat, cell_bulk, cell_tail = 1.0, math.inf, math.inf
    floor = float(thresholds.min_ess_per_chain * chains)
    checks = (
        GateCheck(
            "divergences",
            float(fit.divergences),
            float(thresholds.max_divergences),
            fit.divergences <= thresholds.max_divergences,
            True,
        ),
        GateCheck(
            "parameter_rhat_max",
            parameter_rhat,
            thresholds.max_rhat,
            parameter_rhat <= thresholds.max_rhat,
            True,
        ),
        GateCheck("parameter_ess_bulk_min", parameter_bulk, floor, parameter_bulk >= floor, False),
        GateCheck("parameter_ess_tail_min", parameter_tail, floor, parameter_tail >= floor, False),
        GateCheck(
            "cell_rhat_max",
            cell_rhat,
            thresholds.max_rhat,
            cell_rhat <= thresholds.max_rhat,
            production,
        ),
        GateCheck("cell_ess_bulk_min", cell_bulk, floor, cell_bulk >= floor, production),
        GateCheck("cell_ess_tail_min", cell_tail, floor, cell_tail >= floor, production),
        GateCheck(
            "ppc_coverage_90",
            fit.ppc_coverage_90,
            thresholds.min_ppc_coverage_90,
            fit.ppc_coverage_90 >= thresholds.min_ppc_coverage_90,
            production,
        ),
        GateCheck(
            "reconciliation_anchor_drift_max",
            check.max_anchor_drift,
            check.tolerance,
            check.max_anchor_drift <= check.tolerance,
            production,
        ),
        GateCheck(
            "reconciliation_bound_violations",
            float(check.bound_violations),
            0.0,
            check.bound_violations == 0,
            production,
        ),
    )
    return GateReport(scope=scope, checks=checks, cells_constant=int(np.count_nonzero(~varying)))


def assert_gate_passes(report: GateReport) -> None:
    """Raise `ModelDiagnosticsError` naming every gating check that failed."""
    if not report.passed:
        detail = "; ".join(
            f"{check.name}={check.value} against {check.threshold}"
            for check in report.checks
            if check.gating and not check.passed
        )
        raise ModelDiagnosticsError(f"§11.14's {report.scope} gate failed: {detail}")
