"""§11's state-total model as a `validate.harness.Producer`, so §13 scores it like a baseline.

ONE LOOP SCORES BOTH. `run_pseudo_suppression(data, config=config, producer=StateModelProducer())`
runs the model through the masks, the leakage guard, the masked bounds (`D-087`), §13.2 step 6's
rejection, the primary-like precondition and every §13.5-§13.8 emitter that scored the Stage 4
comparand. The only model-specific parts are the ones below: how a replicate's rows are produced,
and where their intervals come from.

PER REPLICATE (§13.2 steps 3-5 on the model's side):

1. `build_model_data(masked.qcew_monthly)`. The hidden targets are `suppressed` with a null value
   in the masked frame, so the model predicts them and never trains on them (§11.3, §13.4).
2. `fit_state_total_model`, with the production config's sampler settings and seed.
3. `reconcile_fit` against the MASKED system's bounds, then `check_reconciled`.
4. `evaluate_gate(..., scope="replicate")`. The report rides into the manifest's `producer_notes`
   for `validate/promotion.py` and fails nothing here (plan 16, Decision 5).

THE POINT ESTIMATE IS THE MEAN OF THE RECONCILED DRAWS, not their median. Every draw sums to its
month's residual and lies inside its cell's interval, so their mean does too, by linearity and
convexity. `constraint_metrics` then reads the model's adding-up at machine epsilon, as it reads
every baseline's. A per-cell median has neither property, and its adding-up residual would be
scored as the model violating a constraint it never broke. `posterior_summary` still publishes
both (§7.11).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import polars as pl

from ..baselines.runner import assert_within_bounds, release_integers, state_total_bounds
from ..config import Config
from ..contracts import BASELINE_RESULT_SCHEMA, HarmonizedData, assert_declared_provenance
from ..reconcile.scaling import Bounds
from ..validate.harness import Production
from ..validate.metrics import draw_interval_metrics
from ..validate.recover import MaskedSystem
from .arviz_io import draws_digest
from .data import build_model_data
from .diagnostics import evaluate_gate
from .interfaces import MODEL_ID, StateModelConfig
from .reconciliation import ReconciledDraws, check_reconciled, exact_column_means, reconcile_fit
from .state_total import fit_state_total_model


def model_results(
    draws: ReconciledDraws,
    bounds: Bounds,
    config: Config,
    *,
    constraint_set_hash: str | None = None,
) -> pl.DataFrame:
    """`BASELINE_RESULT_SCHEMA` rows for every reconciled cell, estimate = the draws' mean.

    `raw_weight` is the posterior-mean §11.5 score, the model's analogue of a baseline's raw
    weight. The integers come from `baselines.runner.release_integers`, the baselines' own cut, so
    both sources release integers by one rule. `constraint_set_hash` defaults to None because the
    harness path's `run_baselines` writes None there too (`contracts.VALIDATION_SCORE_SCHEMA`).
    """
    means = exact_column_means(draws.values)
    months = np.asarray(draws.reference_months)
    rows: list[dict[str, object]] = []
    for month, anchor in sorted(draws.anchors.items()):
        columns = np.flatnonzero(months == month)
        ids = {draws.state_fips[j]: draws.cell_ids[j] for j in columns}
        estimates = {draws.state_fips[j]: float(means[j]) for j in columns}
        assert_within_bounds(
            estimates,
            bounds,
            cell_ids=ids,
            estimator_id=MODEL_ID,
            reference_month=month,
            tolerance=config.reconciliation.tolerance,
            quantity="estimate",
        )
        integers = release_integers(
            estimates, anchor, bounds, cell_ids=ids, estimator_id=MODEL_ID, config=config
        )
        for j in columns:
            state = draws.state_fips[j]
            rows.append(
                {
                    "estimator_id": MODEL_ID,
                    "cell_id": draws.cell_ids[j],
                    "state_fips": state,
                    "reference_month": month,
                    "raw_weight": float(draws.raw_mean[j]),
                    "estimate": estimates[state],
                    "estimate_integer": integers[state],
                    "weight_basis": "own_estimator",
                    "anchor_basis": anchor.anchor_basis,
                    "reconciliation_status": "anchored_and_reconciled",
                    "decline_reason": None,
                    "decline_kind": None,
                    "residual": anchor.residual,
                    "missing_set_size": len(anchor.missing_cells),
                    "constraint_set_hash": constraint_set_hash,
                }
            )
    results = pl.DataFrame(rows, schema=BASELINE_RESULT_SCHEMA)
    assert_declared_provenance(results)
    return results


def draw_ensembles(draws: ReconciledDraws) -> dict[str, np.ndarray]:
    """Each reconciled cell's draws, keyed by `cell_id`, for `draw_interval_metrics`."""
    return {cell: draws.values[:, j] for j, cell in enumerate(draws.cell_ids)}


@dataclass(frozen=True)
class StateModelProducer:
    """The harness's `Producer` for §11's model: fit, reconcile, gate, and hand back rows."""

    estimator_ids: tuple[str, ...] = (MODEL_ID,)

    def __call__(self, masked: HarmonizedData, system: MaskedSystem, config: Config) -> Production:
        """One replicate's fit on the masked frame, reconciled into the MASKED system's bounds."""
        monthly = masked.qcew_monthly
        fit = fit_state_total_model(
            build_model_data(monthly), StateModelConfig.from_config(config.model)
        )
        bounds = state_total_bounds(system.bounds)
        draws = reconcile_fit(fit, monthly, bounds, config)
        check = check_reconciled(draws, tolerance=config.reconciliation.tolerance)
        report = evaluate_gate(fit, draws, check, config.model.diagnostics, scope="replicate")
        return Production(
            results=model_results(draws, bounds, config),
            interval_metrics=partial(draw_interval_metrics, ensembles=draw_ensembles(draws)),
            notes={"gate": report.to_json(), "draws_sha256": draws_digest(draws)},
        )
