"""§12.7 and INV-012: every draw of a fit reconciled into the feasible set, one month at a time.

`reconcile_draws` (Stage 3) takes ONE month. Its anchor's missing cells are bare `state_fips`, and
its `Bounds` are keyed the same way. A fit's scores cover every suppressed cell in the window, so
this module slices them by month and orders each slice as the month's anchor orders its missing set.
It projects the seven-field bounds to that month (`baselines.runner.month_bounds`) and calls
`reconcile_draws`. Stage 3's roadmap promise is "a single reconciliation entry point every model
draw passes through", and this is that path: no draw is allocated any other way.

THE ANCHOR IS THE ONE EVERY BASELINE ALLOCATES TO. `reconcile/anchor.py::national_residual` builds
it over the frame's own partition, stamped `anchor_basis = 'declared_national_total'`, and the
establishment-closure gate (`closure_audit` + `assert_universe_closes`) must admit it for the whole
window before any month is used. It is a `modeling_assumption` (INV-004), never a constraint row,
and `DrawCheck` reports adding-up to it separately from the hard bounds.

Nothing here summarises. §12.7: "Posterior summaries must be computed from reconciled joint
draws", and `models/summary.py` is where that happens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..baselines.runner import assert_bounds_cover, missing_cell_ids, month_bounds
from ..config import Config
from ..errors import ConceptViolationError
from ..reconcile.anchor import (
    Anchor,
    assert_universe_closes,
    closure_audit,
    national_residual,
    observed_partition,
)
from ..reconcile.draws import PosteriorDraws, ReconciliationInputs, reconcile_draws
from ..reconcile.scaling import Bounds
from .interfaces import MODEL_ID, StateModelFit


def exact_column_means(values: np.ndarray) -> np.ndarray:
    """Each cell's mean over the draw axis, every column summed with `math.fsum`.

    `ndarray.mean(axis=0)` adds a column in an order set by the array's memory layout, so a cell's
    mean can differ in the last place from the same column's own `mean()` (seen while this module
    was written: 41.20440509225164 against 41.20440509225163). `posterior_summary` and the harness
    persist these means into tables whose digests a manifest records, so a mean has to depend on
    the values alone. The same reason is why `validate/metrics.py` reduces through `_exact_sum`.
    """
    if values.shape[0] == 0:
        raise ConceptViolationError("a posterior mean over zero draws is undefined")
    count = values.shape[0]
    return np.array([math.fsum(column) / count for column in values.T.tolist()], dtype=np.float64)


@dataclass(frozen=True)
class ReconciledDraws:
    """A fit's reconciled joint draws over its prediction cells, and what each month was held to.

    `values` is (draws, cells), in `cell_ids` order, which is the fit's own `raw_scores.cell_ids`
    order. `raw_mean` is each cell's posterior-mean §11.5 score, kept for `baseline_results`'s
    `raw_weight` column. `lower` and `upper` are the deterministic bounds each draw was reconciled
    into, with `upper` reading +inf where `selected_upper` is null. `anchors` is keyed by month.
    """

    cell_ids: tuple[str, ...]
    state_fips: tuple[str, ...]
    reference_months: tuple[str, ...]
    values: np.ndarray
    chain: np.ndarray | None
    draw: np.ndarray | None
    raw_mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    anchors: dict[str, Anchor]


@dataclass(frozen=True)
class DrawCheck:
    """INV-012 measured on the reconciled draws themselves, never on a summary of them.

    It holds two obligations in two fields. `bound_violations` counts draw-cell values outside their
    own deterministic interval: those are the HARD constraints (INV-002's per-cell half).
    `max_anchor_drift` is the largest |sum - R_t| over draws and months: adding-up to the declared
    national anchor, which is a `modeling_assumption` (INV-004) and not a hard constraint. Both must
    pass, but a reader must not take one for the other (INV-008's spirit).
    """

    draws_checked: int
    months_checked: int
    cells_checked: int
    max_anchor_drift: float
    bound_violations: int
    tolerance: float

    @property
    def passed(self) -> bool:
        """Whether every draw sums to its month's residual and stays inside each cell's interval."""
        return self.max_anchor_drift <= self.tolerance and self.bound_violations == 0


def reconcile_fit(
    fit: StateModelFit, monthly: pl.DataFrame, bounds: Bounds, config: Config
) -> ReconciledDraws:
    """Reconcile every draw of `fit`, month by month, against the frame's own anchor and bounds.

    `bounds` is keyed by the seven-field `cell_id` (`baselines.runner.state_total_bounds`). The
    harness passes the MASKED system's bounds, never the run directory's (`D-087`), exactly as it
    does for the baselines.

    The closure gate runs first, as in `run_baselines`, so the anchor is admitted for the whole
    window or not at all (§18.3). A cell the fit scored that no month's missing set contains, or a
    missing cell the fit did not score, is refused: either means the model and the anchor disagree
    about which cells are unknown.
    """
    partitions = observed_partition(monthly)
    assert_universe_closes(closure_audit(monthly, partitions))
    raw = np.asarray(fit.raw_scores.values, dtype=np.float64)
    column_of = {cell: index for index, cell in enumerate(fit.raw_scores.cell_ids)}
    reconciled = np.full_like(raw, np.nan)
    filled = np.zeros(raw.shape[1], dtype=bool)
    lower = np.full(raw.shape[1], np.nan)
    upper = np.full(raw.shape[1], np.nan)
    state_of = [""] * raw.shape[1]
    month_of = [""] * raw.shape[1]
    anchors: dict[str, Anchor] = {}
    for month in sorted(partitions):
        anchor = national_residual(monthly, partitions[month], reference_month=month)
        if not anchor.missing_cells:
            continue
        ids = missing_cell_ids(partitions[month])
        unscored = sorted(
            ids[state] for state in anchor.missing_cells if ids[state] not in column_of
        )
        if unscored:
            raise ConceptViolationError(
                f"{month}: the fit scored no draw for missing cell(s) {unscored[:5]}; every cell "
                "the anchor allocates to needs a score, or the residual lands on the others"
            )
        columns = [column_of[ids[state]] for state in anchor.missing_cells]
        assert_bounds_cover(
            dict.fromkeys(anchor.missing_cells, 0.0),
            bounds,
            cell_ids=ids,
            estimator_id=MODEL_ID,
            reference_month=month,
        )
        scoped = month_bounds(bounds, cell_ids=ids, cells=anchor.missing_cells)
        out = reconcile_draws(
            PosteriorDraws(
                cell_ids=anchor.missing_cells,
                values=raw[:, columns],
                chain=fit.raw_scores.chain,
                draw=fit.raw_scores.draw,
            ),
            ReconciliationInputs(anchor=anchor, bounds=scoped),
            config.reconciliation,
        )
        reconciled[:, columns] = out.values
        filled[columns] = True
        for state, column in zip(anchor.missing_cells, columns, strict=True):
            lower[column] = scoped.lower[state]
            upper[column] = scoped.upper_of(state)
            state_of[column] = state
            month_of[column] = month
        anchors[month] = anchor
    if not filled.all():
        orphans = [fit.raw_scores.cell_ids[index] for index in np.flatnonzero(~filled)]
        raise ConceptViolationError(
            f"{len(orphans)} scored cell(s) sit in no month's missing set, first {orphans[:5]}; "
            "a score with no anchor to reconcile against is an estimate §12 never checked"
        )
    return ReconciledDraws(
        cell_ids=tuple(fit.raw_scores.cell_ids),
        state_fips=tuple(state_of),
        reference_months=tuple(month_of),
        values=reconciled,
        chain=fit.raw_scores.chain,
        draw=fit.raw_scores.draw,
        raw_mean=exact_column_means(raw),
        lower=lower,
        upper=upper,
        anchors=anchors,
    )


def check_reconciled(draws: ReconciledDraws, *, tolerance: float) -> DrawCheck:
    """INV-012 on the reconciled draws: each month sums to R_t in every draw, each value in [L, U].

    It sums with `math.fsum`, one draw at a time, so the check does not depend on the order NumPy
    reduces in. The reconciler bisects to a double's resolution, and a re-sum in another order can
    move the last bits: `cli.py::reconcile_command` measured 9.93e-10 against 1e-9 before the
    bisection was fixed.
    """
    months = np.asarray(draws.reference_months)
    worst = 0.0
    for month, anchor in sorted(draws.anchors.items()):
        block = draws.values[:, months == month]
        worst = max(worst, max(abs(math.fsum(row) - anchor.residual) for row in block.tolist()))
    outside = (draws.values < draws.lower[None, :] - tolerance) | (
        draws.values > draws.upper[None, :] + tolerance
    )
    return DrawCheck(
        draws_checked=int(draws.values.shape[0]),
        months_checked=len(draws.anchors),
        cells_checked=int(draws.values.shape[1]),
        max_anchor_drift=float(worst),
        bound_violations=int(np.count_nonzero(outside)),
        tolerance=tolerance,
    )
