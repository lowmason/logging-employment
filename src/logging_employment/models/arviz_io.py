"""The only module that imports ArviZ or xarray: §11.14's statistics and §15.4's store.

It is kept to one module so that an ArviZ API change is a one-file change, and so
`tests/unit/test_arviz_api_probe.py` can pin every call this package makes against the installed
versions. `arviz_stats.rhat` and `arviz_stats.ess` take an array shaped `(chain, draw, ...)` with
`chain_axis=0, draw_axis=1`. Plan 16 read that in the arviz-stats v1.3.3 source and ran it on
1.3.2, which is how it found that tail ESS needs its quantiles passed (`TAIL_PROBABILITIES`).
`method="rank"` is Vehtari et al.'s (2021) rank-normalized, folded split R-hat, the one §11.14's
1.01 threshold is calibrated for.

THE STORE IS AN ARVIZ-SHAPED `DataTree` IN NETCDF (§15.4: "Joint draws SHOULD be stored in an
ArviZ-compatible format ... The storage must preserve draw, chain, state, month, and size
indexes"). It has four groups:

* `posterior`: the monitored parameters, `(chain, draw, ...)`;
* `posterior_predictive`: `reconciled_state_total`, `(chain, draw, cell)`, the reconciled joint
  draws §12.7 summarises. `cell` carries `state_fips` and `reference_month` coordinates, which
  are §15.4's state and month indexes. There is no size index until Stage 6;
* `sample_stats`: `diverging`, `(chain, draw)`;
* `constant_data`: each month's residual and anchor basis, and each cell's deterministic bounds
  and mean raw score. With these, `reconcile` can re-verify INV-012 from the file alone.

Raw scores are NOT stored draw by draw. §12.7 summarises reconciled draws only, and 39 MB of D1
raw draws would buy an audit that `raw_mean` and the parameter trace already support.

WHAT IS DIGESTED IS THE DRAWS, NOT THE FILE. `draws_digest` hashes the reconciled array's float64
bytes, its shape and its cell ids. The netCDF bytes also carry HDF5 and ArviZ metadata that no one
here promises to hold still, so an idempotence check compares digests, never file hashes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xarray as xr
from arviz_stats import ess, rhat

from ..errors import ConceptViolationError
from ..reconcile.anchor import Anchor
from .interfaces import ModelData, StateModelFit
from .reconciliation import ReconciledDraws

STORE_ENGINE = "h5netcdf"
# The named dimension each vector parameter carries after (chain, draw).
PARAMETER_DIMS: dict[str, tuple[str, ...]] = {
    "u": ("state",),
    "rho": ("state",),
    "sigma_eta_state": ("state",),
    "v": ("division",),
    "gamma": ("month_of_year",),
    "delta": ("year",),
}


def rank_rhat(samples: np.ndarray) -> np.ndarray:
    """Rank-normalized split R-hat for every trailing element of a `(chain, draw, ...)` array."""
    return np.asarray(
        rhat(np.asarray(samples, dtype=np.float64), method="rank", chain_axis=0, draw_axis=1),
        dtype=np.float64,
    )


# Vehtari et al. (2021)'s tail ESS is the smaller of the 5% and 95% quantiles' ESS. The array
# interface REQUIRES `prob` for `method="tail"` and raises TypeError without it (run on
# arviz-stats 1.3.2 while plan 16 was written); the xarray interface would default it from the
# global `rcParams["stats.ci_prob"]`, which a gate should not depend on.
TAIL_PROBABILITIES = (0.05, 0.95)


def ess_bulk_tail(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bulk and tail effective sample size for every trailing element of `(chain, draw, ...)`."""
    values = np.asarray(samples, dtype=np.float64)
    bulk = ess(values, method="bulk", chain_axis=0, draw_axis=1)
    tail = ess(values, method="tail", prob=TAIL_PROBABILITIES, chain_axis=0, draw_axis=1)
    return np.asarray(bulk, dtype=np.float64), np.asarray(tail, dtype=np.float64)


def draws_digest(draws: ReconciledDraws) -> str:
    """The sha256 of the reconciled draws' float64 bytes, their shape and their cell ids."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps({"cells": list(draws.cell_ids), "shape": list(draws.values.shape)}).encode(
            "utf-8"
        )
    )
    digest.update(np.ascontiguousarray(draws.values, dtype=np.float64).tobytes())
    return digest.hexdigest()


def write_store(
    path: Path,
    draws: ReconciledDraws,
    fit: StateModelFit,
    data: ModelData,
    *,
    attrs: Mapping[str, str],
) -> str:
    """Write §15.4's store to `path` and return `draws_digest(draws)`.

    `attrs` go on the root group. Every value is a string, which netCDF can hold whatever the
    caller passes.
    """
    chains = fit.chains
    per_chain = draws.values.shape[0] // chains
    if draws.chain is not None and not np.array_equal(
        draws.chain, np.repeat(np.arange(chains), per_chain)
    ):
        raise ConceptViolationError(
            "reconciled draws are not in chain-major order, so the store would mislabel them"
        )
    shared = {"chain": np.arange(chains), "draw": np.arange(per_chain)}
    posterior = xr.Dataset(
        {
            name: (("chain", "draw", *PARAMETER_DIMS.get(name, ())), np.asarray(values))
            for name, values in fit.parameters.items()
        },
        coords={
            **shared,
            "state": list(data.states),
            "division": list(data.divisions),
            "month_of_year": np.arange(1, 13),
            "year": list(data.years),
        },
    )
    predictive = xr.Dataset(
        {
            "reconciled_state_total": (
                ("chain", "draw", "cell"),
                draws.values.reshape(chains, per_chain, -1),
            )
        },
        coords={
            **shared,
            "cell": list(draws.cell_ids),
            "state_fips": ("cell", list(draws.state_fips)),
            "reference_month": ("cell", list(draws.reference_months)),
        },
    )
    stats = xr.Dataset({"diverging": (("chain", "draw"), fit.diverging)}, coords=shared)
    months = sorted(draws.anchors)
    constant = xr.Dataset(
        {
            "residual": (("month",), np.array([draws.anchors[m].residual for m in months])),
            "anchor_basis": (("month",), [draws.anchors[m].anchor_basis for m in months]),
            "deterministic_lower": (("cell",), draws.lower),
            "deterministic_upper": (("cell",), draws.upper),
            "raw_mean": (("cell",), draws.raw_mean),
        },
        coords={"month": months, "cell": list(draws.cell_ids)},
    )
    digest = draws_digest(draws)
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={**dict(attrs), "draws_sha256": digest}),
            "posterior": posterior,
            "posterior_predictive": predictive,
            "sample_stats": stats,
            "constant_data": constant,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.to_netcdf(path, engine=STORE_ENGINE)
    return digest


def store_digest(path: Path) -> str:
    """The `draws_sha256` a store was written with, read from its root group's attributes."""
    tree = xr.open_datatree(path, engine=STORE_ENGINE)
    try:
        return str(tree.attrs["draws_sha256"])
    finally:
        tree.close()


def read_store(path: Path) -> ReconciledDraws:
    """The reconciled draws, bounds and anchors a store holds, read back without the fit.

    `reconcile` calls this to re-verify INV-012 against the persisted artifact rather than the
    in-memory draws `fit-state-model` checked. A defect in writing would otherwise pass unseen.
    Each `Anchor` rebuilt here checks its recorded basis against `contracts.ANCHOR_BASES` as it is
    built (`D-122`), so a store whose basis was altered on disk is refused on read.
    """
    tree = xr.open_datatree(path, engine=STORE_ENGINE)
    try:
        predictive = tree["posterior_predictive"].to_dataset().load()
        constant = tree["constant_data"].to_dataset().load()
    finally:
        tree.close()
    values = np.asarray(predictive["reconciled_state_total"].values, dtype=np.float64)
    chains, per_chain, cells = values.shape
    states = tuple(str(state) for state in predictive["state_fips"].values)
    months = tuple(str(month) for month in predictive["reference_month"].values)
    anchors = {
        str(month): Anchor(
            reference_month=str(month),
            residual=float(residual),
            missing_cells=tuple(s for s, m in zip(states, months, strict=True) if m == str(month)),
            anchor_basis=str(basis),
        )
        for month, residual, basis in zip(
            constant["month"].values,
            constant["residual"].values,
            constant["anchor_basis"].values,
            strict=True,
        )
    }
    return ReconciledDraws(
        cell_ids=tuple(str(cell) for cell in predictive["cell"].values),
        state_fips=states,
        reference_months=months,
        values=values.reshape(chains * per_chain, cells),
        chain=np.repeat(np.arange(chains), per_chain),
        draw=np.tile(np.arange(per_chain), chains),
        raw_mean=np.asarray(constant["raw_mean"].values, dtype=np.float64),
        lower=np.asarray(constant["deterministic_lower"].values, dtype=np.float64),
        upper=np.asarray(constant["deterministic_upper"].values, dtype=np.float64),
        anchors=anchors,
    )
