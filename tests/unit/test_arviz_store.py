"""§15.4's store: the reconciled joint draws in ArviZ's layout, read back exactly."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from logging_employment.errors import ConceptViolationError
from logging_employment.models.arviz_io import draws_digest, read_store, store_digest, write_store
from logging_employment.models.data import build_model_data
from logging_employment.models.reconciliation import reconcile_fit
from logging_employment.reconcile.scaling import Bounds

SEED = sum(map(ord, "tests/arviz-store"))


@pytest.fixture()
def stored(tmp_path: Path, harmonized_toy, appendix_a_config, make_state_fit):
    """The toy's three suppressed cells over two chains of five draws, written to a store."""
    monthly = harmonized_toy.qcew_monthly
    data = build_model_data(monthly)
    cells = data.predict_cell_ids
    rng = np.random.default_rng(SEED)
    parameters = {
        "alpha": rng.normal(size=(2, 5)),
        "u": rng.normal(size=(2, 5, len(data.states))),
        "gamma": rng.normal(size=(2, 5, 12)),
    }
    fit = make_state_fit(rng.gamma(2.0, 20.0, size=(10, len(cells))), cells, parameters=parameters)
    bounds = Bounds(lower=dict.fromkeys(cells, 0.0), upper=dict.fromkeys(cells))
    draws = reconcile_fit(fit, monthly, bounds, appendix_a_config)
    path = tmp_path / "posterior" / "state_total_draws.nc"
    digest = write_store(path, draws, fit, data, attrs={"run_id": "run0"})
    return path, digest, draws


def test_the_store_reads_back_the_reconciled_draws_exactly(stored) -> None:
    path, _digest, draws = stored
    back = read_store(path)
    np.testing.assert_array_equal(back.values, draws.values)
    assert back.cell_ids == draws.cell_ids
    assert back.state_fips == draws.state_fips
    assert back.reference_months == draws.reference_months
    np.testing.assert_array_equal(back.lower, draws.lower)
    np.testing.assert_array_equal(back.upper, draws.upper)
    assert np.isinf(back.upper).all()
    assert back.anchors == draws.anchors


def test_the_digest_covers_the_draws_not_the_file(stored) -> None:
    path, digest, draws = stored
    assert digest == draws_digest(read_store(path)) == store_digest(path)
    moved = replace(draws, values=draws.values.copy())
    moved.values[0, 0] += 1e-9
    assert draws_digest(moved) != digest


def test_the_store_keeps_chain_draw_state_and_month_indexes(stored) -> None:
    """§15.4: "The storage must preserve draw, chain, state, and month indexes"."""
    path, _digest, _draws = stored
    tree = xr.open_datatree(path, engine="h5netcdf")
    try:
        predictive = tree["posterior_predictive"].to_dataset()
        assert predictive["reconciled_state_total"].dims == ("chain", "draw", "cell")
        assert {"state_fips", "reference_month"} <= set(predictive.coords)
        assert tree["posterior"].to_dataset()["u"].dims == ("chain", "draw", "state")
        assert tree["sample_stats"].to_dataset()["diverging"].dims == ("chain", "draw")
    finally:
        tree.close()


def test_draws_out_of_chain_major_order_are_refused(stored, harmonized_toy, make_state_fit) -> None:
    path, _digest, draws = stored
    data = build_model_data(harmonized_toy.qcew_monthly)
    fit = make_state_fit(draws.values, draws.cell_ids)
    with pytest.raises(ConceptViolationError, match="chain-major"):
        write_store(
            path.with_name("other.nc"),
            replace(draws, chain=draws.chain[::-1].copy()),
            fit,
            data,
            attrs={},
        )
