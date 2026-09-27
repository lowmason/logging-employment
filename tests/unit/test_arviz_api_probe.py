"""Every ArviZ and xarray call `models/arviz_io.py` makes, run once against the installed versions.

`arviz_io` is the only module that imports either library, so an API that moved fails here and not
partway through a D1 fit. Plan 16 verified these calls by reading arviz-stats v1.3.3's source; this
module is where they are first run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from arviz_stats import ess, rhat

SEED = sum(map(ord, "tests/arviz-api-probe"))


def _chains(offset: float = 0.0) -> np.ndarray:
    """Four chains of 500 iid normal draws of three elements; chain 3 is shifted by `offset`."""
    values = np.random.default_rng(SEED).normal(size=(4, 500, 3))
    values[3] += offset
    return values


def test_rank_rhat_takes_chain_and_draw_axes_and_sees_a_stuck_chain() -> None:
    mixed = np.asarray(rhat(_chains(), method="rank", chain_axis=0, draw_axis=1))
    stuck = np.asarray(rhat(_chains(3.0), method="rank", chain_axis=0, draw_axis=1))
    assert mixed.shape == (3,)
    assert np.all(mixed < 1.01)
    assert np.all(stuck > 1.1)


def test_bulk_and_tail_ess_take_the_same_axes() -> None:
    """Tail ESS needs its two quantiles spelled out: the array interface has no default for them."""
    bulk = np.asarray(ess(_chains(), method="bulk", chain_axis=0, draw_axis=1))
    tail = np.asarray(ess(_chains(), method="tail", prob=(0.05, 0.95), chain_axis=0, draw_axis=1))
    for values in (bulk, tail):
        assert values.shape == (3,)
        # 2,000 independent draws: anything near the 400 floor would mean the axes were misread.
        assert np.all(values > 1000)


def test_a_datatree_round_trips_through_h5netcdf_bit_for_bit(tmp_path: Path) -> None:
    draws = np.random.default_rng(SEED).normal(size=(2, 5, 3))
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"draws_sha256": "abc"}),
            "posterior_predictive": xr.Dataset(
                {"reconciled_state_total": (("chain", "draw", "cell"), draws)},
                coords={
                    "chain": np.arange(2),
                    "draw": np.arange(5),
                    "cell": ["a", "b", "c"],
                    "state_fips": ("cell", ["01", "02", "04"]),
                },
            ),
        }
    )
    path = tmp_path / "store.nc"
    tree.to_netcdf(path, engine="h5netcdf")
    back = xr.open_datatree(path, engine="h5netcdf")
    try:
        read = back["posterior_predictive"].to_dataset().load()
        assert back.attrs["draws_sha256"] == "abc"
    finally:
        back.close()
    assert read["reconciled_state_total"].dtype == np.float64
    np.testing.assert_array_equal(read["reconciled_state_total"].values, draws)
    assert [str(value) for value in read["state_fips"].values] == ["01", "02", "04"]
