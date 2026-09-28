"""Shared setup for the Stage 3 CLI integration tests.

`staged_repo` gives a command a complete, self-contained repository to run against: a config whose
storage roots point inside `tmp_path`, a staged layer copied from the COMMITTED fixture under
`tests/fixtures/baselines/`, and completed `build-constraints` and `solve-bounds` runs, so both
`schema_manifest.json` and `deterministic_bounds.parquet` exist — `run-baselines` gates on both.

Copying a committed fixture rather than slicing `data/staged` at test time is the point. `data/`
is gitignored, so a fixture derived from it at run time is pinned to untracked, rebuildable data
and is not frozen in any useful sense.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from logging_employment.cli import _input_digests, app
from logging_employment.config import load_config
from logging_employment.models.interfaces import STORE_PATH
from logging_employment.runs import run_dir, run_id

REPO = Path(__file__).resolve().parents[2]
BASELINE_FIXTURES = REPO / "tests" / "fixtures" / "baselines"
STAGED_TABLES = (
    "qcew_monthly",
    "qcew_national_size",
    "cbp_state_size",
    "bridge",
    "qcew_state_parent",
)


@dataclass(frozen=True)
class StagedRepo:
    """A throwaway repository: where its config lives, and where its run outputs land."""

    config_path: Path
    run_dir: Path


def build_staged_repo(
    tmp_path: Path, *, overrides: Mapping[str, Mapping[str, object]] | None = None
) -> StagedRepo:
    """A tmp repo holding the frozen Stage 3 staged layer and a built constraint system.

    `overrides` merges into the shipped config one block at a time, and one level deeper for a
    nested mapping, so a test can shrink the sampler (`{"model": {"draws": 60}}`) without restating
    a block. Plan 16 made this a function so its model tests can ask for a small sampler and a
    loose gate. `staged_repo` is the no-override call every earlier test makes.
    """
    staged = tmp_path / "staged"
    staged.mkdir()
    for name in STAGED_TABLES:
        pl.read_parquet(BASELINE_FIXTURES / f"{name}.parquet").write_parquet(
            staged / f"{name}.parquet"
        )
    raw = yaml.safe_load((REPO / "config.yaml").read_text())
    raw["storage"]["staged_uri"] = str(staged)
    raw["storage"]["raw_uri"] = str(tmp_path / "raw")
    raw["storage"]["output_uri"] = str(tmp_path / "runs")
    raw["storage"]["constraints_uri"] = str(tmp_path / "constraints")
    for block, values in (overrides or {}).items():
        for key, value in values.items():
            if isinstance(value, Mapping):
                raw[block][key] = {**raw[block][key], **value}
            else:
                raw[block][key] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    result = CliRunner().invoke(app, ["build-constraints", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    # `solve-bounds` runs here too, as of R-S5P-3: `run-baselines` now checks every estimate
    # against `deterministic_bounds.parquet` (INV-002's per-cell half) and gates on the file.
    result = CliRunner().invoke(app, ["solve-bounds", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    cfg = load_config(config_path)
    return StagedRepo(
        config_path=config_path, run_dir=run_dir(cfg, run_id(cfg, _input_digests(cfg)))
    )


@pytest.fixture()
def staged_repo(tmp_path: Path) -> StagedRepo:
    """`build_staged_repo` with the shipped config unchanged."""
    return build_staged_repo(tmp_path)


@pytest.fixture(scope="module")
def make_staged_repo(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., StagedRepo]:
    """`build_staged_repo` in a fresh directory per call, for tests that override the config.

    A fixture rather than an import, so a test module never imports a conftest by path. It is
    module-scoped so a module-scoped fixture can build its repository once and share it; each call
    still gets its own directory from `tmp_path_factory`.
    """

    def _build(overrides: Mapping[str, Mapping[str, object]] | None = None) -> StagedRepo:
        return build_staged_repo(tmp_path_factory.mktemp("repo"), overrides=overrides)

    return _build


@pytest.fixture()
def plant_finished_fit() -> Callable[..., None]:
    """Plant what a passing `fit-state-model` leaves in a run, whole or with one artifact damaged.

    Each artifact is the least that `cli.py::_unfinished_fit` accepts, not a real fit: a passing
    report naming `constraint_set_hash`, a store holding only its `draws_sha256` root attribute, a
    few bytes of summary, and a manifest recording both digests. So a command that got past the
    check would crash reading the store's draws, not pass, and a test that damages one artifact
    isolates the refusal to it. `damage` is one of:

    - `manifest_cut_short`: the manifest's first half, as a full disk left it before `os.replace`;
    - `manifest_is_a_directory`: a directory where the manifest was. It exists, and reading it
      raises `OSError`, as a permission or I/O error would. A directory rather than `chmod 000`,
      which root can still read;
    - `store_cut_short`: the store without its last eight bytes;
    - `store_from_another_fit`: a whole store whose digest the manifest does not record;
    - `summary_rewritten`: other bytes in `posterior_summary.parquet`;
    - `summary_is_a_directory`: a directory where the summary was, as for the manifest.

    xarray is imported inside, so an integration test that plants no fit never loads it.
    """

    def _plant(run: Path, constraint_set_hash: str, *, damage: str | None = None) -> None:
        import xarray as xr

        from logging_employment.models.arviz_io import STORE_ENGINE

        def store(draws_sha256: str) -> None:
            root = xr.Dataset(attrs={"draws_sha256": draws_sha256})
            xr.DataTree.from_dict({"/": root}).to_netcdf(run / STORE_PATH, engine=STORE_ENGINE)

        (run / "posterior").mkdir(parents=True, exist_ok=True)
        report = {"passed": True, "failures": [], "constraint_set_hash": constraint_set_hash}
        (run / "posterior" / "diagnostics.json").write_text(json.dumps(report))
        store("d" * 64)
        summary = run / "posterior_summary.parquet"
        summary.write_bytes(b"a posterior summary")
        manifest = run / "state_model_manifest.json"
        recorded = {
            "draws_sha256": "d" * 64,
            "store": STORE_PATH,
            "posterior_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
        }
        manifest.write_text(json.dumps(recorded))
        if damage == "manifest_cut_short":
            manifest.write_bytes(manifest.read_bytes()[: manifest.stat().st_size // 2])
        elif damage == "manifest_is_a_directory":
            manifest.unlink()
            manifest.mkdir()
        elif damage == "store_cut_short":
            (run / STORE_PATH).write_bytes((run / STORE_PATH).read_bytes()[:-8])
        elif damage == "store_from_another_fit":
            store("e" * 64)
        elif damage == "summary_rewritten":
            summary.write_bytes(b"another posterior summary")
        elif damage == "summary_is_a_directory":
            summary.unlink()
            summary.mkdir()
        elif damage is not None:
            raise ValueError(f"no such damage: {damage!r}")

    return _plant
