"""`fit-state-model`, and `reconcile`'s check of the stored draws, on the committed fixture.

§17.4 row 6, "reconcile posterior draws", through the CLI; row 5's fit "on synthetic data" is
`test_state_total_recovery.py`'s. A two-chain sampler of 60 draws and a gate loosened to pass
whatever it measures: these tests are about what the commands write, refuse and reproduce, not about
convergence, which the recovery test and the D1 run own.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from logging_employment.cli import app
from logging_employment.models.arviz_io import draws_digest, read_store
from logging_employment.models.interfaces import STORE_PATH
from logging_employment.models.reconciliation import check_reconciled

REPO = Path(__file__).resolve().parents[2]
FIXTURE_MONTHLY = REPO / "tests" / "fixtures" / "baselines" / "qcew_monthly.parquet"
SMALL_SAMPLER = {"chains": 2, "warmup": 60, "draws": 60}
LOOSE_GATE = {
    "diagnostics": {
        "max_rhat": 100.0,
        "min_ess_per_chain": 1,
        "max_divergences": 1_000_000,
        "min_ppc_coverage_90": 0.0,
    }
}
# No R-hat is below 0.5, so this gate fails every fit, deterministically.
FAILING_GATE = {"diagnostics": {"max_rhat": 0.5}}


def _invoke(command: str, config: Path):
    return CliRunner().invoke(app, [command, "--config", str(config)])


@pytest.fixture()
def fitted(make_staged_repo):
    repo = make_staged_repo({"model": {**SMALL_SAMPLER, **LOOSE_GATE}})
    result = _invoke("fit-state-model", repo.config_path)
    assert result.exit_code == 0, result.output
    return repo


@pytest.mark.slow
def test_the_fit_writes_its_store_its_summary_and_its_manifest(fitted) -> None:
    run = fitted.run_dir
    states = pl.read_parquet(FIXTURE_MONTHLY).filter(pl.col("area_type") == "state")
    summary = pl.read_parquet(run / "posterior_summary.parquet")
    assert summary.height == states.height
    imputed = int((summary["observed_or_imputed"] == "imputed").sum())
    assert imputed == int((states["observation_status"] == "suppressed").sum())
    manifest = json.loads((run / "state_model_manifest.json").read_text())
    written = hashlib.sha256((run / "posterior_summary.parquet").read_bytes()).hexdigest()
    assert manifest["posterior_summary_sha256"] == written
    assert manifest["draws_sha256"] == draws_digest(read_store(run / STORE_PATH))
    assert manifest["anchor_bases"] == ["declared_national_total"]
    assert manifest["code_commit"]
    report = json.loads((run / "posterior" / "diagnostics.json").read_text())
    assert (report["scope"], report["passed"]) == ("production", True)


@pytest.mark.slow
def test_every_stored_draw_satisfies_its_hard_constraints(fitted) -> None:
    """INV-012 on the persisted draws, after transformation, never on a summary of them."""
    check = check_reconciled(read_store(fitted.run_dir / STORE_PATH), tolerance=1e-9)
    assert check.passed
    assert check.draws_checked == 120


@pytest.mark.slow
def test_the_fit_is_idempotent(fitted) -> None:
    """§16.1: same inputs, same draws, same summary bytes."""
    manifest = fitted.run_dir / "state_model_manifest.json"
    first = json.loads(manifest.read_text())
    result = _invoke("fit-state-model", fitted.config_path)
    assert result.exit_code == 0, result.output
    second = json.loads(manifest.read_text())
    for key in ("draws_sha256", "posterior_summary_sha256"):
        assert second[key] == first[key]


@pytest.mark.slow
def test_reconcile_re_verifies_the_draws_on_disk(fitted) -> None:
    for command in ("run-baselines", "reconcile"):
        result = _invoke(command, fitted.config_path)
        assert result.exit_code == 0, result.output
    state_model = json.loads((fitted.run_dir / "reconcile_manifest.json").read_text())[
        "state_model"
    ]
    assert state_model["passed"] is True
    assert state_model["draws_sha256_matches"] is True
    assert state_model["bound_violations"] == 0


@pytest.mark.slow
def test_a_failed_gate_exits_1_and_leaves_only_its_report(make_staged_repo) -> None:
    """Stale artifacts are planted first: a failed re-fit must not leave an old success behind."""
    repo = make_staged_repo({"model": {**SMALL_SAMPLER, **FAILING_GATE}})
    run = repo.run_dir
    stale = (run / STORE_PATH, run / "posterior_summary.parquet", run / "state_model_manifest.json")
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("from an earlier fit")
    result = _invoke("fit-state-model", repo.config_path)
    assert result.exit_code == 1
    report = json.loads((run / "posterior" / "diagnostics.json").read_text())
    assert report["passed"] is False
    assert "parameter_rhat_max" in report["failures"]
    assert not any(path.exists() for path in stale)
    # What `reconcile` and `validate-state-model` compare with the run's before trusting the fit.
    manifest = json.loads((run / "schema_manifest.json").read_text())
    assert report["constraint_set_hash"] == manifest["constraint_set_hash"]


def test_the_fit_requires_solved_bounds(make_staged_repo) -> None:
    repo = make_staged_repo({"model": SMALL_SAMPLER})
    (repo.run_dir / "deterministic_bounds.parquet").unlink()
    result = _invoke("fit-state-model", repo.config_path)
    assert result.exit_code != 0
    # Short tokens only, as `test_baseline_cli.py` explains: Typer boxes and hard-wraps the message.
    assert "solve-bounds" in result.output
    assert "INV-012" in result.output


def test_bounds_from_another_constraint_set_are_refused_before_anything_is_deleted(
    make_staged_repo,
) -> None:
    """`run_id` does not cover code, so `build-constraints` can re-run under the same id after a
    code change and leave `deterministic_bounds.parquet` solved against the old constraint set.
    `constraint_set_hash` is the cross-stage check that catches it, and it runs first: an earlier
    fit's artifacts must survive a refusal."""
    repo = make_staged_repo({"model": SMALL_SAMPLER})
    bounds_path = repo.run_dir / "deterministic_bounds.parquet"
    stale = pl.read_parquet(bounds_path).with_columns(constraint_set_hash=pl.lit("stale0hash"))
    stale.write_parquet(bounds_path)
    earlier = repo.run_dir / "state_model_manifest.json"
    earlier.write_text("from an earlier fit")
    result = _invoke("fit-state-model", repo.config_path)
    assert result.exit_code != 0
    assert "stale0hash" in result.output
    assert "solve-bounds" in result.output
    assert earlier.read_text() == "from an earlier fit"


def test_a_baseline_only_reconcile_writes_the_keys_it_always_wrote(staged_repo) -> None:
    """No store, no `state_model` key: omitted rather than null, as `run_id`'s optional keys are."""
    for command in ("run-baselines", "reconcile"):
        result = _invoke(command, staged_repo.config_path)
        assert result.exit_code == 0, result.output
    manifest = json.loads((staged_repo.run_dir / "reconcile_manifest.json").read_text())
    assert "state_model" not in manifest


@pytest.mark.parametrize(
    ("report", "recorded"),
    [
        ({"passed": True, "constraint_set_hash": "stale0hash"}, "stale0hash"),
        ({"passed": True}, None),
        (None, None),
    ],
    ids=["another_constraint_set", "unrecorded", "no_report"],
)
def test_reconcile_fails_a_fit_from_another_constraint_set_without_reading_it(
    staged_repo, report, recorded
) -> None:
    """A fit can outlive its constraint set: `fit-state-model` keeps it when it refuses stale
    bounds, and `build-constraints` can re-run without a re-fit. Its draws were reconciled into
    bounds that no longer hold, so re-verifying them against themselves would pass. An unrecorded
    hash matches nothing. The planted store is not netCDF: reading it would crash, not record."""
    result = _invoke("run-baselines", staged_repo.config_path)
    assert result.exit_code == 0, result.output
    run = staged_repo.run_dir
    store = run / STORE_PATH
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("from a fit against another constraint set")
    if report is not None:
        (run / "posterior" / "diagnostics.json").write_text(json.dumps(report))
    result = _invoke("reconcile", staged_repo.config_path)
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), result.exception
    assert "fit-state-model" in result.output
    manifest = json.loads((run / "reconcile_manifest.json").read_text())
    current = json.loads((run / "schema_manifest.json").read_text())["constraint_set_hash"]
    assert manifest["within_tolerance"] is True
    assert manifest["state_model"] == {
        "fit_constraint_set_hash": recorded,
        "constraint_set_hash": current,
        "passed": False,
    }


def test_the_cli_starts_without_a_ppl() -> None:
    """The import discipline, kept: `--help` and every baseline command start without JAX.

    `cli.py` imports each `models/` module inside the command that uses it. A subprocess, because
    this test process may already hold `jax` from another test, which would make the check vacuous.
    """
    ppl = ("jax", "numpyro", "arviz_base", "arviz_stats", "xarray", "h5netcdf")
    probe = f"import sys, logging_employment.cli; print(sorted(set({ppl!r}) & set(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
