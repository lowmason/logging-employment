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

from logging_employment.cli import _stale_fit, _stale_reason, _unfinished_fit, app
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
# What a passing fit writes after its report (`cli.py::fit_state_model_command`), and how far an
# interrupted one got: each case is what it had written after its passing report.
FIT_ARTIFACTS = (STORE_PATH, "posterior_summary.parquet", "state_model_manifest.json")
INTERRUPTIONS = {
    "in_the_store": (),
    "before_the_summary": (STORE_PATH,),
    "before_the_manifest": (STORE_PATH, "posterior_summary.parquet"),
    "store_deleted_later": ("posterior_summary.parquet", "state_model_manifest.json"),
}
# Every artifact present, one damaged (`conftest.py::plant_finished_fit`), and what
# `cli.py::_unfinished_fit` finds.
DAMAGE = {
    "manifest_cut_short": {"unreadable": ["state_model_manifest.json"]},
    "manifest_is_a_directory": {"unreadable": ["state_model_manifest.json"]},
    "store_cut_short": {"unreadable": [STORE_PATH]},
    "store_from_another_fit": {"mismatched": [STORE_PATH]},
    "summary_rewritten": {"mismatched": ["posterior_summary.parquet"]},
    "summary_is_a_directory": {"unreadable": ["posterior_summary.parquet"]},
}


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
    # A fit that names no constraint set was reconciled against none this run can see, and the
    # message said "reconciled against constraint set None" (#43's follow-up).
    if recorded is None:
        assert "unrecorded" in result.output
        assert "reconciled against" not in result.output
    else:
        assert recorded in result.output
    manifest = json.loads((run / "reconcile_manifest.json").read_text())
    current = json.loads((run / "schema_manifest.json").read_text())["constraint_set_hash"]
    assert manifest["within_tolerance"] is True
    assert manifest["state_model"] == {
        "fit_constraint_set_hash": recorded,
        "constraint_set_hash": current,
        "passed": False,
    }


@pytest.mark.parametrize("written", list(INTERRUPTIONS.values()), ids=list(INTERRUPTIONS))
def test_reconcile_fails_an_unfinished_fit_without_reading_it(staged_repo, written) -> None:
    """Codex on #41. `fit-state-model` writes its passing report first, so a fit interrupted after
    it leaves `"passed": true` beside artifacts that were never written. `reconcile` keyed on the
    store alone: with no store it read the run as having no fit, and with one it read the store
    whatever else was missing. It fails the fit unread instead, naming what is missing. What the
    fit did write is planted as text, so reading it would crash, not record."""
    result = _invoke("run-baselines", staged_repo.config_path)
    assert result.exit_code == 0, result.output
    run = staged_repo.run_dir
    current = json.loads((run / "schema_manifest.json").read_text())["constraint_set_hash"]
    (run / "posterior").mkdir(parents=True, exist_ok=True)
    report = {"passed": True, "constraint_set_hash": current}
    (run / "posterior" / "diagnostics.json").write_text(json.dumps(report))
    for artifact in written:
        (run / artifact).write_text("from the interrupted fit")
    result = _invoke("reconcile", staged_repo.config_path)
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), result.exception
    assert "fit-state-model" in result.output
    manifest = json.loads((run / "reconcile_manifest.json").read_text())
    assert manifest["within_tolerance"] is True
    assert manifest["state_model"] == {
        "missing": [artifact for artifact in FIT_ARTIFACTS if artifact not in written],
        "passed": False,
    }


@pytest.mark.parametrize(("damage", "found"), list(DAMAGE.items()), ids=list(DAMAGE))
def test_reconcile_fails_a_fit_its_manifest_does_not_vouch_for_without_reading_it(
    staged_repo, plant_finished_fit, damage, found
) -> None:
    """Codex on #42. An artifact's existence does not prove the fit finished writing it: a full
    disk left a manifest cut short, and it still existed. So the manifest must parse, and the
    digests it records must match the store and the summary. `reconcile` fails any other fit
    unread, naming what it found. The planted store holds only its digest, so reading its draws
    would crash, not record."""
    result = _invoke("run-baselines", staged_repo.config_path)
    assert result.exit_code == 0, result.output
    run = staged_repo.run_dir
    current = json.loads((run / "schema_manifest.json").read_text())["constraint_set_hash"]
    plant_finished_fit(run, current, damage=damage)
    result = _invoke("reconcile", staged_repo.config_path)
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), result.exception
    assert "fit-state-model" in result.output
    manifest = json.loads((run / "reconcile_manifest.json").read_text())
    assert manifest["within_tolerance"] is True
    assert manifest["state_model"] == {**found, "passed": False}


@pytest.mark.parametrize("with_store", [True, False], ids=["with_a_store", "without_a_store"])
def test_reconcile_records_a_fit_whose_report_cannot_be_read(staged_repo, with_store) -> None:
    """Codex on #43. A report that exists but cannot be read raised out of `_stale_fit` or
    `_unfinished_fit`, so `reconcile` wrote no manifest at all (§16.1). It records the fit as
    failed instead, and which check names the fault depends on which runs: with a store,
    `_stale_fit` finds no recorded constraint set, and without one, `_unfinished_fit` finds the
    report unreadable. The report is a directory, which fails to read as a permission or I/O error
    would, and does so even as root."""
    result = _invoke("run-baselines", staged_repo.config_path)
    assert result.exit_code == 0, result.output
    run = staged_repo.run_dir
    (run / "posterior" / "diagnostics.json").mkdir(parents=True)
    if with_store:
        (run / STORE_PATH).write_text("from a fit whose report cannot be read")
    result = _invoke("reconcile", staged_repo.config_path)
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), result.exception
    assert "fit-state-model" in result.output
    manifest = json.loads((run / "reconcile_manifest.json").read_text())
    current = json.loads((run / "schema_manifest.json").read_text())["constraint_set_hash"]
    found = (
        {"fit_constraint_set_hash": None, "constraint_set_hash": current}
        if with_store
        else {"unreadable": ["posterior/diagnostics.json"]}
    )
    assert manifest["within_tolerance"] is True
    assert manifest["state_model"] == {**found, "passed": False}
    if with_store:
        assert "unrecorded" in result.output


def test_a_stale_finding_never_names_a_constraint_set_nothing_recorded() -> None:
    """Two recorded sets that differ were reconciled apart. A missing one is unrecorded, whether
    the report or this run's `schema_manifest.json` is absent, unreadable or silent, and the
    remedy starts at the command that writes the missing record."""
    apart = _stale_reason({"fit_constraint_set_hash": "a1", "constraint_set_hash": "b2"})
    assert "'a1'" in apart
    assert "'b2'" in apart
    assert "solve-bounds" in apart
    fit = _stale_reason({"fit_constraint_set_hash": None, "constraint_set_hash": "b2"})
    assert "unrecorded" in fit
    assert "diagnostics.json" in fit
    assert "reconciled against" not in fit
    run = _stale_reason({"fit_constraint_set_hash": "a1", "constraint_set_hash": None})
    assert "unrecorded" in run
    assert "build-constraints" in run
    assert "reconciled against" not in run


def test_a_report_that_is_not_an_object_is_refused_by_both_checks(tmp_path) -> None:
    """A report that parses to anything but a JSON object records neither a pass nor a constraint
    set, so both checks refuse it. `.get` on the list below raised instead."""
    (tmp_path / "posterior").mkdir()
    (tmp_path / "posterior" / "diagnostics.json").write_text("[]")
    (tmp_path / "schema_manifest.json").write_text(json.dumps({"constraint_set_hash": "h"}))
    assert _stale_fit(tmp_path) == {"fit_constraint_set_hash": None, "constraint_set_hash": "h"}
    assert _unfinished_fit(tmp_path) == {"unreadable": ["posterior/diagnostics.json"]}


def test_a_passing_fit_whose_artifacts_agree_is_finished(tmp_path, plant_finished_fit) -> None:
    """The control for every damage case: the same planted fit, undamaged, is accepted, so each
    refusal above is its damage's doing. That a REAL fit's artifacts agree is the slow tests' to
    show (`test_reconcile_re_verifies_the_draws_on_disk`, and `validate-state-model`'s)."""
    plant_finished_fit(tmp_path, "any0hash")
    assert _unfinished_fit(tmp_path) is None


def test_a_failed_fits_report_alone_is_not_an_unfinished_fit(staged_repo) -> None:
    """A failed gate writes its report and nothing else, by design (`fit-state-model`'s
    docstring), so `reconcile` has no draws to re-check and writes the keys it always wrote."""
    result = _invoke("run-baselines", staged_repo.config_path)
    assert result.exit_code == 0, result.output
    run = staged_repo.run_dir
    current = json.loads((run / "schema_manifest.json").read_text())["constraint_set_hash"]
    (run / "posterior").mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "failures": ["parameter_rhat_max"], "constraint_set_hash": current}
    (run / "posterior" / "diagnostics.json").write_text(json.dumps(report))
    result = _invoke("reconcile", staged_repo.config_path)
    assert result.exit_code == 0, result.output
    manifest = json.loads((run / "reconcile_manifest.json").read_text())
    assert "state_model" not in manifest


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
