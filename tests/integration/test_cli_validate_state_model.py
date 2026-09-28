"""`validate-state-model`: §13's harness on the model and §13.10's record, on the committed fixture.

One seed, so one replicate fit per regime (`replicates_per_regime: 3` sizes each mask, as the Stage
4 golden's config does), and the small sampler and loose gate of `test_cli_state_model.py`. Seven
regimes score here, so a passing fit costs seven replicate fits. The verdict on this fixture carries
no meaning: these tests check what the record contains, not which way it went. The refusals fit
nothing, so only the tests that fit are `slow` (plan 16's Decision 13).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from logging_employment.cli import _read_comparand, app
from logging_employment.models.interfaces import STORE_PATH

SMALL_SAMPLER = {"chains": 2, "warmup": 60, "draws": 60}
LOOSE_GATE = {
    "diagnostics": {
        "max_rhat": 100.0,
        "min_ess_per_chain": 1,
        "max_divergences": 1_000_000,
        "min_ppc_coverage_90": 0.0,
    }
}
FIXTURE_VALIDATION = {"replicates_per_regime": 3, "pseudo_suppression_seeds": [1024]}
COMPARAND_TABLES = ("validation_scores", "validation_metrics", "validation_scoreboard")
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
# The comparand `validate` wrote, one part damaged after (`_plant_comparand`), and what
# `cli.py::_read_comparand` finds.
COMPARAND_DAMAGE = {
    "manifest_cut_short": {"unreadable": ["validation_manifest.json"]},
    "manifest_is_a_directory": {"unreadable": ["validation_manifest.json"]},
    "manifest_names_no_scoreboard": {"unreadable": ["validation_manifest.json"]},
    "scores_rewritten": {"mismatched": ["validation_scores.parquet"]},
    "metrics_is_a_directory": {"unreadable": ["validation_metrics.parquet"]},
    "scoreboard_not_parquet": {"unreadable": ["validation_scoreboard.parquet"]},
}
COMPARAND_FILES = (*(f"{table}.parquet" for table in COMPARAND_TABLES), "validation_manifest.json")
# What a `validate-state-model` that failed or was killed partway can leave beside the outputs.
LEFTOVERS = (
    "state_model_validation.partial",
    "state_model_validation.old",
    "promotion_record.json.partial",
)


def _invoke(command: str, config: Path):
    return CliRunner().invoke(app, [command, "--config", str(config)])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plant_comparand(run: Path, damage: str | None = None) -> None:
    """This run's `validate` output: three tables and the manifest recording their digests.

    Each table is a one-row Parquet file, not `validate`'s schema, because the command parses what
    it hashes and compares nothing on the failed-gate path these tests take. `damage` is a key of
    `COMPARAND_DAMAGE`, applied after the manifest is written: a manifest cut short or a directory
    (which fails to read even as root), one that records no digest for a table, a table rewritten
    since, a table that is a directory, and one whose digest matches but which is not Parquet.
    """
    for table in COMPARAND_TABLES:
        path = run / f"{table}.parquet"
        if damage == "scoreboard_not_parquet" and table == "validation_scoreboard":
            path.write_text("a comparand")
        else:
            pl.DataFrame({"cell_id": [table]}).write_parquet(path)
    hashes = {table: _sha256(run / f"{table}.parquet") for table in COMPARAND_TABLES}
    if damage == "manifest_names_no_scoreboard":
        del hashes["validation_scoreboard"]
    manifest = run / "validation_manifest.json"
    manifest.write_text(json.dumps({"estimators": ["a_baseline"], "output_hashes": hashes}))
    if damage == "manifest_cut_short":
        manifest.write_bytes(manifest.read_bytes()[: manifest.stat().st_size // 2])
    elif damage == "manifest_is_a_directory":
        manifest.unlink()
        manifest.mkdir()
    elif damage == "scores_rewritten":
        pl.DataFrame({"cell_id": ["another"]}).write_parquet(run / "validation_scores.parquet")
    elif damage == "metrics_is_a_directory":
        (run / "validation_metrics.parquet").unlink()
        (run / "validation_metrics.parquet").mkdir()
    elif damage not in {None, "scoreboard_not_parquet", "manifest_names_no_scoreboard"}:
        raise ValueError(f"no such damage: {damage!r}")


def _plant_unread_comparand(run: Path) -> None:
    """The comparand and its manifest as text, for a refusal that must come before either is read."""
    for name in COMPARAND_FILES:
        (run / name).write_text("a comparand")


@pytest.fixture(scope="module")
def validated(make_staged_repo):
    """`validate`, `fit-state-model`, then `validate-state-model`, in the documented order."""
    repo = make_staged_repo(
        {"model": {**SMALL_SAMPLER, **LOOSE_GATE}, "validation": FIXTURE_VALIDATION}
    )
    for command in ("validate", "fit-state-model"):
        result = _invoke(command, repo.config_path)
        assert result.exit_code == 0, result.output
    before = {table: _sha256(repo.run_dir / f"{table}.parquet") for table in COMPARAND_TABLES}
    result = _invoke("validate-state-model", repo.config_path)
    assert result.exit_code == 0, result.output
    return repo, before


@pytest.mark.slow
def test_the_record_states_a_verdict_and_what_it_was_measured_against(validated) -> None:
    repo, before = validated
    record = json.loads((repo.run_dir / "promotion_record.json").read_text())
    assert record["verdict"] in {"beat", "not_beaten"}
    assert record["selected_method"] == (
        "state_total_model" if record["verdict"] == "beat" else "section_10_8_hierarchy"
    )
    assert record["provisional"] is True
    assert record["disclosure_review"] == "pending_stage_8"
    assert record["comparand"] == {
        "run_id": record["run_id"],
        **{f"{t}_sha256": h for t, h in before.items()},
    }
    # Which fit the verdict describes: `fit-state-model` can re-run on this run after the record.
    fit_manifest = json.loads((repo.run_dir / "state_model_manifest.json").read_text())
    schema = json.loads((repo.run_dir / "schema_manifest.json").read_text())
    assert record["fit"] == {
        "diagnostics_sha256": _sha256(repo.run_dir / "posterior" / "diagnostics.json"),
        "constraint_set_hash": schema["constraint_set_hash"],
        "draws_sha256": fit_manifest["draws_sha256"],
    }
    assert set(record["gates"]) == {
        "hard_constraints",
        "convergence",
        "coverage",
        "improvement",
        "stratum_degradation",
        "disclosure_review",
    }


@pytest.mark.slow
def test_the_comparand_tables_are_read_never_rewritten(validated) -> None:
    repo, before = validated
    after = {table: _sha256(repo.run_dir / f"{table}.parquet") for table in COMPARAND_TABLES}
    assert after == before


@pytest.mark.slow
def test_the_models_intervals_come_from_its_reconciled_draws(validated) -> None:
    repo, _before = validated
    metrics = pl.read_parquet(
        repo.run_dir / "state_model_validation" / "validation_metrics.parquet"
    )
    coverage = metrics.filter(pl.col("metric_name") == "coverage_0.90")
    assert set(coverage["estimator_id"]) == {"state_total_model"}
    assert set(coverage["interval_source"]) <= {"reconciled_posterior_draws", "none"}
    assert "reconciled_posterior_draws" in set(coverage["interval_source"])


@pytest.mark.slow
def test_every_replicate_fit_left_a_gate_report(validated) -> None:
    """§13.10's convergence gate reads these: one per (regime, seed) that scored."""
    repo, _before = validated
    manifest = json.loads(
        (repo.run_dir / "state_model_validation" / "validation_manifest.json").read_text()
    )
    for entry in manifest["regimes"].values():
        notes = entry.get("producer_notes", [])
        assert len(notes) == entry["replicates"]
        assert all(note["gate"]["scope"] == "replicate" for note in notes)
    record = json.loads((repo.run_dir / "promotion_record.json").read_text())
    replicates = sum(entry["replicates"] for entry in manifest["regimes"].values())
    assert record["gates"]["convergence"]["replicates_checked"] == replicates


@pytest.mark.slow
def test_a_failed_production_fit_is_recorded_without_scoring(make_staged_repo) -> None:
    repo = make_staged_repo(
        {
            "model": {**SMALL_SAMPLER, "diagnostics": {"max_rhat": 0.5}},
            "validation": FIXTURE_VALIDATION,
        }
    )
    assert _invoke("validate", repo.config_path).exit_code == 0
    assert _invoke("fit-state-model", repo.config_path).exit_code == 1
    result = _invoke("validate-state-model", repo.config_path)
    assert result.exit_code == 0, result.output
    record = json.loads((repo.run_dir / "promotion_record.json").read_text())
    assert (record["verdict"], record["selected_method"]) == (
        "not_beaten",
        "section_10_8_hierarchy",
    )
    assert record["gates"]["coverage"]["status"] == "not_evaluated_production_fit_failed"
    assert not (repo.run_dir / "state_model_validation" / "validation_scores.parquet").exists()


def test_the_command_requires_the_comparand(make_staged_repo) -> None:
    repo = make_staged_repo({"model": SMALL_SAMPLER, "validation": FIXTURE_VALIDATION})
    result = _invoke("validate-state-model", repo.config_path)
    assert result.exit_code != 0
    # Short tokens only, as `test_baseline_cli.py` explains: Typer boxes and hard-wraps the message.
    # Not "validate": Typer's "No such command 'validate-state-model'" would contain it too.
    assert "§13.10" in result.output
    assert "comparand" in result.output


@pytest.mark.parametrize("passed", [True, False], ids=["gate_passed", "gate_failed"])
def test_a_fit_from_another_constraint_set_is_refused_before_anything_is_deleted(
    make_staged_repo, passed
) -> None:
    """`fit-state-model` keeps its last fit when it refuses stale bounds, and `build-constraints`
    can re-run without a re-fit. Draws reconciled against another constraint set must not be
    scored, and a failed fit's report must not be written up as not beaten: both refuse before the
    last record is deleted. The comparand is planted as text, since a refusal never reads it."""
    repo = make_staged_repo({"model": SMALL_SAMPLER, "validation": FIXTURE_VALIDATION})
    run = repo.run_dir
    _plant_unread_comparand(run)
    report = {
        "passed": passed,
        "failures": [] if passed else ["parameter_rhat_max"],
        "constraint_set_hash": "stale0hash",
    }
    (run / "posterior").mkdir()
    (run / "posterior" / "diagnostics.json").write_text(json.dumps(report))
    earlier = (
        run / "promotion_record.json",
        run / "state_model_validation" / "validation_manifest.json",
    )
    for path in earlier:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("from an earlier validation")
    result = _invoke("validate-state-model", repo.config_path)
    assert result.exit_code != 0
    # Short tokens only, as `test_baseline_cli.py` explains: Typer boxes and hard-wraps the message.
    assert "stale0hash" in result.output
    assert "fit-state-model" in result.output
    assert all(path.read_text() == "from an earlier validation" for path in earlier)


@pytest.mark.parametrize("written", list(INTERRUPTIONS.values()), ids=list(INTERRUPTIONS))
def test_an_unfinished_fit_is_refused_before_anything_is_deleted(make_staged_repo, written) -> None:
    """Codex on #41. `fit-state-model` writes its passing report first, so a fit interrupted after
    it leaves `"passed": true` beside artifacts that were never written. The command deleted the last
    record and only then failed to read the store, losing a validation of 27 fits. It must refuse
    first, naming what is missing. The comparand and whatever the fit did write are planted as
    text, since a refusal reads neither."""
    repo = make_staged_repo({"model": SMALL_SAMPLER, "validation": FIXTURE_VALIDATION})
    run = repo.run_dir
    _plant_unread_comparand(run)
    current = json.loads((run / "schema_manifest.json").read_text())["constraint_set_hash"]
    (run / "posterior").mkdir()
    report = {"passed": True, "failures": [], "constraint_set_hash": current}
    (run / "posterior" / "diagnostics.json").write_text(json.dumps(report))
    for artifact in written:
        (run / artifact).write_text("from the interrupted fit")
    earlier = (
        run / "promotion_record.json",
        run / "state_model_validation" / "validation_manifest.json",
    )
    for path in earlier:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("from an earlier validation")
    result = _invoke("validate-state-model", repo.config_path)
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), result.exception
    # Short tokens only, as `test_baseline_cli.py` explains: Typer boxes and hard-wraps the message.
    assert "fit-state-model" in result.output
    for artifact in FIT_ARTIFACTS:
        assert (Path(artifact).name in result.output) is (artifact not in written), artifact
    assert all(path.read_text() == "from an earlier validation" for path in earlier)


@pytest.mark.parametrize(("damage", "found"), list(DAMAGE.items()), ids=list(DAMAGE))
def test_a_fit_its_manifest_does_not_vouch_for_is_refused_before_anything_is_deleted(
    make_staged_repo, plant_finished_fit, damage, found
) -> None:
    """Codex on #42. An artifact's existence does not prove the fit finished writing it: a full
    disk left a manifest cut short, it still existed, and the command went on to delete the last
    record and score the store. The manifest must parse and its digests must match the store and
    the summary, or the command refuses first, naming only the artifact at fault. The planted store
    holds only its digest, so reading its draws would crash, not refuse."""
    repo = make_staged_repo({"model": SMALL_SAMPLER, "validation": FIXTURE_VALIDATION})
    run = repo.run_dir
    _plant_unread_comparand(run)
    current = json.loads((run / "schema_manifest.json").read_text())["constraint_set_hash"]
    plant_finished_fit(run, current, damage=damage)
    earlier = (
        run / "promotion_record.json",
        run / "state_model_validation" / "validation_manifest.json",
    )
    for path in earlier:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("from an earlier validation")
    result = _invoke("validate-state-model", repo.config_path)
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), result.exception
    # Short tokens only, as `test_baseline_cli.py` explains: Typer boxes and hard-wraps the message.
    assert "fit-state-model" in result.output
    named = {artifact for artifacts in found.values() for artifact in artifacts}
    for artifact in FIT_ARTIFACTS:
        assert (Path(artifact).name in result.output) is (artifact in named), artifact
    assert all(path.read_text() == "from an earlier validation" for path in earlier)


def test_a_report_that_cannot_be_read_is_refused_before_anything_is_deleted(
    make_staged_repo,
) -> None:
    """Codex on #43. A report that exists but cannot be read raised out of `_stale_fit`, the
    command's first check. It is refused instead, as a report that records no constraint set is,
    before the last record is deleted. The report is a directory, which fails to read as a
    permission or I/O error would, and does so even as root."""
    repo = make_staged_repo({"model": SMALL_SAMPLER, "validation": FIXTURE_VALIDATION})
    run = repo.run_dir
    _plant_unread_comparand(run)
    (run / "posterior" / "diagnostics.json").mkdir(parents=True)
    earlier = (
        run / "promotion_record.json",
        run / "state_model_validation" / "validation_manifest.json",
    )
    for path in earlier:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("from an earlier validation")
    result = _invoke("validate-state-model", repo.config_path)
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), result.exception
    # Short tokens only, as `test_baseline_cli.py` explains: Typer boxes and hard-wraps the message.
    assert "fit-state-model" in result.output
    assert "unrecorded" in result.output
    assert all(path.read_text() == "from an earlier validation" for path in earlier)


def test_a_failed_fits_report_alone_is_written_up_as_not_beaten(make_staged_repo) -> None:
    """A failed gate writes its report and nothing else, by design, so the report alone is not an
    unfinished fit: the command records the model as not beaten without a store (Decision 9). The
    last validation's tables are replaced by none, never left beside a not-beaten record. The record
    names the comparand by the digests of the tables `validate` wrote."""
    repo = make_staged_repo({"model": SMALL_SAMPLER, "validation": FIXTURE_VALIDATION})
    run = repo.run_dir
    _plant_comparand(run)
    current = json.loads((run / "schema_manifest.json").read_text())["constraint_set_hash"]
    (run / "posterior").mkdir()
    report = {"passed": False, "failures": ["parameter_rhat_max"], "constraint_set_hash": current}
    (run / "posterior" / "diagnostics.json").write_text(json.dumps(report))
    (run / "state_model_validation").mkdir()
    (run / "state_model_validation" / "validation_scores.parquet").write_text("from the last run")
    result = _invoke("validate-state-model", repo.config_path)
    assert result.exit_code == 0, result.output
    record = json.loads((run / "promotion_record.json").read_text())
    assert record["verdict"] == "not_beaten"
    assert record["gates"]["convergence"]["production_failures"] == ["parameter_rhat_max"]
    recorded = json.loads((run / "validation_manifest.json").read_text())["output_hashes"]
    assert record["comparand"] == {
        "run_id": record["run_id"],
        **{f"{table}_sha256": recorded[table] for table in COMPARAND_TABLES},
    }
    # The fit the verdict describes: the report it was read from, and no draws, since a failed
    # gate writes none.
    assert record["fit"] == {
        "diagnostics_sha256": _sha256(run / "posterior" / "diagnostics.json"),
        "constraint_set_hash": current,
        "draws_sha256": None,
    }
    assert list((run / "state_model_validation").iterdir()) == []
    assert not any((run / leftover).exists() for leftover in LEFTOVERS)


def test_the_command_requires_validates_manifest(make_staged_repo) -> None:
    """The comparand is vouched for by `validate`'s manifest, so a run with the three tables and no
    manifest has no comparand the command can trust, and is refused as one without the tables is."""
    repo = make_staged_repo({"model": SMALL_SAMPLER, "validation": FIXTURE_VALIDATION})
    run = repo.run_dir
    _plant_comparand(run)
    (run / "validation_manifest.json").unlink()
    result = _invoke("validate-state-model", repo.config_path)
    assert isinstance(result.exception, SystemExit), result.exception
    # Short tokens only, as `test_baseline_cli.py` explains: Typer boxes and hard-wraps the message.
    assert "validation_manifest.json" in result.output
    assert "missing" in result.output


@pytest.mark.parametrize(
    ("damage", "found"), list(COMPARAND_DAMAGE.items()), ids=list(COMPARAND_DAMAGE)
)
def test_a_comparand_validate_did_not_write_is_refused_before_anything_is_deleted(
    make_staged_repo, damage, found
) -> None:
    """#44, found beside Codex's P1. The record names the comparand by digest, and the command
    hashed whatever tables it found, parsed them again from their paths later, and raised on one
    that could not be read. Now each table is read once, before anything moves, and must be what
    `validate` recorded writing in `validation_manifest.json`, or the command refuses, naming only
    the files at fault.
    The report records a failed gate, the path that scores nothing."""
    repo = make_staged_repo({"model": SMALL_SAMPLER, "validation": FIXTURE_VALIDATION})
    run = repo.run_dir
    _plant_comparand(run, damage=damage)
    current = json.loads((run / "schema_manifest.json").read_text())["constraint_set_hash"]
    (run / "posterior").mkdir()
    report = {"passed": False, "failures": ["parameter_rhat_max"], "constraint_set_hash": current}
    (run / "posterior" / "diagnostics.json").write_text(json.dumps(report))
    earlier = (
        run / "promotion_record.json",
        run / "state_model_validation" / "validation_scores.parquet",
    )
    for path in earlier:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("from an earlier validation")
    result = _invoke("validate-state-model", repo.config_path)
    assert isinstance(result.exception, SystemExit), result.exception
    # Short tokens only, as `test_baseline_cli.py` explains: Typer boxes and hard-wraps the message.
    assert "comparand" in result.output
    named = {name for names in found.values() for name in names}
    for name in COMPARAND_FILES:
        assert (name in result.output) is (name in named), name
    assert ("unreadable" in result.output) is ("unreadable" in found)
    assert all(path.read_text() == "from an earlier validation" for path in earlier)
    assert not any((run / leftover).exists() for leftover in LEFTOVERS)


def test_each_comparand_table_is_parsed_from_the_bytes_it_was_hashed_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record names the comparand by digest and the verdict is measured against its frames, so
    both must come from one read. Each table is rewritten on disk the moment it has been read, as a
    `validate` re-run beside the command would rewrite it: parsing it again from its path would
    measure the verdict against a table the record does not name."""
    _plant_comparand(tmp_path)
    planted = {table: pl.read_parquet(tmp_path / f"{table}.parquet") for table in COMPARAND_TABLES}
    recorded = json.loads((tmp_path / "validation_manifest.json").read_text())["output_hashes"]
    read_bytes = Path.read_bytes

    def then_rewritten(self: Path) -> bytes:
        raw = read_bytes(self)
        if self.suffix == ".parquet":
            pl.DataFrame({"cell_id": ["rewritten"]}).write_parquet(self)
        return raw

    monkeypatch.setattr(Path, "read_bytes", then_rewritten)
    comparand = _read_comparand(tmp_path)
    monkeypatch.undo()
    assert comparand.found is None
    assert comparand.sha256 == recorded
    for table in COMPARAND_TABLES:
        assert comparand.frames[table].equals(planted[table]), table


def test_a_store_that_cannot_be_read_fails_before_anything_is_deleted(
    make_staged_repo, plant_finished_fit
) -> None:
    """`read_store` ran after the deletion, so a store that passed every check and still could not
    be read cost the last record and its tables, as anything after it would have, the replicate
    fits included. The planted store holds only its digest: `_unfinished_fit` accepts it, and
    `read_store` then fails on it with `KeyError`."""
    repo = make_staged_repo({"model": SMALL_SAMPLER, "validation": FIXTURE_VALIDATION})
    run = repo.run_dir
    _plant_comparand(run)
    current = json.loads((run / "schema_manifest.json").read_text())["constraint_set_hash"]
    plant_finished_fit(run, current)
    earlier = (
        run / "promotion_record.json",
        run / "state_model_validation" / "validation_scores.parquet",
    )
    for path in earlier:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("from an earlier validation")
    result = _invoke("validate-state-model", repo.config_path)
    assert isinstance(result.exception, KeyError), result.exception
    assert all(path.read_text() == "from an earlier validation" for path in earlier)
    assert not any((run / leftover).exists() for leftover in LEFTOVERS)
