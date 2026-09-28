"""The two Stage 3 commands: what they write, and that a second run writes the same bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from logging_employment.cli import app
from logging_employment.constraints import rows as constraint_rows

runner = CliRunner()


def _rebuild_on_a_changed_system(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`build-constraints` again under the same run id, on a system a code change altered.

    The integrality rows are gone, as a change to `constraints/rows.py` could leave them. `run_id`
    covers config and inputs, not code, so the new `schema_manifest.json` lands beside the bounds
    and results the old constraint set produced.
    """
    monkeypatch.setattr(constraint_rows, "integrality_rows", lambda cells_frame: [])
    result = runner.invoke(app, ["build-constraints", "--config", str(config)])
    monkeypatch.undo()
    assert result.exit_code == 0, result.output


def test_run_baselines_writes_results_and_a_sibling_manifest(staged_repo) -> None:
    result = runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    assert result.exit_code == 0, result.output
    out = staged_repo.run_dir / "baseline_results"
    assert (out / "baseline_results.parquet").exists()
    assert (out / "anchor_audit.parquet").exists()
    manifest = json.loads((staged_repo.run_dir / "baseline_manifest.json").read_text())
    assert "preferred_estimator" in manifest
    assert "output_hashes" in manifest
    assert "weight_basis_counts" in manifest


def test_run_baselines_does_not_touch_the_stage_2_manifest(staged_repo) -> None:
    """`solve-bounds` reads schema_manifest.json's bytes as a precondition gate."""
    before = (staged_repo.run_dir / "schema_manifest.json").read_bytes()
    result = runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    assert result.exit_code == 0, result.output
    assert (staged_repo.run_dir / "schema_manifest.json").read_bytes() == before


def test_a_second_run_is_byte_identical(staged_repo) -> None:
    """§16.1: every command MUST be idempotent for the same inputs."""
    path = staged_repo.run_dir / "baseline_results" / "baseline_results.parquet"
    runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    first = path.read_bytes()
    runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    assert path.read_bytes() == first


def test_the_manifest_records_the_composite_split(staged_repo) -> None:
    """Stage 4 must not report a composite's score as a pure estimator's.

    The split is recorded PER ESTIMATOR. Pooled across all ten it cannot answer the only
    question §10.8's ranking turns on — what fraction of THIS estimator's cells fell back —
    because one pure estimator's own-weighted cells mask another's fallbacks.
    """
    result = runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((staged_repo.run_dir / "baseline_manifest.json").read_text())
    counts = manifest["weight_basis_counts"]
    # A pure estimator: every cell its own.
    assert set(counts["establishment_proportional"]) == {"own_estimator"}
    # A composite: both arms present, and the fallback share recoverable on its own.
    assert set(counts["cbp_intensity"]) >= {"own_estimator", "establishment_fallback"}


def test_declines_are_counted_in_the_manifest(staged_repo) -> None:
    result = runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((staged_repo.run_dir / "baseline_manifest.json").read_text())
    assert "harvest_proportional" in manifest["declines"]
    assert isinstance(manifest["declines"]["harvest_proportional"], dict)


def test_reconcile_writes_a_manifest_like_every_other_command(staged_repo) -> None:
    """§16.1: "Every command MUST write a machine-readable manifest".

    The plan's version only echoed. A verifier that leaves no artifact gives §18.1 nothing to
    reproduce against, and the MUST is not conditional on the command producing data.
    """
    assert (
        runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)]).exit_code
        == 0
    )
    result = runner.invoke(app, ["reconcile", "--config", str(staged_repo.config_path)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((staged_repo.run_dir / "reconcile_manifest.json").read_text())
    assert manifest["within_tolerance"] is True
    assert manifest["checked_pairs"] > 0
    assert manifest["max_residual_drift"] <= manifest["tolerance"]
    # The digest pins WHICH results were checked, so a later edit cannot masquerade as verified.
    assert len(manifest["baseline_results_sha256"]) == 64


def test_reconcile_refuses_before_run_baselines_has_produced_anything(staged_repo) -> None:
    """The precondition gate, matching how `solve-bounds` gates on `build-constraints`."""
    result = runner.invoke(app, ["reconcile", "--config", str(staged_repo.config_path)])
    assert result.exit_code != 0
    assert "run-baselines" in result.output


def test_the_manifest_declares_each_estimators_fallback_intensity(staged_repo) -> None:
    """R-COMP-6. Two baselines answer "how many employees does an establishment carry"
    differently, and R-COMP-7 keeps it that way — so the choice has to be readable from a run's
    output rather than from source."""
    result = runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((staged_repo.run_dir / "baseline_manifest.json").read_text())
    declared = manifest["fallback_intensity"]
    assert declared["share_last_observed"] == "disclosed_qcew"
    assert declared["constrained_regression"] == "disclosed_qcew"
    assert declared["cbp_intensity"] == "national_cbp_march"
    # An estimator that composes nothing declares nothing, and is absent rather than null.
    assert "equal_residual" not in declared
    assert "harvest_proportional" not in declared


def test_declines_are_broken_down_by_kind_not_pooled(staged_repo) -> None:
    """R-COMP-9. §13's scoreboard must be able to tell a considered refusal from a data gap:
    pooled, a baseline that drops months for a data reason is indistinguishable from §10.5."""
    result = runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((staged_repo.run_dir / "baseline_manifest.json").read_text())
    declines = manifest["declines"]
    assert set(declines["harvest_proportional"]) == {"by_design"}
    assert declines["harvest_proportional"]["by_design"] > 0


def test_run_baselines_refuses_a_run_with_no_solved_bounds(staged_repo) -> None:
    """R-S5P-3 makes `solve-bounds` a precondition, rather than skipping INV-002 when it is absent.

    A check that turns itself off when its input is missing is the failure this requirement
    exists to remove: the run would ship `anchored_and_reconciled` rows with nothing having
    looked at §9's intervals, and no artifact would record that.
    """
    (staged_repo.run_dir / "deterministic_bounds.parquet").unlink()
    result = runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    assert result.exit_code != 0
    # Short tokens only: Typer boxes the message and hard-wraps the long tmp_path inside it, so
    # asserting on the full artifact path passes locally and fails under a deeper tmp directory.
    assert "solve-bounds" in result.output
    assert "INV-002" in result.output


def test_the_production_path_scales_an_estimate_into_a_finite_upper_below_it(staged_repo) -> None:
    """The wiring is not vacuous: the bounds the CLI loads really do reach the estimates.

    Every `selected_upper` this fixture solves is null -- its parent table is empty -- so the
    passing run above cannot distinguish a working check from one whose mapping is keyed wrong and
    matches nothing. Tightening ONE cell's upper below the estimate that cell already received is
    the difference. Since `D-111` that no longer halts the run: §12.3 scales the month into the
    bound, so the run succeeds and the cell's released estimate, float and integer, sits at or
    under the new upper. The estimate is read out of the first run rather than assumed.
    """
    assert (
        runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)]).exit_code
        == 0
    )
    results = pl.read_parquet(
        staged_repo.run_dir / "baseline_results" / "baseline_results.parquet"
    ).filter(pl.col("reconciliation_status") == "anchored_and_reconciled")
    row = results.sort(["cell_id", "estimator_id"]).to_dicts()[0]
    bounds_path = staged_repo.run_dir / "deterministic_bounds.parquet"
    bounds = pl.read_parquet(bounds_path)
    assert bounds.filter(pl.col("cell_id") == row["cell_id"]).height == 1
    bounds.with_columns(
        pl.when(pl.col("cell_id") == row["cell_id"])
        .then(pl.lit(row["estimate"] / 2.0))
        .otherwise(pl.col("selected_upper"))
        .alias("selected_upper")
    ).write_parquet(bounds_path)
    result = runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    assert result.exit_code == 0, result.output
    rerun = pl.read_parquet(
        staged_repo.run_dir / "baseline_results" / "baseline_results.parquet"
    ).filter(
        (pl.col("cell_id") == row["cell_id"]) & (pl.col("estimator_id") == row["estimator_id"])
    )
    assert rerun["estimate"].item() <= row["estimate"] / 2.0 < row["estimate"]
    assert rerun["estimate_integer"].item() <= row["estimate"] / 2.0


def test_run_baselines_refuses_bounds_solved_against_another_constraint_set(
    staged_repo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`D-130`. `fit-state-model` refused bounds whose `constraint_set_hash` was not the manifest's;
    `run-baselines` compared nothing, checked every estimate against the old bounds and stamped
    `baseline_results` with the manifest's new hash. It refuses the same way now, before it writes,
    so the last results and their manifest survive the refusal."""
    result = runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    assert result.exit_code == 0, result.output
    run = staged_repo.run_dir
    results = run / "baseline_results" / "baseline_results.parquet"
    manifest = run / "baseline_manifest.json"
    before = (results.read_bytes(), manifest.read_bytes())
    _rebuild_on_a_changed_system(staged_repo.config_path, monkeypatch)
    # The control: the rebuild did move the constraint set out from under the solved bounds.
    schema = json.loads((run / "schema_manifest.json").read_text())["constraint_set_hash"]
    solved = set(pl.read_parquet(run / "deterministic_bounds.parquet")["constraint_set_hash"])
    assert solved != {schema}
    result = runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    assert result.exit_code != 0
    # Short tokens only, as `test_run_baselines_refuses_a_run_with_no_solved_bounds` explains.
    assert "solve-bounds" in result.output
    assert "build-constraints" in result.output
    assert (results.read_bytes(), manifest.read_bytes()) == before


def test_reconcile_refuses_results_rewritten_since_run_baselines(staged_repo) -> None:
    """`D-135`. `reconcile` checked only that `baseline_results.parquet` existed, so it verified
    whatever was there: a file rewritten since, or one a `run-baselines` that failed before its
    manifest left. `run-baselines` records the digest of the results it writes in
    `baseline_manifest.json`, and results with another digest are refused as a precondition, with
    no verdict written."""
    result = runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    assert result.exit_code == 0, result.output
    run = staged_repo.run_dir
    path = run / "baseline_results" / "baseline_results.parquet"
    recorded = hashlib.sha256(path.read_bytes()).hexdigest()
    # The same rows, other bytes: polars' default compression against the deterministic writer's.
    pl.read_parquet(path).write_parquet(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() != recorded
    result = runner.invoke(app, ["reconcile", "--config", str(staged_repo.config_path)])
    assert result.exit_code != 0
    assert "run-baselines" in result.output
    assert "baseline_results.parquet" in result.output
    assert not (run / "reconcile_manifest.json").exists()


def test_reconcile_refuses_results_from_another_constraint_set(
    staged_repo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`D-135`. After `build-constraints` re-runs under the same id, the persisted results were
    produced against a constraint set the run no longer has, and `reconcile` verified them anyway.
    Every row carries the set it was produced under, and a set that is not the manifest's is
    refused, naming the command that ran out of order."""
    result = runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    assert result.exit_code == 0, result.output
    _rebuild_on_a_changed_system(staged_repo.config_path, monkeypatch)
    result = runner.invoke(app, ["reconcile", "--config", str(staged_repo.config_path)])
    assert result.exit_code != 0
    assert "run-baselines" in result.output
    assert "build-constraints" in result.output
    assert not (staged_repo.run_dir / "reconcile_manifest.json").exists()


def test_reconcile_hashes_the_bytes_it_verifies(
    staged_repo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`D-135`. The verdict's `baseline_results_sha256` came from a second read of the path, after
    the parse, so a rewrite between the two recorded a digest of rows the verdict never saw. The
    file is rewritten the moment it has been parsed, as `run-baselines` re-run beside the command
    would rewrite it, and the recorded digest is still of the bytes that were verified. The rewrite
    hangs on `pl.read_parquet`, so the test also asserts it fired once: parsing by any other route
    would never trigger it, and pass."""
    result = runner.invoke(app, ["run-baselines", "--config", str(staged_repo.config_path)])
    assert result.exit_code == 0, result.output
    run = staged_repo.run_dir
    path = run / "baseline_results" / "baseline_results.parquet"
    verified = hashlib.sha256(path.read_bytes()).hexdigest()
    read_parquet = pl.read_parquet
    parsed: list[object] = []

    def then_rewritten(source: object, *args: object, **kwargs: object) -> pl.DataFrame:
        frame = read_parquet(source, *args, **kwargs)
        parsed.append(source)
        frame.write_parquet(path)
        return frame

    monkeypatch.setattr(pl, "read_parquet", then_rewritten)
    result = runner.invoke(app, ["reconcile", "--config", str(staged_repo.config_path)])
    monkeypatch.undo()
    assert result.exit_code == 0, result.output
    assert len(parsed) == 1
    manifest = json.loads((run / "reconcile_manifest.json").read_text())
    assert manifest["baseline_results_sha256"] == verified
    assert hashlib.sha256(path.read_bytes()).hexdigest() != verified
