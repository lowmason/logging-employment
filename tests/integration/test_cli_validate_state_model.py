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

from logging_employment.cli import app

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


def _invoke(command: str, config: Path):
    return CliRunner().invoke(app, [command, "--config", str(config)])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    for table in COMPARAND_TABLES:
        (run / f"{table}.parquet").write_text("a comparand")
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
