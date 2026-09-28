"""The `logging-estimates` command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from .config import load_config

if TYPE_CHECKING:  # annotations only; keeps CLI start-up cheap
    from collections.abc import Mapping

    import polars as pl

    from .config import Config

app = typer.Typer(add_completion=False, help="Monthly state Logging employment estimates.")


@app.callback()
def main() -> None:
    """Pin the app as a command group.

    Typer collapses a one-command app into a bare top-level command, which would make
    `logging-estimates validate-config` invalid today and valid once a second command lands.
    The callback fixes the invocation syntax regardless of how many commands are registered.
    """


@app.command("validate-config")
def validate_config(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Parse the configuration and report the resolved estimand."""
    cfg = load_config(config)
    typer.echo(
        f"OK: {cfg.project.industry_code_used} / {cfg.project.ownership} / "
        f"{cfg.project.geography_universe} / {cfg.project.start_month}..{cfg.project.end_month}"
    )


registry_app = typer.Typer(help="Source registry commands.")
app.add_typer(registry_app, name="registry")


@registry_app.command("verify")
def registry_verify(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Check the seed registry against §7.1 and report every problem found."""
    from .registry.loader import load_registry
    from .registry.validation import verify

    load_config(config)
    seed = Path(__file__).parent / "registry" / "sources.yaml"
    problems = verify(load_registry(seed))
    for problem in problems:
        typer.echo(f"PROBLEM: {problem}")
    if problems:
        raise typer.Exit(code=1)
    typer.echo("OK: registry verified")


@app.command("fetch")
def fetch(
    source: str = typer.Option(..., "--source", help="qcew, qcew_parent, qcew_size, or cbp"),
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Acquire raw bytes for one source into the immutable store and record a snapshot row."""
    from .fetching import KNOWN_SOURCES, fetch_source

    if source not in KNOWN_SOURCES:
        raise typer.BadParameter(f"unknown source {source!r}")

    cfg = load_config(config)
    rows = fetch_source(source, cfg, env_path=Path(".env"))
    typer.echo(f"{source}: {len(rows)} snapshot row(s) -> {cfg.storage.raw_uri}")


@app.command("build-harmonized")
def build_harmonized_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Assemble the harmonized Parquet layer from stored raw bytes and print output hashes."""
    from .build import build_harmonized

    cfg = load_config(config)
    hashes = build_harmonized(
        cfg,
        raw_root=Path(cfg.storage.raw_uri),
        out_root=Path(cfg.storage.staged_uri),
        manifest_path=Path(cfg.storage.output_uri) / "source_manifest.parquet",
    )
    for table, digest in sorted(hashes.items()):
        typer.echo(f"{table} {digest}")


def _constraints_dir(cfg: Config) -> Path:
    """§6.2's `data/constraints/`, read from `storage.constraints_uri`."""
    return Path(cfg.storage.constraints_uri)


def _write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    """Write one run manifest, stamped with the code identity that produced it.

    THE point of funnelling all five manifests through one writer. `run_id` covers config and
    input data but deliberately not source (`runs.code_provenance`), so `code_commit` and
    `uv_lock_sha256` are the only things in `runs/<id>/` that can answer "was this directory
    written by the code I am reading?". A per-command `json.dumps` at each site made adding them
    five edits, and made forgetting them on the sixth manifest the default outcome.

    The stamp is merged LAST so no caller can shadow or drop it, and the JSON keeps the
    `indent=2, sort_keys=True` shape every one of these files already had -- `solve-bounds` reads
    `schema_manifest.json` as a precondition gate and an integration test pins its bytes, so the
    formatting is not free to drift.

    A MANIFEST IS REPLACED WHOLE (Codex on #42). `Path.write_text` truncates before it writes, so
    a full disk or a kill mid-write left an empty or partial manifest where readers look, and
    `_unfinished_fit` read that file's existence as a finished fit. It now parses the manifest and
    checks the digests it records, and a manifest written here is never partial. The text goes to a
    `.partial` sibling, and `os.replace` renames it over the manifest, which is atomic within a
    directory. A write that fails removes its sibling. A process killed outright can leave one, and
    no reader opens it.
    """
    import json
    import os

    from .runs import code_provenance

    text = json.dumps({**payload, **code_provenance()}, indent=2, sort_keys=True)
    partial = path.with_name(f"{path.name}.partial")
    try:
        partial.write_text(text)
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _replace_directory(staged: Path, target: Path) -> None:
    """Put `staged` where `target` is, and delete the old `target` only once `staged` is in place.

    `_write_manifest`'s rule for one file, applied to a directory, which `os.replace` cannot move
    over a non-empty one. So there are two renames, the old directory to a `.old` sibling and then
    `staged` to `target`, and `.old` is deleted only after both. If the second rename fails, the
    first is undone, so a failure here never leaves `target` missing. A kill between the two can,
    and it leaves the old directory at `.old`, so the next swap moves it back before anything else
    rather than deleting it as a leftover. A `.old` beside a `target` is the tail of a swap that
    finished, and is deleted.
    """
    import os
    import shutil

    old = target.with_name(f"{target.name}.old")
    if old.exists():
        if target.exists():
            shutil.rmtree(old)
        else:
            os.replace(old, target)
    if target.exists():
        os.replace(target, old)
    try:
        os.replace(staged, target)
    except BaseException:
        if old.exists():
            os.replace(old, target)
        raise
    shutil.rmtree(old, ignore_errors=True)


def _input_digests(cfg: Config) -> dict[str, str]:
    """A sha256 per harmonized input, which is what makes the run id a function of the data."""
    import hashlib

    staged = Path(cfg.storage.staged_uri)
    return {
        path.stem: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(staged.glob("*.parquet"))
    }


@app.command("build-constraints")
def build_constraints_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Assemble the constraint system from the harmonized layer and persist it."""

    import highspy
    import yaml

    from .build import write_parquet_deterministic
    from .config import resolved_dict
    from .constraints import graph
    from .constraints.system import build_constraint_system
    from .contracts import (
        CONSTRAINT_COEFFICIENT_SCHEMA,
        CONSTRAINT_ROW_SCHEMA,
        TARGET_CELL_SCHEMA,
        HarmonizedData,
        schema_fingerprint,
    )
    from .runs import run_dir, run_id

    cfg = load_config(config)
    data = HarmonizedData.load(Path(cfg.storage.staged_uri))
    built = graph.assign_components(build_constraint_system(data, cfg))

    out = _constraints_dir(cfg)
    hashes = {
        "target_cell": write_parquet_deterministic(built.cells, out / "target_cell.parquet"),
        "constraint_row": write_parquet_deterministic(built.rows, out / "constraint_row.parquet"),
        "constraint_coefficient": write_parquet_deterministic(
            built.coefficients, out / "constraint_coefficient.parquet"
        ),
    }

    run = run_dir(cfg, run_id(cfg, _input_digests(cfg)))
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(resolved_dict(cfg), sort_keys=True))
    _write_manifest(
        run / "schema_manifest.json",
        {
            "constraint_set_hash": built.constraint_set_hash,
            "output_hashes": hashes,
            "schema_fingerprints": {
                "target_cell": schema_fingerprint(TARGET_CELL_SCHEMA),
                "constraint_row": schema_fingerprint(CONSTRAINT_ROW_SCHEMA),
                "constraint_coefficient": schema_fingerprint(CONSTRAINT_COEFFICIENT_SCHEMA),
            },
            "compatibility_report": built.compatibility_report,
            # REQ-029 names the solver among the fail-closed surfaces, and §18.1 wants a run
            # reproducible from its manifest. `config.resolved.yaml` records that the solver is
            # HiGHS; only this records which HiGHS.
            "solver": cfg.constraints.solver,
            "solver_version": highspy.Highs().version(),
        },
    )
    write_parquet_deterministic(
        graph.component_provenance(built), run / "constraint_manifest.parquet"
    )
    typer.echo(f"constraint_set_hash {built.constraint_set_hash}")
    for table, digest in sorted(hashes.items()):
        typer.echo(f"{table} {digest}")


@app.command("solve-bounds")
def solve_bounds_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Solve sharp LP/MILP bounds for every target cell and flag the disclosive ones."""
    import json

    from .build import write_parquet_deterministic
    from .constraints.bounds import solve_bounds
    from .constraints.system import load_system
    from .disclosure.flags import build_flags
    from .runs import run_dir, run_id

    cfg = load_config(config)
    run = run_dir(cfg, run_id(cfg, _input_digests(cfg)))
    manifest = run / "schema_manifest.json"
    if not manifest.exists():
        raise typer.BadParameter(
            f"{manifest} is missing: no `build-constraints` run matches the harmonized inputs "
            "currently in the staged directory. Run `build-constraints` first"
        )
    built = load_system(
        _constraints_dir(cfg),
        expected_hash=json.loads(manifest.read_text())["constraint_set_hash"],
    )
    result = solve_bounds(built, cfg.constraints)
    flags = build_flags(result.bounds, built.cells, cfg.disclosure)

    run.mkdir(parents=True, exist_ok=True)
    # The return values are the outputs' sha256s, exactly as `build-constraints` and
    # `run-baselines` collect them; this command was throwing all three away.
    hashes = {
        "deterministic_bounds": write_parquet_deterministic(
            result.bounds, run / "deterministic_bounds.parquet"
        ),
        "component_rank": write_parquet_deterministic(
            result.components, run / "component_rank.parquet"
        ),
        "disclosure_flags": write_parquet_deterministic(flags, run / "disclosure_flags.parquet"),
    }
    counts = result.bounds.group_by("bound_status").len().sort("bound_status")
    narrow = int(flags["narrow_feasible_interval_flag"].sum())
    exact = int(flags["exact_reconstruction_flag"].sum())
    # §16.1: "Every command MUST write a machine-readable manifest." `solve-bounds` wrote none, so
    # the §9 bounds were the one stage whose outputs a reader could only re-hash by hand, and the
    # §9.8 flag counts -- §14's disclosure surface -- survived only in the terminal scrollback.
    # Written AFTER the three tables and only on the success path: a refused run must leave the
    # directory with no artifact of any kind claiming these inputs were bounded.
    _write_manifest(
        run / "bounds_manifest.json",
        {
            # The gate `load_system` was pinned to, restated where the outputs are. This is what
            # ties `runs/<id>/deterministic_bounds.parquet` to the constraint tables in
            # `data/constraints/`, which the run id does not cover.
            "constraint_set_hash": built.constraint_set_hash,
            "output_hashes": hashes,
            "bound_status_counts": {
                row["bound_status"]: row["len"] for row in counts.iter_rows(named=True)
            },
            "narrow_feasible_interval_flags": narrow,
            "exact_reconstruction_flags": exact,
        },
    )
    for row in counts.iter_rows(named=True):
        typer.echo(f"{row['bound_status']} {row['len']}")
    typer.echo(f"flagged {narrow} narrow, {exact} exact")


@app.command("run-baselines")
def run_baselines_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Run every §10 transparent baseline and persist the results under this run's directory."""
    import json

    import polars as pl

    from .baselines.runner import (
        REGISTRY,
        preferred_estimator,
        preferred_estimator_by_month,
        run_baselines,
        state_total_bounds,
    )
    from .build import write_parquet_deterministic
    from .contracts import (
        ANCHOR_AUDIT_SCHEMA,
        BASELINE_RESULT_SCHEMA,
        HarmonizedData,
        schema_fingerprint,
    )
    from .runs import run_dir, run_id

    cfg = load_config(config)
    data = HarmonizedData.load(Path(cfg.storage.staged_uri))
    run = run_dir(cfg, run_id(cfg, _input_digests(cfg)))
    manifest_path = run / "schema_manifest.json"
    if not manifest_path.exists():
        raise typer.BadParameter(
            f"{manifest_path} is missing: no `build-constraints` run matches the harmonized "
            "inputs currently in the staged directory. Run `build-constraints` first"
        )
    # A SECOND PRECONDITION, new with R-S5P-3. INV-002's per-cell half checks every estimate
    # against §9's solved interval, so `solve-bounds` joins `build-constraints` as a gate rather
    # than the check being skipped when its input happens to be absent. A silently-skipped
    # invariant is the failure this requirement exists to remove, and the documented pipeline
    # order already runs `solve-bounds` before `run-baselines`.
    bounds_path = run / "deterministic_bounds.parquet"
    if not bounds_path.exists():
        raise typer.BadParameter(
            f"{bounds_path} is missing: every baseline estimate is checked against its per-cell "
            "deterministic bounds (INV-002), and this run has none. Run `solve-bounds` first"
        )
    results, audit = run_baselines(
        data,
        cfg,
        constraint_set_hash=json.loads(manifest_path.read_text())["constraint_set_hash"],
        bounds=state_total_bounds(pl.read_parquet(bounds_path)),
    )

    out = run / "baseline_results"
    out.mkdir(parents=True, exist_ok=True)
    hashes = {
        "baseline_results": write_parquet_deterministic(results, out / "baseline_results.parquet"),
        "anchor_audit": write_parquet_deterministic(audit, out / "anchor_audit.parquet"),
    }

    ran = results.filter(pl.col("reconciliation_status") == "anchored_and_reconciled")
    # Nested by estimator, not pooled. A single pooled count cannot answer "what fraction of
    # THIS estimator's cells took the fallback", which is the number §10.8's ranking turns on and
    # the one Task 19 reads back off this command.
    basis_counts: dict[str, dict[str, int]] = {}
    for row in ran.group_by(["estimator_id", "weight_basis"]).len().iter_rows(named=True):
        basis_counts.setdefault(row["estimator_id"], {})[row["weight_basis"]] = row["len"]
    # Nested by kind, not pooled. §13.5-13.8 define no decline metric, so a scoreboard has to be
    # able to separate §10.5's considered refusal from a month an estimator lost to a data gap:
    # pooled, a broken baseline and a benchmark that never runs look identical.
    declines: dict[str, dict[str, int]] = {}
    for row in (
        results.filter(pl.col("reconciliation_status") == "declined")
        .group_by(["estimator_id", "decline_kind"])
        .len()
        .iter_rows(named=True)
    ):
        declines.setdefault(row["estimator_id"], {})[row["decline_kind"]] = row["len"]
    # R-COMP-6: which intensity scaled each estimator's fallback arm, read off the estimators
    # themselves. §10.3 and §10.6 declare the disclosed ratio and §10.4 the national CBP March
    # one, and R-COMP-7 keeps that difference rather than standardising it — so it is reported
    # here, where a reader can check an arm against the name without reading source.
    fallback_intensity = {
        estimator.estimator_id: estimator.fallback_intensity
        for estimator in REGISTRY
        if estimator.fallback_intensity is not None
    }
    # A SIBLING manifest. `schema_manifest.json` is `solve-bounds`'s precondition gate and an
    # idempotence test pins its bytes, so nothing here may write to it.
    _write_manifest(
        run / "baseline_manifest.json",
        {
            "preferred_estimator": preferred_estimator(results),
            "preferred_estimator_by_month": preferred_estimator_by_month(results),
            "output_hashes": hashes,
            "schema_fingerprints": {
                "baseline_results": schema_fingerprint(BASELINE_RESULT_SCHEMA),
                "anchor_audit": schema_fingerprint(ANCHOR_AUDIT_SCHEMA),
            },
            "weight_basis_counts": basis_counts,
            "fallback_intensity": fallback_intensity,
            "declines": declines,
            "anchor": {
                "basis": "declared_national_total",
                "months_gated": audit.height,
                "months_anchored": int(audit["anchored"].sum()),
                "establishment_gap_max": int(audit["establishment_gap"].abs().max()),
            },
        },
    )
    typer.echo(f"preferred {preferred_estimator(results)}")
    for estimator_id, counts in sorted(basis_counts.items()):
        for basis, count in sorted(counts.items()):
            typer.echo(f"{estimator_id} {basis} {count}")
    for estimator_id, kinds in sorted(declines.items()):
        for kind, count in sorted(kinds.items()):
            typer.echo(f"declined {estimator_id} {kind} {count}")


@app.command("fit-state-model")
def fit_state_model_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Fit §11's state-total model, reconcile every draw, gate it (§11.14) and summarise it (§7.11).

    Preconditions are `build-constraints` and `solve-bounds`: every draw is reconciled into §9's
    per-cell interval, so a run without `deterministic_bounds.parquet` has nothing to hold the draws
    to. A RE-FIT REPLACES, NEVER MERGES. The previous fit's artifacts are deleted before sampling,
    so a failed gate cannot leave an earlier success's summary beside its own failed report.
    `run_id` does not cover source code, so a re-fit with new code lands in the same directory.

    `posterior/diagnostics.json` is written BEFORE the gate is enforced, so a failure keeps its
    evidence. The command then exits 1 and writes no store, no summary and no manifest.
    `validate-state-model` reads the report and records the model as not beaten without scoring it.
    A pass writes the store, the summary and, last, the manifest, so a passing report does not by
    itself prove the fit finished: both commands that read a fit check all three are there, the
    manifest parses, and its digests match the other two (`_unfinished_fit`). The report names the
    constraint set the draws were reconciled against, and both compare it with the run's own first
    (`_stale_fit`).
    """
    import json
    import shutil

    import polars as pl

    from .baselines.runner import state_total_bounds
    from .build import write_parquet_deterministic
    from .contracts import POSTERIOR_SUMMARY_SCHEMA, HarmonizedData, schema_fingerprint
    from .errors import ModelDiagnosticsError
    from .models.arviz_io import write_store
    from .models.data import build_model_data
    from .models.diagnostics import assert_gate_passes, evaluate_gate
    from .models.interfaces import MODEL_ID, MODEL_VERSION, STORE_PATH, StateModelConfig
    from .models.reconciliation import check_reconciled, reconcile_fit
    from .models.state_total import fit_state_total_model
    from .models.summary import posterior_summary
    from .runs import run_dir, run_id

    cfg = load_config(config)
    data = HarmonizedData.load(Path(cfg.storage.staged_uri))
    rid = run_id(cfg, _input_digests(cfg))
    run = run_dir(cfg, rid)
    manifest_path = run / "schema_manifest.json"
    bounds_path = run / "deterministic_bounds.parquet"
    for path, command in ((manifest_path, "build-constraints"), (bounds_path, "solve-bounds")):
        if not path.exists():
            raise typer.BadParameter(
                f"{path} is missing: every draw is reconciled into this run's deterministic "
                f"bounds (INV-012), and no run matches the staged inputs. Run `{command}` first"
            )
    # THE BOUNDS MUST BE THIS CONSTRAINT SET'S. `run_id` does not cover code, so `build-constraints`
    # can re-run under the same id after a code change and leave the old `deterministic_bounds`
    # beside a new `schema_manifest.json`. `constraint_set_hash` is the one cross-stage check that
    # fires (CLAUDE.md), and it runs before anything is deleted, so a refusal keeps the last fit.
    # That fit is then stale, and `_stale_fit` keeps `reconcile` and `validate-state-model` off it.
    constraint_set_hash = json.loads(manifest_path.read_text())["constraint_set_hash"]
    bounds = pl.read_parquet(bounds_path)
    solved_against = sorted(set(bounds["constraint_set_hash"].to_list()))
    if solved_against != [constraint_set_hash]:
        raise typer.BadParameter(
            f"{bounds_path} was solved against constraint set {solved_against}, but "
            f"{manifest_path} names {constraint_set_hash!r}: `build-constraints` ran after "
            "`solve-bounds`. Run `solve-bounds` first"
        )
    posterior = run / "posterior"
    shutil.rmtree(posterior, ignore_errors=True)
    for stale in (run / "posterior_summary.parquet", run / "state_model_manifest.json"):
        stale.unlink(missing_ok=True)
    posterior.mkdir(parents=True)

    monthly = data.qcew_monthly
    model_data = build_model_data(monthly)
    fit = fit_state_total_model(model_data, StateModelConfig.from_config(cfg.model))
    draws = reconcile_fit(fit, monthly, state_total_bounds(bounds), cfg)
    check = check_reconciled(draws, tolerance=cfg.reconciliation.tolerance)
    report = evaluate_gate(fit, draws, check, cfg.model.diagnostics, scope="production")
    _write_manifest(
        posterior / "diagnostics.json",
        {**report.to_json(), "run_id": rid, "constraint_set_hash": constraint_set_hash},
    )
    try:
        assert_gate_passes(report)
    except ModelDiagnosticsError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    attrs = {
        "run_id": rid,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "constraint_set_hash": constraint_set_hash,
    }
    digest = write_store(run / STORE_PATH, draws, fit, model_data, attrs=attrs)
    summary = posterior_summary(
        draws, monthly, bounds, run_id=rid, constraint_set_hash=constraint_set_hash
    )
    _write_manifest(
        run / "state_model_manifest.json",
        {
            **attrs,
            "sampler": fit.sampler,
            "draws_sha256": digest,
            "store": STORE_PATH,
            "posterior_summary_sha256": write_parquet_deterministic(
                summary, run / "posterior_summary.parquet"
            ),
            "posterior_summary_schema": schema_fingerprint(POSTERIOR_SUMMARY_SCHEMA),
            # §12.2 (since `D-120`): every row allocated against R_t carries its anchor basis.
            # §7.11's table has no such column, so the fit's manifest and the store carry it.
            "anchor_bases": sorted({anchor.anchor_basis for anchor in draws.anchors.values()}),
            "training_cells": int(model_data.train_y.size),
            "predicted_cells": len(model_data.predict_cell_ids),
            "ppc_coverage_90": fit.ppc_coverage_90,
            "ppc_cells": fit.ppc_cells,
            "posterior_medians": fit.posterior_medians,
            "reconciliation": {
                "draws_checked": check.draws_checked,
                "months_checked": check.months_checked,
                "max_anchor_drift": check.max_anchor_drift,
                "bound_violations": check.bound_violations,
                "tolerance": check.tolerance,
            },
        },
    )
    typer.echo(f"fit {fit.chains} chains; gate passed; {len(draws.cell_ids)} cells reconciled")


def _stale_fit(run: Path) -> dict[str, str | None] | None:
    """The fit's and the run's constraint-set hashes when they differ, or `None` when they agree.

    A fit can outlive its constraint set. `fit-state-model` refuses stale bounds before it deletes
    anything, so that refusal keeps the last fit, and `build-constraints` can re-run under the same
    `run_id` without `fit-state-model` ever running again. Either way the surviving draws were
    reconciled into bounds that no longer hold, and re-verifying them against themselves would
    pass. So `reconcile` and `validate-state-model`, the two commands that read a fit, call this
    first. Deleting the fit on refusal would close only the first of those two routes.

    The fit's hash is `posterior/diagnostics.json`'s. Every fit writes that report first, gate
    passed or not, into the `posterior/` it emptied, so the report and the store come from one fit,
    and reading it needs no netCDF reader. A report that is absent, cannot be read, does not parse
    to an object or lacks the key is `None`, which matches nothing: an unrecorded fit is refused,
    never trusted, and never raised past the caller (Codex on #43), so `reconcile` still writes its
    verdict. This runs first in `validate-state-model`, and in `reconcile` whenever a store exists,
    so a report that cannot be read is refused here as stale. `_unfinished_fit` names the same
    report `unreadable` only in a `reconcile` without a store.
    """
    import json

    def recorded(path: Path) -> str | None:
        """The `constraint_set_hash` a JSON artifact records, or `None` when it records none."""
        try:
            document = json.loads(path.read_text())
        except OSError, ValueError:
            return None
        return document.get("constraint_set_hash") if isinstance(document, dict) else None

    fit = recorded(run / "posterior" / "diagnostics.json")
    current = recorded(run / "schema_manifest.json")
    if fit is not None and fit == current:
        return None
    return {"fit_constraint_set_hash": fit, "constraint_set_hash": current}


def _unfinished_fit(run: Path) -> dict[str, list[str]] | None:
    """Why a passing fit's artifacts are not one finished fit, or `None` when they are.

    `fit-state-model` writes `posterior/diagnostics.json` first, gate passed or not, so a failure
    keeps its evidence. A pass then writes the store, `posterior_summary.parquet` and, last,
    `state_model_manifest.json`, which records the other two's digests. So a fit interrupted after
    its report leaves `"passed": true` beside draws that were never written (Codex on #41).
    `validate-state-model` deleted the last promotion record and only then failed to read the
    store, and `reconcile`, which keyed on the store alone, read such a run as having no fit. Both
    call this after `_stale_fit` and refuse the fit unread.

    EXISTENCE IS NOT PROOF (Codex on #42): a manifest cut short by a full disk still exists. So the
    manifest must parse, and each digest it records must match: `draws_sha256` the store's own, and
    `posterior_summary_sha256` the summary file's. Each key maps to run-relative paths: `missing`,
    `unreadable` and `mismatched`. The store's check is identity, not content. `store_digest` reads
    the digest the store was written with and hashes no draw. HDF5 refuses to open a file shorter
    than the end its superblock records, so a store cut short is `unreadable`. Both commands re-hash
    its draws after this (`draws_sha256_matches`).

    NO READ RAISES PAST THIS (Codex on #43). An artifact that exists but cannot be read is
    `unreadable`: a permission or I/O error, or a directory in its place, raised `OSError` out of
    both commands, and `reconcile` then wrote no verdict at all (§16.1). So is a report or manifest
    that does not parse to an object, and a store that does not open. A report that cannot be read
    is `unreadable`, never `None`, which would let a `reconcile` without a store pass beside it.
    `_stale_fit` refuses the same report first wherever it runs. No report is no fit, and a failed
    report writes no store by design: both are `None`.
    """
    import hashlib
    import json

    from .models.interfaces import STORE_PATH

    report_path = "posterior/diagnostics.json"
    if not (run / report_path).exists():
        return None
    try:
        report = json.loads((run / report_path).read_text())
    except OSError, ValueError:
        report = None
    if not isinstance(report, dict):
        return {"unreadable": [report_path]}
    if report.get("passed") is not True:
        return None
    summary, manifest = "posterior_summary.parquet", "state_model_manifest.json"
    missing = [path for path in (STORE_PATH, summary, manifest) if not (run / path).exists()]
    if missing:
        return {"missing": missing}
    try:
        recorded = json.loads((run / manifest).read_text())
        draws_sha256 = recorded["draws_sha256"]
        summary_sha256 = recorded["posterior_summary_sha256"]
    except OSError, ValueError, KeyError, TypeError:
        return {"unreadable": [manifest]}
    # Only a fit that exists gets this far, so a baseline-only run never loads xarray.
    from .models.arviz_io import store_digest

    found: dict[str, list[str]] = {}
    try:
        if store_digest(run / STORE_PATH) != draws_sha256:
            found["mismatched"] = [STORE_PATH]
    except OSError, KeyError:
        found["unreadable"] = [STORE_PATH]
    try:
        if hashlib.sha256((run / summary).read_bytes()).hexdigest() != summary_sha256:
            found.setdefault("mismatched", []).append(summary)
    except OSError:
        found.setdefault("unreadable", []).append(summary)
    return found or None


def _unfinished_reasons(unfinished: dict[str, list[str]]) -> str:
    """`_unfinished_fit`'s findings as one clause, naming only the artifacts at fault.

    A mismatch is the manifest's word against an artifact's, and the artifact is the one named:
    `mismatched` says "its manifest", never the manifest's file, so every file a refusal names is
    one to distrust.
    """
    labels = {"mismatched": "not what its manifest records"}
    return "; ".join(
        f"{labels.get(kind, kind)}: {', '.join(paths)}" for kind, paths in unfinished.items()
    )


@app.command("reconcile")
def reconcile_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Re-reconcile the persisted baseline estimates and report any drift.

    Stage 3 reconciles inside `run-baselines`, so this command verifies rather than produces: it
    re-sums the persisted estimates against each month's recorded residual and reports the largest
    difference. It does NOT re-run the allocation. Stage 5 makes it load-bearing, when posterior
    draws reconcile separately from the estimators that seeded them.
    """
    import hashlib

    import polars as pl

    from .runs import run_dir, run_id

    cfg = load_config(config)
    run = run_dir(cfg, run_id(cfg, _input_digests(cfg)))
    path = run / "baseline_results" / "baseline_results.parquet"
    if not path.exists():
        raise typer.BadParameter(
            f"{path} is missing: no `run-baselines` run matches the harmonized inputs currently "
            "in the staged directory. Run `run-baselines` first"
        )
    results = pl.read_parquet(path)
    ran = results.filter(pl.col("reconciliation_status") == "anchored_and_reconciled")
    drift = (
        ran.group_by(["estimator_id", "reference_month"])
        .agg(pl.col("estimate").sum().alias("total"), pl.col("residual").first().alias("residual"))
        .with_columns((pl.col("total") - pl.col("residual")).abs().alias("drift"))
    )
    worst = float(drift["drift"].max()) if drift.height else 0.0
    within_tolerance = worst <= cfg.reconciliation.tolerance
    # §16.1: "Every command MUST write a machine-readable manifest and MUST be idempotent for the
    # same inputs." A verifier that only echoes leaves nothing for §18.1 to reproduce against, so
    # the verdict and the digest of what was checked are persisted beside the results.
    payload: dict[str, object] = {
        "checked_pairs": drift.height,
        "max_residual_drift": worst,
        "tolerance": cfg.reconciliation.tolerance,
        "within_tolerance": within_tolerance,
        "baseline_results_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    # Plan 16: INV-012 re-measured on the PERSISTED draws once `fit-state-model` has written them.
    # `fit-state-model` checked the draws it held in memory; this reads the file back, so a defect
    # in writing it cannot pass unseen. The key is OMITTED, never null, before a fit exists, so a
    # baseline-only run writes the keys it always wrote. A fit from another constraint set fails
    # unread: its draws hold to that set's bounds, so checking them would pass (`_stale_fit`). So
    # does a passing fit whose artifacts are not one finished fit (`_unfinished_fit`): keyed on the
    # store alone, a run whose store never landed read as having no fit at all.
    from .models.interfaces import STORE_PATH

    state_model_passed = True
    store = run / STORE_PATH
    stale = _stale_fit(run) if store.exists() else None
    unfinished = _unfinished_fit(run) if stale is None else None
    if unfinished is not None:
        state_model_passed = False
        payload["state_model"] = {**unfinished, "passed": False}
        typer.echo(
            "state-total draws not checked: the fit's report records a pass, but its artifacts are "
            f"not one finished fit ({_unfinished_reasons(unfinished)}). Run `fit-state-model` first",
            err=True,
        )
    elif stale is not None:
        state_model_passed = False
        payload["state_model"] = {**stale, "passed": False}
        typer.echo(
            "state-total draws not checked: the fit was reconciled against constraint set "
            f"{stale['fit_constraint_set_hash']!r}, and this run's is "
            f"{stale['constraint_set_hash']!r}. Run `solve-bounds` and `fit-state-model` first",
            err=True,
        )
    elif store.exists():
        from .models.arviz_io import draws_digest, read_store, store_digest
        from .models.reconciliation import check_reconciled

        draws = read_store(store)
        check = check_reconciled(draws, tolerance=cfg.reconciliation.tolerance)
        digest_matches = draws_digest(draws) == store_digest(store)
        state_model_passed = check.passed and digest_matches
        payload["state_model"] = {
            "draws_checked": check.draws_checked,
            "months_checked": check.months_checked,
            "cells_checked": check.cells_checked,
            "max_anchor_drift": check.max_anchor_drift,
            "bound_violations": check.bound_violations,
            "draws_sha256_matches": digest_matches,
            "passed": state_model_passed,
        }
        typer.echo(
            f"state-total draws: {check.draws_checked} x {check.cells_checked}, max anchor drift "
            f"{check.max_anchor_drift:.3e}, {check.bound_violations} bound violation(s)"
        )
    _write_manifest(run / "reconcile_manifest.json", payload)
    typer.echo(f"checked {drift.height} (estimator, month) pairs")
    typer.echo(f"max residual drift {worst:.3e}")
    if not within_tolerance or not state_model_passed:
        raise typer.Exit(code=1)


def _estimator_override(estimators: str | None) -> dict[str, list[str]] | None:
    """`--estimators a,b` parsed, checked against §10's registry, and canonicalised for the run id.

    Returns the mapping `run_id` folds into its payload, or `None` when no subset was asked for.
    `None` is what keeps this option cheap: `run_id` omits the key entirely, so an un-overridden
    run hashes exactly as it did before the option existed and not one run directory on disk was
    renumbered by adding it.

    The ids come back in REGISTRY order, so `a,b` and `b,a` name ONE run rather than two
    directories holding byte-identical outputs. Raises `ConceptViolationError` on an unknown,
    repeated, or empty subset; the caller turns that into `typer.BadParameter`.
    """
    from .baselines.runner import resolve_estimators

    if estimators is None:
        return None
    names = [name.strip() for name in estimators.split(",") if name.strip()]
    return {"estimators": [estimator.estimator_id for estimator in resolve_estimators(names)]}


@app.command("validate")
def validate_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    estimators: str = typer.Option(
        None,
        "--estimators",
        help=(
            "Comma-separated §10 estimator ids to score, e.g. "
            "'establishment_proportional,share_last_observed'. Omit for the full registry. "
            "A subset changes the run id, so it gets its own run directory rather than "
            "overwriting a full pass's metrics."
        ),
    ),
) -> None:
    """Run §13's pseudo-suppression harness and persist its metrics and scoreboard."""

    from .baselines.runner import resolve_estimators
    from .build import write_parquet_deterministic
    from .contracts import (
        VALIDATION_METRIC_SCHEMA,
        VALIDATION_SCORE_SCHEMA,
        VALIDATION_SCOREBOARD_SCHEMA,
        HarmonizedData,
        assert_required_columns_present,
        validate_frame,
    )
    from .errors import ConceptViolationError
    from .runs import run_dir, run_id
    from .validate.harness import run_pseudo_suppression

    # Resolved BEFORE the staged layer is read, so a mistyped id costs a message rather than a
    # table load, and `typer.BadParameter` reports it the way every other bad option is reported.
    try:
        override = _estimator_override(estimators)
    except ConceptViolationError as error:
        raise typer.BadParameter(str(error), param_hint="--estimators") from error
    chosen = resolve_estimators(None if override is None else override["estimators"])

    cfg = load_config(config)
    data = HarmonizedData.load(Path(cfg.storage.staged_uri))
    run = run_dir(cfg, run_id(cfg, _input_digests(cfg), overrides=override))
    run.mkdir(parents=True, exist_ok=True)

    result = run_pseudo_suppression(data, chosen, cfg)

    # ALL THREE TABLES, gated before anything is written. `validation_scores` used to be excluded
    # with a comment calling its columns "a superset of VALIDATION_SCORE_SCHEMA", which was not
    # true in either direction: measured 2026-09-08, 23 produced against 20 declared with 17 in
    # common, so three declared columns were produced by nothing and six produced ones were
    # declared nowhere. The nullity pass is separate because `validate_frame` compares columns and
    # dtypes only, and the defect that motivated it — a NULL `mask_arm` on every `declines` row —
    # sat inside a table the schema gate already passed.
    for frame, schema, table in (
        (result.scores, VALIDATION_SCORE_SCHEMA, "validation_scores"),
        (result.metrics, VALIDATION_METRIC_SCHEMA, "validation_metrics"),
        (result.scoreboard, VALIDATION_SCOREBOARD_SCHEMA, "validation_scoreboard"),
    ):
        validate_frame(frame, schema, table)
        assert_required_columns_present(frame, table)

    hashes = {
        "validation_scores": write_parquet_deterministic(
            result.scores, run / "validation_scores.parquet"
        ),
        "validation_metrics": write_parquet_deterministic(
            result.metrics, run / "validation_metrics.parquet"
        ),
        "validation_scoreboard": write_parquet_deterministic(
            result.scoreboard, run / "validation_scoreboard.parquet"
        ),
    }
    # Recorded at the TOP level, not only per regime. A reader asking "what did this run score"
    # should not have to open thirteen regime entries and intersect them, and `runs/<id>/` carries
    # no `config.resolved.yaml` for a `validate` run to state it instead.
    manifest = {
        **result.manifest,
        "estimators": [estimator.estimator_id for estimator in chosen],
        "output_hashes": hashes,
    }
    _write_manifest(run / "validation_manifest.json", manifest)
    for regime, entry in sorted(result.manifest["regimes"].items()):
        typer.echo(f"{regime} {entry['disposition']} scored={entry['n_scored']}")


def _publish_state_model_validation(
    run: Path,
    record: Mapping[str, object],
    *,
    tables: Mapping[str, pl.DataFrame],
    validation_manifest: Mapping[str, object] | None,
) -> None:
    """Replace `state_model_validation/`, then `promotion_record.json`, each only once it is whole.

    The last validation is its record and its tables, and nothing of it is touched until the new
    one is written. The tables and their manifest go to a `.partial` sibling, `_replace_directory`
    swaps it in, and the record goes last, through `_write_manifest`. A failure while staging
    removes the staged directory and leaves the last validation as it was. The record carries the
    digests of the tables it describes (`model_validation_hashes`), so a failure between the swap
    and the record, which leaves new tables beside the last record, shows as a mismatch. A failed
    gate publishes no tables and no manifest: the last validation's tables are replaced by none,
    never left beside a not-beaten record.
    """
    import shutil

    from .build import write_parquet_deterministic

    out = run / "state_model_validation"
    staged = out.with_name(f"{out.name}.partial")
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir(parents=True)
    try:
        hashes = {
            table: write_parquet_deterministic(frame, staged / f"{table}.parquet")
            for table, frame in tables.items()
        }
        if validation_manifest is not None:
            _write_manifest(
                staged / "validation_manifest.json",
                {**validation_manifest, "output_hashes": hashes},
            )
            record = {**record, "model_validation_hashes": hashes}
        _replace_directory(staged, out)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    _write_manifest(run / "promotion_record.json", record)


@app.command("validate-state-model")
def validate_state_model_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Score §11's model through §13's harness and write §13.10's promotion record.

    Not in §16.1's list: plan 16 adds it so `validate` stays the comparand's command, byte-identical
    to Stage 4's. The comparand is this run's own `validate` output. The precondition is that
    output, plus `fit-state-model`'s `posterior/diagnostics.json` from this run's constraint set:
    a fit from another is refused before anything is deleted (`_stale_fit`), and so is a passing
    fit whose store, summary and manifest are not all there, or disagree with the digests the
    manifest records (`_unfinished_fit`). A production fit that failed §11.14 is recorded as not
    beaten without running the harness. Otherwise the model is scored through
    `run_pseudo_suppression`'s own loop (`StateModelProducer`), which is 27 fits on D1. Its tables
    go to `state_model_validation/` and the verdict to `promotion_record.json`, and nothing of the
    last validation is touched until both are whole (`_publish_state_model_validation`). Both
    verdicts exit 0, because "deploy the simpler method" is an outcome and not an error.
    """
    import hashlib
    import json

    import polars as pl

    from .contracts import (
        VALIDATION_METRIC_SCHEMA,
        VALIDATION_SCORE_SCHEMA,
        VALIDATION_SCOREBOARD_SCHEMA,
        HarmonizedData,
        assert_required_columns_present,
        validate_frame,
    )
    from .models.arviz_io import draws_digest, read_store, store_digest
    from .models.interfaces import MODEL_ID, MODEL_VERSION, STORE_PATH
    from .models.reconciliation import check_reconciled
    from .models.validation import StateModelProducer
    from .runs import run_dir, run_id
    from .validate.harness import run_pseudo_suppression
    from .validate.promotion import evaluate_promotion, production_failed_record

    cfg = load_config(config)
    rid = run_id(cfg, _input_digests(cfg))
    run = run_dir(cfg, rid)
    comparand = {
        table: run / f"{table}.parquet"
        for table in ("validation_scores", "validation_metrics", "validation_scoreboard")
    }
    diagnostics = run / "posterior" / "diagnostics.json"
    for path, command in [
        *((p, "validate") for p in comparand.values()),
        (diagnostics, "fit-state-model"),
    ]:
        if not path.exists():
            raise typer.BadParameter(
                f"{path} is missing: §13.10 compares the model against this run's own comparand "
                f"and reads its production gate. Run `{command}` first"
            )
    # Before the gate is read and before anything is deleted: a stale fit is neither scored nor
    # written up as not beaten, a passing fit whose artifacts are not one finished fit is not
    # scored, and the last record survives either refusal. Reading the store only after the
    # deletion lost that record whenever the store had never been written (Codex on #41).
    stale = _stale_fit(run)
    if stale is not None:
        raise typer.BadParameter(
            f"{diagnostics} records constraint set {stale['fit_constraint_set_hash']!r}, but "
            f"{run / 'schema_manifest.json'} names {stale['constraint_set_hash']!r}: the fit was "
            "reconciled against another constraint set. Run `solve-bounds` and `fit-state-model` "
            "first"
        )
    unfinished = _unfinished_fit(run)
    if unfinished is not None:
        raise typer.BadParameter(
            f"{diagnostics} records a passing fit, but its artifacts are not one finished fit "
            f"({_unfinished_reasons(unfinished)}). Run `fit-state-model` first"
        )
    # NOTHING OF THE LAST VALIDATION IS TOUCHED UNTIL THE NEW ONE IS WHOLE. Every read, the harness
    # and the verdict come first, and only `_publish_state_model_validation` writes. The last record
    # and its tables were deleted here, so anything that failed after this point, from a comparand
    # that could not be read to an interrupt an hour into D1's 27 fits, cost them (#43's known gap,
    # of Codex's #41 class). Now such a failure leaves both byte for byte and raises as it did.
    envelope = {
        "run_id": rid,
        "model_version": MODEL_VERSION,
        "comparand": {
            "run_id": rid,
            **{
                f"{table}_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                for table, path in comparand.items()
            },
        },
    }
    production_gate = json.loads(diagnostics.read_text())
    if not production_gate["passed"]:
        record = production_failed_record(MODEL_ID, production_gate, cfg.promotion)
        _publish_state_model_validation(
            run, {**record, **envelope}, tables={}, validation_manifest=None
        )
        typer.echo("production fit failed §11.14: not_beaten, section_10_8_hierarchy selected")
        return

    store = run / STORE_PATH
    draws = read_store(store)
    check = check_reconciled(draws, tolerance=cfg.reconciliation.tolerance)
    store_check = {
        "max_anchor_drift": check.max_anchor_drift,
        "bound_violations": check.bound_violations,
        "draws_sha256_matches": draws_digest(draws) == store_digest(store),
    }
    store_check["passed"] = check.passed and bool(store_check["draws_sha256_matches"])

    data = HarmonizedData.load(Path(cfg.storage.staged_uri))
    result = run_pseudo_suppression(data, config=cfg, producer=StateModelProducer())
    for frame, schema, table in (
        (result.scores, VALIDATION_SCORE_SCHEMA, "validation_scores"),
        (result.metrics, VALIDATION_METRIC_SCHEMA, "validation_metrics"),
        (result.scoreboard, VALIDATION_SCOREBOARD_SCHEMA, "validation_scoreboard"),
    ):
        validate_frame(frame, schema, table)
        assert_required_columns_present(frame, table)
    record = evaluate_promotion(
        model_id=MODEL_ID,
        model_scores=result.scores,
        model_metrics=result.metrics,
        comparand_scores=pl.read_parquet(comparand["validation_scores"]),
        comparand_metrics=pl.read_parquet(comparand["validation_metrics"]),
        comparand_scoreboard=pl.read_parquet(comparand["validation_scoreboard"]),
        production_gate=production_gate,
        production_store_check=store_check,
        replicate_gates={
            regime: list(entry["producer_notes"])
            for regime, entry in result.manifest["regimes"].items()
            if entry.get("producer_notes")
        },
        promotion=cfg.promotion,
    )
    _publish_state_model_validation(
        run,
        {**record, **envelope},
        tables={
            "validation_scores": result.scores,
            "validation_metrics": result.metrics,
            "validation_scoreboard": result.scoreboard,
        },
        validation_manifest={**result.manifest, "estimators": [MODEL_ID]},
    )
    typer.echo(f"verdict {record['verdict']}; selected {record['selected_method']}")
