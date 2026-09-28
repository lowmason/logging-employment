"""The `logging-estimates` command-line interface."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import typer

from .config import load_config
from .runs import partial_path as _partial

if TYPE_CHECKING:  # annotations only; keeps CLI start-up cheap
    from collections.abc import Iterator, Mapping

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


def _stage_manifest(path: Path, payload: Mapping[str, object]) -> Path:
    """Write the bytes `_write_manifest` puts at `path` to its `.partial` sibling, and return it.

    The format lives here and only here. `_write_manifest` renames the sibling over `path` at
    once. `_publish_state_model_validation` stages the promotion record this way and renames it
    only once the tables it describes are in place, as its commit point (Codex on #44), and a
    second `json.dumps` for that record would let the two formats drift apart.

    The stamp is merged LAST so no caller can shadow or drop it, and the JSON keeps the
    `indent=2, sort_keys=True` shape every one of these files already had -- `solve-bounds` reads
    `schema_manifest.json` as a precondition gate and an integration test pins its bytes, so the
    formatting is not free to drift. A write that fails removes its sibling.
    """
    import json

    from .runs import code_provenance

    text = json.dumps({**payload, **code_provenance()}, indent=2, sort_keys=True)
    partial = _partial(path)
    try:
        partial.write_text(text)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return partial


def _write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    """Write one run manifest, stamped with the code identity that produced it.

    THE point of funnelling all five manifests through one writer. `run_id` covers config and
    input data but deliberately not source (`runs.code_provenance`), so `code_commit` and
    `uv_lock_sha256` are the only things in `runs/<id>/` that can answer "was this directory
    written by the code I am reading?". A per-command `json.dumps` at each site made adding them
    five edits, and made forgetting them on the sixth manifest the default outcome. The bytes are
    `_stage_manifest`'s, which the promotion record shares.

    A MANIFEST IS REPLACED WHOLE (Codex on #42). `Path.write_text` truncates before it writes, so
    a full disk or a kill mid-write left an empty or partial manifest where readers look, and
    `_unfinished_fit` read that file's existence as a finished fit. It now parses the manifest and
    checks the digests it records, and a manifest written here is never partial. The text goes to a
    `.partial` sibling, and `os.replace` renames it over the manifest, which is atomic within a
    directory. A write that fails removes its sibling. A process killed outright can leave one, and
    no reader opens it.
    """
    import os

    partial = _stage_manifest(path, payload)
    try:
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


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
    from .runs import replace_whole, run_dir, run_id

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
    # Replaced whole (`D-134`): `write_text` truncates first, so a kill or a full disk mid-write
    # left the resolved config cut short in a directory whose manifests are never partial.
    with replace_whole(run / "config.resolved.yaml") as partial:
        partial.write_text(yaml.safe_dump(resolved_dict(cfg), sort_keys=True))
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


def _stale_reason(stale: Mapping[str, str | None]) -> str:
    """`_stale_fit`'s finding as a clause and the command that clears it, claiming no more.

    Two recorded sets that differ were reconciled apart. A missing one is something else: the
    report or this run's `schema_manifest.json` is absent, cannot be read, or names no set. Both
    commands said "reconciled against constraint set None" of it, which no fit ever was (#43's
    follow-up). So it is "unrecorded", one word that Typer's box cannot wrap in two, and the remedy
    starts at the command that writes the missing record.
    """
    fit, current = stale["fit_constraint_set_hash"], stale["constraint_set_hash"]
    if current is None:
        return (
            "this run's constraint set is unrecorded (its schema_manifest.json is absent, cannot be "
            "read, or names none). Run `build-constraints`, `solve-bounds` and `fit-state-model` first"
        )
    if fit is None:
        return (
            "the fit's constraint set is unrecorded (its report, posterior/diagnostics.json, is "
            "absent, cannot be read, or names none). Run `fit-state-model` first"
        )
    return (
        f"the fit was reconciled against constraint set {fit!r}, and this run's is {current!r}. "
        "Run `solve-bounds` and `fit-state-model` first"
    )


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
    one to distrust. `_read_comparand`'s findings share the shape and so the clause.
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
        typer.echo(f"state-total draws not checked: {_stale_reason(stale)}", err=True)
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


_COMPARAND_TABLES = ("validation_scores", "validation_metrics", "validation_scoreboard")


class _Comparand(NamedTuple):
    """This run's `validate` output as `_read_comparand` found it."""

    frames: dict[str, pl.DataFrame]
    sha256: dict[str, str]
    found: dict[str, list[str]] | None


def _read_comparand(run: Path) -> _Comparand:
    """This run's `validate` output, each table read once, and why it is not what `validate` wrote.

    §13.10 compares the model against this run's own comparand, and the record names it by
    digest. The command took both on trust: it hashed whatever tables it found, parsed them again
    from their paths after the harness, and never asked whether `validate` wrote them (#44, found
    beside Codex's P1 on its publish). `validate` records the digest of each table it writes in
    `validation_manifest.json`'s `output_hashes` (`validate_command`), so a table whose bytes are
    not those is `mismatched`, the word `_unfinished_fit` uses for a fit artifact its manifest does
    not vouch for, and no model is scored against a comparand that no `validate` wrote.

    ONE READ PER TABLE. The bytes hashed are the bytes parsed, from memory, so a write between the
    two -- `validate` re-run beside this command -- cannot make the record name one comparand while
    the verdict is measured against another.

    NO READ RAISES PAST THIS, the rule `_unfinished_fit` keeps (Codex on #43). A manifest that
    cannot be read, does not parse to an object, or records no digest for a table, or anything but
    a string for one, is `unreadable`: a `null` compared as a digest would blame the table, the one
    file that is whole. So is a table that cannot be read or, its digest matching, does not parse
    as Parquet, whether polars raises or panics. A panic is `PanicException`, which subclasses
    `BaseException`, not `PolarsError`, and the #44 review measured 5 of 400 corrupted tables
    raising it. `found` maps each kind to run-relative paths, as `_unfinished_fit`'s does, and is
    `None` only when every table is what the manifest records, when `frames` holds all three.
    """
    import hashlib
    import io
    import json

    import polars as pl

    manifest = "validation_manifest.json"
    try:
        recorded = json.loads((run / manifest).read_text())["output_hashes"]
        expected = {table: recorded[table] for table in _COMPARAND_TABLES}
    except OSError, ValueError, KeyError, TypeError:
        return _Comparand({}, {}, {"unreadable": [manifest]})
    if not all(isinstance(digest, str) for digest in expected.values()):
        return _Comparand({}, {}, {"unreadable": [manifest]})
    frames: dict[str, pl.DataFrame] = {}
    sha256: dict[str, str] = {}
    found: dict[str, list[str]] = {}
    for table in _COMPARAND_TABLES:
        path = f"{table}.parquet"
        try:
            raw = (run / path).read_bytes()
        except OSError:
            found.setdefault("unreadable", []).append(path)
            continue
        sha256[table] = hashlib.sha256(raw).hexdigest()
        if sha256[table] != expected[table]:
            found.setdefault("mismatched", []).append(path)
            continue
        try:
            frames[table] = pl.read_parquet(io.BytesIO(raw))
        except pl.exceptions.PolarsError, pl.exceptions.PanicException:
            found.setdefault("unreadable", []).append(path)
    return _Comparand(frames, sha256, found or None)


class _ValidationPaths(NamedTuple):
    """Where a state-model validation and an unfinished publish of one live in a run directory.

    Named once, because `_settle_state_model_validation` reads a publish's progress from exactly
    these five, and a name spelled differently in the publisher would be a leftover it never sees.
    """

    record: Path
    staged_record: Path
    tables: Path
    staged_tables: Path
    old_tables: Path

    @classmethod
    def of(cls, run: Path) -> _ValidationPaths:
        """The five paths in `run`."""
        record = run / "promotion_record.json"
        tables = run / "state_model_validation"
        return cls(
            record=record,
            staged_record=_partial(record),
            tables=tables,
            staged_tables=_partial(tables),
            old_tables=tables.with_name(f"{tables.name}.old"),
        )


def _settle_state_model_validation(run: Path) -> None:
    """Finish or undo a publish of `run`'s state-model validation that did not run to its end.

    ONE FUNCTION FOR AN EXCEPTION AND FOR A KILL, and it reads the disk, not the code path.
    `validate-state-model` calls it first, under the run's lock, before it reads anything, and
    `_publish_state_model_validation` calls it before it writes a byte and as its handler, for a
    publish that raised. Which call raised proves nothing about what moved: an interrupt can land
    after `os.replace` has renamed and before it returns. So the leftovers decide, and running
    this twice does nothing the first run did not.

    THE STAGED RECORD IS THE EVIDENCE. `promotion_record.json.partial` is the last byte a publish
    writes and the first thing an undo removes, so while it exists the publish has not committed
    and the last record is still `promotion_record.json`. The new tables are then wherever the
    renames left them: staged until they were renamed in, `state_model_validation/` after. A staged
    record with no staged directory therefore means they were moved in, and they are moved back
    out -- which is the only way to undo a publish that had no last tables to move to `.old`, the
    first one among them. The last tables come back from `.old`, then the staged record and the
    staged directory are removed, in that order. With no staged record, a `.old` beside the
    tables is the tail of a publish that committed, and is deleted, and a `.old` alone is the last
    tables of a swap killed between its renames by the code before this protocol, and is moved
    back. A staged directory with no staged record is a write that did not finish, or new tables
    an undo moved out and was stopped before deleting, and either way it is deleted.

    IT RAISES RATHER THAN GUESS. Nothing it removes is the last validation, so a failure here
    leaves a state the next call settles. It never removes `.old` quietly: a publish beside one it
    could not delete would fail its first rename, and its undo would stop on
    `UnsettledPublishError` with staged leftovers, blaming two invocations at once for a directory
    that could not be deleted, so the deletion's own error is raised before anything is staged.
    And `os.replace` onto an existing empty directory succeeds without a word, so a `.old` beside
    tables that are not new is refused (`UnsettledPublishError`) rather than renamed over them.

    ONLY UNDER THE RUN'S LOCK (`_run_lock`, `D-138`). Leftovers look the same whether their
    publish was killed or is still running, so every guarantee here holds only while no other
    publish is running on the run. Beside one, a settle undoes its staging under it. Both
    interleavings were measured in scratch against these functions, the first by #44's pre-push
    review and the second after it. During the renames, the publish fails at its commit,
    its handler's settle deletes the last tables as a committed tail, and the last record is left
    beside the new tables with nothing for a later settle to find. During the staging, the settle
    deletes the staging directory as an unfinished write, `build.write_parquet_deterministic`
    recreates it for the next table, and the publish returns without an error, having committed a
    record beside tables missing those written before. `validate_state_model_command` holds the
    lock around every call it makes, its opening settle included.
    """
    import os
    import shutil

    from .errors import UnsettledPublishError

    paths = _ValidationPaths.of(run)
    if paths.staged_record.exists():
        if paths.tables.exists() and not paths.staged_tables.exists():
            os.replace(paths.tables, paths.staged_tables)
        if paths.old_tables.exists():
            if paths.tables.exists():
                raise UnsettledPublishError(
                    f"{paths.old_tables} holds the last state-model validation's tables beside "
                    f"{paths.tables}, which no single publish leaves: settle the two by hand"
                )
            os.replace(paths.old_tables, paths.tables)
        paths.staged_record.unlink()
    elif paths.old_tables.exists():
        if paths.tables.exists():
            shutil.rmtree(paths.old_tables)
        else:
            os.replace(paths.old_tables, paths.tables)
    if paths.staged_tables.exists():
        shutil.rmtree(paths.staged_tables)


def _publish_state_model_validation(
    run: Path,
    record: Mapping[str, object],
    *,
    tables: Mapping[str, pl.DataFrame],
    validation_manifest: Mapping[str, object] | None,
) -> None:
    """Replace `promotion_record.json` and `state_model_validation/` together, or not at all.

    BYTES FIRST, RENAMES LAST (Codex on #44). The last validation is its record and its tables,
    and the order before this -- swap the tables in, delete the last ones, then write the record --
    left the last record beside the new tables whenever the record's write failed, a full disk
    above all. Now every byte is written before anything moves: the tables and their manifest to
    `state_model_validation.partial/`, and the record, through `_stage_manifest`, to
    `promotion_record.json.partial`, last. What follows is renames only: the last tables to
    `.old`, the staged tables in, and the staged record over the last one. THAT RENAME IS THE
    COMMIT: before it the last record stands, and `_settle_state_model_validation` puts its
    tables back; after it the new record stands beside its own tables. `.old` is deleted only
    then, and a failure deleting it costs nothing the next publish does not remove.

    Any exception is raised as it was, once `_settle_state_model_validation` has left one whole
    validation: the last, byte for byte, unless the record's rename ran, and the new one if it
    did. If the settle fails too, its error is raised instead, with the original as its context,
    and the next call settles what it left. A kill anywhere leaves leftovers that the next
    `validate-state-model` settles before it reads anything. Until then, a kill between the first
    rename and the commit leaves the last record beside new tables, or none, and nothing in this
    package reads `state_model_validation/`. An interrupt after the commit exits non-zero with the
    new validation published. Nothing is fsynced, so this is atomic against a process that dies,
    not a machine that does. And all of this holds for one publish at a time: the command calls
    this under the run's lock (`_run_lock`), because another invocation's settle would undo it in
    flight (`_settle_state_model_validation`, `D-138`).

    The record carries the digests of the tables it describes (`model_validation_hashes`). A
    failed gate publishes no tables and no manifest: the last validation's tables are replaced by
    none, never left beside a not-beaten record. The bytes are those the command always wrote:
    nothing staged records its own path, and a rename keeps each file's sha256.
    """
    import os
    import shutil

    from .build import write_parquet_deterministic

    paths = _ValidationPaths.of(run)
    _settle_state_model_validation(run)
    try:
        paths.staged_tables.mkdir(parents=True)
        hashes = {
            table: write_parquet_deterministic(frame, paths.staged_tables / f"{table}.parquet")
            for table, frame in tables.items()
        }
        if validation_manifest is not None:
            _write_manifest(
                paths.staged_tables / "validation_manifest.json",
                {**validation_manifest, "output_hashes": hashes},
            )
            record = {**record, "model_validation_hashes": hashes}
        staged_record = _stage_manifest(paths.record, record)
        if paths.tables.exists():
            os.replace(paths.tables, paths.old_tables)
        os.replace(paths.staged_tables, paths.tables)
        os.replace(staged_record, paths.record)
    except BaseException:
        _settle_state_model_validation(run)
        raise
    shutil.rmtree(paths.old_tables, ignore_errors=True)


@contextmanager
def _run_lock(run: Path) -> Iterator[None]:
    """Hold `run`'s `validate-state-model` lock, or refuse with `RunInUseError` (`D-138`).

    A settle cannot tell a killed publish's leftovers from a running one's, so two invocations on
    one run must never overlap: the second one's opening settle would undo the first's publish
    under it (`_settle_state_model_validation`). An `flock` rather than a lock file's existence,
    because the kernel releases it when its holder exits, killed or not, so no crash leaves a run
    locked and nothing has to guess whether a lock is stale. The file is kept, empty, after the
    lock is released. Deleting it would let an invocation that opened the old file before the
    deletion lock it once released, while another creates and locks a new file at the same path:
    two holders at once. A run directory that does not exist holds nothing to settle or
    publish, and the command refuses it at its first precondition, so it is not created to be
    locked.
    """
    import fcntl

    from .errors import RunInUseError

    if not run.is_dir():
        yield
        return
    lock = run / "validate_state_model.lock"
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RunInUseError(
                f"another validate-state-model is running on {run}, and holds {lock.name}: "
                "a second one beside it could undo its publish. Wait for it to finish"
            ) from error
        yield


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
    go to `state_model_validation/` and the verdict to `promotion_record.json`, both written in full
    before either replaces the last, and the record's rename is the commit
    (`_publish_state_model_validation`). Both verdicts exit 0, because "deploy the simpler method"
    is an outcome and not an error. The whole of it runs under the run's lock (`_run_lock`), and a
    second invocation on the same run is refused before it touches anything.
    """
    from .errors import RunInUseError
    from .runs import run_dir, run_id

    cfg = load_config(config)
    rid = run_id(cfg, _input_digests(cfg))
    run = run_dir(cfg, rid)
    try:
        with _run_lock(run):
            _validate_state_model(cfg, rid, run)
    except RunInUseError as error:
        raise typer.BadParameter(str(error)) from error


def _validate_state_model(cfg: Config, rid: str, run: Path) -> None:
    """`validate_state_model_command`'s body, under the run's lock: check, score and publish."""
    import hashlib
    import json

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
    from .validate.harness import run_pseudo_suppression
    from .validate.promotion import evaluate_promotion, production_failed_record

    # A publish killed last time left its leftovers, and until they are settled the last record
    # can stand beside new tables. Settled here, before any refusal below, so the next run of this
    # command closes that window even when it scores nothing. It restores or finishes; it never
    # removes the last validation. Only under the run's lock: beside another invocation's publish
    # it would undo that publish under it (`_run_lock`, `D-138`).
    _settle_state_model_validation(run)
    diagnostics = run / "posterior" / "diagnostics.json"
    for path, command in [
        *((run / f"{table}.parquet", "validate") for table in _COMPARAND_TABLES),
        (run / "validation_manifest.json", "validate"),
        (diagnostics, "fit-state-model"),
    ]:
        if not path.exists():
            # The file is named run-relative, as a word of its own: Typer's box folds a word
            # longer than its width, so an absolute path's file name can split across two lines.
            raise typer.BadParameter(
                f"{path.relative_to(run)} is missing from {run}: §13.10 compares the model against "
                f"this run's own comparand and reads its production gate. Run `{command}` first"
            )
    # Before the gate is read and before anything is deleted: a stale fit is neither scored nor
    # written up as not beaten, a passing fit whose artifacts are not one finished fit is not
    # scored, and the last record survives either refusal. Reading the store only after the
    # deletion lost that record whenever the store had never been written (Codex on #41).
    stale = _stale_fit(run)
    if stale is not None:
        raise typer.BadParameter(f"the state-total model is not scored: {_stale_reason(stale)}")
    unfinished = _unfinished_fit(run)
    if unfinished is not None:
        raise typer.BadParameter(
            f"{diagnostics} records a passing fit, but its artifacts are not one finished fit "
            f"({_unfinished_reasons(unfinished)}). Run `fit-state-model` first"
        )
    # The comparand is read once, here, and refused unless it is what `validate` recorded writing,
    # so the record names by digest the tables the verdict is measured against (`_read_comparand`).
    comparand = _read_comparand(run)
    if comparand.found is not None:
        raise typer.BadParameter(
            f"this run's comparand is not what `validate` wrote "
            f"({_unfinished_reasons(comparand.found)}). Run `validate` first"
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
            **{f"{table}_sha256": comparand.sha256[table] for table in _COMPARAND_TABLES},
        },
    }
    # WHICH FIT the verdict describes (#44). `fit-state-model` can re-run on this run after the
    # record is written, and nothing in the record said which fit it was. The gate and the digest
    # the record carries come from one read of the report, so the gate is parsed from the bytes the
    # fit is identified by. `_stale_fit` and `_unfinished_fit` read it before, each on its own, so a
    # report replaced between their checks and this read goes unnoticed: the run's lock keeps out
    # another `validate-state-model`, not a `fit-state-model`. A failed gate writes no draws, and
    # the passing branch adds the digest of the draws it checked. Nothing in this package compares
    # these with a later fit yet; they make the record answerable.
    report = diagnostics.read_bytes()
    production_gate = json.loads(report)
    fit = {
        "diagnostics_sha256": hashlib.sha256(report).hexdigest(),
        "constraint_set_hash": production_gate["constraint_set_hash"],
        "draws_sha256": None,
    }
    if not production_gate["passed"]:
        record = production_failed_record(MODEL_ID, production_gate, cfg.promotion)
        _publish_state_model_validation(
            run, {**record, **envelope, "fit": fit}, tables={}, validation_manifest=None
        )
        typer.echo("production fit failed §11.14: not_beaten, section_10_8_hierarchy selected")
        return

    store = run / STORE_PATH
    draws = read_store(store)
    fit["draws_sha256"] = draws_digest(draws)
    check = check_reconciled(draws, tolerance=cfg.reconciliation.tolerance)
    store_check = {
        "max_anchor_drift": check.max_anchor_drift,
        "bound_violations": check.bound_violations,
        "draws_sha256_matches": fit["draws_sha256"] == store_digest(store),
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
        comparand_scores=comparand.frames["validation_scores"],
        comparand_metrics=comparand.frames["validation_metrics"],
        comparand_scoreboard=comparand.frames["validation_scoreboard"],
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
        {**record, **envelope, "fit": fit},
        tables={
            "validation_scores": result.scores,
            "validation_metrics": result.metrics,
            "validation_scoreboard": result.scoreboard,
        },
        validation_manifest={**result.manifest, "estimators": [MODEL_ID]},
    )
    typer.echo(f"verdict {record['verdict']}; selected {record['selected_method']}")
