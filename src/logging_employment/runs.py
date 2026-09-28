"""The `runs/<run_id>/` layout of §6.2, with a run id derived from inputs rather than a clock.

§16.1 requires every command to be idempotent for the same inputs. A timestamp or a UUID would
make each invocation land in a new directory, so "idempotent" could only ever mean "wrote the same
bytes somewhere else". Deriving the id from the resolved configuration and the input digests makes
a repeated run land on its own previous output, where byte-identity is checkable.

Also here: `replace_whole`, how a file under §6.2's roots is written to a `.partial` sibling and
renamed into place, so a reader finds the last whole file or the new one and never one cut short
(`D-134`). It lives here rather than beside `build.write_parquet_deterministic` because the raw
store and the source manifest write no Parquet through that writer, and this module costs them only
`config`. `cli.py`'s manifest writer and its staged publish name their siblings through
`partial_path` too, so an unfinished write looks the same everywhere and no reader opens one.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from .config import Config, resolved_dict

RUN_ID_LENGTH = 12


def run_id(
    config: Config,
    input_digests: Mapping[str, str],
    *,
    overrides: Mapping[str, object] | None = None,
) -> str:
    """A stable identifier for one run: the resolved config, every input's digest, and overrides.

    `overrides` carries a command-line choice that changes WHAT the run computes but that no
    config file declared -- today, `validate --estimators`. It must reach the id: a subset that
    stopped at `argv` would hash to the same directory as a full pass and overwrite its outputs
    with different bytes under one identifier, which is precisely the byte-identity this scheme
    exists to make checkable.

    The key is OMITTED, not emitted as null, when there is no override. That keeps the payload of
    an un-overridden run byte-identical to the payload used before this parameter existed, so
    adding the option renumbered no existing run directory -- `runs/f03023ac9f3a` still resolves
    from `config.yaml`. Emitting `"overrides": null` would have re-identified every run on disk
    and orphaned Stage 4's acceptance artifact for a key that says nothing.
    """
    payload: dict[str, object] = {
        "config": resolved_dict(config),
        "inputs": dict(sorted(input_digests.items())),
    }
    if overrides:
        payload["overrides"] = dict(sorted(overrides.items()))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[
        :RUN_ID_LENGTH
    ]


def run_dir(config: Config, identifier: str) -> Path:
    """The directory this run's outputs belong in."""
    return Path(config.storage.output_uri) / identifier


def partial_path(path: Path) -> Path:
    """The sibling a file or directory is written to before it is renamed over `path`.

    One name for every such sibling, so the manifest writer, the staged promotion record, the
    staged validation tables, the raw store and the source manifest agree on what an unfinished
    write looks like. No reader opens one: every reader names its file in full or globs a suffix
    (`*.parquet`, `*.csv`), which `.partial` never matches.
    """
    return path.with_name(f"{path.name}.partial")


@contextmanager
def replace_whole(path: Path) -> Iterator[Path]:
    """Yield `path`'s `.partial` sibling to write to, and rename it over `path` on a clean exit.

    `Path.write_bytes`, `Path.write_text` and polars' `write_parquet` all truncate before they
    write, so a full disk or a kill mid-write left the file cut short where readers look
    (`D-134`): a raw object whose path claimed a digest its bytes did not have, and which every
    later `put` then reported present; a `runs/source_manifest.parquet` that failed to parse, and
    with it the record of which snapshots a run used; a `config.resolved.yaml` half written.
    `os.replace` is atomic within a directory, so a reader sees the last whole file or the new one
    and never a mix, which is what `cli._write_manifest` already did for every run manifest. A
    write that raises removes its sibling. A process killed outright can leave one, and the next
    write to the path overwrites it. Nothing is fsynced: this is atomic against a process that
    dies, not a machine that does, the limit `cli._publish_state_model_validation` states too.
    """
    partial = partial_path(path)
    try:
        yield partial
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


UNKNOWN_PROVENANCE = "unknown"


def _code_root(start: Path) -> Path | None:
    """The nearest ancestor of `start` holding `uv.lock`, or `None` when no ancestor does.

    `uv.lock` is the anchor because it is the one file that must exist in a checkout of THIS
    project and must not exist in a wheel built from it -- `[tool.hatch.build.targets.wheel]`
    ships `src/logging_employment` and nothing else. Anchoring on `.git` instead would find the
    enclosing repository of a site-packages copy that some unrelated checkout happens to sit
    inside, and stamp a commit that never produced the running code. One anchor for both stamps is
    the point: when this returns `None`, "which code ran" is honestly unanswerable and both keys
    say so together rather than one of them guessing.
    """
    for candidate in (start, *start.parents):
        if (candidate / "uv.lock").is_file():
            return candidate
    return None


def _git_output(root: Path, *args: str) -> str | None:
    """`git -C root <args>` stripped stdout, or `None` when git could not answer.

    `None` for every way the question goes unanswered -- git absent from PATH (`OSError`), the
    command hanging (`SubprocessError`, hence the timeout), or a non-zero exit, which is what
    `root` not being a repository looks like. `check=False` is the classification, not an
    oversight: a probe that reads the return code cannot also let it raise.
    """
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def code_provenance(start: Path | None = None) -> dict[str, str]:
    """Which source and which locked dependencies produced a run, for stamping into its manifest.

    RECORDS, NEVER RAISES -- the one deliberate exception to §18.3's fail-closed default. That
    default guards values that would corrupt an estimate; a commit id corrupts nothing, it only
    tells a reader whether `runs/<id>/` still corresponds to the code in front of them. Halting a
    pipeline because `git` is missing would turn a diagnostic into an outage and make the package
    unusable from an sdist or a vendored copy, so an unanswerable probe records `"unknown"` --
    which is itself the finding, since a manifest that cannot name its commit is one a reviewer
    must not treat as reproducible.

    Nothing here reaches `run_id`, and that is the whole design. Hashing the commit into the id
    would rename every directory under `runs/` on every commit, so §16.1's "idempotent for the
    same inputs" would become unobservable and Stage 4's acceptance artifact would orphan itself
    on the next commit. Staleness -- CLAUDE.md's "a run directory can be stale w.r.t. your code"
    -- is made DETECTABLE here, not impossible.

    A DIRTY TREE IS NOT ITS COMMIT. `git rev-parse HEAD` answers on a dirty worktree, and a bare
    sha from one is a false "this run matches that commit": exactly the silent failure the stamp
    exists to prevent. The `-dirty` suffix (`git describe --dirty`'s convention) puts that in the
    value a reader compares, rather than in a sibling boolean they can forget to read. Untracked
    files count as dirty because Python imports whatever is on disk, so an uncommitted module is
    part of the code that ran. An unanswerable `git status` after an answerable `rev-parse` also
    marks dirty -- "could not verify clean" must not read as "clean".

    `start` is where the upward walk begins; the default is this module's own directory, so the
    stamp describes the code that is executing rather than the caller's cwd. Tests pass a path
    outside any checkout to reach the unknown branch without mocking `subprocess` or `PATH`.
    """
    root = _code_root(Path(__file__).resolve().parent if start is None else start)
    if root is None:
        return {"code_commit": UNKNOWN_PROVENANCE, "uv_lock_sha256": UNKNOWN_PROVENANCE}
    lock = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    commit = _git_output(root, "rev-parse", "HEAD")
    if commit is None:
        return {"code_commit": UNKNOWN_PROVENANCE, "uv_lock_sha256": lock}
    status = _git_output(root, "status", "--porcelain")
    return {"code_commit": commit if status == "" else f"{commit}-dirty", "uv_lock_sha256": lock}
