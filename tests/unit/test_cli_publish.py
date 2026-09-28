"""`cli._publish_state_model_validation` and `cli._settle_state_model_validation`.

A state-model validation is its record and its tables, and `validate-state-model` replaces them
together or not at all (Codex on #44). Every test here asks one question of a publish that raised
or was killed at some step: is exactly one whole validation left, the last or the new one? A tree
is compared whole, empty directories included, because `os.replace` onto an empty directory
succeeds without a word. The designs these tests pin were checked first in scratch against 3,456
injected faults and 256 real kills, and four planted mutants of the protocol each turn one red.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from logging_employment import build, runs
from logging_employment.cli import (
    _publish_state_model_validation,
    _settle_state_model_validation,
)
from logging_employment.errors import UnsettledPublishError

SCORES = pl.DataFrame({"cell_id": ["b", "a"], "estimate": [2.0, 1.0]})
# What an unfinished publish leaves beside the run's outputs.
LEFTOVERS = (
    "state_model_validation.partial",
    "state_model_validation.old",
    "promotion_record.json.partial",
)
# The three renames of a publish, in order. The last is its commit.
RENAMES = [
    ("state_model_validation", "state_model_validation.old"),
    ("state_model_validation.partial", "state_model_validation"),
    ("promotion_record.json.partial", "promotion_record.json"),
]


def _plant_the_last_validation(run: Path) -> dict[Path, str]:
    """The last `validate-state-model`'s record and one of its tables, as text."""
    last = {
        run / "promotion_record.json": "the last record",
        run / "state_model_validation" / "validation_scores.parquet": "the last scores",
    }
    for path, text in last.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return last


def _tree(run: Path) -> dict[str, bytes | None]:
    """Every path under `run`, a directory as `None`, so an empty directory counts."""
    return {
        path.relative_to(run).as_posix(): None if path.is_dir() else path.read_bytes()
        for path in sorted(run.rglob("*"))
    }


def _publish(run: Path, gate_passed: bool = True) -> None:
    """A publish of the passing branch's shape, or of the failed gate's: no tables, no manifest."""
    if gate_passed:
        _publish_state_model_validation(
            run,
            {"verdict": "beat"},
            tables={"validation_scores": SCORES},
            validation_manifest={"regimes": {}},
        )
    else:
        _publish_state_model_validation(
            run, {"verdict": "not_beaten"}, tables={}, validation_manifest=None
        )


def _failing_at_the_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """`os.replace` fails when it renames the staged record, and only then."""
    replace = os.replace

    def refuse(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
        if Path(src).name == "promotion_record.json.partial":
            raise OSError(errno.EIO, "Input/output error")
        replace(src, dst)

    monkeypatch.setattr(os, "replace", refuse)


def test_a_validation_is_published_with_the_digests_of_its_own_tables(tmp_path: Path) -> None:
    """The tables are staged and renamed into place, so the digests the validation manifest and
    the record carry are the bytes readers find, and nothing is left beside them."""
    _plant_the_last_validation(tmp_path)
    _publish(tmp_path)
    out = tmp_path / "state_model_validation"
    table = out / "validation_scores.parquet"
    written = {"validation_scores": hashlib.sha256(table.read_bytes()).hexdigest()}
    record = json.loads((tmp_path / "promotion_record.json").read_text())
    assert record["model_validation_hashes"] == written
    assert json.loads((out / "validation_manifest.json").read_text())["output_hashes"] == written
    assert sorted(child.name for child in tmp_path.iterdir()) == [
        "promotion_record.json",
        "state_model_validation",
    ]


def test_the_commit_is_the_last_mutation_of_a_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every byte is written before anything moves, and the record's rename is the last move.
    The settle reads a publish's progress from exactly this order, so a step added to it must be
    walked through the settle and the kill test below before it is pinned here."""
    _plant_the_last_validation(tmp_path)
    moves: list[str] = []
    replace, rmtree = os.replace, shutil.rmtree

    def logged_replace(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
        moves.append(f"{Path(src).name} -> {Path(dst).name}")
        replace(src, dst)

    def logged_rmtree(path: os.PathLike[str] | str, *args: object, **kwargs: object) -> None:
        moves.append(f"delete {Path(path).name}")
        rmtree(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", logged_replace)
    monkeypatch.setattr(shutil, "rmtree", logged_rmtree)
    _publish(tmp_path)
    monkeypatch.undo()
    assert moves == [
        "validation_manifest.json.partial -> validation_manifest.json",
        "state_model_validation -> state_model_validation.old",
        "state_model_validation.partial -> state_model_validation",
        "promotion_record.json.partial -> promotion_record.json",
        "delete state_model_validation.old",
    ]


def test_a_publish_that_fails_while_staging_leaves_the_last_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A table write that fails, a full disk here, leaves the last record and tables as they were
    and nothing staged."""
    last = _plant_the_last_validation(tmp_path)

    def disk_full(frame: pl.DataFrame, path: Path) -> str:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(build, "write_parquet_deterministic", disk_full)
    with pytest.raises(OSError, match="No space left"):
        _publish(tmp_path)
    monkeypatch.undo()
    assert {path: path.read_text() for path in last} == last
    assert sorted(child.name for child in tmp_path.iterdir()) == [
        "promotion_record.json",
        "state_model_validation",
    ]


def test_a_record_that_cannot_be_written_leaves_the_last_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex on #44. The tables were swapped in and the last ones deleted before the record was
    written, so a record write that failed, a full disk here, left the last record beside the new
    tables. The record is now staged before anything moves, and a failure there moves nothing."""
    last = _plant_the_last_validation(tmp_path)
    write_text = Path.write_text

    def disk_full(self: Path, data: str, *args: object, **kwargs: object) -> int:
        if self.name.startswith("promotion_record.json"):
            write_text(self, data[: len(data) // 2])
            raise OSError(errno.ENOSPC, "No space left on device")
        return write_text(self, data, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", disk_full)
    with pytest.raises(OSError, match="No space left"):
        _publish(tmp_path)
    monkeypatch.undo()
    assert {path: path.read_text() for path in last} == last
    assert sorted(child.name for child in tmp_path.iterdir()) == [
        "promotion_record.json",
        "state_model_validation",
    ]


@pytest.mark.parametrize("gate_passed", [True, False], ids=["tables", "failed_gate"])
@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("rename", RENAMES, ids=lambda rename: f"{rename[0]}->{rename[1]}")
def test_an_interrupt_at_any_rename_leaves_one_whole_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rename: tuple[str, str],
    when: str,
    gate_passed: bool,
) -> None:
    """An interrupt can land before a rename, or after it has renamed and before it returns, so
    the handler cannot trust which call raised and reads the disk instead. Only an interrupt after
    the record's rename finds the new validation; every other finds the last, byte for byte. The
    failed gate matters: its tables are an empty directory, which `os.replace` overwrites quietly."""
    new_run = tmp_path / "new"
    new_run.mkdir()
    _plant_the_last_validation(new_run)
    _publish(new_run, gate_passed)
    new = _tree(new_run)
    run = tmp_path / "run"
    run.mkdir()
    _plant_the_last_validation(run)
    last = _tree(run)
    replace = os.replace

    def interrupted(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
        if (Path(src).name, Path(dst).name) == rename and Path(dst).parent == run:
            if when == "after":
                replace(src, dst)
            raise KeyboardInterrupt
        replace(src, dst)

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _publish(run, gate_passed)
    monkeypatch.undo()
    committed = rename == RENAMES[-1] and when == "after"
    assert _tree(run) == (new if committed else last)


def test_a_first_publish_that_fails_at_its_commit_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no last tables there is no `.old` to tell the undo what moved: the staged record with
    no staged directory does, and the new tables are moved back out."""
    _failing_at_the_commit(monkeypatch)
    with pytest.raises(OSError, match="Input/output"):
        _publish(tmp_path)
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


def test_the_last_tables_left_at_old_by_a_kill_are_put_back_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3accb25's swap killed between its renames left the last tables at `.old` and no target.
    The next publish puts them back before anything else, so even when it fails at its commit,
    the last tables are where readers look."""
    (tmp_path / "promotion_record.json").write_text("the last record")
    old = tmp_path / "state_model_validation.old"
    old.mkdir()
    (old / "validation_scores.parquet").write_text("the last scores")
    _failing_at_the_commit(monkeypatch)
    with pytest.raises(OSError, match="Input/output"):
        _publish(tmp_path)
    monkeypatch.undo()
    table = tmp_path / "state_model_validation" / "validation_scores.parquet"
    assert table.read_text() == "the last scores"
    assert (tmp_path / "promotion_record.json").read_text() == "the last record"
    assert not any((tmp_path / leftover).exists() for leftover in LEFTOVERS)


@pytest.mark.parametrize("gate_passed", [True, False], ids=["tables", "failed_gate"])
def test_a_superseded_old_that_cannot_be_deleted_stops_the_next_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate_passed: bool
) -> None:
    """A publish that committed and was killed while deleting `.old` leaves it beside its own
    tables. The next publish deletes it before it stages anything, and if it cannot, it stops:
    carrying on would leave a stale `.old` for its own undo to put back as the last tables."""
    last = _plant_the_last_validation(tmp_path)
    old = tmp_path / "state_model_validation.old"
    old.mkdir()
    (old / "validation_scores.parquet").write_text("superseded")
    rmtree = shutil.rmtree

    def refuse(path: Path, ignore_errors: bool = False, *args: object, **kwargs: object) -> None:
        if Path(path) == old:
            if ignore_errors:
                return None
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return rmtree(path, ignore_errors, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(shutil, "rmtree", refuse)
    with pytest.raises(PermissionError):
        _publish(tmp_path, gate_passed)
    monkeypatch.undo()
    assert {path: path.read_text() for path in last} == last
    assert sorted(child.name for child in tmp_path.iterdir()) == [
        "promotion_record.json",
        "state_model_validation",
        "state_model_validation.old",
    ]


def test_last_tables_beside_tables_of_no_publish_are_refused(tmp_path: Path) -> None:
    """A state no single publish leaves: a staged record, staged tables, tables and `.old`.
    Settling it would rename `.old` over the tables, which `os.replace` does quietly when they are
    an empty directory, so it is refused by name and nothing moves."""
    for name in ("state_model_validation", "state_model_validation.old"):
        (tmp_path / name).mkdir()
    (tmp_path / "state_model_validation.partial").mkdir()
    (tmp_path / "state_model_validation.old" / "validation_scores.parquet").write_text("stale")
    (tmp_path / "promotion_record.json.partial").write_text("{}")
    before = _tree(tmp_path)
    with pytest.raises(UnsettledPublishError, match=r"state_model_validation\.old"):
        _settle_state_model_validation(tmp_path)
    assert _tree(tmp_path) == before


KILLER = """
import os
import shutil
import sys
from pathlib import Path

import polars as pl

from logging_employment import build, cli, runs

runs.code_provenance = lambda start=None: {"code_commit": "c", "uv_lock_sha256": "u"}
run, n, mode = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
steps = [0]


def counted(real):
    def step(*args, **kwargs):
        if steps[0] == n:
            os._exit(137)
        steps[0] += 1
        return real(*args, **kwargs)

    return step


os.replace = counted(os.replace)
shutil.rmtree = counted(shutil.rmtree)
Path.mkdir = counted(Path.mkdir)
Path.write_text = counted(Path.write_text)
Path.unlink = counted(Path.unlink)
build.write_parquet_deterministic = counted(build.write_parquet_deterministic)
if mode == "tables":
    cli._publish_state_model_validation(
        run,
        {"verdict": "beat"},
        tables={"validation_scores": pl.DataFrame({"cell_id": ["b", "a"], "estimate": [2.0, 1.0]})},
        validation_manifest={"regimes": {}},
    )
else:
    cli._publish_state_model_validation(
        run, {"verdict": "not_beaten"}, tables={}, validation_manifest=None
    )
os._exit(0)
"""


def _settle_interrupted_at(run: Path, step: int) -> bool:
    """Settle `run`, raising at its `step`-th rename or removal; whether that step was reached."""
    steps = [0]

    def counted(real):  # type: ignore[no-untyped-def]
        def counting(*args, **kwargs):  # type: ignore[no-untyped-def]
            if steps[0] == step:
                raise KeyboardInterrupt
            steps[0] += 1
            return real(*args, **kwargs)

        return counting

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "replace", counted(os.replace))
        patch.setattr(shutil, "rmtree", counted(shutil.rmtree))
        patch.setattr(Path, "unlink", counted(Path.unlink))
        try:
            _settle_state_model_validation(run)
        except KeyboardInterrupt:
            return True
    return False


@pytest.mark.parametrize("gate_passed", [True, False], ids=["tables", "failed_gate"])
def test_a_publish_killed_at_any_step_is_settled_by_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate_passed: bool
) -> None:
    """A real kill, not a simulated one: a child process publishes and dies by `os._exit` at its
    n-th mutation, so no handler runs. Settling what it left gives the last validation or, only
    once the record's rename ran, the new one. A settle itself cut short at any of its own steps
    (it has no handler, so a raise nothing catches leaves what a kill would) settles to the same
    place next time. And a publish after any kill leaves the new validation."""
    monkeypatch.setattr(
        runs, "code_provenance", lambda start=None: {"code_commit": "c", "uv_lock_sha256": "u"}
    )
    script = tmp_path / "killer.py"
    script.write_text(KILLER)
    new_run = tmp_path / "new"
    new_run.mkdir()
    _plant_the_last_validation(new_run)
    _publish(new_run, gate_passed)
    new = _tree(new_run)
    mode = "tables" if gate_passed else "failed_gate"
    settled_new = False
    for n in range(100):
        run = tmp_path / f"run{n}"
        run.mkdir()
        _plant_the_last_validation(run)
        last = _tree(run)
        child = subprocess.run([sys.executable, str(script), str(run), str(n), mode], check=False)
        if child.returncode == 0:
            break
        assert child.returncode == 137, n
        killed = tmp_path / f"killed{n}"
        shutil.copytree(run, killed)
        _settle_state_model_validation(run)
        settled = _tree(run)
        assert settled in (last, new), n
        settled_new |= settled == new
        for step in range(100):
            again = tmp_path / f"again{n}-{step}"
            shutil.copytree(killed, again)
            if not _settle_interrupted_at(again, step):
                break
            _settle_state_model_validation(again)
            assert _tree(again) == settled, (n, step)
        _publish(run, gate_passed)
        assert _tree(run) == new, n
    else:
        pytest.fail("the publish never ran to its end")
    assert settled_new, "no kill landed after the commit and before `.old` was deleted"
