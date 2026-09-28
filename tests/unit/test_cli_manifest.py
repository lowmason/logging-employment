"""`cli._write_manifest`, `cli._replace_directory` and `cli._publish_state_model_validation`: a
run's outputs are replaced whole, never left half-written or merged with the last run's."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path

import polars as pl
import pytest

from logging_employment import build
from logging_employment.cli import (
    _publish_state_model_validation,
    _replace_directory,
    _write_manifest,
)


def test_a_write_that_fails_partway_leaves_the_last_manifest_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex on #42. `Path.write_text` truncates before it writes, so a full disk or a kill
    mid-write left an empty or partial manifest where readers look, and `_unfinished_fit` read
    that file's existence as a finished fit. The writer now writes a sibling and replaces the
    manifest whole, so a write that fails partway leaves the previous manifest as it was, and no
    sibling behind. The failure is simulated in `Path.write_text`, the call the writer makes."""
    path = tmp_path / "state_model_manifest.json"
    _write_manifest(path, {"generation": 1})
    before = path.read_bytes()
    write_text = Path.write_text

    def disk_full(self: Path, data: str, *args: object, **kwargs: object) -> int:
        write_text(self, data[: len(data) // 2])
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "write_text", disk_full)
    with pytest.raises(OSError, match="No space left"):
        _write_manifest(path, {"generation": 2})
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert [child.name for child in tmp_path.iterdir()] == [path.name]


def test_a_manifest_is_never_visible_until_it_is_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replace is the only step that makes a manifest visible. Interrupted before it, the
    first manifest a command writes does not exist at all, and its sibling is removed."""
    path = tmp_path / "reconcile_manifest.json"

    def interrupted(*args: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _write_manifest(path, {"generation": 1})
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


def _directory(path: Path, generation: str) -> Path:
    path.mkdir()
    (path / "validation_manifest.json").write_text(generation)
    return path


def _refusing_to_move(staged: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`os.replace` fails, as a full disk or a kill would, when it moves `staged` in, and only
    then, so every other rename still happens."""
    replace = os.replace

    def refuse(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
        if Path(src) == staged:
            raise OSError(errno.EIO, "Input/output error")
        replace(src, dst)

    monkeypatch.setattr(os, "replace", refuse)


def test_a_directory_is_replaced_whole(tmp_path: Path) -> None:
    """The staged directory takes the target's place, and nothing is left beside it."""
    target = _directory(tmp_path / "state_model_validation", "last")
    staged = _directory(tmp_path / "state_model_validation.partial", "new")
    _replace_directory(staged, target)
    assert (target / "validation_manifest.json").read_text() == "new"
    assert [child.name for child in tmp_path.iterdir()] == [target.name]


def test_a_swap_that_fails_puts_the_last_directory_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last directory is moved aside before the staged one moves in. If that second rename
    fails, the first is undone, so the target is never left missing by a failure."""
    target = _directory(tmp_path / "state_model_validation", "last")
    staged = _directory(tmp_path / "state_model_validation.partial", "new")
    _refusing_to_move(staged, monkeypatch)
    with pytest.raises(OSError, match="Input/output"):
        _replace_directory(staged, target)
    monkeypatch.undo()
    assert (target / "validation_manifest.json").read_text() == "last"
    assert not target.with_name(f"{target.name}.old").exists()


def test_a_swap_killed_between_its_renames_is_undone_before_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill between the two renames leaves no target and the last directory at `.old`. The next
    swap puts it back before anything else, so even when that swap fails too, the last directory
    is where readers look rather than deleted as a leftover."""
    target = tmp_path / "state_model_validation"
    _directory(target.with_name(f"{target.name}.old"), "last")
    staged = _directory(tmp_path / "state_model_validation.partial", "new")
    _refusing_to_move(staged, monkeypatch)
    with pytest.raises(OSError, match="Input/output"):
        _replace_directory(staged, target)
    monkeypatch.undo()
    assert (target / "validation_manifest.json").read_text() == "last"


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


def test_a_validation_is_published_with_the_digests_of_its_own_tables(tmp_path: Path) -> None:
    """The tables are written to a staged directory and renamed into place, so the digests the
    validation manifest and the record carry are the bytes readers find."""
    _plant_the_last_validation(tmp_path)
    scores = pl.DataFrame({"cell_id": ["b", "a"], "estimate": [2.0, 1.0]})
    _publish_state_model_validation(
        tmp_path,
        {"verdict": "beat"},
        tables={"validation_scores": scores},
        validation_manifest={"regimes": {}},
    )
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


def test_a_publish_that_fails_while_staging_leaves_the_last_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing of the last validation is touched until the new one is whole. A table write that
    fails, a full disk here, leaves the last record and tables as they were and no staging."""
    last = _plant_the_last_validation(tmp_path)

    def disk_full(frame: pl.DataFrame, path: Path) -> str:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(build, "write_parquet_deterministic", disk_full)
    with pytest.raises(OSError, match="No space left"):
        _publish_state_model_validation(
            tmp_path,
            {"verdict": "beat"},
            tables={"validation_scores": pl.DataFrame({"estimate": [1.0]})},
            validation_manifest={"regimes": {}},
        )
    monkeypatch.undo()
    assert {path: path.read_text() for path in last} == last
    assert sorted(child.name for child in tmp_path.iterdir()) == [
        "promotion_record.json",
        "state_model_validation",
    ]
