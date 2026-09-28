"""`cli._write_manifest` and `cli._stage_manifest`: a run manifest is replaced whole.

The state-model validation's publish, which stages its record through `_stage_manifest`, has its
own module, `test_cli_publish.py`.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from logging_employment.cli import _stage_manifest, _write_manifest


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


def test_a_staged_manifest_holds_the_bytes_write_manifest_writes(tmp_path: Path) -> None:
    """The promotion record is staged, not written, so that its rename can be a publish's commit
    (Codex on #44). Its bytes must still be `_write_manifest`'s: one format, not two that drift."""
    payload = {"verdict": "beat", "model_validation_hashes": {"validation_scores": "ab"}}
    (tmp_path / "written").mkdir()
    (tmp_path / "staged").mkdir()
    _write_manifest(tmp_path / "written" / "promotion_record.json", payload)
    staged = _stage_manifest(tmp_path / "staged" / "promotion_record.json", payload)
    assert staged == tmp_path / "staged" / "promotion_record.json.partial"
    written = tmp_path / "written" / "promotion_record.json"
    assert staged.read_bytes() == written.read_bytes()
    assert not (tmp_path / "staged" / "promotion_record.json").exists()
