"""`cli._write_manifest`, the one writer every run manifest goes through."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from logging_employment.cli import _write_manifest


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
