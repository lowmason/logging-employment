"""The raw store is content-addressed, immutable, and secret-free."""

from __future__ import annotations

import errno
import hashlib
from pathlib import Path

import pytest

from logging_employment.errors import StoredObjectMismatchError
from logging_employment.ingest.base import FetchedBytes
from logging_employment.store import RawStore, assert_no_secret


def _fetched(content: bytes) -> FetchedBytes:
    return FetchedBytes(
        url="https://example.invalid/x.csv",
        params={},
        content=content,
        http_status=200,
        retrieved_at_utc="2026-09-05T00:00:00+00:00",
    )


def test_identical_bytes_produce_one_retrieval_id(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    first = store.put("qcew", _fetched(b"a,b\n1,2\n"), "slice.csv")
    second = store.put("qcew", _fetched(b"a,b\n1,2\n"), "slice.csv")
    assert first.retrieval_id == second.retrieval_id
    assert first.was_already_present is False
    assert second.was_already_present is True


def test_different_bytes_produce_different_retrieval_ids(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    a = store.put("qcew", _fetched(b"one"), "slice.csv")
    b = store.put("qcew", _fetched(b"two"), "slice.csv")
    assert a.retrieval_id != b.retrieval_id


def test_a_stored_object_is_not_rewritten(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    stored = store.put("qcew", _fetched(b"original"), "slice.csv")
    mtime = stored.raw_path.stat().st_mtime_ns
    store.put("qcew", _fetched(b"original"), "slice.csv")
    assert stored.raw_path.stat().st_mtime_ns == mtime
    assert stored.raw_path.read_bytes() == b"original"


def test_assert_no_secret_raises_on_a_leaked_value() -> None:
    with pytest.raises(ValueError, match="secret"):
        assert_no_secret("url=...&key=abc123", ["abc123"])


def test_assert_no_secret_ignores_empty_secrets() -> None:
    assert_no_secret("anything at all", ["", None])  # type: ignore[list-item]


def test_an_interrupted_write_leaves_no_object_where_its_digest_claims_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`D-134`. `write_bytes` truncates before it writes, so a full disk or a kill mid-write left
    an object cut short at the path that names its digest, and every later `put` of the same
    response saw it exist, left it alone and reported it present. The bytes go to a `.partial`
    sibling first: an interrupted write leaves nothing at the object's path and no sibling, and
    the next `put` stores the object."""
    store = RawStore(tmp_path)
    content = b"a,b\n1,2\n"
    write_bytes = Path.write_bytes

    def disk_full(self: Path, data: bytes) -> int:
        write_bytes(self, data[: len(data) // 2])
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "write_bytes", disk_full)
    with pytest.raises(OSError, match="No space left"):
        store.put("qcew", _fetched(content), "slice.csv")
    monkeypatch.undo()
    expected = store.path_for("qcew", hashlib.sha256(content).hexdigest(), "slice.csv")
    assert not expected.exists()
    assert list(expected.parent.iterdir()) == []
    stored = store.put("qcew", _fetched(content), "slice.csv")
    assert stored.was_already_present is False
    assert stored.raw_path.read_bytes() == content


def test_an_object_whose_bytes_are_not_its_digest_is_refused(tmp_path: Path) -> None:
    """An object the code before `D-134` left cut short still exists at the path that claims its
    digest. `put` hashes what it finds before trusting it, refuses by name rather than reporting
    the object present, and leaves it as it is: a stored object is never rewritten, so the remedy
    is a human's."""
    store = RawStore(tmp_path)
    whole = store.put("qcew", _fetched(b"original"), "slice.csv")
    whole.raw_path.write_bytes(b"orig")
    with pytest.raises(StoredObjectMismatchError, match="slice.csv"):
        store.put("qcew", _fetched(b"original"), "slice.csv")
    assert whole.raw_path.read_bytes() == b"orig"
