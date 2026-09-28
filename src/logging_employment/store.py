"""The immutable, content-addressed raw store and the `source_snapshot` row it produces."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .errors import SecretInPayloadError, StoredObjectMismatchError
from .ingest.base import FetchedBytes
from .runs import replace_whole


@dataclass(frozen=True)
class StoredObject:
    """One immutable object in the raw store."""

    retrieval_id: str
    content_sha256: str
    byte_count: int
    raw_path: Path
    was_already_present: bool


def assert_no_secret(payload: str, secrets: Sequence[str | None]) -> None:
    """Raise `SecretInPayloadError` if any non-empty secret value appears in the payload (§7.2, D3).

    The message names no value, because the value is the secret (`D-140`).
    """
    for secret in secrets:
        if secret and secret in payload:
            raise SecretInPayloadError(
                "a secret value reached a manifest payload; refusing to write it"
            )


class RawStore:
    """`data/raw/<source_id>/<retrieval_id>/<filename>`, written once and never rewritten.

    `retrieval_id` is the content sha256, so the same bytes always land in the same place and a
    rerun against an unchanged source is a no-op rather than a second copy (§6.2).
    """

    def __init__(self, root: Path) -> None:
        """Anchor the store at `root`, creating it if absent."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, source_id: str, content_sha256: str, filename: str) -> Path:
        """Where an object with this hash lives, whether or not it exists yet."""
        return self.root / source_id / content_sha256 / filename

    def put(self, source_id: str, fetched: FetchedBytes, filename: str) -> StoredObject:
        """Store bytes verbatim and return their identity. Existing objects are left untouched.

        WRITTEN WHOLE OR NOT AT ALL (`D-134`). `write_bytes` truncates before it writes, so a kill
        or a full disk mid-write left an object cut short at the path that claims its digest, and
        every later `put` of the same response saw it exist, left it alone and reported it
        `was_already_present`. The bytes go to a `.partial` sibling and are renamed over the path
        (`runs.replace_whole`), so the path holds the whole object or nothing, and a `.partial` a
        kill leaves matches no reader's glob. An object already there is hashed before it is
        trusted: one cut short by the code before this, or by anything since, is refused by name
        (`StoredObjectMismatchError`) rather than reported present, and left for a human, because
        a stored object is never rewritten.
        """
        digest = hashlib.sha256(fetched.content).hexdigest()
        path = self.path_for(source_id, digest, filename)
        already = path.exists()
        if already:
            found = hashlib.sha256(path.read_bytes()).hexdigest()
            if found != digest:
                raise StoredObjectMismatchError(
                    f"{path} holds bytes whose sha256 is {found}, not the {digest} its path "
                    "claims: the object was cut short or altered after it was stored. Remove it "
                    "and fetch again"
                )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with replace_whole(path) as partial:
                partial.write_bytes(fetched.content)
        return StoredObject(
            retrieval_id=digest,
            content_sha256=digest,
            byte_count=len(fetched.content),
            raw_path=path,
            was_already_present=already,
        )


def snapshot_row(
    *,
    source_id: str,
    fetched: FetchedBytes,
    stored: StoredObject,
    reference_start: str,
    reference_end: str,
    release_status: str,
    naics_vintage: str,
    schema_fingerprint: str,
    parser_version: str,
    source_publication_date: str | None,
    secrets: Sequence[str | None],
) -> dict[str, object]:
    """Build one `source_snapshot` row, refusing to emit it if a secret is present.

    `source_publication_date` is the response's `Last-Modified` header or None (D-100); the note
    beside `contracts.SOURCE_SNAPSHOT_SCHEMA` says what that column can and cannot be read as.
    """
    params_json = json.dumps(fetched.params, sort_keys=True)
    assert_no_secret(fetched.url, secrets)
    assert_no_secret(params_json, secrets)
    return {
        "snapshot_id": stored.retrieval_id,
        "source_id": source_id,
        "request_url_or_file": fetched.url,
        "request_parameters_json": params_json,
        "retrieved_at_utc": fetched.retrieved_at_utc,
        "source_publication_date": source_publication_date,
        "reference_start": reference_start,
        "reference_end": reference_end,
        "release_status": release_status,
        "naics_vintage": naics_vintage,
        "schema_fingerprint": schema_fingerprint,
        "content_sha256": stored.content_sha256,
        "byte_count": stored.byte_count,
        "http_status": fetched.http_status,
        "parser_version": parser_version,
        "raw_path": str(stored.raw_path),
    }
