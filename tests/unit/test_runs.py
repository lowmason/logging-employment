"""`run_id`'s payload: what goes into it, and what deliberately stays out of it."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path

import pytest

from logging_employment.config import Config, resolved_dict
from logging_employment.runs import (
    RUN_ID_LENGTH,
    code_provenance,
    partial_path,
    replace_whole,
    run_id,
)

DIGESTS = {"qcew_monthly": "aa" * 32, "bridge": "bb" * 32}


def _payload_before_overrides_existed(cfg: Config) -> str:
    """The id as it was computed before `run_id` grew an `overrides` parameter.

    Re-derived from `resolved_dict` rather than pinned as a literal: a literal would encode
    today's `config.yaml` and would have to be re-typed on every unrelated config change, which
    is exactly how a compatibility pin stops checking compatibility and starts checking nothing.
    """
    payload = json.dumps(
        {"config": resolved_dict(cfg), "inputs": dict(sorted(DIGESTS.items()))}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:RUN_ID_LENGTH]


def test_a_run_with_no_override_hashes_exactly_as_it_did_before_the_parameter_existed(
    appendix_a_config: Config,
) -> None:
    """The whole reason `overrides` is omitted rather than emitted as null.

    Emitting `"overrides": null` would have re-identified every directory under `runs/` for a key
    that says nothing -- orphaning Stage 4's acceptance artifact, and making `solve-bounds` and
    `run-baselines` refuse until a full rebuild. This is the test that keeps that from happening
    by accident later.
    """
    assert run_id(appendix_a_config, DIGESTS) == _payload_before_overrides_existed(
        appendix_a_config
    )


def test_an_empty_override_mapping_is_the_same_as_none(appendix_a_config: Config) -> None:
    """`{}` means "nothing was overridden", so it must not fork the id either."""
    assert run_id(appendix_a_config, DIGESTS, overrides={}) == run_id(appendix_a_config, DIGESTS)


def test_an_estimator_subset_gets_its_own_run_id(appendix_a_config: Config) -> None:
    """The point of routing the subset through the id: it cannot land on a full pass's outputs.

    Same config, same inputs, different estimators -- if these collided, the two runs would write
    different bytes to one `runs/<id>/validation_metrics.parquet` under a single identifier.
    """
    full = run_id(appendix_a_config, DIGESTS)
    subset = run_id(appendix_a_config, DIGESTS, overrides={"estimators": ["equal_residual"]})
    other = run_id(
        appendix_a_config, DIGESTS, overrides={"estimators": ["establishment_proportional"]}
    )
    assert len({full, subset, other}) == 3


def test_the_code_stamp_records_unknown_rather_than_halting_a_run(tmp_path: Path) -> None:
    """R-S5P-5: the one deliberate exception to §18.3's fail-closed default, pinned as a test.

    `tmp_path` has no `uv.lock` in any ancestor, which is the tarball/sdist install -- reached
    without mocking `subprocess` or `PATH`. Fail-closed guards values that would corrupt an
    estimate; a missing commit id corrupts none, and refusing to run without `git` on PATH turns a
    provenance diagnostic into an outage. BOTH keys go unknown together: `uv.lock` is the anchor
    the git probe is rooted at, so half an answer here would mean a lock digest from one checkout
    beside a commit from whatever repository happened to enclose it.
    """
    assert code_provenance(tmp_path) == {"code_commit": "unknown", "uv_lock_sha256": "unknown"}


def test_a_tree_with_a_lock_and_no_git_history_stamps_half_an_answer(tmp_path: Path) -> None:
    """The sdist case, and the ONLY test that reaches the `git` probe's failure path.

    hatchling ships `uv.lock` (it is tracked) and never ships `.git`, so an sdist install has a
    readable lock digest and an unanswerable commit. Half an answer is the right answer: guessing
    the commit from whatever repository encloses the install directory would be a lie, and halting
    would make the package unusable exactly where it is most often installed. Measured: `git -C`
    on a directory outside any repository exits 128, which is the `returncode != 0` branch.
    """
    (tmp_path / "uv.lock").write_bytes(b"lock bytes")
    assert code_provenance(tmp_path) == {
        "code_commit": "unknown",
        "uv_lock_sha256": hashlib.sha256(b"lock bytes").hexdigest(),
    }


def test_no_code_stamp_reaches_the_payload_the_run_id_hashes(appendix_a_config: Config) -> None:
    """The HARD constraint: stamping code identity must renumber no run directory.

    Two assertions, guarding two different routes in. The first is the id itself, still equal to
    the derivation that predates provenance -- that is the one that fires if `run_id` starts
    folding the stamp into its own payload. The second guards the route the first cannot see:
    a `code_commit` or `uv_lock_sha256` field added to `Config` would reach `resolved_dict`, and
    both the id and its pre-provenance re-derivation would move together, silently and in step.
    So the second reconstructs the payload from `resolved_dict` and names the keys that must not
    appear in it, at the point of entry rather than as a changed digest nobody can attribute.
    """
    assert run_id(appendix_a_config, DIGESTS) == _payload_before_overrides_existed(
        appendix_a_config
    )
    payload = json.dumps(
        {"config": resolved_dict(appendix_a_config), "inputs": DIGESTS}, sort_keys=True
    )
    assert not set(code_provenance()) & set(json.loads(payload)["config"])
    assert "code_commit" not in payload
    assert "uv_lock_sha256" not in payload


def test_replace_whole_leaves_the_last_file_when_the_write_fails_partway(tmp_path: Path) -> None:
    """`D-134`. `write_bytes`, `write_text` and `write_parquet` truncate before they write, so a
    full disk or a kill mid-write left the file cut short where readers look. The write goes to
    the `.partial` sibling, and a write that raises leaves the last file as it was and removes
    the sibling."""
    path = tmp_path / "source_manifest.parquet"
    path.write_bytes(b"the last manifest")
    with pytest.raises(OSError, match="No space left"), replace_whole(path) as partial:
        partial.write_bytes(b"half of the")
        raise OSError(errno.ENOSPC, "No space left on device")
    assert path.read_bytes() == b"the last manifest"
    assert [child.name for child in tmp_path.iterdir()] == [path.name]


def test_replace_whole_makes_nothing_visible_until_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename is the only step that makes the file visible. Interrupted before it, a file
    written for the first time does not exist at all, and its sibling is removed."""
    path = tmp_path / "config.resolved.yaml"

    def interrupted(*args: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt), replace_whole(path) as partial:
        partial.write_text("project: {}")
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


def test_replace_whole_renames_the_sibling_over_the_path(tmp_path: Path) -> None:
    path = tmp_path / "object.csv"
    path.write_bytes(b"the last object")
    with replace_whole(path) as partial:
        assert partial == partial_path(path) == tmp_path / "object.csv.partial"
        partial.write_bytes(b"the new object")
    assert path.read_bytes() == b"the new object"
    assert [child.name for child in tmp_path.iterdir()] == [path.name]


def test_the_partial_sibling_has_one_name_across_the_package() -> None:
    """One name for every unfinished write, so no reader anywhere opens one: the manifest writer,
    the staged promotion record and the staged validation tables in `cli.py` name theirs through
    `runs.partial_path` too."""
    from logging_employment.cli import _partial

    assert _partial(Path("runs/x/promotion_record.json")) == partial_path(
        Path("runs/x/promotion_record.json")
    )
    assert partial_path(Path("runs/x/state_model_validation")) == Path(
        "runs/x/state_model_validation.partial"
    )
