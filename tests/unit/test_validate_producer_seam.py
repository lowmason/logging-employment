"""`run_pseudo_suppression`'s producer seam: one loop scores §10's registry and a model alike.

Driven on the committed fixture (`tests/fixtures/baselines/`) with one cheap estimator and one seed,
so the loop runs whole in seconds. The byte-identity of the baseline path itself is
`tests/integration/test_validation_golden.py`'s to prove: its golden did not move when the seam went
in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import pytest

from logging_employment.baselines.runner import REGISTRY, resolve_estimators
from logging_employment.config import Config, load_config
from logging_employment.contracts import HarmonizedData
from logging_employment.errors import ConceptViolationError
from logging_employment.validate.harness import (
    BaselineProducer,
    Production,
    ValidationResult,
    run_pseudo_suppression,
)
from logging_employment.validate.recover import MaskedSystem

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "baselines"
CHEAP = resolve_estimators(["establishment_proportional"])


def _config() -> Config:
    cfg = load_config(REPO / "config.yaml")
    return cfg.model_copy(
        update={
            "validation": cfg.validation.model_copy(
                update={"replicates_per_regime": 3, "pseudo_suppression_seeds": [1024]}
            )
        }
    )


@dataclass
class _Recording:
    """A producer that delegates to §10's registry and records what the harness handed it."""

    inner: BaselineProducer
    seen: list[tuple[pl.DataFrame, MaskedSystem]] = field(default_factory=list)

    @property
    def estimator_ids(self) -> tuple[str, ...]:
        """The delegate's ids, so the manifest names what actually scored."""
        return self.inner.estimator_ids

    def __call__(self, masked: HarmonizedData, system: MaskedSystem, config: Config) -> Production:
        """Record the masked frame and system, then produce exactly what the delegate does."""
        self.seen.append((masked.qcew_monthly, system))
        production = self.inner(masked, system, config)
        return Production(
            results=production.results,
            interval_metrics=production.interval_metrics,
            notes={"call": len(self.seen)},
        )


@pytest.fixture(scope="module")
def recorded() -> tuple[_Recording, ValidationResult]:
    producer = _Recording(BaselineProducer(CHEAP))
    result = run_pseudo_suppression(
        HarmonizedData.load(FIXTURE), config=_config(), producer=producer
    )
    return producer, result


def test_the_producer_sees_each_replicates_masked_frame_and_system(recorded) -> None:
    producer, result = recorded
    scored = [entry for entry in result.manifest["regimes"].values() if entry["replicates"]]
    assert len(producer.seen) == sum(entry["replicates"] for entry in scored)
    hashes = sorted(h for entry in scored for h in entry["hashes"])
    assert sorted(system.constraint_set_hash for _frame, system in producer.seen) == hashes
    original = HarmonizedData.load(FIXTURE).qcew_monthly
    suppressed = int((original["observation_status"] == "suppressed").sum())
    for frame, _system in producer.seen:
        assert int((frame["observation_status"] == "suppressed").sum()) > suppressed


def test_producer_notes_ride_into_the_manifest_with_their_seed(recorded) -> None:
    _producer, result = recorded
    for entry in result.manifest["regimes"].values():
        notes = entry.get("producer_notes", [])
        assert len(notes) == entry["replicates"]
        assert all(note["seed"] == 1024 and note["call"] >= 1 for note in notes)


def test_a_producer_without_notes_writes_the_manifest_it_always_wrote() -> None:
    result = run_pseudo_suppression(
        HarmonizedData.load(FIXTURE), config=_config(), producer=BaselineProducer(CHEAP)
    )
    assert not any("producer_notes" in entry for entry in result.manifest["regimes"].values())


def test_estimators_and_a_producer_together_are_refused() -> None:
    with pytest.raises(ConceptViolationError, match="not both"):
        run_pseudo_suppression(
            HarmonizedData.load(FIXTURE),
            CHEAP,
            _config(),
            producer=BaselineProducer(CHEAP),
        )


def test_the_default_producer_is_the_whole_registry_in_order() -> None:
    assert BaselineProducer(REGISTRY).estimator_ids == tuple(e.estimator_id for e in REGISTRY)
