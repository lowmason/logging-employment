"""Every config float existing code reads as a check threshold or tolerance is finite at load.

A module of its own rather than a block in `test_config.py`: plan 16 rewrites that file's import
block and its closing run-id pin by literal diff, and a block appended there would stop those
hunks applying.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from logging_employment.config import (
    Config,
    ConstraintsConfig,
    DisclosureConfig,
    ReconciliationConfig,
    load_config,
)

SHIPPED = Path(__file__).resolve().parents[2] / "config.yaml"

# The blocks whose floats existing code reads as a check threshold or a tolerance, named by CLASS
# and resolved to their `Config` key, so the location each test below expects is the one pydantic
# reports rather than a second spelling of it. `BaselinesConfig` is absent on purpose: its one
# float, `regression_ridge_penalty`, is a regularization strength, not a threshold.
# `PromotionConfig` is inert here and belongs to Stage 5 (plan 16).
THRESHOLD_MODELS = (ConstraintsConfig, ReconciliationConfig, DisclosureConfig)
THRESHOLD_FLOATS = sorted(
    (block, name)
    for block, info in Config.model_fields.items()
    if info.annotation in THRESHOLD_MODELS
    for name, field in info.annotation.model_fields.items()
    if field.annotation is float
)
# The four of those that are a tolerance or a floor, and so are also `gt=0.0`; each class's
# docstring says why. The other three -- the MILP switch and both narrowness widths -- read zero
# as their "off" position, and `OFF_AT_ZERO`'s test pins that.
POSITIVE_FLOATS = (
    ("constraints", "feasibility_tolerance"),
    ("constraints", "rank_tolerance"),
    ("reconciliation", "tolerance"),
    ("reconciliation", "zero_seed_floor"),
)
OFF_AT_ZERO = sorted(set(THRESHOLD_FLOATS) - set(POSITIVE_FLOATS))


def _dotted(pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...]) -> list[str]:
    """`block.name` ids, so a failing case names the config key an operator would edit."""
    return [f"{block}.{name}" for block, name in pairs]


def _with(tmp_path: Path, block: str, name: str, value: float) -> Path:
    """`config.yaml` with one float replaced, written back as YAML.

    Round-tripped rather than validated as a dict, so `.inf` and `.nan` reach `load_config` the
    way an operator's file would deliver them.
    """
    payload = yaml.safe_load(SHIPPED.read_text())
    payload[block][name] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def _refusals(path: Path) -> list[tuple[Any, str]]:
    """Every `(location, error type)` `load_config` raises for `path`, which must raise."""
    with pytest.raises(ValidationError) as caught:
        load_config(path)
    return [(e["loc"], e["type"]) for e in caught.value.errors()]


def test_the_threshold_discovery_finds_all_seven_floats() -> None:
    """Pinned, so the parametrized tests below cannot pass by iterating over nothing.

    Three in `constraints:`, two in `reconciliation:`, two in `disclosure:`. A float added to any
    of the three blocks moves this count, and the tests below then hold it to the finiteness rule
    without being edited; `POSITIVE_FLOATS` must name real fields for its own test to mean anything.
    """
    assert len(THRESHOLD_FLOATS) == 7
    assert set(POSITIVE_FLOATS) <= set(THRESHOLD_FLOATS)


@pytest.mark.parametrize("spelling", [".inf", "-.inf", ".nan"])
@pytest.mark.parametrize(("block", "name"), THRESHOLD_FLOATS, ids=_dotted(THRESHOLD_FLOATS))
def test_a_non_finite_threshold_or_tolerance_is_refused_at_load(
    tmp_path: Path, block: str, name: str, spelling: str
) -> None:
    """YAML reads `.inf`, `-.inf` and `.nan` as floats, and a plain `float` admitted all three.

    A comparison against a non-finite threshold answers the same way for every value compared:
    at a NaN or +inf `reconciliation.tolerance`, `baselines.runner.leaves_its_interval` found no
    estimate outside its interval, so INV-002's per-cell check passed vacuously. The error TYPE is
    asserted beside the location so the refusal is shown to be the finiteness rule, not `gt=0.0`
    catching `-.inf` by accident.
    """
    value = yaml.safe_load(spelling)
    assert isinstance(value, float) and not math.isfinite(value), (
        f"precondition failed: YAML read {spelling!r} as {value!r}, not a non-finite float"
    )
    assert ((block, name), "finite_number") in _refusals(_with(tmp_path, block, name, value))


@pytest.mark.parametrize("value", [0.0, -1.0e-9])
@pytest.mark.parametrize(("block", "name"), POSITIVE_FLOATS, ids=_dotted(POSITIVE_FLOATS))
def test_a_tolerance_or_floor_at_or_below_zero_is_refused_at_load(
    tmp_path: Path, block: str, name: str, value: float
) -> None:
    """A tolerance or floor at or below zero cannot do the job its reader gives it.

    HiGHS refuses a feasibility tolerance there and keeps its own; `matrix_rank` counts float
    residue as rank; `math.isclose` raises on a negative `abs_tol`; and §12.4 asks for a "small
    positive floor". The config docstrings carry the measurements.
    """
    assert ((block, name), "greater_than") in _refusals(_with(tmp_path, block, name, value))


@pytest.mark.parametrize(("block", "name"), OFF_AT_ZERO, ids=_dotted(OFF_AT_ZERO))
def test_a_threshold_outside_the_positive_four_still_loads_at_zero(
    tmp_path: Path, block: str, name: str
) -> None:
    """Zero is the "off" position of the MILP switch and of each narrowness arm, not a bad value.

    At zero `_needs_milp` re-solves nothing, which is `enforce_integrality: false`'s outcome, and a
    narrowness arm flags no interval wider than a point -- the kind of "off" position
    `tests/integration/test_d1_acceptance.py` sets each arm to in turn. Where "narrow" begins is the
    governance owner's policy (§21), so no sign bound is invented for these three.
    """
    cfg = load_config(_with(tmp_path, block, name, 0.0))
    assert getattr(getattr(cfg, block), name) == 0.0
