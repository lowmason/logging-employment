"""Appendix A's `model:` block (§11) as `ModelConfig`: what it pins, accepts and refuses."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from logging_employment.config import (
    ModelConfig,
    StateModelDiagnostics,
    StateModelPriors,
    load_config,
    resolved_dict,
)
from logging_employment.runs import run_id

REPO = Path(__file__).resolve().parents[2]


def test_the_shipped_config_pins_every_model_key_at_its_default() -> None:
    """`config.yaml` writes every key out, so a changed default cannot move a run silently."""
    raw = yaml.safe_load((REPO / "config.yaml").read_text())["model"]
    assert set(raw) == set(ModelConfig.model_fields)
    assert set(raw["priors"]) == set(StateModelPriors.model_fields)
    assert set(raw["diagnostics"]) == set(StateModelDiagnostics.model_fields)
    assert load_config(REPO / "config.yaml").model == ModelConfig()


@pytest.mark.parametrize(
    "switch", ["include_change_points", "include_harvest_factor", "include_ces"]
)
def test_an_unimplemented_model_component_is_refused_at_load(switch: str) -> None:
    """`true` would record a component in the run's config that no code fits."""
    with pytest.raises(ValidationError, match=switch):
        ModelConfig.model_validate({switch: True})


def test_only_the_implemented_backend_loads() -> None:
    with pytest.raises(ValidationError, match="backend"):
        ModelConfig.model_validate({"backend": "cmdstanpy"})


@pytest.mark.parametrize("multipliers", [[1.5, 2.0], [0.5, 1.0], [1.0, 1.0]])
def test_the_variance_multipliers_must_hold_the_fitted_model(multipliers: list[float]) -> None:
    """§11.13 inflates variance: 1.0 must be present, nothing below it, nothing repeated."""
    with pytest.raises(ValidationError, match="suppressed_variance_multipliers"):
        ModelConfig.model_validate({"suppressed_variance_multipliers": multipliers})


def test_the_fitted_model_alone_is_a_valid_multiplier_list() -> None:
    assert ModelConfig.model_validate({"suppressed_variance_multipliers": [1.0]})


def test_one_chain_is_refused_because_rhat_needs_two() -> None:
    with pytest.raises(ValidationError, match="chains"):
        ModelConfig.model_validate({"chains": 1})


# The priors that are not scales or concentrations, each with its own range below.
BOUNDED_PRIORS = {"intercept_mean", "persistence_max", "innovation_df"}


def test_every_prior_scale_and_concentration_must_be_positive_and_finite() -> None:
    """A non-positive or non-finite scale would reach NumPyro as an invalid distribution and fail
    as a backend error or a non-finite density, not as a refusal naming the value (§18.3). Read off
    the model, not listed, so a prior added later is covered until someone decides otherwise. The
    slopes' scale is Appendix A's own key, so it sits at the block's top level, not in `priors`."""
    scales = [
        (StateModelPriors, name)
        for name in sorted(set(StateModelPriors.model_fields) - BOUNDED_PRIORS)
    ]
    scales.append((ModelConfig, "standardized_beta_sd"))
    assert len(scales) == 10
    for model, name in scales:
        for value in (0.0, -1.0, math.inf, math.nan):
            with pytest.raises(ValidationError, match=name):
                model.model_validate({name: value})


def test_the_bounded_priors_refuse_values_outside_their_ranges() -> None:
    """The intercept's mean may be any finite number. The cap is a share of 1, so 1.0 removes it.
    The degrees of freedom must leave the innovations a variance, because eta's first month starts
    at the variance-matched scale sigma / sqrt(1 - rho^2)."""
    for name, value in (
        ("intercept_mean", math.inf),
        ("intercept_mean", math.nan),
        ("persistence_max", 0.0),
        ("persistence_max", 1.01),
        ("innovation_df", 2.0),
        ("innovation_df", math.inf),
    ):
        with pytest.raises(ValidationError, match=name):
            StateModelPriors.model_validate({name: value})
    accepted = StateModelPriors.model_validate(
        {"intercept_mean": -0.5, "persistence_max": 1.0, "innovation_df": 2.5}
    )
    assert (accepted.intercept_mean, accepted.persistence_max) == (-0.5, 1.0)


def test_no_float_in_the_model_block_may_be_infinite_or_nan() -> None:
    """YAML reads `.inf` and `.nan` as floats. A non-finite prior reaches NumPyro as an invalid
    distribution. An infinite gate threshold passes its check on every fit, and a NaN one fails it
    on every fit, but only after the fit has sampled. So each is refused at load, naming the field
    (§18.3). Read off the three models, so a float added later is covered, and the count fails
    first. `[nan, 1.0]` is the order `min()` lets past the multipliers' own check: it returns the
    NaN, and `nan < 1.0` is false."""
    floats = [
        (model, name)
        for model in (ModelConfig, StateModelPriors, StateModelDiagnostics)
        for name, field in model.model_fields.items()
        if field.annotation is float
    ]
    assert len(floats) == 16
    for model, name in floats:
        for value in (math.inf, -math.inf, math.nan):
            with pytest.raises(ValidationError, match=name):
                model.model_validate({name: value})
    for multipliers in ([1.0, math.inf], [math.nan, 1.0]):
        with pytest.raises(ValidationError, match="suppressed_variance_multipliers"):
            ModelConfig.model_validate({"suppressed_variance_multipliers": multipliers})


def test_the_integer_gate_thresholds_cannot_fail_open() -> None:
    """Codex on #41. The ESS floor is `min_ess_per_chain * chains` (`models/diagnostics.py`), so a
    floor of 0 or below passes every fit's ESS check however badly it mixed. A negative
    `max_divergences` fails every fit, but only after it has sampled. Both are refused at load,
    naming the field (§18.3), and the loosest values that still gate something load."""
    for name, value in (
        ("min_ess_per_chain", 0),
        ("min_ess_per_chain", -100),
        ("max_divergences", -1),
    ):
        with pytest.raises(ValidationError, match=name):
            StateModelDiagnostics.model_validate({name: value})
    StateModelDiagnostics.model_validate({"min_ess_per_chain": 1, "max_divergences": 0})


def test_no_integer_in_the_model_block_accepts_a_negative_but_the_seed() -> None:
    """The integer counterpart of the float test above, read off the same three models, so an
    integer added later without a floor fails here, and the count fails first. The seed names a
    random stream and gates nothing, so any integer is a seed."""
    ints = [
        (model, name)
        for model in (ModelConfig, StateModelPriors, StateModelDiagnostics)
        for name, field in model.model_fields.items()
        if field.annotation is int
    ]
    assert len(ints) == 6
    for model, name in ints:
        if name == "seed":
            continue
        with pytest.raises(ValidationError, match=name):
            model.model_validate({name: -1})


def test_a_misspelled_model_key_is_refused() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        ModelConfig.model_validate({"chain": 4})


def test_the_seed_is_derived_from_a_name_not_chosen() -> None:
    assert ModelConfig().seed == sum(map(ord, "logging-employment/state-total-model"))


def test_the_model_block_reaches_the_run_id() -> None:
    """In `resolved_dict` on purpose (Decision 1): a different draw count writes different draws."""
    cfg = load_config(REPO / "config.yaml")
    assert resolved_dict(cfg)["model"]["seed"] == 3645
    fewer = cfg.model_copy(update={"model": cfg.model.model_copy(update={"draws": 500})})
    assert run_id(fewer, {}) != run_id(cfg, {})
