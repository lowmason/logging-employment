"""§11.1-§11.3 and §11.5 in NumPyro: determinism, shapes, the likelihood's scope, the priors.

The hermetic tier's one MCMC module. `TINY` samples a twelve-cell panel with two chains of 40 draws,
so its fits cost compile time, not sampling time. Recovery on a panel big enough to test it is
`tests/integration/test_state_total_recovery.py`, marked slow.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from scipy import stats

from logging_employment.config import ModelConfig
from logging_employment.errors import ConceptViolationError
from logging_employment.models.interfaces import ModelData, StateModelConfig, StateModelFit
from logging_employment.models.state_total import (
    fit_state_total_model,
    one_step_scale,
    prior_predictive,
    raw_scores,
    state_total_model,
)

SEED = sum(map(ord, "tests/state-total-toy"))
TINY = replace(StateModelConfig.from_config(ModelConfig()), chains=2, warmup=40, draws=40)
# (state index, month index). State 2 is suppressed in months 1 and 3.
TRAIN = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3)]
PREDICT = [(2, 1), (2, 3)]
# D1's observed log employees per establishment spans -0.511 to 2.342 (measured 2026-09-26).
D1_RESPONSE_RANGE = (-0.511, 2.342)


def _toy_data() -> ModelData:
    """Three states in two divisions over four months of one year."""
    rng = np.random.default_rng(SEED)
    train_a = rng.integers(3, 40, size=len(TRAIN)).astype(np.float64)
    predict_a = rng.integers(1, 10, size=len(PREDICT)).astype(np.float64)
    log_a = np.log(np.concatenate([train_a, predict_a]))
    centre, scale = float(log_a.mean()), float(log_a.std())
    return ModelData(
        states=("01", "02", "04"),
        months=("2023-01", "2023-02", "2023-03", "2023-04"),
        divisions=("division_a", "division_b"),
        years=("2023",),
        state_division=np.array([0, 0, 1], dtype=np.int64),
        month_of_year=np.arange(4, dtype=np.int64),
        year_of_month=np.zeros(4, dtype=np.int64),
        train_state=np.array([s for s, _ in TRAIN], dtype=np.int64),
        train_month=np.array([t for _, t in TRAIN], dtype=np.int64),
        train_y=1.5 + 0.2 * rng.standard_normal(len(TRAIN)),
        train_x=(np.log(train_a) - centre) / scale,
        predict_state=np.array([s for s, _ in PREDICT], dtype=np.int64),
        predict_month=np.array([t for _, t in PREDICT], dtype=np.int64),
        predict_exposure=predict_a,
        predict_x=(np.log(predict_a) - centre) / scale,
        predict_cell_ids=("toy|04|2023-02", "toy|04|2023-04"),
        log_exposure_centre=centre,
        log_exposure_scale=scale,
    )


@pytest.fixture(scope="module")
def toy_data() -> ModelData:
    return _toy_data()


@pytest.fixture(scope="module")
def tiny_fit(toy_data: ModelData) -> StateModelFit:
    return fit_state_total_model(toy_data, TINY)


def test_importing_the_model_turns_on_float64() -> None:
    """`reconcile_draws` checks adding-up to 1e-9, which float32 cannot hold at employment scale."""
    assert jnp.asarray(0.0).dtype == jnp.float64


def test_scores_are_positive_finite_and_chain_major(tiny_fit: StateModelFit) -> None:
    scores = tiny_fit.raw_scores
    assert scores.values.shape == (80, len(PREDICT))
    assert scores.values.dtype == np.float64
    assert np.all(np.isfinite(scores.values))
    assert np.all(scores.values > 0.0)
    assert scores.chain.tolist() == [0] * 40 + [1] * 40
    assert scores.draw.tolist() == list(range(40)) * 2
    assert scores.cell_ids == ("toy|04|2023-02", "toy|04|2023-04")


def test_the_same_seed_gives_bit_identical_draws(
    tiny_fit: StateModelFit, toy_data: ModelData
) -> None:
    """§16.1's idempotence, at the source of every later byte."""
    again = fit_state_total_model(toy_data, TINY)
    np.testing.assert_array_equal(again.raw_scores.values, tiny_fit.raw_scores.values)
    for name, values in tiny_fit.parameters.items():
        np.testing.assert_array_equal(again.parameters[name], values)


def test_the_seed_is_what_the_draws_depend_on(tiny_fit: StateModelFit, toy_data: ModelData) -> None:
    """Without this, a fit that ignored the configured seed would pass the test above."""
    other = fit_state_total_model(toy_data, replace(TINY, seed=TINY.seed + 1))
    assert not np.array_equal(other.raw_scores.values, tiny_fit.raw_scores.values)


def test_monitored_parameters_keep_their_chain_and_draw_axes(tiny_fit: StateModelFit) -> None:
    assert tiny_fit.parameters["u"].shape == (2, 40, 3)
    assert tiny_fit.parameters["v"].shape == (2, 40, 2)
    assert tiny_fit.parameters["gamma"].shape == (2, 40, 12)
    assert tiny_fit.parameters["rho"].shape == (2, 40, 3)
    # A one-year panel has no year contrast to sample, so neither year site exists.
    assert "delta" not in tiny_fit.parameters
    assert "sigma_delta" not in tiny_fit.parameters
    np.testing.assert_allclose(tiny_fit.parameters["gamma"].sum(axis=-1), 0.0, atol=1e-10)


def test_persistence_never_reaches_its_cap(tiny_fit: StateModelFit) -> None:
    """rho = persistence_max * Beta, so no draw may sit at or above the cap."""
    rho = tiny_fit.parameters["rho"]
    assert float(rho.max()) < TINY.priors.persistence_max
    assert float(rho.min()) > 0.0


def test_the_fit_records_its_sampler_and_its_predictive_check(tiny_fit: StateModelFit) -> None:
    assert tiny_fit.sampler["chain_method"] == "vectorized"
    assert tiny_fit.sampler["seed"] == TINY.seed
    # A saturated tree is 1023 steps; the share is the evidence the D1 profile made necessary.
    assert 1.0 <= tiny_fit.sampler["mean_leapfrog_steps"] <= 1023.0
    assert 0.0 <= tiny_fit.sampler["tree_depth_saturation_share"] <= 1.0
    assert tiny_fit.chains == 2
    assert tiny_fit.ppc_cells == len(TRAIN)
    assert 0.0 <= tiny_fit.ppc_coverage_90 <= 1.0
    assert set(tiny_fit.posterior_medians) == {"sigma_eta", "rho"}


def test_the_score_is_section_11_5_literally() -> None:
    """q = A * exp(mu): exposures 2 and 5 at mu = log 3 and log 4 score 6 and 20."""
    scores = raw_scores(np.array([2.0, 5.0]), np.log(np.array([[3.0, 4.0]])))
    np.testing.assert_allclose(scores, [[6.0, 20.0]], rtol=1e-15)


def test_no_suppressed_cell_is_an_observation(toy_data: ModelData) -> None:
    """§11.3: the likelihood has one term per training cell, and nothing else observes."""
    trace = handlers.trace(handlers.seed(state_total_model, rng_seed=0)).get_trace(
        toy_data, TINY, y=jnp.asarray(toy_data.train_y)
    )
    observed = [name for name, site in trace.items() if site.get("is_observed")]
    assert observed == ["y"]
    assert trace["y"]["fn"].batch_shape == (len(TRAIN),)
    # One innovation per cell the training values do not pin: the two suppressed cells.
    assert trace["z"]["value"].shape == (len(PREDICT),)
    # Every toy state has a training cell, so every state is sampled by its pinned level.
    assert trace["level"]["value"].shape == (3,)
    assert "u_raw" not in trace
    assert trace["mu_predict"]["value"].shape == (len(PREDICT),)


def test_the_likelihood_is_the_ar1_density_of_the_pinned_path(toy_data: ModelData) -> None:
    """§11.3 exact, §11.2 Student-t AR(1): each training cell scores the innovation y - m implies.

    The expected values are computed here by a plain loop and scipy, independent of `_path`'s scan.
    State 2's months 1 and 3 are suppressed, so its month-2 innovation is scored against a path
    built from an innovation, and its month-3 cell is built, never scored. Every toy state has a
    training cell, so each is sampled by its level and log scale, and m = level + gamma +
    beta * (x - x_bar).
    """
    rng = np.random.default_rng(SEED + 1)
    values = {
        "alpha": 1.4,
        "beta": 0.3,
        # Enters every level, and the levels are substituted, so it cannot move the likelihood.
        "beta_between": 0.2,
        "sigma_u": 0.4,
        "sigma_v": 0.2,
        "sigma_gamma": 0.1,
        "sigma_eta": 0.05,
        "tau": 0.3,
        "level": np.array([1.6, 1.1, 1.9]),
        "log_scale": np.log(np.array([0.04, 0.06, 0.05])),
        "rho_raw": np.array([0.7, 0.9, 0.8]),
        "v_raw": np.array([0.6, -0.3]),
        "gamma": rng.normal(scale=0.1, size=12),
        "z": np.array([1.3, -0.7]),
    }
    trace = handlers.trace(
        handlers.substitute(state_total_model, data={k: jnp.asarray(v) for k, v in values.items()})
    ).get_trace(toy_data, TINY, y=jnp.asarray(toy_data.train_y))
    scored = np.asarray(trace["y"]["fn"].log_prob(trace["y"]["value"]))

    df, cap = TINY.priors.innovation_df, TINY.priors.persistence_max
    rho = cap * values["rho_raw"]
    scale = np.exp(values["log_scale"])
    x = np.zeros((3, 4))
    x[toy_data.train_state, toy_data.train_month] = toy_data.train_x
    x[toy_data.predict_state, toy_data.predict_month] = toy_data.predict_x
    # Every toy cell is published (trained or suppressed), so x_bar is each row's mean.
    x_bar = x.mean(axis=1)
    y = dict(zip(TRAIN, toy_data.train_y, strict=True))
    innovation = dict(zip(PREDICT, values["z"], strict=True))
    expected, eta = {}, np.zeros((3, 4))
    for state in range(3):
        for month in range(4):
            within = x[state, month] - x_bar[state]
            m = values["level"][state] + values["gamma"][month] + values["beta"] * within
            if month == 0:
                location, width = 0.0, scale[state] / np.sqrt(1.0 - rho[state] ** 2)
            else:
                location, width = rho[state] * eta[state, month - 1], scale[state]
            if (state, month) in y:
                eta[state, month] = y[(state, month)] - m
                expected[(state, month)] = stats.t.logpdf(
                    eta[state, month], df, loc=location, scale=width
                )
            else:
                eta[state, month] = location + width * innovation[(state, month)]
    np.testing.assert_allclose(scored, [expected[cell] for cell in TRAIN], rtol=1e-12)


def test_the_one_step_scale_is_stationary_in_the_first_month() -> None:
    """At t = 0 there is no month before, so the prediction's scale is the stationary one."""
    got = one_step_scale(np.array([[0.6]]), np.array([[0.1]]), np.array([0, 0]), np.array([0, 5]))
    np.testing.assert_allclose(got, [[0.125, 0.1]], rtol=1e-15)


def test_the_prior_predictive_puts_the_response_where_logging_lives(toy_data: ModelData) -> None:
    """Checked before any fit is trusted: the priors alone must cover D1's observed range.

    They must not put material mass above 1,000 employees per establishment (log 6.9), which no
    logging state has ever averaged.
    """
    draws = prior_predictive(toy_data, TINY, num_samples=500)
    assert draws.shape == (500, len(TRAIN))
    low, median, high, extreme = np.quantile(draws, [0.025, 0.5, 0.975, 0.995])
    assert low < D1_RESPONSE_RANGE[0]
    assert high > D1_RESPONSE_RANGE[1]
    assert 1.0 < median < 2.0
    assert extreme < np.log(1000.0)


def test_an_unknown_backend_is_refused_before_anything_samples(toy_data: ModelData) -> None:
    with pytest.raises(ConceptViolationError, match="cmdstanpy"):
        fit_state_total_model(toy_data, replace(TINY, backend="cmdstanpy"))
