"""§17.5's state-total rows: the fit recovers the parameters it was simulated from (§17.4 row 5).

The panel is drawn from §11.1-§11.3's own generative process AT D1'S REGIME, because a recovery test
at an easier one says nothing about D1 (plan 16, Decision 15). What makes D1 hard is reproduced:

* training cells are exact (§11.3), and innovation scales are D1's: sigma_eta 0.035, dispersed
  across states by tau 0.8, so a state's scale runs from about 0.01 to 0.1;
* x, the standardized log establishment count, moves between states and barely within one. It is
  constant within a quarter and steps at quarter boundaries, as QCEW's exposure does, and its
  within-state and between-state slopes differ in sign, as on D1 (posterior means -1.31 and
  +0.34), so the model must split them;
* suppression takes whole quarters: two states have no training cell, two keep two or three
  quarters, and the rest lose runs of one to four quarters.

18 states in 6 divisions over 96 months, with three injected 8-sigma shocks. "Within tolerances
appropriate to sample size" (§17.5) is read one row at a time, and each assertion's comment gives
the arithmetic its bound comes from.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from logging_employment.config import ModelConfig
from logging_employment.models.arviz_io import rank_rhat
from logging_employment.models.interfaces import ModelData, StateModelConfig, StateModelFit
from logging_employment.models.state_total import fit_state_total_model, state_mean_exposure

pytestmark = pytest.mark.slow

SEED = sum(map(ord, "tests/state-total-recovery"))
SAMPLER = replace(StateModelConfig.from_config(ModelConfig()), chains=4, warmup=500, draws=500)
DIVISIONS, PER_DIVISION, YEARS = 6, 3, 8
STATES, MONTHS = DIVISIONS * PER_DIVISION, 12 * YEARS
ALPHA, SIGMA_ETA, TAU, DF = 1.6, 0.035, 0.8, 5.0
BETA, BETA_BETWEEN, SIGMA_U = -1.2, 0.4, 0.35
V = np.array([0.6, 0.3, 0.0, -0.1, -0.3, -0.5])
GAMMA = 0.02 * np.sin(2.0 * np.pi * np.arange(12) / 12.0)
DELTA = np.array([0.06, 0.04, 0.035, 0.02, 0.0, -0.03, -0.05, -0.075])
DELTA = DELTA - DELTA.mean()
# Two persistence groups, alternating so neither lines up with a division. Both sit under the cap.
RHO = np.where(np.arange(STATES) % 2 == 0, 0.9, 0.6)
SHOCKS = ((2, 30), (7, 55), (12, 80))
UNTRAINED = (4, 13)
SPARSE = {9: (5, 6), 16: (10, 11, 12)}  # state -> the only quarters it trains on


def _mask(rng: np.random.Generator) -> np.ndarray:
    """Training cells, whole quarters at a time, in D1's pattern."""
    quarters = MONTHS // 3
    trained = np.ones((STATES, quarters), dtype=bool)
    for state in range(STATES):
        if state in UNTRAINED:
            trained[state] = False
        elif state in SPARSE:
            trained[state] = False
            trained[state, list(SPARSE[state])] = True
        else:
            for _ in range(rng.integers(0, 3)):
                start, length = rng.integers(0, quarters), rng.integers(1, 5)
                trained[state, start : start + length] = False
    return np.repeat(trained, 3, axis=1)


def _simulate() -> tuple[ModelData, dict[str, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    division = np.repeat(np.arange(DIVISIONS), PER_DIVISION)
    base = np.log(rng.integers(5, 500, size=STATES).astype(float))
    steps = rng.normal(0.0, 0.03, size=(STATES, MONTHS // 3))
    log_a = np.repeat(base[:, None] + np.cumsum(steps, axis=1), 3, axis=1)
    centre, scale = float(log_a.mean()), float(log_a.std())
    x = (log_a - centre) / scale
    x_bar = x.mean(axis=1)
    u = rng.normal(0.0, SIGMA_U, size=STATES)
    state_scale = SIGMA_ETA * np.exp(TAU * rng.standard_normal(STATES))
    z = rng.standard_t(DF, size=(STATES, MONTHS))
    for state, month in SHOCKS:
        z[state, month] = 8.0
    eta = np.empty((STATES, MONTHS))
    eta[:, 0] = state_scale * z[:, 0] / np.sqrt(1.0 - RHO**2)
    for month in range(1, MONTHS):
        eta[:, month] = RHO * eta[:, month - 1] + state_scale * z[:, month]
    month_of_year = np.arange(MONTHS) % 12
    year_of_month = np.arange(MONTHS) // 12
    mu = (
        ALPHA
        + (u + V[division])[:, None]
        + GAMMA[month_of_year][None, :]
        + DELTA[year_of_month][None, :]
        + BETA * (x - x_bar[:, None])
        + BETA_BETWEEN * x_bar[:, None]
        + eta
    )
    trained = _mask(rng)
    train_s, train_t = np.nonzero(trained)
    predict_s, predict_t = np.nonzero(~trained)
    exposure = np.exp(log_a)
    data = ModelData(
        states=tuple(f"{s:02d}" for s in range(STATES)),
        months=tuple(f"{2017 + t // 12}-{t % 12 + 1:02d}" for t in range(MONTHS)),
        divisions=tuple(f"division_{d}" for d in range(DIVISIONS)),
        years=tuple(str(2017 + year) for year in range(YEARS)),
        state_division=division.astype(np.int64),
        month_of_year=month_of_year.astype(np.int64),
        year_of_month=year_of_month.astype(np.int64),
        train_state=train_s.astype(np.int64),
        train_month=train_t.astype(np.int64),
        train_y=mu[train_s, train_t],
        train_x=x[train_s, train_t],
        predict_state=predict_s.astype(np.int64),
        predict_month=predict_t.astype(np.int64),
        predict_exposure=exposure[predict_s, predict_t],
        predict_x=x[predict_s, predict_t],
        predict_cell_ids=tuple(f"{s}|{t}" for s, t in zip(predict_s, predict_t, strict=True)),
        log_exposure_centre=centre,
        log_exposure_scale=scale,
    )
    truth = {
        "level": ALPHA + u + V[division] + BETA_BETWEEN * x_bar,
        "state_scale": state_scale,
        "mu_hidden": mu[predict_s, predict_t],
        "trained": trained.any(axis=1),
    }
    return data, truth


@pytest.fixture(scope="module")
def recovered() -> tuple[ModelData, dict[str, np.ndarray], StateModelFit]:
    data, truth = _simulate()
    return data, truth, fit_state_total_model(data, SAMPLER)


def _flat(fit: StateModelFit, name: str) -> np.ndarray:
    """A parameter's draws with chains pooled: (chains * draws, ...)."""
    values = fit.parameters[name]
    return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])


def _level(data: ModelData, fit: StateModelFit) -> np.ndarray:
    """Each state's level alpha + u + v[r(s)] + beta_between * x_bar[s], per draw."""
    v = _flat(fit, "v")[:, data.state_division]
    between = _flat(fit, "beta_between")[:, None] * state_mean_exposure(data)[None, :]
    return _flat(fit, "alpha")[:, None] + _flat(fit, "u") + v + between


def test_the_sampler_mixed_well_enough_for_recovery_to_mean_anything(recovered) -> None:
    """Looser than §11.14's 1.01: this test asks about recovery, and D1 owns the gate."""
    _data, _truth, fit = recovered
    worst = max(
        float(np.max(rank_rhat(values.reshape(values.shape[0], values.shape[1], -1))))
        for values in fit.parameters.values()
    )
    assert worst <= 1.05
    assert fit.divergences == 0


def test_trained_state_levels_are_recovered(recovered) -> None:
    """Exact innovations pin a trained state's level to about scale / ((1 - rho) * sqrt(n)), 0.04
    at rho 0.9 and 96 months, against a spread of about 0.5 across states. So the correlation should
    sit near 0.99, and 95% intervals allow one miss among the sixteen trained states."""
    data, truth, fit = recovered
    level = _level(data, fit)
    trained = truth["trained"]
    assert np.corrcoef(level.mean(axis=0)[trained], truth["level"][trained])[0, 1] >= 0.95
    low, high = np.quantile(level, [0.025, 0.975], axis=0)
    hit = (truth["level"] >= low) & (truth["level"] <= high)
    assert int(np.sum(hit[trained])) >= int(trained.sum()) - 1


def test_untrained_state_levels_fall_inside_their_wide_intervals(recovered) -> None:
    """A state with no training cell is known only through beta_between * x_bar, v and the u
    prior, so its interval spans about +/- 2 sigma_u. Both untrained states' true levels must fall
    inside. With one beta for both slopes, one fell 2.5 log points below its interval's centre."""
    data, truth, fit = recovered
    low, high = np.quantile(_level(data, fit), [0.025, 0.975], axis=0)
    for state in UNTRAINED:
        assert low[state] <= truth["level"][state] <= high[state]


def test_beta_is_recovered_from_the_quarterly_steps(recovered) -> None:
    """Hundreds of trained quarter steps of sd 0.03 in log A, scored against innovations of sd about
    0.035, identify beta: its interval must hold the truth and be narrow beside |beta| = 1.2."""
    _data, _truth, fit = recovered
    low, high = np.quantile(_flat(fit, "beta"), [0.025, 0.975])
    assert low <= BETA <= high
    assert high - low <= 0.5


def test_the_between_state_slope_is_recovered(recovered) -> None:
    """Sixteen trained levels, spread about 1 in x_bar around a residual sd of 0.35, identify
    beta_between to about 0.35 / 4 = 0.09. Its interval must hold the truth and exclude the within
    slope, which is the confusion the split exists to prevent."""
    _data, _truth, fit = recovered
    low, high = np.quantile(_flat(fit, "beta_between"), [0.025, 0.975])
    assert low <= BETA_BETWEEN <= high
    assert low > BETA


def test_seasonality_and_year_effects_are_recovered(recovered) -> None:
    """Both rest on thousands of exact innovations, so their errors are a few hundredths at most."""
    _data, _truth, fit = recovered
    assert float(np.max(np.abs(_flat(fit, "gamma").mean(axis=0) - GAMMA))) <= 0.01
    assert float(np.max(np.abs(_flat(fit, "delta").mean(axis=0) - DELTA))) <= 0.03


def test_ar_persistence_separates_the_two_groups(recovered) -> None:
    """The Beta prior pulls every state toward its mean, so exact recovery is not the test. The
    posterior means should still separate 0.9 from 0.6 by more than 0.1."""
    _data, _truth, fit = recovered
    rho = _flat(fit, "rho").mean(axis=0)
    assert float(rho[RHO == 0.9].mean() - rho[RHO == 0.6].mean()) >= 0.1


def test_state_innovation_scales_are_recovered(recovered) -> None:
    """Student-t innovations absorb the three 8-sigma jumps, so each trained state's scale stays
    within 40% of its truth, except at most two: a sparse state's scale rests on a few steps."""
    _data, truth, fit = recovered
    ratio = np.median(_flat(fit, "sigma_eta_state"), axis=0) / truth["state_scale"]
    trained = truth["trained"]
    within = (ratio[trained] >= 0.6) & (ratio[trained] <= 1.4)
    assert int(np.sum(within)) >= int(trained.sum()) - 2


def test_hidden_cells_are_predicted_within_their_intervals(recovered) -> None:
    """Predictive recovery, on the latent mean the score is built from: log(q / A) = mu."""
    data, truth, fit = recovered
    mu = np.log(fit.raw_scores.values / data.predict_exposure[None, :])
    low, high = np.quantile(mu, [0.05, 0.95], axis=0)
    covered = float(np.mean((truth["mu_hidden"] >= low) & (truth["mu_hidden"] <= high)))
    assert covered >= 0.8
    in_trained = truth["trained"][data.predict_state]
    error = np.abs(np.median(mu, axis=0) - truth["mu_hidden"])
    assert float(np.median(error[in_trained])) <= 0.1


def test_the_one_step_check_is_near_nominal(recovered) -> None:
    """The one-step check replays the fit's own innovation law on data drawn from that law, so its
    coverage should sit near 0.9."""
    _data, _truth, fit = recovered
    assert 0.85 <= fit.ppc_coverage_90 <= 0.97
