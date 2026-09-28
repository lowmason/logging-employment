"""§11.1-§11.3 and §11.5: the robust hierarchical state-intensity model, in NumPyro.

THE MODEL. mu[s, t] = m[s, t] + eta[s, t], where

    m[s, t] = alpha + u[s] + v[r(s)] + gamma[m(t)] + delta[y(t)]
              + beta * (x[s, t] - x_bar[s]) + beta_between * x_bar[s]

is §11.1's mean less lambda_H * H and less the path. X is the standardized log establishment count
split into its within-state and between-state parts, §11.1's "nonredundant predictors": on D1 the
two slopes differ (a state's intensity falls as its own count steps up, -1.31 per unit, and rises
across states with count, +0.34: posterior means, plan 16's design pass), and with one beta the
state effects had to absorb the difference. u then tracked x_bar (correlation 0.91 in a D1-like
simulation), which breaks the exchangeable prior u ~ N(0, sigma_u) exactly where it matters: a state
with no training cell was predicted from that prior, 2.5 log points from its true level.
`include_harvest_factor` is false, and Stage 7 adds the factor and re-runs §13.10's gate. eta is an
AR(1) per state with Student-t innovations of scale sigma_eta[s] and persistence rho[s] (§11.2),
started at its stationary scale:

    eta[s, 0] ~ t_nu(0, sigma_eta[s] / sqrt(1 - rho[s]^2))
    eta[s, t] ~ t_nu(rho[s] * eta[s, t-1], sigma_eta[s])

TRAINING CELLS ARE EXACT (§11.3's first sentence: "Published QCEW values are exact observations
... there is no extra arbitrary measurement error"). At a training cell mu = y, so the path is
PINNED there, eta = y - m, and the likelihood is the AR(1) density of the innovation that pinning
implies. y -> eta is a shift, so its Jacobian is 1. Every other cell's eta is built from a unit
innovation, rho * eta[t-1] + sigma_eta * z. There is no sigma_y. §11.3's t(mu, sigma_y) "practical
response model" was plan 16's first implementation, and on D1 it failed §11.14's gate: sigma_y
settled near 0.004, so the training cells pinned mu in all but name, and chains split between
allocating state levels to u and to near-unit-root paths (plan 16, Decision 15). Pinning is that
model's sigma_y -> 0 limit, taken in closed form.

COORDINATES WHERE THE DATA PIN. Exact innovations pin three things for a state with a training cell:
its path at those cells, its innovation scale sigma_eta[s], and its level L = alpha + u + v +
beta_between * x_bar (`state_mean_exposure`). So those states sample `level` and `log_scale`
directly, each from its own prior, and u = L - alpha - v - beta_between * x_bar is derived (a shear,
Jacobian 1: the same model as u ~ N(0, sigma_u)). In the non-centred coordinates §11.14 prefers,
alpha or sigma_u could move only if all 44 of D1's trained states' raw effects moved with it, and
every tree hit NUTS's 1,023-step ceiling. In these, trees average about 31 steps and a production
fit takes about two minutes. A state with no training cell is non-centred throughout (u = sigma_u *
u_raw, log scale = log sigma_eta + tau * w), as §11.14's SHOULD says, because nothing pins it.

PERSISTENCE IS CAPPED: rho = persistence_max * Beta(c1, c0), §11.12's "transformed Beta prior".
§11.2 prefers the AR(1) to a random walk because Logging intensity is "persistent but plausibly
mean-reverting". As rho -> 1 a state's own path pins its level less and less, and the year effects
can trade against slow drift in the paths. On D1, uncapped, rho reached 0.978, and a production
fit failed §11.14's R-hat on one of two seeds (delta and rho at 1.011). Capped at 0.95, both
seeds passed at 1.005.

gamma and delta are ZeroSumNormal, so the month effects sum to zero as §11.1 requires and the year
effects are identified against alpha.

THE LATENT PATH IS SAMPLED, NOT MARGINALIZED. A Kalman filter would integrate eta out, but only for
Gaussian innovations. §11.2's Student-t shocks are the reason the unpinned path is a parameter.

THE SCORE IS §11.5 LITERALLY, q = A * exp(mu), computed in `raw_scores` and nowhere else.

FLOAT64 AT IMPORT. `numpyro.enable_x64()` runs when this module is imported, and `_fit_numpyro`
checks JAX's working precision before sampling. If some earlier import had initialised JAX in
float32, every draw would be silently downcast. `reconcile_draws` checks adding-up to 1e-9, and
float32 cannot represent that at employment scale.

THE CHAIN METHOD IS `vectorized`, meaning every chain runs in one `vmap` on one device. `parallel`
needs `numpyro.set_host_device_count` to run before JAX initialises. A CLI imported by a test runner
cannot guarantee that, and NumPyro silently falls back to `sequential` when it fails. `vectorized`
gives the same draws in any process for the same key, and every fit records the method in its
`sampler` notes.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

from ..errors import ConceptViolationError
from ..reconcile.draws import PosteriorDraws
from .interfaces import MODEL_ID, MODEL_VERSION, ModelData, StateModelConfig, StateModelFit

numpyro.enable_x64()

CHAIN_METHOD = "vectorized"
# NUTS's default depth, passed explicitly so the saturation share below has a named ceiling. A
# saturated tree takes 2**10 - 1 = 1023 leapfrog steps. While plan 16 was written, a
# production-config fit of the D1 panel under §11.3's practical response model hit the ceiling in
# 99.95% of its iterations and failed §11.14's gate, which is why every fit records how often it
# did.
MAX_TREE_DEPTH = 10
EXPOSURE_OFFSET = 0.0
MONTHS_PER_YEAR = 12
# The parameters §11.14's R-hat is computed over, and §15.4's store keeps. The latent innovations
# `z`, the raw draws behind the non-centered effects and the capped persistence, and the cell means
# are left out: they are either re-derivable from these or never summarised for release.
MONITORED: tuple[str, ...] = (
    "alpha",
    "beta",
    "beta_between",
    "sigma_u",
    "u",
    "sigma_v",
    "v",
    "sigma_gamma",
    "gamma",
    "sigma_delta",
    "delta",
    "rho",
    "sigma_eta",
    "tau",
    "sigma_eta_state",
)


def training_grid(data: ModelData) -> np.ndarray:
    """The `states x months` mask of training cells: the cells whose path is pinned."""
    grid = np.zeros((len(data.states), len(data.months)), dtype=bool)
    grid[data.train_state, data.train_month] = True
    return grid


def _path(
    pinned: np.ndarray,
    target: jnp.ndarray,
    innovations: jnp.ndarray,
    rho: jnp.ndarray,
    scale: jnp.ndarray,
    df: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """§11.2's eta month by month: pinned to `target` where `pinned`, built from `innovations` elsewhere.

    Returns the path, each pinned cell's innovation log density (0 elsewhere), and each cell's
    one-step location rho * eta[t-1] (0 at t = 0). All three are `states x months`.

    eta[:, 0] carries the AR(1)'s stationary scale, scale / sqrt(1 - rho^2). That is exact for
    Gaussian innovations. For Student-t innovations it is a variance-matching approximation, because
    their stationary law is not itself a scaled t. The alternative, starting every state at eta = 0,
    would pull January 2017 toward the state effect for no reason in the data.
    """
    stationary = scale / jnp.sqrt(jnp.clip(1.0 - rho**2, 1e-12))
    first = jnp.where(pinned[:, 0], target[:, 0], stationary * innovations[:, 0])
    first_lp = jnp.where(
        pinned[:, 0], dist.StudentT(df, 0.0, stationary).log_prob(target[:, 0]), 0.0
    )

    def step(
        previous: jnp.ndarray, month: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """One month: pin a training cell and score its innovation, or build the cell."""
        is_pinned, pinned_value, innovation = month
        location = rho * previous
        current = jnp.where(is_pinned, pinned_value, location + scale * innovation)
        log_prob = jnp.where(
            is_pinned, dist.StudentT(df, location, scale).log_prob(pinned_value), 0.0
        )
        return current, (current, log_prob, location)

    _, (rest, rest_lp, rest_location) = jax.lax.scan(
        step, first, (pinned[:, 1:].T, target[:, 1:].T, innovations[:, 1:].T)
    )
    eta = jnp.concatenate([first[:, None], rest.T], axis=1)
    log_prob = jnp.concatenate([first_lp[:, None], rest_lp.T], axis=1)
    location = jnp.concatenate([jnp.zeros_like(first)[:, None], rest_location.T], axis=1)
    return eta, log_prob, location


def _fixed_part(
    data: ModelData, x: np.ndarray, x_bar: np.ndarray, effects: dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """§11.1's mu less lambda_H * H and less eta, on the `states x months` grid."""
    return (
        effects["alpha"]
        + effects["u"][:, None]
        + effects["v"][data.state_division][:, None]
        + effects["gamma"][data.month_of_year][None, :]
        + effects["delta"][data.year_of_month][None, :]
        + effects["beta"] * (x - x_bar[:, None])
        + effects["beta_between"] * x_bar[:, None]
    )


def _exposure_grid(data: ModelData) -> np.ndarray:
    """Standardized log exposure on the grid. A cell with no published row reads 0, and no mu there
    is ever used: such a cell is neither trained on nor scored."""
    x = np.zeros((len(data.states), len(data.months)))
    x[data.train_state, data.train_month] = data.train_x
    x[data.predict_state, data.predict_month] = data.predict_x
    return x


def state_mean_exposure(data: ModelData) -> np.ndarray:
    """Each state's mean standardized log exposure over its published cells, x_bar[s].

    It splits X into the between-state part x_bar, which enters a state's level through
    `beta_between`, and the within-state part x - x_bar, which moves its path month to month through
    `beta`.
    """
    states = np.concatenate([data.train_state, data.predict_state])
    x = np.concatenate([data.train_x, data.predict_x])
    counts = np.bincount(states, minlength=len(data.states))
    return np.bincount(states, weights=x, minlength=len(data.states)) / np.maximum(counts, 1)


def state_total_model(
    data: ModelData, config: StateModelConfig, y: jnp.ndarray | None = None
) -> None:
    """§11.1-§11.3 as a NumPyro program. With `y=None` it draws `y` instead of conditioning on it.

    Every prior reads `config.priors` or `config.standardized_beta_sd`, because §11.12 says "Every
    prior must be exposed in resolved configuration". The likelihood site `y` has one term per
    TRAINING cell, its pinned innovation's density. A prediction cell's eta is built, never pinned,
    so no suppressed cell is ever an observation (§11.3). With `y=None` no cell is pinned, and `y`
    is the deterministic mu at the training cells: the prior predictive.

    A state with a training cell is sampled in the coordinates its exact innovations pin: its level
    L = alpha + u + v + beta_between * x_bar and its log innovation scale, each drawn from its own
    prior (`level` ~ N(alpha + v + beta_between * x_bar, sigma_u) is u ~ N(0, sigma_u) sheared,
    Jacobian 1). A
    state with none is non-centred. The module docstring gives the measured reason.
    """
    priors = config.priors
    n_states, n_months = len(data.states), len(data.months)
    pinned = training_grid(data) if y is not None else np.zeros((n_states, n_months), dtype=bool)
    unpinned = np.flatnonzero(~pinned.ravel())
    held = np.flatnonzero(pinned.any(axis=1))
    free = np.flatnonzero(~pinned.any(axis=1))
    x = _exposure_grid(data)
    x_bar = state_mean_exposure(data)
    alpha = numpyro.sample("alpha", dist.Normal(priors.intercept_mean, priors.intercept_sd))
    beta = numpyro.sample("beta", dist.Normal(0.0, config.standardized_beta_sd))
    beta_between = numpyro.sample("beta_between", dist.Normal(0.0, config.standardized_beta_sd))
    sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(priors.state_scale_sd))
    sigma_v = numpyro.sample("sigma_v", dist.HalfNormal(priors.region_scale_sd))
    sigma_gamma = numpyro.sample("sigma_gamma", dist.HalfNormal(priors.month_scale_sd))
    sigma_eta = numpyro.sample("sigma_eta", dist.HalfNormal(priors.innovation_scale_sd))
    tau = numpyro.sample("tau", dist.HalfNormal(priors.innovation_dispersion_sd))

    with numpyro.plate("state", n_states):
        rho_raw = numpyro.sample(
            "rho_raw",
            dist.Beta(priors.persistence_concentration1, priors.persistence_concentration0),
        )
    with numpyro.plate("division", len(data.divisions)):
        v_raw = numpyro.sample("v_raw", dist.Normal(0.0, 1.0))
    v = numpyro.deterministic("v", sigma_v * v_raw)
    # A plate cannot be empty, so a group with no state has no site, as a one-year panel has no
    # year effect: in the prior predictive nothing is held, and a fully trained panel frees nothing.
    level = log_scale = u_raw = w = jnp.zeros(0)
    if held.size:
        with numpyro.plate("held_state", held.size):
            level = numpyro.sample(
                "level",
                dist.Normal(
                    alpha + v[data.state_division[held]] + beta_between * x_bar[held], sigma_u
                ),
            )
            log_scale = numpyro.sample("log_scale", dist.Normal(jnp.log(sigma_eta), tau))
    if free.size:
        with numpyro.plate("free_state", free.size):
            u_raw = numpyro.sample("u_raw", dist.Normal(0.0, 1.0))
            w = numpyro.sample("w", dist.Normal(0.0, 1.0))
    gamma = numpyro.sample("gamma", dist.ZeroSumNormal(sigma_gamma, event_shape=(MONTHS_PER_YEAR,)))
    if len(data.years) > 1:
        sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(priors.year_scale_sd))
        delta = numpyro.sample(
            "delta", dist.ZeroSumNormal(sigma_delta, event_shape=(len(data.years),))
        )
    else:
        # A single year has nothing to contrast: a zero-sum effect over one level is identically 0.
        delta = jnp.zeros(1)
    z = numpyro.sample(
        "z", dist.StudentT(priors.innovation_df, 0.0, 1.0).expand([unpinned.size]).to_event(1)
    )

    rho = numpyro.deterministic("rho", priors.persistence_max * rho_raw)
    u_held = level - alpha - v[data.state_division[held]] - beta_between * x_bar[held]
    u = numpyro.deterministic(
        "u", jnp.zeros(n_states).at[held].set(u_held).at[free].set(sigma_u * u_raw)
    )
    scale = numpyro.deterministic(
        "sigma_eta_state",
        jnp.exp(
            jnp.zeros(n_states).at[held].set(log_scale).at[free].set(jnp.log(sigma_eta) + tau * w)
        ),
    )
    effects = {
        "alpha": alpha,
        "beta": beta,
        "beta_between": beta_between,
        "u": u,
        "v": v,
        "gamma": gamma,
        "delta": delta,
    }
    m = _fixed_part(data, x, x_bar, effects)
    target = jnp.zeros((n_states, n_months))
    if y is not None:
        target = target.at[data.train_state, data.train_month].set(y) - m
    innovations = jnp.zeros(n_states * n_months).at[unpinned].set(z).reshape(n_states, n_months)
    eta, log_prob, location = _path(pinned, target, innovations, rho, scale, priors.innovation_df)
    mu = m + eta
    numpyro.deterministic("mu_predict", mu[data.predict_state, data.predict_month])
    if y is None:
        numpyro.deterministic("y", mu[data.train_state, data.train_month])
        return
    numpyro.deterministic("one_step_location", (m + location)[data.train_state, data.train_month])
    numpyro.factor("y", log_prob[data.train_state, data.train_month])


def raw_scores(exposure: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """§11.5's positive score, q = (A + eps_A) * exp(mu). This is the one place it is computed.

    `mu` is shaped (draws, cells) and `exposure` (cells,). `EXPOSURE_OFFSET` (eps_A) is 0.0 because
    `build_model_data` refuses every A <= 0. §11.5's offset exists for zero exposure, which cannot
    reach here. It is kept as a named constant so the formula reads as the spec writes it.

    NOTHING IS MISSING FROM q. mu at a suppressed cell includes the state-month path eta, which
    carries §11.2's process variation into every suppressed month, and training cells are exact
    (§11.3), so the model has no separate observation noise that the score could omit.
    """
    return (np.asarray(exposure, dtype=np.float64) + EXPOSURE_OFFSET) * np.exp(
        np.asarray(mu, dtype=np.float64)
    )


def one_step_scale(
    rho: np.ndarray, scale: np.ndarray, train_state: np.ndarray, train_month: np.ndarray
) -> np.ndarray:
    """Each training cell's one-step predictive scale, per draw: (draws, states) -> (draws, cells).

    The innovation scale sigma_eta[s], or at t = 0 the stationary one, as `_path` scores them.
    """
    stationary = scale / np.sqrt(np.clip(1.0 - rho**2, 1e-12, None))
    return np.where(train_month == 0, stationary[:, train_state], scale[:, train_state])


def _ppc_coverage_90(
    location: np.ndarray, scale: np.ndarray, df: float, y: np.ndarray, seed: int
) -> float:
    """The share of training cells inside their central 90% ONE-STEP-AHEAD predictive interval.

    This is §11.14's "posterior predictive checks on observed cells", reduced to one number the gate
    can compare. Training cells are exact (§11.3), so an in-sample replicate of mu IS y, and a check
    of y against it would pass by construction. The check is instead the one a state-space model
    admits: each draw predicts a training cell from the month before, m + rho * eta[t-1] plus a t_nu
    innovation of the cell's `one_step_scale`, and the share of y inside its 90% interval is
    reported. That tests the innovation law the likelihood scores. A NumPy generator seeded from the
    fit's seed draws the innovations, so the check is as reproducible as the draws. With no training
    cell it is NaN, and the gate reads NaN as a failure.
    """
    if location.shape[1] == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    replicate = location + scale * rng.standard_t(df, size=location.shape)
    low, high = np.quantile(replicate, [0.05, 0.95], axis=0)
    return float(np.mean((y >= low) & (y <= high)))


def _fit_numpyro(data: ModelData, config: StateModelConfig) -> StateModelFit:
    """Sample with NumPyro's NUTS, then turn the draws into §11.5 scores and diagnostic inputs."""
    if jnp.asarray(0.0).dtype != jnp.float64:
        raise ConceptViolationError(
            "JAX is running in float32. reconcile_draws checks adding-up to "
            "reconciliation.tolerance, which float32 cannot represent at employment scale; "
            "import logging_employment.models.state_total before any other JAX work"
        )
    mcmc = MCMC(
        NUTS(
            state_total_model,
            target_accept_prob=config.target_accept,
            max_tree_depth=MAX_TREE_DEPTH,
        ),
        num_warmup=config.warmup,
        num_samples=config.draws,
        num_chains=config.chains,
        chain_method=CHAIN_METHOD,
        progress_bar=False,
    )
    mcmc.run(
        jax.random.key(config.seed),
        data,
        config,
        y=jnp.asarray(data.train_y, dtype=jnp.float64),
        extra_fields=("diverging", "num_steps"),
    )
    samples = {
        name: np.asarray(value) for name, value in mcmc.get_samples(group_by_chain=True).items()
    }
    extra = mcmc.get_extra_fields(group_by_chain=True)
    diverging = np.asarray(extra["diverging"], dtype=bool)
    num_steps = np.asarray(extra["num_steps"])
    chains, draws = diverging.shape
    scores = raw_scores(data.predict_exposure, samples["mu_predict"].reshape(chains * draws, -1))
    return StateModelFit(
        raw_scores=PosteriorDraws(
            cell_ids=data.predict_cell_ids,
            values=scores,
            chain=np.repeat(np.arange(chains), draws),
            draw=np.tile(np.arange(draws), chains),
        ),
        parameters={name: samples[name] for name in MONITORED if name in samples},
        diverging=diverging,
        ppc_coverage_90=_ppc_coverage_90(
            samples["one_step_location"].reshape(chains * draws, -1),
            one_step_scale(
                samples["rho"].reshape(chains * draws, -1),
                samples["sigma_eta_state"].reshape(chains * draws, -1),
                data.train_state,
                data.train_month,
            ),
            config.priors.innovation_df,
            data.train_y,
            config.seed,
        ),
        ppc_cells=int(data.train_y.shape[0]),
        posterior_medians={name: float(np.median(samples[name])) for name in ("sigma_eta", "rho")},
        sampler={
            "backend": config.backend,
            "chain_method": CHAIN_METHOD,
            "chains": chains,
            "warmup": config.warmup,
            "draws": draws,
            "target_accept": config.target_accept,
            "seed": config.seed,
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "numpyro_version": numpyro.__version__,
            "jax_version": jax.__version__,
            "max_tree_depth": MAX_TREE_DEPTH,
            "mean_leapfrog_steps": float(np.mean(num_steps)),
            "tree_depth_saturation_share": float(np.mean(num_steps >= 2**MAX_TREE_DEPTH - 1)),
        },
    )


# §2.2's backend row: NumPyro first, CmdStanPy "a supported alternative". This mapping is that
# slot. A second backend adds an entry here and a value to `ModelConfig.backend`'s Literal, in one
# change.
BACKENDS: dict[str, Callable[[ModelData, StateModelConfig], StateModelFit]] = {
    "numpyro": _fit_numpyro,
}


def fit_state_total_model(data: ModelData, config: StateModelConfig) -> StateModelFit:
    """§16.2's fit: the posterior's §11.5 scores as joint draws, plus §11.14's diagnostic inputs.

    It dispatches on `config.backend` through `BACKENDS`. An unknown backend is refused here as well
    as by the config's `Literal`, because a `StateModelConfig` can be built without a config file.
    """
    backend = BACKENDS.get(config.backend)
    if backend is None:
        raise ConceptViolationError(
            f"backend {config.backend!r} has no implementation; BACKENDS carries {sorted(BACKENDS)}"
        )
    return backend(data, config)


def prior_predictive(data: ModelData, config: StateModelConfig, *, num_samples: int) -> np.ndarray:
    """The training-cell response drawn from the priors alone, shaped (num_samples, N).

    This is the bayesian-workflow prior predictive check: the priors in `config.priors` must put the
    response where log employees per establishment plausibly lives before any fit is trusted. With
    `y=None` no cell is pinned, so every path is drawn from its innovations and `y` is mu itself.
    """
    draws = Predictive(state_total_model, num_samples=num_samples)(
        jax.random.key(config.seed), data, config, y=None
    )
    return np.asarray(draws["y"])
