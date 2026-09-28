"""The NumPyro and JAX mechanisms `models/state_total.py` is built on, run once on a toy model.

The model scores exact training cells by pinning a scanned AR(1) path and adding the innovations'
Student-t log density through `numpyro.factor`. With nothing to pin, it draws the path from unit
innovations and records it with `numpyro.deterministic`, which is what `Predictive` returns. The
toy below does both, beside a zero-sum seasonal vector, so an API that moved fails here, in seconds,
before Task 4 builds the state-total model on it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

numpyro.enable_x64()

SEED = sum(map(ord, "tests/numpyro-api-probe"))


def _model(y: jnp.ndarray | None = None) -> None:
    """An AR(1) path around a location, pinned to `y` and scored when given, drawn when not."""
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    rho = numpyro.sample("rho", dist.Beta(8.0, 2.0))
    scale = numpyro.sample("scale", dist.HalfNormal(0.1))
    numpyro.sample("season", dist.ZeroSumNormal(0.5, event_shape=(12,)))
    if y is None:
        z = numpyro.sample("z", dist.StudentT(5.0, 0.0, 1.0).expand([5]).to_event(1))

        def step(previous: jnp.ndarray, innovation: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            current = rho * previous + scale * innovation
            return current, current

        _, path = jax.lax.scan(step, jnp.zeros(()), z)
        numpyro.deterministic("y", mu + path)
        return
    innovations = (y[1:] - mu) - rho * (y[:-1] - mu)
    numpyro.factor("y", dist.StudentT(5.0, 0.0, scale).log_prob(innovations).sum())


def test_enable_x64_makes_jax_arrays_float64() -> None:
    assert jnp.asarray(0.0).dtype == jnp.float64


def test_vectorized_chains_take_a_typed_key_and_report_their_extra_fields_by_chain() -> None:
    mcmc = MCMC(
        NUTS(_model),
        num_warmup=20,
        num_samples=10,
        num_chains=2,
        chain_method="vectorized",
        progress_bar=False,
    )
    mcmc.run(jax.random.key(SEED), y=jnp.zeros(5), extra_fields=("diverging", "num_steps"))
    samples = mcmc.get_samples(group_by_chain=True)
    assert samples["mu"].shape == (2, 10)
    assert samples["season"].shape == (2, 10, 12)
    np.testing.assert_allclose(np.asarray(samples["season"]).sum(axis=-1), 0.0, atol=1e-10)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    assert np.asarray(extra["diverging"]).shape == (2, 10)
    # Leapfrog steps per iteration: at least one, at most 2**10 - 1 under the default tree depth.
    steps = np.asarray(extra["num_steps"])
    assert steps.shape == (2, 10)
    assert steps.min() >= 1
    assert steps.max() <= 1023


def test_predictive_draws_the_unobserved_site() -> None:
    draws = Predictive(_model, num_samples=7)(jax.random.key(SEED))
    assert np.asarray(draws["y"]).shape == (7, 5)
