"""§16.2's model interfaces: what a fit consumes and what it returns, with no PPL in sight.

§16.2: "PPL-specific objects must remain behind model interfaces." Nothing here imports JAX or
NumPyro, so `ModelData` can be built and tested in a process that never initialises a sampler, and
a second backend (`state_total.BACKENDS`) would consume the same objects.

`PosteriorDraws` IS IMPORTED, NEVER REDECLARED. Stage 3 ships it (`reconcile/draws.py`), and
`reconcile_draws` both takes and returns it. A second class of the same name here would make two
types where §16.2 names one.

DEVIATION FROM §16.2, RECORDED. `fit_state_total_model` returns a `StateModelFit`, not a bare
`PosteriorDraws`. §11.14 says "The model interface MUST return joint draws and standard
diagnostics", and a `PosteriorDraws` has no slot for a divergence count or a parameter trace.
`StateModelFit.raw_scores` is the `PosteriorDraws` that §16.2 names.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ModelConfig, StateModelPriors
from ..reconcile.draws import PosteriorDraws

__all__ = [
    "MODEL_ID",
    "MODEL_VERSION",
    "STORE_PATH",
    "ModelData",
    "PosteriorDraws",
    "StateModelConfig",
    "StateModelFit",
]

# §7.11's `model_id` and `model_version`. The version names the model's SPECIFICATION: the
# likelihood, the priors' form and §11.5's score. Bump it when one of those changes. The code
# that ran is recorded separately, as `code_commit` in every manifest (`runs.code_provenance`).
MODEL_ID = "state_total_model"
MODEL_VERSION = "student_t_ar1.1"
# The store's place in a run directory: under `posterior/`, where the roadmap's Stage 5 `Produces`
# puts it. Declared here rather than in `arviz_io` so `reconcile` can find the store without
# importing xarray when there is none.
STORE_PATH = "posterior/state_total_draws.nc"


@dataclass(frozen=True)
class ModelData:
    """One month-by-state panel as the state-total model reads it (§11.1, §11.3, §11.5).

    It holds two disjoint cell sets over a `states x months` latent grid:

    * TRAINING cells are §11.3's "disclosed training cells": `observation_status == 'observed'`,
      each with a response `y = log(E / A)`.
    * PREDICTION cells are the suppressed ones. They carry exposure `A` and NO response. That is
      §11.3's "Suppressed cells have no pseudo-observation", held by the type rather than by a
      filter: no array exists that a suppressed cell's value could be written into.

    A true-zero cell (27 on D1, each with `A = 0`) is in neither set. It is published, so nothing
    imputes it, and log(E / A) is undefined there. A (state, month) with no published row at all
    (on D1, 48 ND months and 36 DE months) is on the grid only through the latent path, which must
    be contiguous in time for §11.2's AR(1). DC publishes no row in any month and so is not a
    state of the grid.

    `train_x` and `predict_x` are standardized log establishment counts, §11.1's X. Their centre and
    scale come from training and prediction cells together. `qtrly_establishments` is published for
    suppressed cells, so standardizing over both reads nothing a suppression withholds.
    """

    states: tuple[str, ...]
    months: tuple[str, ...]
    divisions: tuple[str, ...]
    years: tuple[str, ...]
    state_division: np.ndarray
    month_of_year: np.ndarray
    year_of_month: np.ndarray
    train_state: np.ndarray
    train_month: np.ndarray
    train_y: np.ndarray
    train_x: np.ndarray
    predict_state: np.ndarray
    predict_month: np.ndarray
    predict_exposure: np.ndarray
    predict_x: np.ndarray
    predict_cell_ids: tuple[str, ...]
    log_exposure_centre: float
    log_exposure_scale: float


@dataclass(frozen=True)
class StateModelConfig:
    """§16.2's `StateModelConfig`: the sampler settings and priors one fit runs under.

    A frozen dataclass built from `config.model` by `from_config`, rather than the pydantic block
    itself, for two reasons. A test can shrink the sampler with `dataclasses.replace` without
    writing a config file. And the fit never sees `diagnostics`, which belongs to the gate
    (`models/diagnostics.py`), not to the sampler.
    """

    backend: str
    chains: int
    warmup: int
    draws: int
    target_accept: float
    seed: int
    standardized_beta_sd: float
    priors: StateModelPriors

    @classmethod
    def from_config(cls, model: ModelConfig) -> StateModelConfig:
        """The fit settings `config.model` declares, and nothing else from it."""
        return cls(
            backend=model.backend,
            chains=model.chains,
            warmup=model.warmup,
            draws=model.draws,
            target_accept=model.target_accept,
            seed=model.seed,
            standardized_beta_sd=model.standardized_beta_sd,
            priors=model.priors,
        )


@dataclass(frozen=True)
class StateModelFit:
    """One fit's joint draws, plus the diagnostics §11.14 requires beside them.

    `raw_scores` is §11.5's q for every prediction cell, with one row per retained draw. Its
    `chain` and `draw` arrays keep the index §15.4's store must preserve. Its `cell_ids` are the
    seven-field `cell_id`s of `ModelData.predict_cell_ids`, in that order. These are NOT
    estimates. §11.5 says "These scores are not final estimates", and `models/reconciliation.py`
    maps each draw into the feasible set before anything summarises it.

    `parameters` maps each monitored parameter's name to an array shaped `(chains, draws, ...)`,
    the layout `arviz_io` reads. The latent innovations are excluded: there is one per cell no
    training value pins, 1,338 per draw on D1, and no released summary is computed from them.

    `posterior_medians` holds two scalars a reviewer needs to read the fit, each a median over
    draws and states: the innovation scale `sigma_eta` and the persistence `rho`. Persistence
    decides how far a suppressed gap reverts toward the state level, and a median pressed against
    `persistence_max` says the cap binds.
    """

    raw_scores: PosteriorDraws
    parameters: dict[str, np.ndarray]
    diverging: np.ndarray
    ppc_coverage_90: float
    ppc_cells: int
    posterior_medians: dict[str, float]
    sampler: dict[str, object]

    @property
    def chains(self) -> int:
        """The number of chains, read off the divergence array's leading axis."""
        return int(self.diverging.shape[0])

    @property
    def divergences(self) -> int:
        """Post-warmup divergent transitions, summed over every chain."""
        return int(np.count_nonzero(self.diverging))
