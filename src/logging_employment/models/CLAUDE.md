# `models/` — §11's state-total model

Fits §11.1-§11.3's robust hierarchical state-intensity model in NumPyro, turns every draw into
§11.5's score, reconciles each draw into the deterministic feasible set month by month (§12.7,
INV-012), gates the fit on §11.14, and writes §7.11's `posterior_summary` and §15.4's store. Built by
plan 16 (roadmap Stage 5). Spec map: §11.1-§11.3 and §11.5 → `state_total`, §11.12's priors →
`config.StateModelPriors`, §11.14 → `diagnostics`, §12.7 → `reconciliation`, §7.11 → `summary`,
§15.4 → `arviz_io`, §16.2 → `interfaces`, §13's scoring of the model → `validation`. §11's
size-composition model is Stage 6's and does not live here yet.

Module docstrings are the design record, and each states its alternatives. Read the one you are
about to change before you change it.

## Read this first

- **Only two modules touch a PPL.** `interfaces`, `data`, `reconciliation` and `summary` are NumPy
  and polars. `state_total` alone imports JAX and NumPyro, and `arviz_io` alone imports ArviZ and
  xarray. `diagnostics` reaches ArviZ through `arviz_io`, and `validation` reaches JAX through the
  fit. `cli.py` imports each model module inside the command that needs it, so `--help` and the
  baseline commands never initialise JAX. Keep it that way (§16.2: "PPL-specific objects must
  remain behind model interfaces").
- **`state_total` turns on float64 when it is imported** (`numpyro.enable_x64()`), and
  `_fit_numpyro` refuses to sample in float32. `reconcile_draws` checks adding-up to 1e-9, which
  float32 cannot represent at employment scale. Chains run `vectorized`, in one process, so the
  same key gives the same draws in any process and `numpyro.set_host_device_count` is never called.
- **A raw score is not an estimate.** §11.5's `q = A * exp(mu)` is computed in
  `state_total.raw_scores` and nowhere else. Every draw becomes an estimate only through
  `reconciliation.reconcile_fit`, which calls `reconcile/draws.py::reconcile_draws` once per month.
  `summary.posterior_summary` reads reconciled draws only (§12.7: "Posterior summaries must be
  computed from reconciled joint draws").
- **The anchor is a declared modeling assumption, never a constraint.** `reconcile_fit` admits the
  window through `reconcile/anchor.py`'s `closure_audit` + `assert_universe_closes`, then allocates
  each month against `national_residual`, stamped `anchor_basis = 'declared_national_total'`
  (§12.2 since `D-120`). Its implied ceiling never enters `deterministic_bounds`. §7.11's table has
  no `anchor_basis` column, so the basis rides in `state_model_manifest.json` (`anchor_bases`) and
  in the store's `constant_data`.
- **Training cells are exact; there is no σ_y.** §11.3's first sentence governs: at a training cell
  mu = y, the path is pinned there, and the likelihood is the AR(1) density of the innovation that
  implies. §11.3's t(mu, sigma_y) "practical response model" was plan 16's first implementation and
  failed §11.14's gate on D1 (Decision 15). So the PPC is one-step-ahead (`_ppc_coverage_90`).
- **Three more choices, each with a measured reason (Decision 15).** A trained state samples its
  pinned `level` and `log_scale` directly; persistence is capped (`persistence_max`); X is split
  into `beta` (within a state) and `beta_between` (across states). The first two are what let D1's
  fit pass §11.14's gate; the third keeps a state with no training cell from being predicted off
  its true level. Undo one and re-run the D1 acceptance before trusting a result.
- **`fit_state_total_model` returns `StateModelFit`, not a bare `PosteriorDraws`** — a recorded
  deviation from §16.2 (`interfaces.py`'s module docstring). `PosteriorDraws` is imported from
  `reconcile/draws.py`, never redeclared.
- **Means are summed with `math.fsum`** (`reconciliation.exact_column_means`), because
  `ndarray.mean(axis=0)` moved a persisted mean in its last place with memory layout. The summary's
  and the harness's point estimates are digested, so they must depend on the values alone.
- **Measured on D1 on 2026-09-27 (plan 16, Task 15), a dated witness.** `runs/dd7337e89047`'s
  production fit (seed 3645, jax 0.11.2, numpyro 0.22.0) passed §11.14's gate with 0 divergences,
  parameter R-hat 1.0067, cell R-hat 1.0045, cell ESS bulk 1,010 and tail 434 against a floor of
  400, one-step 90% coverage 0.904, anchor drift 4.5e-13 and 0 bound violations. `fit-state-model`
  took 139.5 s wall, at 31.0 leapfrog steps a draw with a tree-depth saturation share of 0.0. Of the
  1,227 imputed cells, 189 have a zero-width 90% interval on their §9 upper bound. §13.10's verdict
  is `not_beaten`, so `section_10_8_hierarchy` is selected, provisionally until Stage 7. Three gates
  failed:
  - coverage, pooled 940 of 1,253 = 0.750;
  - improvement, where 5 of 9 regimes are worse than their comparand on matched WAPE;
  - convergence, where 1 of 27 replicates had parameter R-hat 1.0187.

  Recompute before citing a number. See `specs/findings/stage-5-log.md`.

## The model's structural choices (plan 16, Decision 7)

- No harvest factor H: `include_harvest_factor` is `Literal[False]`, and Stage 7 adds it and re-runs
  §13.10's gate. The three `include_*` switches are refused at load if set true.
- X is standardized log establishment count, centred and scaled over training AND prediction cells.
  `qtrly_establishments` is published for suppressed cells, so this reads nothing a suppression
  withholds. It enters twice: beta on x - x_bar[s] and beta_between on x_bar[s]
  (`state_total.state_mean_exposure`), because D1's within-state and between-state slopes differ in
  sign (Decision 15).
- One latent AR(1) path per state crosses the 2022-01 NAICS seam: 113310 maps one-to-one across the
  two vintages, each cell keeps its own vintage inside its `cell_id`, and the year effects absorb a
  level shift (`data.py`'s docstring).
- True zeros are neither trained on nor predicted: they are published, and log(E / A) is undefined
  there. `build_model_data` refuses any training or prediction cell with `A <= 0`, a suppressed cell
  with a value, a non-final training row, and more than one industry or ownership code.
- Non-centred only where no training cell pins a state: a trained state samples its level and its
  log innovation scale directly (Decision 15). Month and year effects are ZeroSumNormal. Persistence
  is 0.95 * Beta(8, 2), prior mean 0.76, where §11.12 says "centered near 0.8" (flagged, Decision 7).
  Every prior is a config field, so it is in `resolved_dict` and in `run_id`: the slopes' scale is
  `ModelConfig.standardized_beta_sd`, Appendix A's own key, and the rest are `StateModelPriors`.
  No float in the `model:` block may be infinite or NaN, and each is refused at load, by name.

## The gate and its two scopes

`diagnostics.evaluate_gate` measures divergences, parameter R-hat, cell R-hat and bulk and tail
ESS over the imputed cells' reconciled draws, one-step-ahead 90% predictive coverage over the
training cells (they are exact, so an in-sample check would pass by construction), and the two
reconciliation checks. `production` gates on all of them and `fit-state-model` raises
`ModelDiagnosticsError` after writing `posterior/diagnostics.json`, which also records the run's
`constraint_set_hash`. `replicate`, the harness's 27
fits, records everything and gates only on divergences and parameter R-hat (Decision 5);
`validate/promotion.py` reads those reports. A cell pinned by a one-cell missing set never varies,
so it is counted in `cells_constant` and left out of R-hat and ESS. A NaN anywhere else fails its
check. Tail ESS passes `prob=(0.05, 0.95)` explicitly: arviz-stats' array interface requires it.

## The store

`posterior/state_total_draws.nc` (`interfaces.STORE_PATH`) is an ArviZ-shaped `DataTree` written
with h5netcdf: `posterior` (the monitored parameters), `posterior_predictive`
(`reconciled_state_total`, `(chain, draw, cell)` with `state_fips` and `reference_month`
coordinates), `sample_stats` (`diverging`) and `constant_data` (each month's residual and anchor
basis, each cell's bounds and mean raw score). `draws_sha256` digests the reconciled array, its
shape and its cell ids, never the file: HDF5 metadata is not promised to hold still. `reconcile`
re-reads the store and re-checks INV-012 against it, and `validate-state-model` does the same
before it scores anything. Both first compare `diagnostics.json`'s `constraint_set_hash` with
`schema_manifest.json`'s and refuse a fit from another constraint set unread
(`cli.py::_stale_fit`): the draws hold to that set's bounds, so re-checking them would pass.
`fit-state-model` keeps the last fit when it refuses stale bounds, and `build-constraints` can
re-run without a re-fit, so such a fit is a normal state of a run directory, not a corruption.

## Tests and commands

```bash
uv run pytest tests/unit/test_numpyro_api_probe.py tests/unit/test_arviz_api_probe.py \
  tests/unit/test_model_data.py tests/unit/test_state_total_model.py \
  tests/unit/test_model_reconciliation.py tests/unit/test_posterior_summary.py \
  tests/unit/test_arviz_store.py tests/unit/test_model_diagnostics.py \
  tests/unit/test_model_validation.py   # hermetic tier: no data/, toy fits of 2 chains x 40 draws
uv run pytest -m slow tests/integration/test_cli_state_model.py \
  tests/integration/test_state_total_recovery.py tests/integration/test_cli_validate_state_model.py

logging-estimates fit-state-model --config config.yaml       # after solve-bounds
logging-estimates reconcile --config config.yaml             # also re-verifies the stored draws
logging-estimates validate-state-model --config config.yaml  # after validate; 27 more fits on D1
```

`tests/unit/conftest.py::make_state_fit` builds a `StateModelFit` from a raw-score array with no
sampler behind it, so reconciliation, the gate, the summary and the store are tested without JAX.
`tests/integration/test_state_total_recovery.py` simulates from the model's own generative process
at D1's regime (exact cells, D1's innovation scales, whole-quarter suppression, two untrained
states) and checks §17.5's recovery rows. Its bounds were first run on arviz-stats 1.3.2 and NumPyro 0.21
while plan 16 was written, and its docstrings give the arithmetic behind each one.
