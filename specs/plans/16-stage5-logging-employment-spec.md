# Stage 5: State-Total Bayesian Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> **subagent-driven-development** (the default) — or **executing-plans** when your human partner
> chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/logging-employment-spec-roadmap.md, Stage 5 — on plan completion, tick the stage and re-validate later stages against what shipped.

> **Decision 15 was answered on 2026-09-26, and the model changed before execution.** While this
> plan was written, a production-config fit of the model as §11 writes it failed §11.14's gate on D1
> (parameter R-hat 3.76, cell bulk ESS 5.7 against 400). Your human partner chose a design pass
> first. The pass changed four things: training cells are exact, trained states are sampled in the
> coordinates the data pin, persistence is capped, and X is split into within-state and
> between-state parts. The revised model passed the production gate on D1 on two seeds, through the
> real fit → reconcile → gate, at about 130 s a fit. Decision 15 has the evidence. Decision 7 flags
> the four spec readings the change takes.

**Goal:** Fit §11's robust hierarchical state-intensity model, reconcile every joint draw to the hard
constraints, enforce §11.14's diagnostics in code, and promote the model over the Stage 4 preferred
baseline only if it clears §13.10's gates, writing the verdict either way.

**Architecture:** A new `models/` package behind §16.2's interface.
- `models/data.py` builds the panel.
- `models/state_total.py` is the only JAX/NumPyro module.
- `models/reconciliation.py` passes every draw through Stage 3's `reconcile_draws`, month by month, behind the same closure gate as the baselines.
- `models/diagnostics.py` enforces §11.14.
- `models/summary.py` writes §7.11's `posterior_summary`.
- `models/arviz_io.py` stores the draws for §15.4.

`validate/harness.py` gains a producer seam, so the model is scored by the loop that scored the comparand, and `validate/promotion.py` writes §13.10's record. Two commands: `fit-state-model` (§16.1) and `validate-state-model`, which is new so that `validate` stays the comparand's command.

**Tech Stack:** Python ≥ 3.14, uv + hatchling, Polars, NumPy, SciPy, HiGHS via `highspy`, Typer,
pytest; new here: JAX, NumPyro, arviz-base, arviz-stats (with xarray) and h5netcdf.

**Requirements input:** read at `40b2688`.
- `specs/logging-employment-spec-roadmap.md`'s Stage 5 block, every field, `Consumes` in full.
- Two facts from other blocks: Stage 4's SHIPPED point (3), that `preferred_baseline` may return `None` ("no comparand"), and Stage 7's Produces, that this stage's promotion decision is provisional until Stage 7 re-runs §13.10.
- The spec sections on the block's `Spec:` line.
- `specs/findings/stage-5-log.md`.
- `specs/deferred_items.md`, where `D-109`, `D-115` and `D-116` name this stage.

`D-120`, the §12.2/§15.2 amendment the brief anticipated, landed before this plan was saved, so Decision 2 cites it.

**Provenance of the code in this plan.** Every code block in Tasks 1–14 was executed before this plan was saved, in plan order, as one commit per task in a scratch repository built from `40b2688`. The whole chain was rebuilt and re-run after Decision 15's pass.
- Each red output shown was observed by running that task's tests against the previous task's commit.
- Every commit passed `ruff format --check`, `ruff check`, `interrogate`, its own tests, and the non-slow suite without `data/`.
- The environment was Python 3.14 with jax 0.11.1, numpyro 0.21.0, arviz-base 1.3.0, arviz-stats 1.3.2 and xarray 2026.7.0, one release behind each pin, with NumPy 2.5.3 and Polars 1.44.2 against the lock's 2.5.2 and 1.44.1. Installing the pinned versions needs network access this session did not use. Task 1's probes exist to catch the difference in seconds.
- The observed lines are quoted where they appear, as "observed".

**Suite counts are stated as deltas, and every delta was measured.**
- Base, with `data/`: `1570 passed` for the whole suite (12 min) and `1543 passed, 27 deselected` for the non-slow tier (2.5 min), at `40b2688`, measured while this plan was written.
- Tests added: 135, of which 21 are `slow`, and every one runs without `data/`.
- Per-task deltas on the non-slow suite: Task 1 +6, Task 2 +14, Task 3 +14, Task 4 +13, Task 5 +8, Task 6 +9, Task 7 +4, Task 8 +12, Task 9 +3, Task 10 +0 (10 slow), Task 11 +14, Task 12 +17, Task 13 +0 (6 slow).
- The scratch's hermetic count after Task 13 was `1607 passed, 45 skipped, 49 deselected`. With the five h5netcdf tests below, that becomes the `1612 passed, 45 skipped, 48 deselected` Task 14 expects.

**What was NOT executed, and why:**

- **Task 1 Steps 3–4, `uv add` and the lock.** They need PyPI, and no package was downloaded. The pins are the current releases, and their cp314 wheels were confirmed (Decision 10).
- **Thirteen tests that need h5netcdf**, which no local Python 3.14 environment had:
  - Task 1's netCDF round-trip probe;
  - Task 7's four store tests;
  - the four Task 9 tests on the `fitted` fixture;
  - the four Task 13 tests on the `validated` fixture.

  Everything they call below the netCDF layer ran.
- **Every `requires_staged` test at Tasks 1–14.** The per-task gates ran without `data/`, so these were collected and skipped. The non-slow ones then ran once at Task 14's code with `data/` linked: `1652 passed, 49 deselected`, which is 1543 + 114 less the five h5netcdf tests, the moved pin included. The slow D1 tier was not re-run on plan code; Task 15 Step 4's byte comparison is its check.
- **Task 14's two measured counts,** predicted from the scratch's: a bare run at Task 14's code without `data/`, leaving out the 13 tests that need h5netcdf, gave `1620 passed, 72 skipped`, and those 13 make it the `1633 passed, 72 skipped` Task 14 expects. The hermetic count is the one above.
- **Task 15, apart from three pieces.** Its production fit, reconciliation and gate ran on D1 through the scratch code on two seeds, and passed (Decision 15). Seven replicate masks ran through the same fit, reconcile and gate code, outside the CLI. Step 4's byte comparison was dry-run against a doctored copy of the comparand: it accepted exactly the declared differences and named two planted ones. The comparand re-run itself, the CLI's 27 replicate fits, and every fit on the locked versions happen first at execution.

---

## Decisions this plan makes

The brief leaves these open, or flags them for an explicit decision. Each gives its evidence so a reviewer can reject the decision rather than rediscover it. The three the brief flags are 1, 2 and 3. Decisions 4, 5 and 7 are the readings of the spec a reviewer is most likely to want changed. Decision 15 was a question only your human partner could answer. They answered it with a design pass, and it now records how that pass changed the model.

1. **Appendix A's `model:` block goes INTO `resolved_dict`, so the run id moves, exactly once.** `specs/completed/stage5-preconditions.md` §4 leaves the choice to this plan. `exclude=True` is the remedy for a key no stage reads (`SourcesConfig`'s seven inactive sources). Every key here changes the draws `fit-state-model` writes, and excluding them would let two fits under different priors share one `runs/<id>/`. The cost is one re-id. Both ids were computed from the scratch tree after Decision 15's pass settled the `priors` block, and the staged pin was confirmed by running its own test with `data/` linked. The config-only canary `run_id(load_config("config.yaml"), {})` moves `39d1d0859838` → `14352bb8e56e`. The staged pin over `data/staged`'s five tables moves `4cf47a918dd8` → `dd7337e89047`. `promotion.catastrophic_stratum_coverage_alpha` (Decision 4) lands in the same task, so the id moves once rather than twice.

   **The comparand is re-run under the new id and checked byte for byte (Task 15)**: `build-constraints` → `solve-bounds` → `run-baselines` → `reconcile` → `validate`. The checks are: every parquet file identical by sha256; every JSON manifest identical except `code_commit` and `uv_lock_sha256` (no manifest in `runs/4cf47a918dd8` names its own run id, checked); and `config.resolved.yaml` differing only by the `model:` block and the new promotion key. `runs/4cf47a918dd8` stays on disk untouched.

   **`D-116` does not ride along.** Its revisit-if ("the Stage 5 comparand is re-run for another reason") fires again. It is declined for the reason its own text records from the last time: a new propensity predictor re-draws every regime's mask. That would turn this re-run's byte-identity check into an attribution problem, and it would move the comparand this stage is judged against.

   **`D-115` is not taken either.** Its revisit-if asks whether §13.10 "needs tighter state bounds than the parent alone gives". It does not: the model and the comparand are scored on one identification set, so a tighter bound would move both.

2. **The anchor is named by symbol, and §12.2's superseded wording appears nowhere in this plan's constraints.** Every allocation target is `reconcile/anchor.py::national_residual`, stamped `anchor_basis = 'declared_national_total'`. The establishment-closure gate `closure_audit` + `assert_universe_closes` admits it, and §12.2 has named it that way since `D-120`. `models/reconciliation.py::reconcile_fit` runs that gate over the whole window before it builds any anchor, exactly as `run_baselines` does. So the model inherits `D-123`'s identity check on a fully disclosed month, and every `Anchor` refuses an undeclared basis as it is built (`D-122`). Both landed on main in PR #34 (`3baf6ec`) while this plan was being written. That covers the anchors `models/arviz_io.py::read_store` rebuilds from disk too. `D-124` (PR #35, `40b2688`) then made `closure_audit`'s `anchored` column agree with that gate. The model path reads the gate, not the column, so nothing here changes.

   **Flagged for review:** §12.2 says every row whose allocation target is R_t carries `anchor_basis`. §7.11's `posterior_summary` has no such column, and this plan keeps §7.11's 23 fields in order. The basis is recorded in `state_model_manifest.json` (`anchor_bases`) and in the store's `constant_data`. The model's harness rows (`models/validation.py::model_results`) carry it in `anchor_basis` as every `baseline_results` row does. The alternative is a §7.11 amendment that adds the column, which moves `POSTERIOR_SUMMARY_SCHEMA`'s fingerprint.

3. **`D-109` closes in Task 12, on its own done-when.** `validate/promotion.py` reads all three Appendix A keys, plus Decision 4's fourth, and is the only module that does. The tripwire `test_the_promotion_keys_are_still_unread_and_the_docstring_still_says_so` reddens that day and names the keys, as Task 12 Step 4 shows. It is then converted rather than deleted: `test_the_promotion_keys_are_read_only_by_the_promotion_record` pins `validate/promotion.py` as the sole reader. Coverage is compared exactly, as hits over `calibration_sample_size` in `Fraction` against `Fraction(repr(tolerance))`. `tests/unit/test_validate_promotion.py::test_seventeen_of_twenty_is_within_five_points_of_ninety` pins 17/20 against 0.90 ± 0.05 as within, where the float comparison says no. `PromotionConfig`'s docstring is rewritten to say who reads the keys.

4. **§13.10's readings. FLAGGED FOR REVIEW; all live in `validate/promotion.py`'s module docstring, and each reaches the record beside its evidence.** §13.10 states six gates and no procedure.
   - **Coverage is pooled** over every regime and seed, as exact integer hits over exact samples, never a mean of ratios. Per-regime coverage is reported and gates nothing. A perfectly calibrated model passes a ±5-point test in all nine regimes separately with probability 0.106 at the comparand's sample sizes, and pooled over its n = 1,253 with probability ≈ 1.0. The pool is lopsided: `whole_seasonal_blocks` is about 69% of it (`D-106`), and `pool_share` records each regime's part.
   - **"Does not fail catastrophically in any major stratum"** means P(X ≤ hits) < α for X ~ Binomial(n, 0.9), with α = `catastrophic_stratum_coverage_alpha` = 0.001, a key this plan originates. It is tested per regime and per Census division pooled across regimes. That is 18 tests, with a family-wise false-alarm rate of 0.0122 for a calibrated model. For scale, the tail probability is 7.2e-6 at 10 of 20, 0.0024 at 13 of 20, 1.7e-5 at 160 of 200, and 0.00078 at 165 of 200. The comparand's pooled division samples run from 63 (`middle_atlantic`) to 225 (`south_atlantic`).
   - **WAPE is compared on matched (seed, cell) pairs, per regime**, as a relative improvement of at least `minimum_wape_improvement` over `scoreboard.preferred_baseline`'s choice. The model never declines where a comparand may, so pooled scoreboard WAPEs would compare two cell sets, and `n_model_only` counts what matching drops.
   - **Degradation is per Census division, pooled across regimes**, relative, failing above `maximum_major_stratum_wape_degradation`. "Without documented substantive benefit" is not machine-evaluable, so a degraded division fails and is named.
   - **The "clearly superior calibrated uncertainty" alternative** passes a regime whose WAPE gate failed only when three things all hold: the pooled coverage passes and the regime is not catastrophic for the model; the comparand has no interval there or its own pooled coverage is catastrophic; and the model's CRPS is no worse wherever the comparand has one.
   - **A regime whose `preferred_baseline` is `None` is `not_applicable`**, neither a pass nor a failure (Stage 4's SHIPPED point (3)). At least one regime must have a comparand.
   - **The verdict** is `beat` or `not_beaten`. `selected_method` is `state_total_model` or `section_10_8_hierarchy` (§13.10's last line). `provisional` is always true, because Stage 7 adds the harvest factor and re-runs the gate (Stage 7's Produces). Disclosure review is `pending_stage_8`, never passed.
   - **Hard constraints and convergence include every replicate fit.** A model that converged on the full panel and not on a masked one has not shown that its convergence holds.

5. **§11.14's gate has two scopes. FLAGGED FOR REVIEW.** `production` (`fit-state-model`) gates on every check:
   - divergences (0);
   - rank R-hat ≤ 1.01 over every monitored parameter element, and over every imputed cell's reconciled draws (§11.14's "stable posterior summaries across independent seeds/chains");
   - bulk and tail ESS ≥ `min_ess_per_chain` × chains = 400 on the imputed cells, the "release-relevant summaries";
   - one-step-ahead 90% predictive coverage ≥ 0.85 over the training cells (below);
   - both reconciliation checks.

   `replicate` (the harness's 27 fits) records all of these and gates only on divergences and parameter R-hat. Over roughly 1,400 quantities and 27 fits, a max-statistic gate would let one cell at 1.011 in one replicate decide promotion. That was measured, not only argued. Decision 15's pass ran seven replicate masks. All seven passed this scope, and one (`small_cell_biased`, seed 1024) would have failed production scope on cell tail ESS: 259 against 400. §11.14 gives no ESS number, and 100 per chain is the Vehtari et al. convention. A cell a one-cell missing set pins never varies, so it is counted in `cells_constant` and excluded. Any other NaN fails its check.

   **The PPC check is one-step-ahead and one-sided, and flagged.** Training cells are exact (Decision 15), so a training cell's in-sample replicate is y itself, and an in-sample check would pass by construction. The check is the one a state-space model admits. Each draw predicts each training cell from the month before: m + ρ·η[t−1], plus a Student-t innovation at the cell's scale (the stationary scale in the first month). The share of cells whose y falls inside its central 90% interval must reach `min_ppc_coverage_90`. That tests the innovation law the likelihood scores, and on D1 it scored 0.904 and 0.905 on two seeds. `min_ppc_coverage_90` is a key this plan originates (Appendix A's `model:` block has no diagnostics keys), and coverage above the nominal is not scored. Under §11's original σ_y model, the in-sample version scored 1.0 on D1's failed fit, a symptom of chains that disagreed. That model and its check no longer exist. A two-sided reading needs a second originated key. Decided before Task 2, it rides the same run-id move.

6. **§11.5's score is kept literal: `q = A · exp(mu)`.** mu includes the state-month path eta, so §11.2's process variation reaches every suppressed month. It is the only per-cell variation the model has. Training cells are exact (Decision 15), so there is no observation scale for the score to include or leave out. The question this decision first answered, whether §11.3's σ_y belonged in the score, went away with σ_y. The fit records the posterior medians of σ_η and ρ in `state_model_manifest.json`, so a reviewer can read the fit's innovation scale and persistence without opening the store. On D1 they were 0.036 and 0.838.

7. **The model's structure.**
   - H is omitted: `include_harvest_factor` is `Literal[False]`, and Stage 7 adds it.
   - X is standardized log `qtrly_establishments`, centred and scaled over training and prediction cells together. It is published where employment is suppressed. It enters as two terms, β·(x − x̄_s) + β_between·x̄_s, both slopes N(0, `standardized_beta_sd`) (reading 4 below).
   - One latent AR(1) path per state crosses the 2022-01 NAICS seam. `113310` maps one-to-one across both vintages, the year effects absorb a level shift, and each cell keeps its own vintage in its `cell_id`.
   - True zeros are neither trained on nor predicted, and `A > 0` is asserted on every cell that is.
   - The priors are `config.StateModelPriors`:
     - intercept N(1.5, 1), centred on D1's observed mean log employees per establishment of 1.46 (SD 0.48, over the 3,462 training cells);
     - state, region, month and year scales HalfNormal(0.5, 0.5, 0.25, 0.25);
     - persistence `persistence_max` · Beta(8, 2) = 0.95 · Beta(8, 2), prior mean 0.76 (reading 2 below);
     - innovation scale HalfNormal(0.1) with a log-scale dispersion HalfNormal(0.5);
     - the innovations' Student-t degrees of freedom fixed at 5 (§11.12). There is no observation scale (reading 1).
   - The parameterization: a state with a training cell samples its level and its log innovation scale directly, and a state without one is non-centred (reading 3). Month and year effects are ZeroSumNormal, and eta starts at a variance-matched stationary scale.
   - Float64 is enabled at import, chains run `vectorized`, and the seed is `sum(map(ord, "logging-employment/state-total-model"))` = 3645.

   **Four readings of the spec, FLAGGED FOR REVIEW.** A D1 measurement in Decision 15's pass forced each one. Each is a place a reviewer may prefer a spec amendment to this plan's reading.
   1. **§11.3: exact observation governs, and the "practical response model" is dropped.** §11.3 opens "Published QCEW values are exact observations … there is no extra arbitrary measurement error". It offers y ~ t(μ, σ_y) only as "a practical response model". On D1 the practical model failed §11.14's gate with σ_y near 0.004, and re-parameterizing it did not rescue it (R-hat 2.89). An amendment would state that training cells pin the latent path.
   2. **§11.12: persistence is capped at 0.95, so its prior mean is 0.76, where §11.12 says "centered near 0.8".** `persistence_max` is a key this plan originates, inside the `priors` block that Decision 1 already moves the run id for. Uncapped, a production-length fit failed R-hat on one of two seeds, with δ and ρ at 1.011.
   3. **§11.14: its non-centred SHOULD is set aside for trained states.** A SHOULD admits a measured reason. Non-centred, every tree hit the 1,023-step ceiling. In the pinned coordinates none does, and a fit takes about 130 s instead of 1,091. Untrained states keep the SHOULD.
   4. **§11.1: X holds two predictors built from one count, a within-state part x − x̄_s and a between-state part x̄_s.** §11.1's X_{s,t}ᵀβ admits a vector of "standardized, nonredundant predictors", and these two are nonredundant by construction. Neither is re-standardized: both are in units of standardized x, so `standardized_beta_sd` means the same for each. §11.1 does not say what X holds. This plan chose it, and until the design pass it was one predictor, standardized log A. The split is flagged because x̄_s is constant within a state, so it competes with u_s for each state's level. With one slope, a D1-regime simulation mispredicted the untrained states: their hidden-cell coverage was 0.50, and one state's true level fell 2.5 log points from its posterior mean. On D1 the two slopes have opposite signs, −1.31 and +0.34.

8. **`fit_state_total_model` returns `StateModelFit`, not a bare `PosteriorDraws`**, a deviation from §16.2 recorded in `models/interfaces.py`. §11.14 says the interface "MUST return joint draws and standard diagnostics", and `PosteriorDraws` has no slot for a divergence count or a parameter trace. `StateModelFit.raw_scores` is the `PosteriorDraws` §16.2 names, imported from `reconcile/draws.py` and never redeclared.

9. **Commands.**
   - `fit-state-model` is §16.1's, with preconditions `schema_manifest.json` and `deterministic_bounds.parquet`. A re-fit deletes the previous fit's artifacts first. `posterior/diagnostics.json` is written BEFORE the gate is enforced, so a failure exits 1 and keeps its evidence.
   - `reconcile` gains a second job. When a store exists, it re-reads it and re-checks INV-012 plus the draws' digest, recorded under `state_model` in `reconcile_manifest.json`. The key is omitted, never null, when there is no store, so a baseline-only run writes exactly the keys it always wrote.
   - `validate-state-model` is NOT in §16.1. It is added so that `validate` stays the comparand's command and its output stays byte-identical to Stage 4's. It needs `validate`'s three tables and `posterior/diagnostics.json`, and writes `state_model_validation/` plus `promotion_record.json`. Either verdict exits 0, because "deploy the simpler method" is an outcome, not an error. A failed production fit gets a `not_beaten` record without the 27 replicate fits being spent.

10. **Dependencies:** `jax>=0.11.2`, `numpyro>=0.22.0`, `arviz-base>=1.3.1`, `arviz-stats[xarray]>=1.3.3`, `h5netcdf>=1.8.1`, the current releases when this plan was written. Before locking, their cp314 wheels for `macosx_11_0_arm64` and `manylinux_2_27_x86_64` (jaxlib) and `manylinux_2_28_x86_64` (h5py) were confirmed on PyPI. Task 1 guards the lock: no package already in `uv.lock` may change version. A moved numpy, scipy or polars could move a float golden, and this plan re-pins none.

11. **The harness's point estimate is the MEAN of the reconciled draws**, and every mean over draws is summed with `math.fsum` (`models/reconciliation.py::exact_column_means`). Every draw sums to its month's residual and lies in its interval, so their mean does too, by linearity and convexity. A per-cell median has neither property, and `constraint_metrics` would score its adding-up gap as the model breaking a constraint it never broke. `ndarray.mean(axis=0)` moved one cell's mean in the last place in the scratch run (41.20440509225164 against the column's own 41.20440509225163). The means are digested (`posterior_summary_sha256`), so `fsum` makes them a function of the values alone, as `validate/metrics._exact_sum` does for the metrics. `posterior_summary` publishes both mean and median (§7.11).

    **The mean is the less stable summary in the sparsest states. This is flagged, not changed.** On D1, two seeds' reconciled posterior medians agree within 3.7% in every imputed cell. Their means differ by up to 12.2%, and the 5 cells over 5% are all in NV, which has no training cell, and RI, which has six. There, a raw score is exp of a Student-t path through a long unpinned stretch. Its mean is heavy-tailed, and only reconciliation bounds it. The harness scores the mean, for the adding-up reasons above, so a WAPE in those states carries that seed noise. Which summary is published is a question for §7.11 and Stage 8, and `posterior_summary` carries both.

12. **§10.7's time-ordered interval is not built.** The model's intervals come from its own reconciled draws (`interval_source = 'reconciled_posterior_draws'`, a third value of `contracts.INTERVAL_SOURCES`, now enforced by `assert_declared_provenance`). The baselines' leave-one-out ensemble stays as it is under its own name. `validate/intervals.py`'s docstring, which said Stage 5 would build the rolling version, is corrected.

13. **The per-task gate is the non-slow suite plus that task's slow tests, by path.** `uv run pytest -q -m "not slow"` runs everything else, and a task that adds a slow module runs it by name. The whole suite, slow tier included, runs at Task 14. The `slow` tier holds every NUTS fit larger than a toy.

14. **Execution runs in a worktree, with `data/` and `runs/` symlinked from the main checkout (Task 0).** Another session edits `specs/` in the main checkout concurrently, so this branch stays out of it. `.gitignore`'s `data/` and `runs/` match directories only, so the two symlinks are excluded through the common `info/exclude`. The symlinks make every `requires_staged` test run at every gate, so the pin moved in Task 2 is observed at once.

15. **The model was redesigned before execution, and the redesign passes §11.14's gate on D1.** Your human partner answered this decision's question with (b), a design pass first, on 2026-09-26. This is that pass's record. Four of its choices read the spec differently from its text, and Decision 7 flags each for review.

    **What failed: §11's model as written.** The scratch code of Tasks 3–8 made one production-config fit of the D1 panel: 50 states, 96 months, 3,462 training and 1,227 prediction cells, 4 chains of 1,000 warmup and 1,000 draws, seed 3645, `target_accept` 0.9, tree depth 10.
    - It took 1,091 s, and 99.95% of its trees hit the depth ceiling.
    - The production gate failed five checks: 2 divergences, parameter R-hat 3.76, cell R-hat 1.93, and cell bulk and tail ESS 5.7 and 13.9 against 400. Reconciliation did its job (anchor drift 4.5e-13, 0 bound violations).
    - Two 4 × 300/300 fits showed chains that disagree about where the posterior is, not chains that are slow. β's chain means were −0.01, −1.19, 0.01 and −1.15, and σ_u ran from 0.18 to 1.41. Centring η made it worse: 170 divergences, and σ_u collapsed to 0.0004–0.013 as the paths absorbed the state levels.
    - Every chain put σ_y between 0.003 and 0.006. At that scale the training cells pin μ in all but name.

    **The design pass.** Fourteen variants, the two diagnostic fits above among them, were fitted to the same D1 panel. Each was screened at 4 × 300/300, where R-hat ≤ 1.1 decided only whether to keep it. A variant was accepted only through the real fit → reconcile → production gate on two seeds, plus recovery at D1's regime. Four changes survived. Each is listed with the measurement that kept it.
    1. **Training cells are exact** (§11.3's first sentence). At a training cell the path is pinned, η = y − m, and the likelihood is the AR(1) density of the innovation that pinning implies. There is no σ_y, and `priors` loses its two `observation_*` keys. The control variant kept σ_y and changed only the parameterization, centring η wherever a state has at least 48 training months. It still split the chains: R-hat 2.89, three chains with σ_u under 0.006 and ρ up to 0.999, and one with σ_u 0.70. Exact observation made all four chains agree (β −1.21, σ_u 1.15), but every tree still saturated.
    2. **Trained states are sampled in the coordinates the data pin.** For a state with a training cell, exact innovations pin its level L = α + u + v + β_between·x̄ and its log innovation scale. So the 44 trained states sample those two directly, and u = L − α − v − β_between·x̄ is derived. That is a shear with Jacobian 1: the same model in other coordinates. The six untrained states stay non-centred. In non-centred coordinates every tree hit the 1,023-step ceiling. In these, trees average 31 steps, none saturates, and a production fit takes about 130 s instead of 1,091.
    3. **Persistence is capped:** ρ = `persistence_max` · Beta(8, 2), with `persistence_max` 0.95, which is §11.12's "transformed Beta prior" with mean 0.76. Uncapped, the largest state's posterior-mean ρ on D1 was 0.978, with 6% of states above 0.95, and a production-length fit failed §11.14's R-hat on seed 3645, with δ and ρ at 1.011. Capped, both seeds passed at 1.005. This was measured with change 2 in place, before change 4. Change 4 does not touch persistence.
    4. **X enters as a within-state part and a between-state part:** β·(x − x̄_s) + β_between·x̄_s, where x̄_s is the state's mean exposure over training and prediction cells, and β_between ~ N(0, `standardized_beta_sd`). With one β, Task 10's D1-regime test failed on prediction, not on mixing. The simulated panel's u tracked x̄ (correlation 0.91). An untrained state's true level fell 2.5 log points from its posterior mean, outside its 95% interval. Hidden-cell 90% coverage was 0.50 in the untrained states. On D1 the split finds β −1.31 and β_between +0.34 (posterior means). The correlation of u with x̄ over trained states is 0.013, and σ_u falls from 1.12 to 0.31.

    **The predictive check becomes one-step-ahead** (Decision 5). An exact training cell's in-sample replicate is y itself, so an in-sample check would pass by construction.

    **Acceptance, measured on the older local environment (jax 0.11.1, numpyro 0.21.0):**
    - **D1, the production gate, through the real fit → reconcile → gate, on two seeds.**
      - Seed 3645 PASSED: 0 divergences, parameter R-hat 1.0047, cell R-hat 1.0042, cell ESS bulk 975 and tail 1,176, one-step coverage 0.904, anchor drift 4.5e-13, 0 bound violations. The fit took 129.5 s and reconciliation 37.7 s, at 31 leapfrog steps a draw with no saturated tree.
      - Seed 20260926 PASSED: parameter R-hat 1.0086, cell R-hat 1.0049, cell ESS 1,268 and 723, one-step coverage 0.905, fit 130.7 s.
      - The posterior medians were σ_η 0.036 and ρ 0.838 on both seeds.
    - **Recovery.**
      - Task 10's D1-regime panel: 10 of 10.
      - The real model at D1's own mask and x, simulated from its D1 posterior means and refitted: 42 of 44 trained and 6 of 6 untrained levels fall inside their 95% intervals. β's and β_between's intervals hold their truths. Hidden-cell 90% coverage is 0.90 in trained states' gaps and 1.00 in untrained states.
    - **§13's replicate masks.** Seven regime and seed pairs went through mask → §9 solve → fit → reconcile → gate. Six ran on the final model, and `long_consecutive_runs` ran on the model before change 4.
      - All seven pass the replicate gate: parameter R-hat 1.0045–1.0055, 0 divergences, each fit 114–146 s.
      - Read at production scope, for information, six of seven pass. `small_cell_biased` seed 1024 has cell tail ESS 259 against 400.

    **What the evidence does not settle, and where each goes:**
    - **Margins.** Seed 20260926's parameter R-hat was 1.0086 against 1.01. Execution samples on the locked jax 0.11.2 and numpyro 0.22.0, so its draws will differ. Task 15 keeps its fail branch.
    - **Means in the sparsest states.** From seed to seed, reconciled posterior medians agree within 3.7% in every cell, but means differ by up to 12.2%, in NV and RI. Decision 11 covers this.
    - **Cells on a bound.** 177 imputed cells (14%) have a zero-width 90% interval in both seeds, each on its §9 upper bound: VT 58 of its 96 predicted cells, NM 27, NH 21, CO 19, UT 16, and 36 more across seven states. In a seed-3645 refit, 192 cells had their reconciled 5% quantile on the bound.
      - In 159 of those 192, more than half the raw draws already exceeded the bound, so the model overshoots it.
      - Elsewhere the raw draws mostly sit below the bound, and reconciliation's scaling toward the month's residual lifts them onto it. That is the pattern in WV and IA.

      §11.14 checks neither. Task 15 reports the count. Whether the model adds anything where §9 binds is a question for §13 and Stage 8.
    - **Cost.** `validate-state-model`'s 27 replicate fits come to about 27 × (130 s to fit + 40 s to reconcile and gate), roughly 77 minutes at the measured speed.

    **Where the evidence lives.** The pass's harness scripts (`variants.py`, `d1_accept.py`, `replicate_check.py`, `recover_real.py`, `bound_cause.py`) and its experiment ledger were scratch work. They are not part of this plan or the repo. A copy is kept at `~/.cache/logging-employment/plan16-scratch/` on the machine that wrote this plan. An executor needs none of them: Task 4's tests, Task 10 and Task 15 re-measure everything above that the plan depends on.

    **Tasks the answer changed:**
    - Task 1: the NumPyro probe's toy model now uses the model's own mechanisms.
    - Task 2: no `observation_*` priors, a new `persistence_max`, and both run-id pins.
    - Task 3: `posterior_medians`.
    - Task 4: the model.
    - Task 5: the fixture's medians.
    - Task 8: the predictive check's wording.
    - Task 10: the D1-regime panel.
    - Task 14: the guides and the counts.
    - Task 15: it expects a pass and keeps the fail branch.

    Tasks 6, 7, 9 and 11–13 consume `StateModelFit` and did not change.

## Global Constraints

- `requires-python = ">=3.14"`; author `Lowell Mason <mason.lowell@mac.com>`; MIT (Rollout D4).
- **`run_id` moves exactly once, in Task 2, and deliberately.** The config-only canary becomes `14352bb8e56e` and the staged pin `dd7337e89047`. No later task adds a `Config` field or a key to `config.yaml`. `runs/4cf47a918dd8` (the §13.10 comparand) and `runs/f03023ac9f3a` (Stage 4's acceptance run) are never written to: Task 15's re-run lands in `runs/dd7337e89047`.
- **Network: Task 1's `uv add` only** (PyPI). Every other task reads `data/` and writes `runs/`, and nothing else, with one exception. Task 15's `build-constraints` rewrites the main checkout's `data/constraints/`, and Step 4 proves the bytes identical against a copy taken in Step 2. No credential is needed or typed. `CENSUS_API_KEY` and `BLS_CONTACT_EMAIL` are never read by this plan.
- **The model is Decision 15's.** A production fit of §11's model as written failed §11.14's gate on D1 while this plan was written. Your human partner answered Decision 15 with a design pass (2026-09-26), and the pass changed the model before this plan was executed: training cells are exact, trained states are sampled in the coordinates the data pin, persistence is capped, and X is split into within-state and between-state parts. Decision 7 flags the four spec readings this takes. No task re-opens them. A task whose observed output contradicts Decision 15's measurements stops and reports.
- **Import discipline (§16.2: "PPL-specific objects must remain behind model interfaces").** `models/state_total.py` is the only module that imports JAX or NumPyro, and `models/arviz_io.py` the only one that imports ArviZ or xarray. `cli.py` imports every `models/` module inside the command that uses it, never at module level, so `--help` and the baseline commands never initialise JAX.
- **Float64, always.** `models/state_total.py` calls `numpyro.enable_x64()` at import, and `_fit_numpyro` refuses to sample in float32.
- **Seeds are descriptive**, `sum(map(ord, "<name>"))`, never a bare constant. The model's is 3645, from `"logging-employment/state-total-model"`.
- **Format and lint scope is `src tests`, never `.`**: `uv run ruff format src tests`, `uv run ruff check src tests`. `uv run interrogate src` is `fail-under = 100`, and every new callable's docstring is in its code block.
- **Fail closed with a named error** (§18.3). One class is new, `errors.ModelDiagnosticsError`. Everything else raises an existing `errors.py` class, mostly `ConceptViolationError`, and every message carries the offending value.
- **Schemas are ordered literals.** `contracts.POSTERIOR_SUMMARY_SCHEMA` is §7.11's 23 fields in §7.11's order. Every persisted table goes through `validate_frame`, then `assert_declared_provenance`, then `build.write_parquet_deterministic`.
- **INV-008.** Deterministic and posterior intervals are different columns and never one another. INV-012: every summary and every scored estimate is computed from reconciled draws, never from raw scores.
- **Suite arithmetic is the check, stated as deltas.** Each gate names the tests its task adds: `passed` rises by exactly that count, and `skipped` does not move. With `data/` linked (Task 0), a data-bound test runs; without it, it skips.
- **Do not edit `specs/logging-employment-spec-roadmap.md` or `specs/deferred_items.md` in any task.** Ticking Stage 5 and filing deferred items belong to the Plan Completion Protocol, after Task 15. `specs/findings/stage-5-log.md` is appended in Task 15.
- **Commit per task on this branch; never push.** Cite deferred items by `D-nnn`. Never use a bare `git stash`; the stash stack is shared with other sessions.

## File Structure

| File | Task | Responsibility |
|---|---|---|
| `pyproject.toml`, `uv.lock` | 1 | five new dependencies; no locked version of an existing package moves |
| `tests/unit/test_numpyro_api_probe.py`, `tests/unit/test_arviz_api_probe.py` | 1 | the JAX, NumPyro, ArviZ and xarray calls this plan builds on, run once in seconds |
| `src/logging_employment/config.py` | 2, 12 | `StateModelPriors`, `StateModelDiagnostics`, `ModelConfig`, `Config.model`; `PromotionConfig`'s fourth key and, in Task 12, its docstring |
| `config.yaml` | 2 | the `model:` block and `promotion.catastrophic_stratum_coverage_alpha` |
| `tests/unit/test_config_model_block.py`, `tests/unit/test_config.py`, `tests/unit/test_config_validation_block.py`, `tests/integration/test_stage4_acceptance.py` | 2, 12 | the block's refusals; the fence loads whole; both run-id pins; the tripwire |
| `src/logging_employment/models/__init__.py`, `models/interfaces.py` | 3 | §16.2's `ModelData`, `StateModelConfig`, `StateModelFit`; `MODEL_ID`, `MODEL_VERSION`, `STORE_PATH` |
| `src/logging_employment/models/data.py` | 3 | `build_model_data`, `state_cell_ids`: the panel, split into training and prediction cells, fail-closed |
| `src/logging_employment/baselines/runner.py` | 3, 11 | `missing_cell_ids` made public; `release_integers` extracted verbatim |
| `src/logging_employment/models/state_total.py` | 4 | the NumPyro model (Decision 15's), §11.5's score, the one-step check, the fit, `BACKENDS` (§2.2's CmdStanPy slot) |
| `src/logging_employment/models/reconciliation.py` | 5 | `reconcile_fit` (every draw, every month, through `reconcile_draws`), `check_reconciled`, `exact_column_means` |
| `tests/unit/conftest.py` | 5 | `make_state_fit`: a `StateModelFit` from a raw-score array, no sampler |
| `src/logging_employment/contracts.py` | 6, 11 | `POSTERIOR_SUMMARY_SCHEMA`, `OBSERVED_OR_IMPUTED`; `INTERVAL_SOURCES` gains the draws and is enforced |
| `src/logging_employment/models/summary.py` | 6 | §7.11's `posterior_summary` |
| `src/logging_employment/models/arviz_io.py` | 7 | R-hat and ESS; §15.4's store; `draws_digest` |
| `src/logging_employment/errors.py`, `models/diagnostics.py` | 8 | `ModelDiagnosticsError`; §11.14's gate with its two scopes |
| `src/logging_employment/cli.py` | 9, 13 | `fit-state-model`; `reconcile`'s check of the stored draws; `validate-state-model` |
| `tests/integration/conftest.py` | 9 | `build_staged_repo` with config overrides; module-scoped `make_staged_repo` |
| `tests/integration/test_state_total_recovery.py` | 10 | §17.5's state-total recovery rows |
| `src/logging_employment/validate/harness.py`, `validate/metrics.py`, `validate/intervals.py` | 11 | the `Producer` seam; `draw_interval_metrics`; the corrected docstring |
| `src/logging_employment/models/validation.py` | 11 | `StateModelProducer`, `model_results`, `draw_ensembles` |
| `src/logging_employment/validate/promotion.py` | 12 | §13.10's record |
| `CLAUDE.md`, `src/logging_employment/{models,validate,reconcile,baselines}/CLAUDE.md` | 14 | the new package, commands, counts and contracts |
| `specs/findings/stage-5-log.md` | 15 | the re-run's byte comparison, the fit, the verdict |

Test files are listed in each task. Every test this plan adds runs without `data/` except the moved pin in `test_stage4_acceptance.py`.

---

### Task 0: Worktree, shared data, and the pre-plan baseline

**Implements:** Decision 14. No tracked file changes.

**Files:** none tracked. Appends two lines to the common `info/exclude`.

**Interfaces:**
- Consumes: the main checkout's gitignored `data/` (~564 MB) and `runs/`.
- Produces: a worktree on `40b2688` or a descendant, where `data/` and `runs/` resolve to the main checkout's, and a measured baseline every later gate is a delta from.

- [ ] **Step 1: Confirm the base, and that nothing this plan edits has moved**

```bash
git log --oneline -1
git fetch -q origin && git diff --stat 40b2688 origin/main -- src tests config.yaml pyproject.toml uv.lock .github
```

Expected: `40b2688 Merge pull request #35 …` or a descendant, and an empty stat. If `origin/main` has moved inside `src/`, `tests/`, `config.yaml`, `pyproject.toml`, `uv.lock` or `.github/`, fast-forward first. Then check that every diff in this plan still applies to its task's predecessor (`git apply --check` per block) before Task 1. **Stop and report any that do not.**

- [ ] **Step 2: Link the shared data and runs, and keep git blind to the links**

`.gitignore` ignores `data/` and `runs/` with a trailing slash, which matches directories only, and a symlink is not a directory to git. So the links go in the common `info/exclude`, which every worktree of this repo reads.

```bash
MAIN="$(git worktree list --porcelain | awk '/^worktree /{print $2; exit}')"
ln -s "$MAIN/data" data
ln -s "$MAIN/runs" runs
EXCLUDE="$(git rev-parse --git-common-dir)/info/exclude"
grep -qx '/data' "$EXCLUDE" || echo '/data' >> "$EXCLUDE"
grep -qx '/runs' "$EXCLUDE" || echo '/runs' >> "$EXCLUDE"
git status --short
ls data/staged
```

Expected: `git status --short` prints nothing. `ls` lists the five staged tables: `bridge.parquet`, `cbp_state_size.parquet`, `qcew_monthly.parquet`, `qcew_national_size.parquet`, `qcew_state_parent.parquet`.

- [ ] **Step 3: Measure the baseline with the data present**

```bash
uv run pytest -q -p no:cacheprovider
uv run pytest -q -p no:cacheprovider -m "not slow"
uv run python -c "from pathlib import Path; from logging_employment.config import load_config; from logging_employment.runs import run_id; from logging_employment.cli import _input_digests; c = load_config(Path('config.yaml')); print(run_id(c, {}), run_id(c, _input_digests(c)))"
```

Expected, as measured at `40b2688` on this Mac while this plan was written:

- `1570 passed` for the whole suite, in about 12 minutes. Its 27 `slow` tests are D1 integration tests, which run because `data/` is linked.
- `1543 passed, 27 deselected` for the non-slow suite, in about 2.5 minutes.
- `39d1d0859838 4cf47a918dd8`: the config-only canary and the staged pin, before Task 2 moves them.

**If Step 1 found `origin/main` moved, measure again and use your own numbers.** Every later gate is a delta from these, not an absolute.

Every later gate states its count as this Step's non-slow number plus the tests added since. **The delta is the check; a mismatch halts.**

---

### Task 1: Dependencies, and a probe of the PPL calls this plan builds on

**Implements:** Decision 10. It serves §21's PPL row and §2.2's backend row (NumPyro first; CmdStanPy is a slot, `models/state_total.py::BACKENDS`).

This plan was written without JAX, NumPyro, ArviZ or h5netcdf in the project's environment. The NumPyro and ArviZ calls were read in the tagged sources: numpyro 0.22.0 `infer/mcmc.py`, `util.py` and `distributions/continuous.py`; arviz-stats v1.3.3 `sampling_diagnostics.py` and `__init__.py`; arviz-base v1.3.1 `io_dict.py`. They were then run in a second local Python 3.14 environment one release behind each pin (jax 0.11.1, numpyro 0.21.0, arviz-base 1.3.0, arviz-stats 1.3.2, xarray 2026.7.0) that has no h5netcdf. That run found one call the source reading had missed: `ess(..., method="tail")` on an array needs `prob`. These probes pin those calls against the versions the lock actually installs, in seconds, before Task 4 builds on them. The NumPyro probe's toy uses the state-total model's own mechanisms (Decision 15): a scanned AR(1) path, pinned and scored through `numpyro.factor` when data are given, and drawn from unit innovations and recorded with `numpyro.deterministic` when not.

**Files:**
- Modify: `pyproject.toml` (`[project].dependencies`), `uv.lock` (written by `uv add` only)
- Create: `tests/unit/test_numpyro_api_probe.py`, `tests/unit/test_arviz_api_probe.py`

**Interfaces:**
- Consumes: nothing from this plan.
- Produces: an environment where `jax`, `numpyro`, `arviz_base`, `arviz_stats`, `xarray` and `h5netcdf` import, and a record of the exact calls later tasks make:
  - `numpyro.enable_x64()`;
  - `MCMC(NUTS(model, target_accept_prob=..., max_tree_depth=10), num_warmup=..., num_samples=..., num_chains=..., chain_method="vectorized", progress_bar=False)`;
  - `mcmc.run(jax.random.key(seed), ..., extra_fields=("diverging", "num_steps"))`;
  - `mcmc.get_samples(group_by_chain=True)` and `mcmc.get_extra_fields(group_by_chain=True)`;
  - `dist.ZeroSumNormal(scale, event_shape=(k,))` and `Predictive(model, num_samples=n)(key, ...)`;
  - `arviz_stats.rhat(x, method="rank", chain_axis=0, draw_axis=1)`;
  - `arviz_stats.ess(x, method="bulk", ...)` and `arviz_stats.ess(x, method="tail", prob=(0.05, 0.95), ...)`, both with `chain_axis=0, draw_axis=1`;
  - `xr.DataTree.from_dict`, `.to_netcdf(path, engine="h5netcdf")` and `xr.open_datatree(path, engine="h5netcdf")`.

- [ ] **Step 1: Write the two probes**

`tests/unit/test_numpyro_api_probe.py`:

```python
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
```

`tests/unit/test_arviz_api_probe.py`:

```python
"""Every ArviZ and xarray call `models/arviz_io.py` makes, run once against the installed versions.

`arviz_io` is the only module that imports either library, so an API that moved fails here and not
partway through a D1 fit. Plan 16 verified these calls by reading arviz-stats v1.3.3's source; this
module is where they are first run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from arviz_stats import ess, rhat

SEED = sum(map(ord, "tests/arviz-api-probe"))


def _chains(offset: float = 0.0) -> np.ndarray:
    """Four chains of 500 iid normal draws of three elements; chain 3 is shifted by `offset`."""
    values = np.random.default_rng(SEED).normal(size=(4, 500, 3))
    values[3] += offset
    return values


def test_rank_rhat_takes_chain_and_draw_axes_and_sees_a_stuck_chain() -> None:
    mixed = np.asarray(rhat(_chains(), method="rank", chain_axis=0, draw_axis=1))
    stuck = np.asarray(rhat(_chains(3.0), method="rank", chain_axis=0, draw_axis=1))
    assert mixed.shape == (3,)
    assert np.all(mixed < 1.01)
    assert np.all(stuck > 1.1)


def test_bulk_and_tail_ess_take_the_same_axes() -> None:
    """Tail ESS needs its two quantiles spelled out: the array interface has no default for them."""
    bulk = np.asarray(ess(_chains(), method="bulk", chain_axis=0, draw_axis=1))
    tail = np.asarray(ess(_chains(), method="tail", prob=(0.05, 0.95), chain_axis=0, draw_axis=1))
    for values in (bulk, tail):
        assert values.shape == (3,)
        # 2,000 independent draws: anything near the 400 floor would mean the axes were misread.
        assert np.all(values > 1000)


def test_a_datatree_round_trips_through_h5netcdf_bit_for_bit(tmp_path: Path) -> None:
    draws = np.random.default_rng(SEED).normal(size=(2, 5, 3))
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"draws_sha256": "abc"}),
            "posterior_predictive": xr.Dataset(
                {"reconciled_state_total": (("chain", "draw", "cell"), draws)},
                coords={
                    "chain": np.arange(2),
                    "draw": np.arange(5),
                    "cell": ["a", "b", "c"],
                    "state_fips": ("cell", ["01", "02", "04"]),
                },
            ),
        }
    )
    path = tmp_path / "store.nc"
    tree.to_netcdf(path, engine="h5netcdf")
    back = xr.open_datatree(path, engine="h5netcdf")
    try:
        read = back["posterior_predictive"].to_dataset().load()
        assert back.attrs["draws_sha256"] == "abc"
    finally:
        back.close()
    assert read["reconciled_state_total"].dtype == np.float64
    np.testing.assert_array_equal(read["reconciled_state_total"].values, draws)
    assert [str(value) for value in read["state_fips"].values] == ["01", "02", "04"]
```

- [ ] **Step 2: Run them to watch them fail**

Run: `uv run pytest tests/unit/test_numpyro_api_probe.py tests/unit/test_arviz_api_probe.py -q`

Expected (observed with this repo's pre-plan environment):

```text
E   ModuleNotFoundError: No module named 'jax'
E   ModuleNotFoundError: No module named 'xarray'
ERROR tests/unit/test_numpyro_api_probe.py
ERROR tests/unit/test_arviz_api_probe.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!!
2 errors
```

- [ ] **Step 3: Add the five dependencies, keeping the old lock to compare against**

```bash
cp uv.lock /tmp/uv.lock.pre-plan16
uv add 'jax>=0.11.2' 'numpyro>=0.22.0' 'arviz-base>=1.3.1' 'arviz-stats[xarray]>=1.3.3' 'h5netcdf>=1.8.1'
git diff pyproject.toml
```

Expected: `[project].dependencies` gains exactly the five lines above, beside the ten it had. `uv add` places them in the list's sorted order, and their order is not load-bearing. This step needs network access to PyPI. If `uv add` cannot resolve, stop and report the resolver's message. Do not loosen a floor to make it resolve.

- [ ] **Step 4: Prove no locked version of an existing package moved**

A moved numpy, scipy or polars could move a float golden, and this plan re-pins none.

```bash
uv run python - <<'EOF'
import tomllib


def versions(path):
    with open(path, "rb") as handle:
        return {p["name"]: p["version"] for p in tomllib.load(handle)["package"] if "version" in p}


before, after = versions("/tmp/uv.lock.pre-plan16"), versions("uv.lock")
moved = {name: (before[name], after[name]) for name in before if name in after and before[name] != after[name]}
print("added:", sorted(set(after) - set(before)))
print("moved:", moved)
print("removed:", sorted(set(before) - set(after)))
raise SystemExit(1 if moved or set(before) - set(after) else 0)
EOF
```

Expected: `moved: {}` and `removed: []`, exit 0. `added` lists `jax`, `jaxlib`, `numpyro`, `arviz-base`, `arviz-stats`, `xarray`, `h5netcdf`, `h5py` and their own transitive dependencies. **If anything moved, stop and report the pairs.** That is a decision for your human partner, not a pin to override.

- [ ] **Step 5: Run the probes to see them pass**

Run: `uv run pytest tests/unit/test_numpyro_api_probe.py tests/unit/test_arviz_api_probe.py -q`

Expected: `6 passed`. Five of the six were observed passing on the older local environment (`5 passed, 1 deselected`, with the h5netcdf round trip deselected there). The round trip runs for the first time here.

- [ ] **Step 6: Gates**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
uv run pytest -q -p no:cacheprovider -m "not slow"
```

Expected: all three gates clean. Non-slow suite: Task 0's count **+ 6 passed**, skipped unchanged.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock tests/unit/test_numpyro_api_probe.py tests/unit/test_arviz_api_probe.py
git commit -m "build: add JAX, NumPyro, ArviZ and h5netcdf, and probe the calls plan 16 builds on"
```

---

### Task 2: Appendix A's `model:` block, the fourth promotion key, and the one re-id

**Implements:** Decision 1; §11.12 ("Every prior must be exposed in resolved configuration"); Appendix A's `model:` block; and the fence-test rewrite the roadmap's Stage 5 `Consumes` asks for since `D-121`.

Both config additions land in this one task because both feed `resolved_dict`. Split across two tasks, they would move the run id twice. `PromotionConfig`'s docstring gets an interim paragraph for the fourth key here and is rewritten in Task 12, once a module reads the keys. The tripwire stays green in between, because a field declaration is not a read.

**Files:**
- Create: `tests/unit/test_config_model_block.py`
- Modify: `tests/unit/test_config.py`, `tests/unit/test_config_validation_block.py`, `tests/integration/test_stage4_acceptance.py`
- Modify: `src/logging_employment/config.py` (`SourcesConfig`'s docstring, `PromotionConfig`, new `StateModelPriors` / `StateModelDiagnostics` / `ModelConfig`, `Config.model`), `config.yaml`

**Interfaces:**
- Consumes: nothing from this plan.
- Produces:
  - `config.StateModelPriors`: 12 floats, as Decision 7 lists them. There is no observation scale (Decision 15), and `persistence_max` caps ρ;
  - `config.StateModelDiagnostics`: `max_rhat: float = 1.01`, `min_ess_per_chain: int = 100`, `max_divergences: int = 0`, `min_ppc_coverage_90: float = 0.85`;
  - `config.ModelConfig`:
    - `backend: Literal["numpyro"]`;
    - `chains: int >= 2` (4), `warmup` (1000), `draws` (1000), `target_accept` in (0, 1) (0.9);
    - `state_dynamic: Literal["student_t_ar1"]`;
    - three `include_*` switches, each `Literal[False]`;
    - `standardized_beta_sd` (0.5), `suppressed_variance_multipliers` ([1.0, 1.5, 2.0]), `seed` (3645);
    - `priors: StateModelPriors`, `diagnostics: StateModelDiagnostics`;
  - `Config.model: ModelConfig = ModelConfig()`;
  - `PromotionConfig.catastrophic_stratum_coverage_alpha: float = 0.001`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_config_model_block.py` (new; 13 tests):

```python
"""Appendix A's `model:` block (§11) as `ModelConfig`: what it pins, accepts and refuses."""

from __future__ import annotations

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
```

`tests/unit/test_config.py`: the fence now loads whole, and the canary moves. `_appendix_a_made_loadable` and `_error_locations` go, since nothing is left to pop or to list.

```diff
diff --git a/tests/unit/test_config.py b/tests/unit/test_config.py
index dadcd51..6b08f0a 100644
--- a/tests/unit/test_config.py
+++ b/tests/unit/test_config.py
@@ -10,14 +10,14 @@ import pytest
 import yaml
 from pydantic import ValidationError
 
-from logging_employment.config import Config, load_config, resolved_dict
+from logging_employment.config import Config, ModelConfig, load_config, resolved_dict
 from logging_employment.runs import run_id
 
 # NOT Appendix A's fence. Measured against `specs/logging-employment-spec.md`'s `## Appendix A`
 # block, this constant differs from it in four ways, every one of which is this package's doing
 # rather than the spec's:
-#   1. it OMITS `model:` -- Stage 5's block, for which `Config` has no field. Adding one would put
-#      a key in `resolved_dict` and re-identify every run directory under `runs/`.
+#   1. it OMITS `model:`, which `Config` defaults to `ModelConfig()` -- Appendix A's eleven values
+#      plus the three keys plan 16 originated.
 #   2. it OMITS `promotion:` and `validation:`, which `Config` defaults.
 #   3. it ADDS five `reconciliation:` keys this package originated (`tolerance`,
 #      `max_bisection_iterations`, `max_projection_iterations`, `zero_seed_floor`,
@@ -220,6 +220,21 @@ SPEC = REPO_ROOT / "specs" / "logging-employment-spec.md"
 # Appendix A's seven declared-inactive source entries, in the order the spec lists them.
 INACTIVE_SOURCES = ("tpo", "fia", "ces", "susb", "bds", "nonemployer", "bea")
 
+# Appendix A's `model:` keys, as the spec lists them. `ModelConfig` adds three more of its own.
+APPENDIX_A_MODEL_KEYS = (
+    "backend",
+    "chains",
+    "warmup",
+    "draws",
+    "target_accept",
+    "state_dynamic",
+    "include_change_points",
+    "include_harvest_factor",
+    "include_ces",
+    "standardized_beta_sd",
+    "suppressed_variance_multipliers",
+)
+
 
 def appendix_a_fence() -> dict[str, Any]:
     """Appendix A's example configuration, parsed out of the spec file itself.
@@ -243,44 +258,25 @@ def appendix_a_fence() -> dict[str, Any]:
     raise AssertionError("Appendix A carries no closed fenced block")
 
 
-def _appendix_a_made_loadable() -> dict[str, Any]:
-    """Appendix A's fence minus `model:`, the one block `Config` has no field for.
-
-    Popped HERE and not accepted in `config.py`, because a `model:` field would re-identify every
-    run directory for a stage that does not exist yet. Until `D-121` this also patched in
-    `baselines:` and the two `disclosure:` widths, which the spec did not state; Appendix A now
-    carries all three, the widths labelled as the governance owner's policy (§21).
-    """
-    fence = appendix_a_fence()
-    fence.pop("model")
-    return fence
-
-
-def _error_locations(payload: dict[str, Any]) -> list[tuple[str, str]]:
-    """Every `(dotted location, error type)` `Config` reports for `payload`, sorted."""
-    with pytest.raises(ValidationError) as caught:
-        Config.model_validate(payload)
-    return sorted((".".join(str(p) for p in e["loc"]), e["type"]) for e in caught.value.errors())
-
-
-def test_the_spec_fence_loads_but_for_stage_5s_model_block() -> None:
-    """Appendix A's own fence loads, except for the one block `Config` has no field for.
+def test_the_spec_fence_loads_whole() -> None:
+    """Appendix A's own fence loads with no patching, `model:` included.
 
     Before Appendix A's seven `enabled: false` sources were declared on `SourcesConfig`, this
     fence produced ELEVEN errors: those seven, `model`, and three `missing` keys -- `baselines:`
     and the two `disclosure:` widths. Code could fix only the seven; the three were a spec gap,
-    and defaulting them in `config.py` would have invented policy the spec did not state.
-    `D-121` (2026-09-26) closed that gap in the spec instead: Appendix A now carries all three,
-    the widths labelled as the governance owner's policy under §21. What is left is `model:`,
-    Stage 5's block. Adding a field for it would put a key in `resolved_dict` and re-identify
-    every run directory in `runs/`, so the first assertion pins that it is the ONLY error and the
-    second, after popping it, that the rest of the fence loads clean.
+    which `D-121` (2026-09-26) closed in the spec. `model:` was the last: plan 16 gave it
+    `ModelConfig`, accepting the one re-identification of every run directory that a field in
+    `resolved_dict` costs (its Decision 1). Until then this test pinned `model` as the fence's only
+    error, as `test_the_spec_fence_loads_but_for_stage_5s_model_block`.
+
+    The fence's block must be exactly Appendix A's eleven keys at `ModelConfig`'s defaults, so the
+    three keys plan 16 originated (`seed`, `priors`, `diagnostics`) stay recognisable as additions
+    rather than blending into the spec's own.
     """
     fence = appendix_a_fence()
-    assert _error_locations(fence) == [("model", "extra_forbidden")]
-
-    fence.pop("model")
-    Config.model_validate(fence)
+    cfg = Config.model_validate(fence)
+    assert sorted(fence["model"]) == sorted(APPENDIX_A_MODEL_KEYS)
+    assert cfg.model == ModelConfig()
 
 
 def test_the_spec_fence_declares_seven_inactive_sources_and_all_seven_load() -> None:
@@ -289,7 +285,7 @@ def test_the_spec_fence_declares_seven_inactive_sources_and_all_seven_load() ->
     assert sorted(sources) == sorted(("qcew", "qcew_size", "cbp", *INACTIVE_SOURCES))
     assert all(sources[name] == {"enabled": False} for name in INACTIVE_SOURCES)
 
-    cfg = Config.model_validate(_appendix_a_made_loadable())
+    cfg = Config.model_validate(appendix_a_fence())
     assert all(getattr(cfg.sources, name).enabled is False for name in INACTIVE_SOURCES)
     assert cfg.sources.qcew.release_status == "final"
 
@@ -334,13 +330,14 @@ def test_an_inactive_source_reaches_neither_the_resolved_config_nor_the_run_id(
     assert run_id(with_seven, {}) == run_id(without, {})
 
 
-def test_the_shipped_configs_run_id_is_unmoved_by_the_inactive_source_fields() -> None:
+def test_the_shipped_configs_run_id_is_pinned() -> None:
     """A literal pin on the id `config.yaml` derives, because the cost of moving it is external.
 
-    `runs/f03023ac9f3a` is Stage 4's acceptance artifact and is derived from this config plus the
-    staged inputs. Those inputs are gitignored, so the digest map here is empty and the pinned
-    value is not that directory's name -- it is a canary over the same `resolved_dict` input,
-    which is the half of the id a config change can move. Measured before this change and
-    unchanged by it.
+    Run directories are derived from this config plus the staged inputs. Those inputs are
+    gitignored, so the digest map here is empty and the pinned value is no directory's name -- it
+    is a canary over the same `resolved_dict` input, which is the half of the id a config change
+    can move. MOVED ONCE, deliberately, by plan 16's config task: `model:` joined `resolved_dict`
+    (Decision 1). It read `39d1d0859838` from the inactive-source fields (R-S5P-6) until then, and
+    the seven inactive sources still move nothing.
     """
-    assert run_id(load_config(REPO_ROOT / "config.yaml"), {}) == "39d1d0859838"
+    assert run_id(load_config(REPO_ROOT / "config.yaml"), {}) == "14352bb8e56e"
```

`tests/unit/test_config_validation_block.py`: the fourth key joins `PROMOTION_KEYS`, which the unchanged tripwire watches, and gets its own value test (+1).

```diff
diff --git a/tests/unit/test_config_validation_block.py b/tests/unit/test_config_validation_block.py
index 59d5ca1..bb1c4c7 100644
--- a/tests/unit/test_config_validation_block.py
+++ b/tests/unit/test_config_validation_block.py
@@ -34,11 +34,22 @@ def test_promotion_gates_carry_appendix_a_defaults():
     assert p.nominal_coverage_tolerance == 0.05
 
 
+def test_the_catastrophic_coverage_alpha_is_one_in_a_thousand():
+    """Plan 16's origination: §13.10 says "does not fail catastrophically" and states no number.
+
+    At 0.001 over 18 strata (9 regimes, 9 divisions), a calibrated model is flagged somewhere with
+    probability 0.0122 (the plan's Decision 4).
+    """
+    assert PromotionConfig().catastrophic_stratum_coverage_alpha == 0.001
+
+
 PROMOTION_KEYS = frozenset(
     {
         "minimum_wape_improvement",
         "maximum_major_stratum_wape_degradation",
         "nominal_coverage_tolerance",
+        # Plan 16's, and watched by the same tripwire from the day it was declared.
+        "catastrophic_stratum_coverage_alpha",
     }
 )
 
```

`tests/integration/test_stage4_acceptance.py`: the staged pin moves (a `requires_staged` test, so it runs only with `data/` linked):

```diff
diff --git a/tests/integration/test_stage4_acceptance.py b/tests/integration/test_stage4_acceptance.py
index 8344a2d..e05267b 100644
--- a/tests/integration/test_stage4_acceptance.py
+++ b/tests/integration/test_stage4_acceptance.py
@@ -227,9 +227,13 @@ def test_the_shipped_config_resolves_to_the_parent_margin_comparand_run():
     MOVED ONCE, deliberately, by plan 15 Task 4: the fifth staged table (`qcew_state_parent`) and
     `D-114`'s rewrite of `cbp_state_size.parquet` re-id every run. The previous pin, `f03023ac9f3a`,
     is Stage 4's acceptance run; it stays on disk as the comparand plan 15 Task 9 measures against.
+
+    MOVED AGAIN by plan 16's config task: Appendix A's `model:` block joined `resolved_dict`. The
+    pin before it, `4cf47a918dd8`, is the §13.10 comparand. Plan 16 re-runs it under this id,
+    checks the two byte for byte, and keeps the old directory on disk.
     """
     cfg = load_config(REPO / "config.yaml")
-    assert run_id(cfg, _input_digests(cfg)) == "4cf47a918dd8"
+    assert run_id(cfg, _input_digests(cfg)) == "dd7337e89047"
 
 
 def test_no_scoring_regime_gained_or_lost_a_score(fixture_run):
```

- [ ] **Step 2: Run them to watch them fail**

Run: `uv run pytest tests/unit/test_config_model_block.py tests/unit/test_config.py tests/unit/test_config_validation_block.py tests/integration/test_stage4_acceptance.py -q`

Expected (observed):

```text
E   ImportError: cannot import name 'ModelConfig' from 'logging_employment.config' (…/src/logging_employment/config.py)
ERROR tests/unit/test_config_model_block.py
ERROR tests/unit/test_config.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!!
2 errors
```

- [ ] **Step 3: Implement the block**

`src/logging_employment/config.py`:

```diff
diff --git a/src/logging_employment/config.py b/src/logging_employment/config.py
index 8dcd9b0..f4706b6 100644
--- a/src/logging_employment/config.py
+++ b/src/logging_employment/config.py
@@ -109,9 +109,9 @@ class SourcesConfig(_Strict):
     `nonemployer`, `bea` -- carry `enabled: false` and belong to Stages 4-8; before they were
     declared here, `extra="forbid"` made the spec's own reference configuration unloadable, which
     is `R-S5P-6`'s defect (`specs/completed/stage5-preconditions.md`) and what this fixes. `R-S5P-6` scoped
-    only these seven. Appendix A's `model:` block is still left failing on purpose, and
-    `tests/unit/test_config.py` asserts it is the only thing that fails; the fence's three MISSING
-    keys were a spec gap, which `D-121` closed in Appendix A rather than with defaults here.
+    only these seven. The fence's three MISSING keys were a spec gap, which `D-121` closed in
+    Appendix A rather than with defaults here, and its `model:` block loads since plan 16 gave it
+    `ModelConfig`. So Appendix A's fence now loads whole, and `tests/unit/test_config.py` pins that.
 
     They are declared as fields rather than admitted by `extra="allow"`
     so that a MISSPELLED source name is still a load-time error and so that `enabled: true` on an
@@ -287,13 +287,13 @@ class ValidationConfig(_Strict):
 class PromotionConfig(_Strict):
     """§13.10's gates. Configurable engineering thresholds, not findings.
 
-    ALL THREE KEYS ARE INERT TODAY, and this note is the record R-S5G-3 requires rather than a
+    ALL FOUR KEYS ARE INERT TODAY, and this note is the record R-S5G-3 requires rather than a
     disclaimer. Each reaches `config.resolved.yaml` and folds into `runs.run_id`, so an unread key
     is a claim in a run's record that no code backs -- the defect `D-064` names for four other
     keys. The convention there is to write inertness into the code (`constraints/matrix.py`'s "NO
     REAL INPUT UNTIL STAGE 6", `errors.py::NoHarvestFactorError`), which is what this is.
 
-    They are inert for TWO different reasons, and the distinction is the useful half:
+    They are inert for different reasons, and the distinction is the useful half:
 
     - `maximum_major_stratum_wape_degradation` and `nominal_coverage_tolerance` have their INPUT as
       of R-S5G-1. `validate/metrics.py` now emits a per-census-division WAPE and a per-division
@@ -304,6 +304,10 @@ class PromotionConfig(_Strict):
       preferred transparent baseline and Stage 5 produces the model.
     - `minimum_wape_improvement` is missing both. Its comparison needs a second scoreboard, and
       `validation_scoreboard.parquet` exists in one copy -- the baseline one.
+    - `catastrophic_stratum_coverage_alpha` is plan 16's. §13.10's "does not fail catastrophically
+      in any major stratum" states no number, and this is the exact binomial lower-tail level the
+      promotion record plan 16 builds will read. It is declared with the `model:` block, in one
+      commit, so `runs.run_id` moves once rather than twice.
 
     NO EVALUATOR IS BUILT HERE, deliberately. A function whose primary argument is Stage 5's
     not-yet-designed output would fix that signature by guessing it, and three of §13.10's six
@@ -319,6 +323,118 @@ class PromotionConfig(_Strict):
     minimum_wape_improvement: float = 0.05
     maximum_major_stratum_wape_degradation: float = 0.02
     nominal_coverage_tolerance: float = 0.05
+    # Originated by plan 16. §13.10 says "does not fail catastrophically" and gives no number.
+    catastrophic_stratum_coverage_alpha: float = 0.001
+
+
+class StateModelPriors(_Strict):
+    """§11.12's starting priors for the state-total model, each one a configured value.
+
+    §11.12: "Every prior must be exposed in resolved configuration." These are in `resolved_dict`
+    and so in `runs.run_id`, deliberately: a prior changes every draw the model writes. The values
+    are weakly informative on the log employees-per-establishment scale, where D1's observed cells
+    have mean 1.46 and SD 0.48 (measured 2026-09-26), so `intercept_mean` centres the intercept
+    there. Both Student-t degrees of freedom are fixed at 5, §11.12's "fixed near 5 for the first
+    implementation".
+
+    PERSISTENCE IS `persistence_max * Beta(8, 2)`: capped at 0.95, mean 0.76, §11.12's "transformed
+    Beta prior centered near 0.8". §11.2 prefers the AR(1) because Logging intensity is "plausibly
+    mean-reverting". Uncapped, D1's persistence reached 0.978 and a production fit failed §11.14's
+    R-hat on one of two seeds; capped, both passed (plan 16, Decision 15). 1.0 removes the cap.
+
+    THERE IS NO OBSERVATION SCALE. Training cells are exact (§11.3), so §11.3's `sigma_y` and its
+    degrees of freedom are not parameters of the model.
+    """
+
+    intercept_mean: float = 1.5
+    intercept_sd: float = 1.0
+    state_scale_sd: float = 0.5
+    region_scale_sd: float = 0.5
+    month_scale_sd: float = 0.25
+    year_scale_sd: float = 0.25
+    persistence_concentration1: float = 8.0
+    persistence_concentration0: float = 2.0
+    persistence_max: float = Field(default=0.95, gt=0.0, le=1.0)
+    innovation_scale_sd: float = 0.1
+    innovation_dispersion_sd: float = 0.5
+    innovation_df: float = 5.0
+
+
+class StateModelDiagnostics(_Strict):
+    """§11.14's diagnostic gate as thresholds, enforced by `models/diagnostics.py`.
+
+    §11.14 gives R-hat "at or below 1.01" and asks for "adequate effective sample size" without a
+    number. `min_ess_per_chain` fills that gap with the convention of 100 per chain, so the floor is
+    400 at Appendix A's four chains. `min_ppc_coverage_90` is a floor on the share of training cells
+    inside their 90% ONE-STEP-AHEAD predictive interval, predicted from the month before. Training
+    cells are exact (§11.3), so an in-sample replicate of a training cell is y itself and could not
+    fail. It is a floor only: coverage above nominal means the innovation law is wider than the
+    data's month-to-month moves, which widens intervals without understating them, while a fit
+    that misses more than 15% of its training cells' moves understates them. `max_divergences` is 0
+    because §11.14 says "no unresolved divergent transitions".
+    """
+
+    max_rhat: float = 1.01
+    min_ess_per_chain: int = 100
+    max_divergences: int = 0
+    min_ppc_coverage_90: float = 0.85
+
+
+class ModelConfig(_Strict):
+    """Appendix A's `model:` block (§11), plus the keys a fit needs that Appendix A omits.
+
+    IN `resolved_dict`, NOT `exclude=True`. The seven inactive sources are excluded because they
+    contribute no bytes to any stage. Every key here changes the draws `fit-state-model` writes, so
+    `runs.run_id` must see it. Adding the block re-identifies every run directory ONCE. Plan 16
+    re-runs the §13.10 comparand `runs/4cf47a918dd8` under the new id and checks the re-run byte for
+    byte against it (`specs/findings/stage-5-log.md`; the plan's Decision 1).
+
+    Appendix A's eleven keys are declared verbatim, and each default is Appendix A's value. Three
+    are originated here: `seed`, `priors` (§11.12) and `diagnostics` (§11.14).
+
+    THE THREE `include_*` SWITCHES ARE `Literal[False]`, refused at load like an inactive source.
+    Nothing implements them. §11.2's change points "MAY be added only if validation shows material
+    gains". §11.4's harvest factor and §11.11's CES row are Stage 7's. Accepting `true` would
+    record a model component in the run's config that no code fits.
+
+    `backend` admits one value. `models/state_total.py::BACKENDS` is §2.2's CmdStanPy slot: a
+    second backend adds a value here and an entry there together.
+
+    `suppressed_variance_multipliers` is §11.13's sensitivity list. This stage fits the 1.0 model
+    only, and the other values are Stage 7's sensitivity runs, so the validator requires 1.0 to be
+    present and refuses a multiplier below 1.0 or a repeated one. The key is otherwise inert here,
+    and that is recorded rather than hidden: `models/state_total.py` never reads it.
+
+    `seed` is `sum(map(ord, "logging-employment/state-total-model"))`, a descriptive seed rather
+    than a bare constant. It is in the run id because a different seed writes different draws.
+    """
+
+    backend: Literal["numpyro"] = "numpyro"
+    chains: int = Field(default=4, ge=2)
+    warmup: int = Field(default=1000, ge=1)
+    draws: int = Field(default=1000, ge=1)
+    target_accept: float = Field(default=0.9, gt=0.0, lt=1.0)
+    state_dynamic: Literal["student_t_ar1"] = "student_t_ar1"
+    include_change_points: Literal[False] = False
+    include_harvest_factor: Literal[False] = False
+    include_ces: Literal[False] = False
+    standardized_beta_sd: float = Field(default=0.5, gt=0.0)
+    suppressed_variance_multipliers: list[float] = [1.0, 1.5, 2.0]
+    seed: int = 3645
+    priors: StateModelPriors = StateModelPriors()
+    diagnostics: StateModelDiagnostics = StateModelDiagnostics()
+
+    @field_validator("suppressed_variance_multipliers")
+    @classmethod
+    def _the_fitted_multiplier_is_present(cls, value: list[float]) -> list[float]:
+        """Refuse a list without 1.0, with a multiplier below 1.0, or with a repeat (§11.13)."""
+        if 1.0 not in value or min(value) < 1.0 or len(set(value)) != len(value):
+            raise ValueError(
+                f"suppressed_variance_multipliers {value}: §11.13 inflates the suppressed-cell "
+                "process variance, so every multiplier must be >= 1.0, the fitted model's 1.0 must "
+                "be present, and no multiplier may repeat"
+            )
+        return value
 
 
 class Config(_Strict):
@@ -331,6 +447,7 @@ class Config(_Strict):
     reconciliation: ReconciliationConfig
     baselines: BaselinesConfig
     disclosure: DisclosureConfig
+    model: ModelConfig = ModelConfig()
     validation: ValidationConfig = ValidationConfig()
     promotion: PromotionConfig = PromotionConfig()
 
```

`config.yaml`:

```diff
diff --git a/config.yaml b/config.yaml
index 45791f6..bab0217 100644
--- a/config.yaml
+++ b/config.yaml
@@ -9,12 +9,13 @@
 # loads. They stay out of this file because declaring `enabled: false` on a source with no
 # ingest module adds nothing a reader can act on -- and because every one of the seven is
 # `exclude=True`, writing them in here would not move `run_id` either way. Appendix A's
-# `model:` block IS still omitted, because `Config` has no field for it and adding one would
-# re-id every run directory. Its `validation:` and `promotion:` blocks are NOT omitted — both are
-# present below and pin every key (a 2026-09-10 correction; this comment claimed all three were
-# absent, which was already untrue when Stage 4 added them). The constraints/reconciliation/
-# disclosure blocks are present; `reconciliation:` additionally carries five numerical
-# keys Appendix A does not specify (see ReconciliationConfig).
+# `model:` block is present since plan 16 (`ModelConfig`), with its eleven keys verbatim plus
+# three this package originated. Adding it re-identifies every run directory once, so plan 16
+# re-runs Stage 4's comparand under the new id. Its `validation:` and `promotion:` blocks are
+# NOT omitted — both are present below and pin every key (a 2026-09-10 correction; this comment
+# claimed all three were absent, which was already untrue when Stage 4 added them). The
+# constraints/reconciliation/disclosure blocks are present; `reconciliation:` additionally
+# carries five numerical keys Appendix A does not specify (see ReconciliationConfig).
 project:
   name: 'logging-state-employment'
   industry_code_supplied: '1113310'
@@ -104,7 +105,43 @@ validation:
   naics_seam_month: '2022-01'
   naics_seam_halfwidth_months: 3
 
+model:
+  # Appendix A's eleven keys, verbatim.
+  backend: 'numpyro'
+  chains: 4
+  warmup: 1000
+  draws: 1000
+  target_accept: 0.9
+  state_dynamic: 'student_t_ar1'
+  include_change_points: false
+  include_harvest_factor: false
+  include_ces: false
+  standardized_beta_sd: 0.5
+  suppressed_variance_multipliers: [1.0, 1.5, 2.0]
+  # Originated by plan 16, not by Appendix A (see ModelConfig).
+  seed: 3645  # sum(map(ord, "logging-employment/state-total-model"))
+  priors:
+    intercept_mean: 1.5
+    intercept_sd: 1.0
+    state_scale_sd: 0.5
+    region_scale_sd: 0.5
+    month_scale_sd: 0.25
+    year_scale_sd: 0.25
+    persistence_concentration1: 8.0
+    persistence_concentration0: 2.0
+    persistence_max: 0.95
+    innovation_scale_sd: 0.1
+    innovation_dispersion_sd: 0.5
+    innovation_df: 5.0
+  diagnostics:
+    max_rhat: 1.01
+    min_ess_per_chain: 100
+    max_divergences: 0
+    min_ppc_coverage_90: 0.85
+
 promotion:
   minimum_wape_improvement: 0.05
   maximum_major_stratum_wape_degradation: 0.02
   nominal_coverage_tolerance: 0.05
+  # Originated by plan 16: §13.10's "does not fail catastrophically" states no threshold.
+  catastrophic_stratum_coverage_alpha: 0.001
```

- [ ] **Step 4: Run the tests to see them pass**

Run: the Step 2 command.

Expected: every test passes. Observed without `data/`: `53 passed, 2 skipped`. With `data/` linked, the moved pin in `test_stage4_acceptance.py` runs too.

- [ ] **Step 5: Read both ids back, the canary and the pin**

```bash
uv run python -c "from pathlib import Path; from logging_employment.config import load_config; from logging_employment.runs import run_id; from logging_employment.cli import _input_digests; c = load_config(Path('config.yaml')); print(run_id(c, {}), run_id(c, _input_digests(c)))"
uv run logging-estimates validate-config --config config.yaml
```

Expected: `14352bb8e56e dd7337e89047`, then `validate-config` exits 0. If the second id is not `dd7337e89047`, `data/staged` is not the five tables Task 0 listed. **Stop**: Task 15's byte comparison assumes exactly this re-id and no other.

- [ ] **Step 6: Gates**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
uv run pytest -q -p no:cacheprovider -m "not slow"
```

Expected: gates clean. Non-slow suite: Task 1's count **+ 14 passed** (13 new here, 1 new in `test_config_validation_block.py`; `test_config.py` renames two and adds none).

- [ ] **Step 7: Commit**

```bash
git add config.yaml src/logging_employment/config.py tests/unit/test_config_model_block.py tests/unit/test_config.py tests/unit/test_config_validation_block.py tests/integration/test_stage4_acceptance.py
git commit -m "feat(config): declare Appendix A's model block and re-id every run once (plan 16)"
```

---

### Task 3: `ModelData`: the panel the model reads, and nothing it may not

**Implements:**
- §11.1's panel;
- §11.3's "Suppressed cells have no pseudo-observation", held by the type: a prediction cell has no response array to write into;
- §11.11's QCEW rows: disclosed final QCEW is the exact target observation, preliminary QCEW is
  kept out of a final fit, and suppressed QCEW gets no pseudo-value;
- §16.2's `ModelData` and `StateModelConfig`, with `PosteriorDraws` imported, never redeclared;
- Decision 7's exposure, seam and true-zero choices.

`baselines/runner.py::_cell_ids` becomes the public `missing_cell_ids(partition)`. Its unused `anchor` argument goes, and so does the second copy of the seven-field id rule this task would otherwise need.

**Files:**
- Create: `src/logging_employment/models/__init__.py`, `src/logging_employment/models/interfaces.py`, `src/logging_employment/models/data.py`
- Modify: `src/logging_employment/baselines/runner.py` (`_cell_ids` → `missing_cell_ids`, its one call site, one import)
- Test: `tests/unit/test_model_data.py` (new; 14 tests)

**Interfaces:**
- Consumes: `constraints.cells.{cell_id, KIND_STATE_TOTAL, TOTAL_SIZE_CLASS}`, `validate.regimes.DIVISION_OF`, `reconcile.draws.PosteriorDraws`, Task 2's `ModelConfig` / `StateModelPriors`.
- Produces:
  - `models.interfaces.ModelData`, a frozen dataclass:
    - `states`, `months`, `divisions`, `years`, all `tuple[str, ...]`;
    - int64 arrays `state_division`, `month_of_year`, `year_of_month`, `train_state`, `train_month`, `predict_state`, `predict_month`;
    - float arrays `train_y` (log E/A), `train_x`, `predict_exposure`, `predict_x`;
    - `predict_cell_ids: tuple[str, ...]` (seven-field `cell_id`s, in (month, state) order);
    - `log_exposure_centre`, `log_exposure_scale`.
  - `models.interfaces.StateModelConfig.from_config(model: ModelConfig)`, a frozen dataclass of the fit's settings.
  - `models.interfaces.StateModelFit`: `raw_scores: PosteriorDraws`, `parameters: dict[str, np.ndarray]` shaped `(chains, draws, ...)`, `diverging: np.ndarray` `(chains, draws)`, `ppc_coverage_90: float`, `ppc_cells: int`, `posterior_medians: dict[str, float]` (σ_η and ρ, Decision 15), `sampler: dict[str, object]`, and the properties `chains`, `divergences`.
  - The constants `MODEL_ID = "state_total_model"`, `MODEL_VERSION = "student_t_ar1.1"`, `STORE_PATH = "posterior/state_total_draws.nc"`.
  - `models.data.build_model_data(monthly: pl.DataFrame) -> ModelData`, `models.data.state_cell_ids(frame) -> tuple[str, ...]`, and `TRAINING_STATUS = "observed"` / `PREDICTION_STATUS = "suppressed"`.
  - `baselines.runner.missing_cell_ids(partition: Partition) -> dict[str, str]`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_model_data.py`:

```python
"""`models/data.py::build_model_data`: which cells train, which predict, and what is refused."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from logging_employment.baselines.runner import missing_cell_ids
from logging_employment.errors import ConceptViolationError
from logging_employment.models.data import build_model_data
from logging_employment.reconcile.anchor import observed_partition


def test_observed_cells_train_and_suppressed_cells_predict(harmonized_toy) -> None:
    """The toy's months: '04' suppressed in both, '06' in February; everything else observed."""
    data = build_model_data(harmonized_toy.qcew_monthly)
    assert data.states == ("01", "02", "04", "06")
    assert data.months == ("2023-01", "2023-02")
    assert data.train_y.shape == (5,)
    assert [
        (data.states[s], data.months[t])
        for s, t in zip(data.predict_state, data.predict_month, strict=True)
    ] == [("04", "2023-01"), ("04", "2023-02"), ("06", "2023-02")]


def test_the_response_is_log_employees_per_establishment(harmonized_toy) -> None:
    data = build_model_data(harmonized_toy.qcew_monthly)
    observed = harmonized_toy.qcew_monthly.filter(
        (pl.col("area_type") == "state") & (pl.col("observation_status") == "observed")
    ).sort("reference_month", "state_fips")
    expected = np.log(
        observed["employment_value"].to_numpy() / observed["qtrly_establishments"].to_numpy()
    )
    np.testing.assert_allclose(data.train_y, expected, rtol=1e-15, atol=0)


def test_a_suppressed_cell_is_never_a_response(harmonized_toy) -> None:
    """§11.3: no pseudo-observation. The two sets share no (state, month)."""
    data = build_model_data(harmonized_toy.qcew_monthly)
    train = set(zip(data.train_state.tolist(), data.train_month.tolist(), strict=True))
    predict = set(zip(data.predict_state.tolist(), data.predict_month.tolist(), strict=True))
    assert not train & predict
    assert data.predict_exposure.shape == (len(predict),)


def test_prediction_cells_carry_the_runners_seven_field_ids(harmonized_toy) -> None:
    """The join key a fit's draws, `deterministic_bounds` and `baseline_results` all share."""
    monthly = harmonized_toy.qcew_monthly
    data = build_model_data(monthly)
    expected: list[str] = []
    for _month, partition in sorted(observed_partition(monthly).items()):
        ids = missing_cell_ids(partition)
        expected.extend(ids[state] for state in sorted(ids))
    assert list(data.predict_cell_ids) == expected


def test_a_true_zero_cell_is_in_neither_set(make_monthly) -> None:
    monthly = make_monthly(
        {"state_fips": "01", "area_fips": "01000"},
        {
            "state_fips": "02",
            "area_fips": "02000",
            "observation_status": "true_zero",
            "employment_value": 0,
            "qtrly_establishments": 0,
            "is_true_zero": True,
        },
        {
            "state_fips": "04",
            "area_fips": "04000",
            "observation_status": "suppressed",
            "employment_value": None,
        },
    )
    data = build_model_data(monthly)
    assert data.train_y.shape == (1,)
    assert [data.states[s] for s in data.predict_state] == ["04"]
    # Still a state of the grid: its latent path runs through the month it published a zero.
    assert "02" in data.states


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (
            {"observation_status": "suppressed", "employment_value": 40},
            "pseudo-observation",
        ),
        (
            {
                "observation_status": "suppressed",
                "employment_value": None,
                "qtrly_establishments": 0,
            },
            "no establishments",
        ),
        ({"release_status": "preliminary"}, "final-vintage"),
        ({"employment_value": 0}, "no positive employment"),
        ({"industry_code": "113"}, "industry_code"),
        ({"state_fips": "72", "area_fips": "72000"}, "Census division"),
    ],
)
def test_a_cell_the_model_cannot_read_is_refused_by_name(make_monthly, row, reason) -> None:
    monthly = make_monthly(
        {"state_fips": "01", "area_fips": "01000"},
        {"state_fips": "04", "area_fips": "04000", **row},
    )
    with pytest.raises(ConceptViolationError, match=reason):
        build_model_data(monthly)


def test_calendar_indexes_follow_the_month_string(make_monthly) -> None:
    monthly = make_monthly(
        {
            "reference_month": "2021-12",
            "reference_quarter": "2021Q4",
            "naics_vintage": "NAICS 2017",
        },
        {"reference_month": "2022-01", "reference_quarter": "2022Q1"},
    )
    data = build_model_data(monthly)
    assert data.months == ("2021-12", "2022-01")
    assert data.month_of_year.tolist() == [11, 0]
    assert data.years == ("2021", "2022")
    assert data.year_of_month.tolist() == [0, 1]


def test_exposure_is_standardized_over_both_sets(harmonized_toy) -> None:
    data = build_model_data(harmonized_toy.qcew_monthly)
    x = np.concatenate([data.train_x, data.predict_x])
    assert abs(float(np.mean(x))) < 1e-12
    assert abs(float(np.std(x)) - 1.0) < 1e-12


def test_row_order_does_not_move_a_single_array(harmonized_toy) -> None:
    monthly = harmonized_toy.qcew_monthly
    reversed_rows = build_model_data(monthly.reverse())
    data = build_model_data(monthly)
    for name in ("train_state", "train_month", "train_y", "predict_state", "predict_exposure"):
        np.testing.assert_array_equal(getattr(reversed_rows, name), getattr(data, name))
    assert reversed_rows.predict_cell_ids == data.predict_cell_ids
```

- [ ] **Step 2: Run them to watch them fail**

Run: `uv run pytest tests/unit/test_model_data.py -q`

Expected (observed):

```text
E   ImportError: cannot import name 'missing_cell_ids' from 'logging_employment.baselines.runner' (…/src/logging_employment/baselines/runner.py)
ERROR tests/unit/test_model_data.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error
```

- [ ] **Step 3: Make the runner's identifier public**

```diff
diff --git a/src/logging_employment/baselines/runner.py b/src/logging_employment/baselines/runner.py
index 6a6ba81..16390f3 100644
--- a/src/logging_employment/baselines/runner.py
+++ b/src/logging_employment/baselines/runner.py
@@ -51,6 +51,7 @@ from ..errors import (
 )
 from ..reconcile.allocate import allocate
 from ..reconcile.anchor import (
+    Partition,
     assert_universe_closes,
     closure_audit,
     national_residual,
@@ -172,9 +173,12 @@ def resolve_estimators(declared: Sequence[str] | None) -> tuple[Estimator, ...]:
     return tuple(e for e in REGISTRY if e.estimator_id in chosen)
 
 
-def _cell_ids(partition, anchor) -> dict[str, str]:
+def missing_cell_ids(partition: Partition) -> dict[str, str]:
     """The shipped seven-field `cell_id` for each missing cell, keyed by state.
 
+    Public since plan 16: `models/reconciliation.py` keys a fit's draws by the same identifier, and
+    a second copy of this function would be a second place for the seven fields to drift.
+
     `anchor.missing_cells` carries bare `state_fips`, but Stage 2's identifier is
     `kind|state|month|ownership|industry|naics_vintage|size_class`. Building a shorter string here
     would produce a `baseline_results.cell_id` that never joins `target_cell`,
@@ -415,7 +419,7 @@ def run_baselines(
         anchor = national_residual(data.qcew_monthly, partitions[month], reference_month=month)
         if not anchor.missing_cells:
             continue
-        ids = _cell_ids(partitions[month], anchor)
+        ids = missing_cell_ids(partitions[month])
         for estimator in estimators:
             try:
                 outcome = estimator.weights(context, anchor)
```

- [ ] **Step 4: Write the package, its interfaces and the panel builder**

`src/logging_employment/models/__init__.py`. The docstring names modules later tasks add. It states the package's import rule, and later tasks keep to it.

```python
"""§11's Bayesian models. Stage 5 ships the state-total model; Stage 6 adds the size model.

NOTHING HERE IMPORTS JAX AT PACKAGE IMPORT. `interfaces`, `data`, `reconciliation` and `summary`
are plain NumPy and polars, so the CLI, the harness and the promotion gate can import them without
initialising a JAX backend (§16.2: "PPL-specific objects must remain behind model interfaces").
`state_total` is the only module that imports NumPyro and JAX, and `arviz_io` the only one that
imports ArviZ and xarray. `diagnostics` reaches ArviZ through `arviz_io`, and `validation` reaches
both through the fit, so `cli.py` imports each of those inside the command that needs it.
"""
```

`src/logging_employment/models/interfaces.py`:

```python
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
```

`src/logging_employment/models/data.py`:

```python
"""`ModelData` from the harmonized QCEW panel (§11.1, §11.3, and §11.11's QCEW rows).

It reads `observation_status` and nothing else to decide which cells may carry a response. So the
pseudo-suppression mask (`validate.mask.apply_mask`, which rewrites a hidden cell to `suppressed`
with a null value) moves a cell from training to prediction without this module knowing a mask
exists. That is how the §13 harness fits the model on exactly what a real suppression would leave
public.

ONE LATENT PATH CROSSES THE 2022-01 NAICS SEAM, deliberately. §10.3's share baselines refuse to
cross it (`BaselinesConfig.historical_may_cross_naics_vintage`), because a share is a ratio of two
classification periods. Here the seam is one month on a continuous state series.
`harmonize/naics.py::assert_113310_survives_the_window` establishes that 113310 pairs one-to-one,
under the same title, across the 2017 and 2022 vintages, so the two vintages are not incompatible
in INV-007's sense. A level shift at the seam is absorbed by the year effects. Each cell keeps its
own `naics_vintage` inside its `cell_id`, so no value is relabelled from one vintage to the other.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..constraints.cells import KIND_STATE_TOTAL, TOTAL_SIZE_CLASS, cell_id
from ..errors import ConceptViolationError
from ..validate.regimes import DIVISION_OF
from .interfaces import ModelData

TRAINING_STATUS = "observed"
PREDICTION_STATUS = "suppressed"
FINAL = "final"


def _refuse(frame: pl.DataFrame, reason: str) -> None:
    """Raise naming the first cells of `frame`, when it has any."""
    if frame.height:
        cells = frame.select("state_fips", "reference_month").rows()[:5]
        raise ConceptViolationError(f"{frame.height} cell(s) {reason}; first {cells}")


def state_cell_ids(frame: pl.DataFrame) -> tuple[str, ...]:
    """The seven-field `cell_id` of every row, read from the row's own published fields.

    Public because `models/summary.py` keys §7.11's rows the same way. The fields are read off each
    row rather than hardcoded, as `baselines/runner.py::missing_cell_ids` does, so a vintage change
    follows the data.
    """
    return tuple(
        cell_id(
            KIND_STATE_TOTAL,
            state_fips=str(row["state_fips"]),
            reference_month=str(row["reference_month"]),
            ownership_code=str(row["ownership_code"]),
            industry_code=str(row["industry_code"]),
            naics_vintage=str(row["naics_vintage"]),
            size_class=TOTAL_SIZE_CLASS,
        )
        for row in frame.iter_rows(named=True)
    )


def _indices(frame: pl.DataFrame, column: str, lookup: dict[str, int]) -> np.ndarray:
    """`frame[column]` as positions in `lookup`, the integer index the model gathers by."""
    return np.array([lookup[str(v)] for v in frame[column].to_list()], dtype=np.int64)


def build_model_data(monthly: pl.DataFrame) -> ModelData:
    """The state-total panel of `monthly`, split into training and prediction cells.

    FAIL-CLOSED REFUSALS, each naming the offending cells:

    * a training or prediction cell with `qtrly_establishments <= 0`. §11.5's score is
      `A * exp(mu)`, and at `A = 0` that is zero, which `reconcile_draws` floors to a 1e-12 weight
      rather than refusing. Measured on D1 on 2026-09-26: no such cell (observed A >= 3,
      suppressed A >= 1).
    * a training cell whose `employment_value` is null or `<= 0`, where log(E / A) is undefined.
    * a suppressed cell carrying a non-null `employment_value`. That is the pseudo-observation §11.3
      forbids, or a mask that failed to clear the value.
    * a training row whose `release_status` is not `final`. §11.11 says preliminary QCEW is a
      "separate as-of observation; do not mix with final hard controls". D1 is final throughout.
    * more than one `industry_code` or `ownership_code` among the state rows, because one fit is one
      estimand (§3).
    * a state outside `validate/regimes.py::DIVISION_OF`, which carries §17.5's region effect.

    Cells are ordered by (month, state) in both sets, so the same frame always gives the same
    arrays.
    """
    states_frame = monthly.filter(pl.col("area_type") == "state")
    for column in ("industry_code", "ownership_code"):
        values = sorted(str(v) for v in states_frame[column].unique().to_list())
        if len(values) != 1:
            raise ConceptViolationError(
                f"the state panel carries {len(values)} {column} values {values}; one fit is "
                "one estimand (§3)"
            )
    states = tuple(sorted(str(s) for s in states_frame["state_fips"].unique().to_list()))
    unmapped = [state for state in states if state not in DIVISION_OF]
    if unmapped:
        raise ConceptViolationError(
            f"state_fips {unmapped} fall in no Census division; §17.5's region effect partitions "
            "on validate/regimes.py::CENSUS_DIVISIONS"
        )
    months = tuple(sorted(str(m) for m in monthly["reference_month"].unique().to_list()))
    divisions = tuple(sorted({DIVISION_OF[state] for state in states}))
    years = tuple(sorted({month[:4] for month in months}))

    order = ["reference_month", "state_fips"]
    train = states_frame.filter(pl.col("observation_status") == TRAINING_STATUS).sort(order)
    predict = states_frame.filter(pl.col("observation_status") == PREDICTION_STATUS).sort(order)

    _refuse(
        train.filter(pl.col("release_status") != FINAL),
        "are not final-vintage, and §11.11 keeps preliminary QCEW out of a final fit",
    )
    _refuse(
        train.filter(pl.col("employment_value").is_null() | (pl.col("employment_value") <= 0)),
        "are observed with no positive employment, so log(E / A) is undefined",
    )
    _refuse(
        predict.filter(pl.col("employment_value").is_not_null()),
        "are suppressed yet carry a value, which §11.3 forbids as a pseudo-observation",
    )
    for name, frame in (("training", train), ("prediction", predict)):
        _refuse(
            frame.filter(
                pl.col("qtrly_establishments").is_null() | (pl.col("qtrly_establishments") <= 0)
            ),
            f"in the {name} set have no establishments, so §11.5's A * exp(mu) is zero there",
        )

    state_index = {state: index for index, state in enumerate(states)}
    month_index = {month: index for index, month in enumerate(months)}
    train_exposure = train["qtrly_establishments"].cast(pl.Float64).to_numpy()
    predict_exposure = predict["qtrly_establishments"].cast(pl.Float64).to_numpy()
    log_exposure = np.log(np.concatenate([train_exposure, predict_exposure]))
    centre = float(np.mean(log_exposure)) if log_exposure.size else 0.0
    spread = float(np.std(log_exposure)) if log_exposure.size else 0.0
    # A constant exposure carries no information for beta; a scale of 1.0 leaves x at zero rather
    # than dividing by zero, and beta's posterior is then its prior, which is the truth of it.
    scale = spread if spread > 0.0 else 1.0

    return ModelData(
        states=states,
        months=months,
        divisions=divisions,
        years=years,
        state_division=np.array(
            [divisions.index(DIVISION_OF[state]) for state in states], dtype=np.int64
        ),
        month_of_year=np.array([int(month[5:7]) - 1 for month in months], dtype=np.int64),
        year_of_month=np.array([years.index(month[:4]) for month in months], dtype=np.int64),
        train_state=_indices(train, "state_fips", state_index),
        train_month=_indices(train, "reference_month", month_index),
        train_y=np.log(train["employment_value"].cast(pl.Float64).to_numpy() / train_exposure),
        train_x=(np.log(train_exposure) - centre) / scale,
        predict_state=_indices(predict, "state_fips", state_index),
        predict_month=_indices(predict, "reference_month", month_index),
        predict_exposure=predict_exposure,
        predict_x=(np.log(predict_exposure) - centre) / scale,
        predict_cell_ids=state_cell_ids(predict),
        log_exposure_centre=centre,
        log_exposure_scale=scale,
    )
```

- [ ] **Step 5: Run the tests to see them pass, and the runner's own tests unchanged**

Run: `uv run pytest tests/unit/test_model_data.py tests/unit/test_baselines_bounds.py tests/integration/test_baseline_golden.py -q`

Expected: all pass. `test_model_data.py` observed: `14 passed`. The golden proves the rename moved no row of `baseline_results`.

- [ ] **Step 6: Gates**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
uv run pytest -q -p no:cacheprovider -m "not slow"
```

Expected: gates clean. Non-slow suite: Task 2's count **+ 14 passed**.

- [ ] **Step 7: Commit**

```bash
git add src/logging_employment/models/__init__.py src/logging_employment/models/interfaces.py src/logging_employment/models/data.py src/logging_employment/baselines/runner.py tests/unit/test_model_data.py
git commit -m "feat(models): ModelData and the fit's interfaces, with suppressed cells unable to train (§11.3)"
```

---

### Task 4: The robust hierarchical state-intensity model, in NumPyro

**Implements:** Decision 15's model. Four of its choices read the spec differently from its text, and Decision 7 flags each.
- §11.1's mean, less H (Decision 7), with X as a within-state and a between-state part (Decision 15);
- §11.2's AR(1) with Student-t innovations and state-specific persistence and scale, persistence capped at `persistence_max` (Decision 15);
- §11.3's first sentence: training cells are exact. The path is pinned there, and the likelihood is the AR(1) density of the innovations the pinning implies. There is no σ_y (Decision 15);
- §11.5's score, literally (Decision 6);
- §11.12's priors, all from config;
- §11.14's "vectorized state/time operations", and its non-centred SHOULD for every state no training cell pins. A trained state samples its level and log scale directly (Decision 15);
- §11.14's posterior predictive check, as the one-step-ahead check an exact-observation model admits (Decision 5);
- §2.2's backend row, as `BACKENDS`;
- §16.2's `fit_state_total_model`.

The toy tests exercise determinism, shape, the likelihood's scope, the likelihood's VALUE against SciPy, the cap and the priors. They are part of the hermetic tier: two chains of 40 draws on a twelve-cell panel cost compile time, about 13 s for the module when observed. Recovery is Task 10's.

**Files:**
- Create: `src/logging_employment/models/state_total.py`
- Test: `tests/unit/test_state_total_model.py` (new; 13 tests)

**Interfaces:**
- Consumes: Task 3's `ModelData`, `StateModelConfig`, `StateModelFit`, `MODEL_ID`, `MODEL_VERSION`; `reconcile.draws.PosteriorDraws`.
- Produces:
  - `fit_state_total_model(data: ModelData, config: StateModelConfig) -> StateModelFit`;
  - `state_total_model(data, config, y=None)`, the NumPyro program;
  - `raw_scores(exposure, mu) -> np.ndarray`, §11.5's q, shaped (draws, cells);
  - `training_grid(data) -> np.ndarray`, the states × months mask of pinned cells;
  - `state_mean_exposure(data) -> np.ndarray`, x̄ per state, over training and prediction cells;
  - `one_step_scale(rho, scale, train_state, train_month) -> np.ndarray`, each training cell's one-step predictive scale per draw;
  - `prior_predictive(data, config, *, num_samples) -> np.ndarray`;
  - `MONITORED`, the 15 parameter names whose R-hat §11.14 gates;
  - `BACKENDS: dict[str, Callable]`, `CHAIN_METHOD = "vectorized"`, `MAX_TREE_DEPTH = 10`.
  - Draws are chain-major: row `c * draws + d` is chain c, draw d. `fit.sampler` records the backend, the chain method, the sizes, the seed, the model and library versions, and since the D1 profile (Decision 15) `mean_leapfrog_steps` and `tree_depth_saturation_share`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_state_total_model.py`:

```python
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
```

- [ ] **Step 2: Run them to watch them fail**

Run: `uv run pytest tests/unit/test_state_total_model.py -q`

Expected (observed):

```text
E   ModuleNotFoundError: No module named 'logging_employment.models.state_total'
ERROR tests/unit/test_state_total_model.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error
```

- [ ] **Step 3: Write the model**

`src/logging_employment/models/state_total.py`:

```python
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
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/unit/test_state_total_model.py -q`

Expected: `13 passed`, observed on the older local environment as `13 passed`. If `test_the_same_seed_gives_bit_identical_draws` fails, something reached the model unseeded. Find it; do not loosen the test to `allclose`.

- [ ] **Step 5: Gates**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
uv run pytest -q -p no:cacheprovider -m "not slow"
```

Expected: gates clean. Non-slow suite: Task 3's count **+ 13 passed**.

- [ ] **Step 6: Commit**

```bash
git add src/logging_employment/models/state_total.py tests/unit/test_state_total_model.py
git commit -m "feat(models): the Student-t AR(1) state-intensity model and its literal §11.5 score"
```

---

### Task 5: Every draw reconciled, month by month, through Stage 3's one entry point

**Implements:**
- §12.7 ("Posterior summaries must be computed from reconciled joint draws", and "joint draws MUST remain available");
- INV-012 measured on the draws themselves;
- INV-013 (the draw axis is never reduced);
- REQ-018 (every draw);
- Decision 2's anchor path;
- Decision 11's exact means.

`reconcile/draws.py::reconcile_draws` takes one month, keyed by bare `state_fips`. A fit scores every suppressed cell of the window. So `reconcile_fit` does four things. It runs the closure gate over the whole window, including `D-123`'s identity check. It builds each month's anchor with `national_residual`, over the frame's own partition. It orders the fit's columns as that anchor orders its missing set, and projects the seven-field bounds with `runner.month_bounds`. Then it calls `reconcile_draws`. `check_reconciled` sums each draw with `math.fsum`, so the check does not depend on NumPy's reduction order. That order once put a D1 drift at 9.93e-10 against 1e-9.

**Files:**
- Create: `src/logging_employment/models/reconciliation.py`
- Modify: `tests/unit/conftest.py` (a `make_state_fit` fixture: a `StateModelFit` built from a raw-score array, with no sampler behind it)
- Test: `tests/unit/test_model_reconciliation.py` (new; 8 tests)

**Interfaces:**
- Consumes:
  - `reconcile.anchor.{observed_partition, closure_audit, assert_universe_closes, national_residual, Anchor}`;
  - `reconcile.draws.{reconcile_draws, ReconciliationInputs, PosteriorDraws}`, `reconcile.scaling.Bounds`;
  - `baselines.runner.{missing_cell_ids, month_bounds, assert_bounds_cover}`;
  - Task 3's `StateModelFit`, `MODEL_ID`, `state_cell_ids`.
- Produces:
  - `reconcile_fit(fit, monthly, bounds: Bounds, config) -> ReconciledDraws`. `bounds` is keyed by the seven-field `cell_id`, as `runner.state_total_bounds` returns them.
  - `ReconciledDraws`:
    - `cell_ids`, `state_fips`, `reference_months`, all tuples in the fit's cell order;
    - `values` (draws, cells);
    - `chain`, `draw`;
    - `raw_mean` (cells,);
    - `lower`, `upper` (cells,), where `upper` is +inf for a null `selected_upper`;
    - `anchors: dict[month, Anchor]`.
  - `check_reconciled(draws, *, tolerance) -> DrawCheck`, where `DrawCheck` carries `draws_checked`, `months_checked`, `cells_checked`, `max_anchor_drift`, `bound_violations`, `tolerance` and `passed`.
  - `exact_column_means(values) -> np.ndarray`.
  - The fixture `make_state_fit(values, cell_ids, *, chains=2, parameters=None, diverging=None, ppc_coverage_90=0.9) -> StateModelFit`.

- [ ] **Step 1: Write the fixture and the failing tests**

`tests/unit/conftest.py`:

```diff
diff --git a/tests/unit/conftest.py b/tests/unit/conftest.py
index 21514e0..a6d47ea 100644
--- a/tests/unit/conftest.py
+++ b/tests/unit/conftest.py
@@ -9,6 +9,7 @@ from __future__ import annotations
 
 from collections.abc import Callable
 
+import numpy as np
 import polars as pl
 import pytest
 
@@ -19,6 +20,8 @@ from logging_employment.contracts import (
     QCEW_NATIONAL_SIZE_SCHEMA,
     HarmonizedData,
 )
+from logging_employment.models.interfaces import StateModelFit
+from logging_employment.reconcile.draws import PosteriorDraws
 
 _MONTHLY_DEFAULTS: dict[str, object] = {
     "snapshot_id": "2024q1",
@@ -219,3 +222,43 @@ def harmonized_toy(make_monthly) -> HarmonizedData:
         cbp_state_size=cbp,
         bridge=pl.DataFrame([], schema=BRIDGE_SCHEMA),
     )
+
+
+@pytest.fixture()
+def make_state_fit() -> Callable[..., StateModelFit]:
+    """Build a `StateModelFit` from raw-score draws alone, with no sampler behind it.
+
+    The fit's consumers -- reconciliation, the gate, the summary, the store -- read arrays, not a
+    sampler, so their tests need not pay for MCMC or import JAX. Draws are chain-major, as
+    `models/state_total.py` lays them out: `values[c * per_chain + d]` is chain c, draw d.
+    """
+
+    def _build(
+        values: np.ndarray,
+        cell_ids: tuple[str, ...],
+        *,
+        chains: int = 2,
+        parameters: dict[str, np.ndarray] | None = None,
+        diverging: np.ndarray | None = None,
+        ppc_coverage_90: float = 0.9,
+    ) -> StateModelFit:
+        values = np.asarray(values, dtype=np.float64)
+        per_chain = values.shape[0] // chains
+        return StateModelFit(
+            raw_scores=PosteriorDraws(
+                cell_ids=tuple(cell_ids),
+                values=values,
+                chain=np.repeat(np.arange(chains), per_chain),
+                draw=np.tile(np.arange(per_chain), chains),
+            ),
+            parameters={} if parameters is None else parameters,
+            diverging=(
+                np.zeros((chains, per_chain), dtype=bool) if diverging is None else diverging
+            ),
+            ppc_coverage_90=ppc_coverage_90,
+            ppc_cells=0,
+            posterior_medians={"sigma_eta": 0.1, "rho": 0.8},
+            sampler={"chain_method": "vectorized"},
+        )
+
+    return _build
```

`tests/unit/test_model_reconciliation.py`:

```python
"""`models/reconciliation.py`: every draw reconciled month by month, and INV-012 measured on draws.

The toy panel's missing sets are '04' in January (residual 50) and '04' and '06' in February
(residual 80), so a January draw has one cell to hold the whole residual and a February draw two.
No sampler runs here: the fits are raw-score arrays (`make_state_fit`).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from logging_employment.errors import ConceptViolationError, WeightDomainError
from logging_employment.models.data import state_cell_ids
from logging_employment.models.reconciliation import (
    check_reconciled,
    exact_column_means,
    reconcile_fit,
)
from logging_employment.reconcile.scaling import Bounds

SEED = sum(map(ord, "tests/model-reconciliation"))
TOLERANCE = 1e-9


def _predicted(harmonized_toy) -> tuple[str, ...]:
    """Suppressed toy cells as seven-field ids, in `build_model_data`'s (month, state) order."""
    monthly = harmonized_toy.qcew_monthly
    suppressed = monthly.filter(
        (pl.col("area_type") == "state") & (pl.col("observation_status") == "suppressed")
    ).sort("reference_month", "state_fips")
    return state_cell_ids(suppressed)


def _bounds(cells: tuple[str, ...], upper: dict[str, float] | None = None) -> Bounds:
    """Nonnegativity for every cell, and a finite upper only where `upper` names one."""
    return Bounds(
        lower=dict.fromkeys(cells, 0.0),
        upper={cell: (upper or {}).get(cell) for cell in cells},
    )


def _raw(draws: int, cells: int) -> np.ndarray:
    return np.random.default_rng(SEED).gamma(2.0, 20.0, size=(draws, cells))


def test_every_draw_adds_up_and_stays_inside_its_bounds(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    cells = _predicted(harmonized_toy)
    fit = make_state_fit(_raw(6, 3), cells)
    draws = reconcile_fit(fit, harmonized_toy.qcew_monthly, _bounds(cells), appendix_a_config)
    check = check_reconciled(draws, tolerance=TOLERANCE)
    assert check.passed
    assert (check.draws_checked, check.months_checked, check.cells_checked) == (6, 2, 3)
    # January's missing set is one cell, so every draw pins it to the residual.
    np.testing.assert_allclose(draws.values[:, 0], 50.0, rtol=0.0, atol=TOLERANCE)
    for row in draws.values.tolist():
        assert math.isclose(row[1] + row[2], 80.0, rel_tol=0.0, abs_tol=TOLERANCE)


def test_a_finite_upper_bound_scales_every_draw_into_it(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    """§12.3 on each draw: '06' in February may hold at most 10, so '04' takes the rest."""
    cells = _predicted(harmonized_toy)
    fit = make_state_fit(np.tile([[5.0, 1.0, 9.0]], (4, 1)), cells)
    draws = reconcile_fit(
        fit, harmonized_toy.qcew_monthly, _bounds(cells, {cells[2]: 10.0}), appendix_a_config
    )
    assert np.all(draws.values[:, 2] <= 10.0 + TOLERANCE)
    np.testing.assert_allclose(draws.values[:, 1], 70.0, atol=TOLERANCE)
    assert check_reconciled(draws, tolerance=TOLERANCE).passed


def test_a_negative_draw_is_refused_rather_than_floored(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    """Stage 3's refusal reaches the model path unchanged (§18.3)."""
    cells = _predicted(harmonized_toy)
    raw = _raw(2, 3)
    raw[1, 2] = -1.0
    with pytest.raises(WeightDomainError, match="negative"):
        reconcile_fit(
            make_state_fit(raw, cells),
            harmonized_toy.qcew_monthly,
            _bounds(cells),
            appendix_a_config,
        )


def test_a_missing_cell_the_fit_never_scored_is_refused(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    cells = _predicted(harmonized_toy)
    fit = make_state_fit(_raw(2, 2), cells[:2])
    with pytest.raises(ConceptViolationError, match="scored no draw"):
        reconcile_fit(fit, harmonized_toy.qcew_monthly, _bounds(cells), appendix_a_config)


def test_a_scored_cell_in_no_missing_set_is_refused(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    """A score with no anchor to reconcile against is an estimate §12 never checked."""
    cells = _predicted(harmonized_toy)
    stray = cells[0].replace("|04|", "|01|")
    fit = make_state_fit(_raw(2, 4), (*cells, stray))
    with pytest.raises(ConceptViolationError, match="no month's missing set"):
        reconcile_fit(fit, harmonized_toy.qcew_monthly, _bounds(cells), appendix_a_config)


def test_the_draw_axis_and_its_chain_index_survive(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    """§12.7: nothing here summarises, so the draws keep the index §15.4's store must preserve."""
    cells = _predicted(harmonized_toy)
    raw = _raw(6, 3)
    fit = make_state_fit(raw, cells)
    draws = reconcile_fit(fit, harmonized_toy.qcew_monthly, _bounds(cells), appendix_a_config)
    assert draws.chain.tolist() == [0, 0, 0, 1, 1, 1]
    assert draws.draw.tolist() == [0, 1, 2, 0, 1, 2]
    np.testing.assert_allclose(draws.raw_mean, raw.mean(axis=0), rtol=1e-12, atol=0.0)
    assert draws.state_fips == ("04", "04", "06")
    assert draws.reference_months == ("2023-01", "2023-02", "2023-02")
    assert {month: anchor.anchor_basis for month, anchor in draws.anchors.items()} == {
        "2023-01": "declared_national_total",
        "2023-02": "declared_national_total",
    }


def test_a_column_mean_depends_on_the_values_alone() -> None:
    """Neither the draws' order nor the array's memory layout moves a persisted mean."""
    rng = np.random.default_rng(SEED)
    values = rng.gamma(2.0, 20.0, size=(4000, 7))
    reference = exact_column_means(values)
    assert np.array_equal(reference, exact_column_means(np.asfortranarray(values)))
    assert np.array_equal(reference, exact_column_means(values[rng.permutation(4000)]))
    np.testing.assert_allclose(reference, values.mean(axis=0), rtol=1e-12, atol=0.0)
    with pytest.raises(ConceptViolationError, match="zero draws"):
        exact_column_means(np.empty((0, 3)))


def test_the_check_reads_the_draws_not_a_summary(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    """One drifted draw and one escaped value are each caught, and reported apart."""
    cells = _predicted(harmonized_toy)
    draws = reconcile_fit(
        make_state_fit(_raw(4, 3), cells),
        harmonized_toy.qcew_monthly,
        _bounds(cells, {cells[2]: 1000.0}),
        appendix_a_config,
    )
    draws.values[3, 1] += 1.0
    drifted = check_reconciled(draws, tolerance=TOLERANCE)
    assert drifted.max_anchor_drift == pytest.approx(1.0)
    assert drifted.bound_violations == 0
    assert not drifted.passed
    draws.values[3, 1] -= 1.0
    draws.values[2, 2] = 1001.0
    escaped = check_reconciled(draws, tolerance=TOLERANCE)
    assert escaped.bound_violations == 1
    assert not escaped.passed
```

- [ ] **Step 2: Run them to watch them fail**

Run: `uv run pytest tests/unit/test_model_reconciliation.py -q`

Expected (observed):

```text
E   ModuleNotFoundError: No module named 'logging_employment.models.reconciliation'
ERROR tests/unit/test_model_reconciliation.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error
```

- [ ] **Step 3: Write the reconciliation module**

`src/logging_employment/models/reconciliation.py`:

```python
"""§12.7 and INV-012: every draw of a fit reconciled into the feasible set, one month at a time.

`reconcile_draws` (Stage 3) takes ONE month. Its anchor's missing cells are bare `state_fips`, and
its `Bounds` are keyed the same way. A fit's scores cover every suppressed cell in the window, so
this module slices them by month and orders each slice as the month's anchor orders its missing set.
It projects the seven-field bounds to that month (`baselines.runner.month_bounds`) and calls
`reconcile_draws`. Stage 3's roadmap promise is "a single reconciliation entry point every model
draw passes through", and this is that path: no draw is allocated any other way.

THE ANCHOR IS THE ONE EVERY BASELINE ALLOCATES TO. `reconcile/anchor.py::national_residual` builds
it over the frame's own partition, stamped `anchor_basis = 'declared_national_total'`, and the
establishment-closure gate (`closure_audit` + `assert_universe_closes`) must admit it for the whole
window before any month is used. It is a `modeling_assumption` (INV-004), never a constraint row,
and `DrawCheck` reports adding-up to it separately from the hard bounds.

Nothing here summarises. §12.7: "Posterior summaries must be computed from reconciled joint
draws", and `models/summary.py` is where that happens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..baselines.runner import assert_bounds_cover, missing_cell_ids, month_bounds
from ..config import Config
from ..errors import ConceptViolationError
from ..reconcile.anchor import (
    Anchor,
    assert_universe_closes,
    closure_audit,
    national_residual,
    observed_partition,
)
from ..reconcile.draws import PosteriorDraws, ReconciliationInputs, reconcile_draws
from ..reconcile.scaling import Bounds
from .interfaces import MODEL_ID, StateModelFit


def exact_column_means(values: np.ndarray) -> np.ndarray:
    """Each cell's mean over the draw axis, every column summed with `math.fsum`.

    `ndarray.mean(axis=0)` adds a column in an order set by the array's memory layout, so a cell's
    mean can differ in the last place from the same column's own `mean()` (seen while this module
    was written: 41.20440509225164 against 41.20440509225163). `posterior_summary` and the harness
    persist these means into tables whose digests a manifest records, so a mean has to depend on
    the values alone. The same reason is why `validate/metrics.py` reduces through `_exact_sum`.
    """
    if values.shape[0] == 0:
        raise ConceptViolationError("a posterior mean over zero draws is undefined")
    count = values.shape[0]
    return np.array([math.fsum(column) / count for column in values.T.tolist()], dtype=np.float64)


@dataclass(frozen=True)
class ReconciledDraws:
    """A fit's reconciled joint draws over its prediction cells, and what each month was held to.

    `values` is (draws, cells), in `cell_ids` order, which is the fit's own `raw_scores.cell_ids`
    order. `raw_mean` is each cell's posterior-mean §11.5 score, kept for `baseline_results`'s
    `raw_weight` column. `lower` and `upper` are the deterministic bounds each draw was reconciled
    into, with `upper` reading +inf where `selected_upper` is null. `anchors` is keyed by month.
    """

    cell_ids: tuple[str, ...]
    state_fips: tuple[str, ...]
    reference_months: tuple[str, ...]
    values: np.ndarray
    chain: np.ndarray | None
    draw: np.ndarray | None
    raw_mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    anchors: dict[str, Anchor]


@dataclass(frozen=True)
class DrawCheck:
    """INV-012 measured on the reconciled draws themselves, never on a summary of them.

    It holds two obligations in two fields. `bound_violations` counts draw-cell values outside their
    own deterministic interval: those are the HARD constraints (INV-002's per-cell half).
    `max_anchor_drift` is the largest |sum - R_t| over draws and months: adding-up to the declared
    national anchor, which is a `modeling_assumption` (INV-004) and not a hard constraint. Both must
    pass, but a reader must not take one for the other (INV-008's spirit).
    """

    draws_checked: int
    months_checked: int
    cells_checked: int
    max_anchor_drift: float
    bound_violations: int
    tolerance: float

    @property
    def passed(self) -> bool:
        """Whether every draw sums to its month's residual and stays inside each cell's interval."""
        return self.max_anchor_drift <= self.tolerance and self.bound_violations == 0


def reconcile_fit(
    fit: StateModelFit, monthly: pl.DataFrame, bounds: Bounds, config: Config
) -> ReconciledDraws:
    """Reconcile every draw of `fit`, month by month, against the frame's own anchor and bounds.

    `bounds` is keyed by the seven-field `cell_id` (`baselines.runner.state_total_bounds`). The
    harness passes the MASKED system's bounds, never the run directory's (`D-087`), exactly as it
    does for the baselines.

    The closure gate runs first, as in `run_baselines`, so the anchor is admitted for the whole
    window or not at all (§18.3). A cell the fit scored that no month's missing set contains, or a
    missing cell the fit did not score, is refused: either means the model and the anchor disagree
    about which cells are unknown.
    """
    partitions = observed_partition(monthly)
    assert_universe_closes(closure_audit(monthly, partitions))
    raw = np.asarray(fit.raw_scores.values, dtype=np.float64)
    column_of = {cell: index for index, cell in enumerate(fit.raw_scores.cell_ids)}
    reconciled = np.full_like(raw, np.nan)
    filled = np.zeros(raw.shape[1], dtype=bool)
    lower = np.full(raw.shape[1], np.nan)
    upper = np.full(raw.shape[1], np.nan)
    state_of = [""] * raw.shape[1]
    month_of = [""] * raw.shape[1]
    anchors: dict[str, Anchor] = {}
    for month in sorted(partitions):
        anchor = national_residual(monthly, partitions[month], reference_month=month)
        if not anchor.missing_cells:
            continue
        ids = missing_cell_ids(partitions[month])
        unscored = sorted(
            ids[state] for state in anchor.missing_cells if ids[state] not in column_of
        )
        if unscored:
            raise ConceptViolationError(
                f"{month}: the fit scored no draw for missing cell(s) {unscored[:5]}; every cell "
                "the anchor allocates to needs a score, or the residual lands on the others"
            )
        columns = [column_of[ids[state]] for state in anchor.missing_cells]
        assert_bounds_cover(
            dict.fromkeys(anchor.missing_cells, 0.0),
            bounds,
            cell_ids=ids,
            estimator_id=MODEL_ID,
            reference_month=month,
        )
        scoped = month_bounds(bounds, cell_ids=ids, cells=anchor.missing_cells)
        out = reconcile_draws(
            PosteriorDraws(
                cell_ids=anchor.missing_cells,
                values=raw[:, columns],
                chain=fit.raw_scores.chain,
                draw=fit.raw_scores.draw,
            ),
            ReconciliationInputs(anchor=anchor, bounds=scoped),
            config.reconciliation,
        )
        reconciled[:, columns] = out.values
        filled[columns] = True
        for state, column in zip(anchor.missing_cells, columns, strict=True):
            lower[column] = scoped.lower[state]
            upper[column] = scoped.upper_of(state)
            state_of[column] = state
            month_of[column] = month
        anchors[month] = anchor
    if not filled.all():
        orphans = [fit.raw_scores.cell_ids[index] for index in np.flatnonzero(~filled)]
        raise ConceptViolationError(
            f"{len(orphans)} scored cell(s) sit in no month's missing set, first {orphans[:5]}; "
            "a score with no anchor to reconcile against is an estimate §12 never checked"
        )
    return ReconciledDraws(
        cell_ids=tuple(fit.raw_scores.cell_ids),
        state_fips=tuple(state_of),
        reference_months=tuple(month_of),
        values=reconciled,
        chain=fit.raw_scores.chain,
        draw=fit.raw_scores.draw,
        raw_mean=exact_column_means(raw),
        lower=lower,
        upper=upper,
        anchors=anchors,
    )


def check_reconciled(draws: ReconciledDraws, *, tolerance: float) -> DrawCheck:
    """INV-012 on the reconciled draws: each month sums to R_t in every draw, each value in [L, U].

    It sums with `math.fsum`, one draw at a time, so the check does not depend on the order NumPy
    reduces in. The reconciler bisects to a double's resolution, and a re-sum in another order can
    move the last bits: `cli.py::reconcile_command` measured 9.93e-10 against 1e-9 before the
    bisection was fixed.
    """
    months = np.asarray(draws.reference_months)
    worst = 0.0
    for month, anchor in sorted(draws.anchors.items()):
        block = draws.values[:, months == month]
        worst = max(worst, max(abs(math.fsum(row) - anchor.residual) for row in block.tolist()))
    outside = (draws.values < draws.lower[None, :] - tolerance) | (
        draws.values > draws.upper[None, :] + tolerance
    )
    return DrawCheck(
        draws_checked=int(draws.values.shape[0]),
        months_checked=len(draws.anchors),
        cells_checked=int(draws.values.shape[1]),
        max_anchor_drift=float(worst),
        bound_violations=int(np.count_nonzero(outside)),
        tolerance=tolerance,
    )
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/unit/test_model_reconciliation.py -q`

Expected: `8 passed` (observed: `8 passed`).

- [ ] **Step 5: Gates**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
uv run pytest -q -p no:cacheprovider -m "not slow"
```

Expected: gates clean. Non-slow suite: Task 4's count **+ 8 passed**.

- [ ] **Step 6: Commit**

```bash
git add src/logging_employment/models/reconciliation.py tests/unit/conftest.py tests/unit/test_model_reconciliation.py
git commit -m "feat(models): reconcile every draw month by month and check INV-012 on the draws"
```

---

### Task 6: §7.11's `posterior_summary`, from reconciled draws only

**Implements:**
- §7.11's table, 23 fields in the spec's order;
- REQ-012's posterior half and INV-008: deterministic and posterior intervals in distinct columns, never relabelled;
- §12.7 (summaries from reconciled joint draws);
- INV-001 (a published cell carries its value exactly);
- the brief's warning that a status this stage invents would reach `posterior_summary.parquet` unchecked "unless this stage calls the guard itself". `posterior_summary` calls `assert_declared_provenance`, and `observed_or_imputed` joins the closed sets it checks.

Three columns are null, each for a stated reason:
- `model_sensitivity_low` and `model_sensitivity_high` are Stage 7's (§11.13), as Stage 7's block says.
- `probability_thresholds_json`: §7.11 names the column, and no section states a threshold, so inventing one would be §21's policy.

**Files:**
- Modify: `src/logging_employment/contracts.py` (`OBSERVED_OR_IMPUTED`, `POSTERIOR_SUMMARY_SCHEMA`, one entry in `assert_declared_provenance`'s loop)
- Create: `src/logging_employment/models/summary.py`
- Test: `tests/unit/test_posterior_summary.py` (new; 9 tests)

**Interfaces:**
- Consumes: Task 5's `ReconciledDraws`, `reconcile_fit` and `exact_column_means`; Task 3's `state_cell_ids`, `TRAINING_STATUS`, `PREDICTION_STATUS`, `MODEL_ID`, `MODEL_VERSION`; `baselines.runner.state_total_bounds`.
- Produces:
  - `models.summary.posterior_summary(draws, monthly, bounds: pl.DataFrame, *, run_id: str, constraint_set_hash: str) -> pl.DataFrame`, validated against `contracts.POSTERIOR_SUMMARY_SCHEMA`. There is one row per state cell with a published row: observed, true zero or suppressed. `bounds` is the run's `deterministic_bounds` frame, not a `Bounds`.
  - `models.summary.LEVELS`: `(0.5, "ci50")`, `(0.8, "ci80")`, `(0.9, "ci90")`, `(0.95, "ci95")`, as equal-tailed quantiles.
  - `contracts.OBSERVED_OR_IMPUTED = ("observed", "imputed")`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_posterior_summary.py`:

```python
"""§7.11's `posterior_summary`, from reconciled draws (§12.7) and the run's deterministic bounds."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from logging_employment.baselines.runner import state_total_bounds
from logging_employment.contracts import (
    DETERMINISTIC_BOUNDS_SCHEMA,
    POSTERIOR_SUMMARY_SCHEMA,
    assert_declared_provenance,
)
from logging_employment.errors import ConceptViolationError
from logging_employment.models.data import state_cell_ids
from logging_employment.models.reconciliation import reconcile_fit
from logging_employment.models.summary import LEVELS, posterior_summary

SEED = sum(map(ord, "tests/posterior-summary"))
POSTERIOR_COLUMNS = ["posterior_mean", "posterior_median"] + [
    f"{stem}_{end}" for _level, stem in LEVELS for end in ("low", "high")
]


def _states(monthly: pl.DataFrame) -> pl.DataFrame:
    return monthly.filter(pl.col("area_type") == "state")


def _bounds_frame(monthly: pl.DataFrame, upper: dict[str, float]) -> pl.DataFrame:
    """A `deterministic_bounds` table of every state cell: published ones pinned, others [0, U]."""
    states = _states(monthly)
    rows = []
    for identifier, row in zip(state_cell_ids(states), states.iter_rows(named=True), strict=True):
        published = row["observation_status"] != "suppressed"
        low = float(row["employment_value"]) if published else 0.0
        high = float(row["employment_value"]) if published else upper.get(identifier)
        rows.append(
            {
                "cell_id": identifier,
                "component_id": "c0",
                "rank": 0,
                "nullity": 0,
                "lp_lower": low,
                "lp_upper": high,
                "milp_lower": None,
                "milp_upper": None,
                "selected_lower": low,
                "selected_upper": high,
                "bound_status": "observed" if published else "unbounded",
                "exactly_identified": published,
                "integer_exactly_identified": published,
                "solver_status": "optimal",
                "solver_tolerance": 1e-7,
                "constraint_set_hash": "hash0",
            }
        )
    return pl.DataFrame(rows, schema=DETERMINISTIC_BOUNDS_SCHEMA)


@pytest.fixture()
def summarised(harmonized_toy, appendix_a_config, make_state_fit):
    """The toy's three suppressed cells reconciled over 40 draws, '06' in February capped at 60."""
    monthly = harmonized_toy.qcew_monthly
    suppressed = _states(monthly).filter(pl.col("observation_status") == "suppressed")
    cells = state_cell_ids(suppressed.sort("reference_month", "state_fips"))
    bounds = _bounds_frame(monthly, {cells[2]: 60.0})
    fit = make_state_fit(np.random.default_rng(SEED).gamma(2.0, 20.0, size=(40, 3)), cells)
    draws = reconcile_fit(fit, monthly, state_total_bounds(bounds), appendix_a_config)
    frame = posterior_summary(draws, monthly, bounds, run_id="run0", constraint_set_hash="hash0")
    return monthly, draws, bounds, frame


def test_one_row_per_state_cell_in_section_7_11s_field_order(summarised) -> None:
    monthly, _draws, _bounds, frame = summarised
    assert list(frame.schema.items()) == list(POSTERIOR_SUMMARY_SCHEMA.items())
    assert frame.height == _states(monthly).height == 8
    assert frame["cell_id"].n_unique() == 8


def test_a_published_cell_carries_its_value_in_every_posterior_column(summarised) -> None:
    """INV-001: a posterior interval around a published number would invent uncertainty."""
    monthly, _draws, _bounds, frame = summarised
    observed = _states(monthly).filter(pl.col("observation_status") == "observed")
    values = dict(
        zip(state_cell_ids(observed), observed["employment_value"].to_list(), strict=True)
    )
    rows = frame.filter(pl.col("observed_or_imputed") == "observed")
    assert rows.height == 5
    for row in rows.iter_rows(named=True):
        assert {row[column] for column in POSTERIOR_COLUMNS} == {float(values[row["cell_id"]])}
        assert row["reconciliation_status"] == "observed"


def test_an_imputed_cells_summaries_come_from_its_draws_and_nest(summarised) -> None:
    _monthly, draws, _bounds, frame = summarised
    for j, cell in enumerate(draws.cell_ids):
        row = frame.filter(pl.col("cell_id") == cell).row(0, named=True)
        column = draws.values[:, j]
        assert row["posterior_median"] == float(np.quantile(column, 0.5))
        assert row["posterior_mean"] == pytest.approx(float(column.mean()), rel=1e-12, abs=0.0)
        assert row["ci95_low"] <= row["ci90_low"] <= row["ci80_low"] <= row["ci50_low"]
        assert row["ci50_high"] <= row["ci80_high"] <= row["ci90_high"] <= row["ci95_high"]
        assert row["observed_or_imputed"] == "imputed"
        assert row["reconciliation_status"] == "anchored_and_reconciled"


def test_deterministic_and_posterior_intervals_are_separate_columns(summarised) -> None:
    """INV-008: §9's interval is copied through, and the posterior one sits inside it."""
    _monthly, draws, _bounds, frame = summarised
    imputed = frame.filter(pl.col("observed_or_imputed") == "imputed").sort("cell_id")
    capped = imputed.filter(pl.col("cell_id") == draws.cell_ids[2]).row(0, named=True)
    assert (capped["deterministic_lower"], capped["deterministic_upper"]) == (0.0, 60.0)
    assert capped["ci95_high"] <= 60.0
    uncapped = imputed.filter(pl.col("cell_id") != draws.cell_ids[2])
    assert uncapped["deterministic_upper"].null_count() == uncapped.height
    assert (imputed["ci95_low"] >= imputed["deterministic_lower"]).all()


def test_the_columns_later_owners_fill_are_null(summarised) -> None:
    """Sensitivity is Stage 7's (§11.13); §7.11's thresholds have no stated value anywhere."""
    frame = summarised[3]
    for column in (
        "model_sensitivity_low",
        "model_sensitivity_high",
        "probability_thresholds_json",
    ):
        assert frame[column].null_count() == frame.height


def test_the_source_vintage_set_is_a_json_list(summarised) -> None:
    frame = summarised[3]
    assert {json.loads(value)[0] for value in frame["source_vintage_set"].to_list()} == {"2024q1"}


def test_a_suppressed_cell_with_no_draws_is_refused(summarised) -> None:
    monthly, draws, bounds, _frame = summarised
    doctored = monthly.with_columns(
        pl.when((pl.col("state_fips") == "01") & (pl.col("reference_month") == "2023-01"))
        .then(pl.lit("suppressed"))
        .otherwise(pl.col("observation_status"))
        .alias("observation_status")
    )
    with pytest.raises(ConceptViolationError, match="no reconciled draws"):
        posterior_summary(draws, doctored, bounds, run_id="run0", constraint_set_hash="hash0")


def test_draws_for_a_cell_that_is_not_suppressed_are_refused(summarised) -> None:
    monthly, draws, bounds, _frame = summarised
    doctored = monthly.with_columns(
        pl.when((pl.col("state_fips") == "06") & (pl.col("reference_month") == "2023-02"))
        .then(pl.lit("observed"))
        .otherwise(pl.col("observation_status"))
        .alias("observation_status"),
        pl.when((pl.col("state_fips") == "06") & (pl.col("reference_month") == "2023-02"))
        .then(pl.lit(30))
        .otherwise(pl.col("employment_value"))
        .alias("employment_value"),
    )
    with pytest.raises(ConceptViolationError, match="match no suppressed row"):
        posterior_summary(draws, doctored, bounds, run_id="run0", constraint_set_hash="hash0")


def test_an_undeclared_observed_or_imputed_value_is_refused() -> None:
    with pytest.raises(ConceptViolationError, match="observed_or_imputed"):
        assert_declared_provenance(pl.DataFrame({"observed_or_imputed": ["guessed"]}))
```

- [ ] **Step 2: Run them to watch them fail**

Run: `uv run pytest tests/unit/test_posterior_summary.py -q`

Expected (observed):

```text
E   ImportError: cannot import name 'POSTERIOR_SUMMARY_SCHEMA' from 'logging_employment.contracts' (…/src/logging_employment/contracts.py)
ERROR tests/unit/test_posterior_summary.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error
```

- [ ] **Step 3: Declare the schema and the closed set**

`src/logging_employment/contracts.py`:

```diff
diff --git a/src/logging_employment/contracts.py b/src/logging_employment/contracts.py
index f1bed3d..b40b9f3 100644
--- a/src/logging_employment/contracts.py
+++ b/src/logging_employment/contracts.py
@@ -235,9 +235,9 @@ DECLINE_KINDS: tuple[str, ...] = ("by_design", "data_gap", "reconciliation_failu
 def assert_declared_provenance(frame: pl.DataFrame) -> None:
     """Refuse a provenance value outside its declared tuple.
 
-    The seven tuples this checks (`RECONCILIATION_STATUSES`, `WEIGHT_BASES`, `ANCHOR_BASES`,
-    `DECLINE_KINDS`, `SUPPRESSION_TYPES`, `STRATUM_KINDS`, `MASK_ARMS`) are the closed sets a
-    baseline row's provenance may draw from, but
+    The eight tuples this checks (`RECONCILIATION_STATUSES`, `WEIGHT_BASES`, `ANCHOR_BASES`,
+    `DECLINE_KINDS`, `SUPPRESSION_TYPES`, `STRATUM_KINDS`, `MASK_ARMS`, `OBSERVED_OR_IMPUTED`) are
+    the closed sets a row's provenance may draw from, but
     `BASELINE_RESULT_SCHEMA` checks dtypes only -- `pl.String` accepts any string. `weight_basis`
     is the live exposure: `run_baselines` copies it from an estimator's own `outcome.basis`, so a
     third-party estimator's typo reached `baseline_results.parquet` and passed every test. Nulls
@@ -259,6 +259,8 @@ def assert_declared_provenance(frame: pl.DataFrame) -> None:
         # a literal at the emit sites, so its value comes from data. Both harness calls pass a
         # frame that carries it: the scored frame and the assembled metrics.
         ("mask_arm", MASK_ARMS),
+        # Plan 16. §7.11 names the column and gives no values; `models/summary.py` writes two.
+        ("observed_or_imputed", OBSERVED_OR_IMPUTED),
     ):
         if column not in frame.columns:
             continue
@@ -381,6 +383,43 @@ ANCHOR_AUDIT_SCHEMA: dict[str, pl.DataType] = {
 }
 
 
+# §7.11's `observed_or_imputed`, which the spec names and does not enumerate. A published or
+# true-zero cell is `observed` and carries its exact value in every posterior column (INV-001); a
+# suppressed cell is `imputed` from reconciled draws.
+OBSERVED_OR_IMPUTED: tuple[str, ...] = ("observed", "imputed")
+
+# §7.11, in the spec's field order, which is load-bearing (`schema_fingerprint`). One row per
+# state-total cell the panel publishes a row for. Deterministic and posterior intervals are distinct
+# columns and never one another (INV-008): `deterministic_*` is §9's interval copied through, `ci*`
+# the reconciled draws' equal-tailed quantiles. `probability_thresholds_json` and the two
+# `model_sensitivity_*` columns are null in Stage 5 for the reasons `models/summary.py` gives.
+POSTERIOR_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
+    "cell_id": pl.String,
+    "model_id": pl.String,
+    "model_version": pl.String,
+    "run_id": pl.String,
+    "posterior_mean": pl.Float64,
+    "posterior_median": pl.Float64,
+    "ci50_low": pl.Float64,
+    "ci50_high": pl.Float64,
+    "ci80_low": pl.Float64,
+    "ci80_high": pl.Float64,
+    "ci90_low": pl.Float64,
+    "ci90_high": pl.Float64,
+    "ci95_low": pl.Float64,
+    "ci95_high": pl.Float64,
+    "probability_thresholds_json": pl.String,
+    "deterministic_lower": pl.Float64,
+    "deterministic_upper": pl.Float64,
+    "model_sensitivity_low": pl.Float64,
+    "model_sensitivity_high": pl.Float64,
+    "observed_or_imputed": pl.String,
+    "reconciliation_status": pl.String,
+    "constraint_set_hash": pl.String,
+    "source_vintage_set": pl.String,
+}
+
+
 HOLDOUT_REGIMES: tuple[str, ...] = (
     "small_cell_biased",
     "concentration_proxy",
```

- [ ] **Step 4: Write the summary**

`src/logging_employment/models/summary.py`:

```python
"""§7.11's `posterior_summary`, computed from reconciled joint draws and nothing else (§12.7).

ONE ROW PER STATE-TOTAL CELL THE PANEL PUBLISHES A ROW FOR: 4,716 on D1. An observed or true-zero
cell carries its published value in every posterior column, with `observed_or_imputed = 'observed'`
and `reconciliation_status = 'observed'`. INV-001 requires disclosed values to be preserved exactly,
and a posterior interval around a published number would imply uncertainty the source does not
have. A suppressed cell summarises its reconciled draws, with `observed_or_imputed = 'imputed'` and
`reconciliation_status = 'anchored_and_reconciled'`.

DETERMINISTIC AND POSTERIOR INTERVALS ARE DIFFERENT COLUMNS AND NEVER ONE ANOTHER (INV-008).
`deterministic_lower` / `deterministic_upper` are §9's `selected_lower` / `selected_upper` copied
through, with a null upper where no public fact bounds the cell. The `ci*` columns are equal-tailed
quantiles of the reconciled draws. `models/reconciliation.py::DrawCheck` has already checked that
every draw lies inside the deterministic interval, so each posterior interval nests inside it, and
nothing here widens, clips or relabels either one.

THREE COLUMNS ARE NULL, EACH FOR A DECLARED REASON:

* `model_sensitivity_low` / `_high`: §11.13's sensitivity variants are Stage 7's
  (`models/suppression_sensitivity.py` in its Produces). Stage 7's roadmap block is where "later
  stages may assume `model_sensitivity_low` and `model_sensitivity_high` for every release cell".
* `probability_thresholds_json`: §7.11 names the column, and no section of the spec states a
  threshold for it. Inventing one here would be the policy §21 leaves to its owner.

`source_vintage_set` is a JSON list. For an observed cell it is the cell's own `release_vintage`.
For an imputed cell it is every `release_vintage` among the training cells, because a
hierarchical fit informs each imputed cell from the whole panel.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from ..baselines.runner import state_total_bounds
from ..contracts import POSTERIOR_SUMMARY_SCHEMA, assert_declared_provenance, validate_frame
from ..errors import ConceptViolationError
from .data import PREDICTION_STATUS, TRAINING_STATUS, state_cell_ids
from .interfaces import MODEL_ID, MODEL_VERSION
from .reconciliation import ReconciledDraws, exact_column_means

PUBLISHED_STATUSES: tuple[str, ...] = (TRAINING_STATUS, "true_zero")
# (level, column stem): equal-tailed central intervals, §7.11's four.
LEVELS: tuple[tuple[float, str], ...] = (
    (0.50, "ci50"),
    (0.80, "ci80"),
    (0.90, "ci90"),
    (0.95, "ci95"),
)


def posterior_summary(
    draws: ReconciledDraws,
    monthly: pl.DataFrame,
    bounds: pl.DataFrame,
    *,
    run_id: str,
    constraint_set_hash: str,
) -> pl.DataFrame:
    """§7.11's table for every state cell of `monthly`, validated and provenance-checked.

    `bounds` is the run's `deterministic_bounds` frame. A state cell with a published row whose
    status is neither published nor suppressed is refused, as is an imputed cell with no suppressed
    row to hang it on. Either one would drop a cell from the release without a signal.
    """
    states = monthly.filter(pl.col("area_type") == "state")
    identifiers = state_cell_ids(states)
    deterministic = state_total_bounds(bounds)
    column_of = {cell: index for index, cell in enumerate(draws.cell_ids)}
    quantiles = [0.5]
    for level, _stem in LEVELS:
        quantiles += [(1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0]
    q = np.quantile(draws.values, quantiles, axis=0) if draws.values.size else None
    means = exact_column_means(draws.values) if draws.values.size else None
    training = sorted(
        str(v)
        for v in states.filter(pl.col("observation_status") == TRAINING_STATUS)["release_vintage"]
        .unique()
        .to_list()
    )
    rows: list[dict[str, object]] = []
    used: set[str] = set()
    for identifier, row in zip(identifiers, states.iter_rows(named=True), strict=True):
        status = row["observation_status"]
        base = {
            "cell_id": identifier,
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "run_id": run_id,
            "probability_thresholds_json": None,
            "deterministic_lower": deterministic.lower.get(identifier),
            "deterministic_upper": deterministic.upper.get(identifier),
            "model_sensitivity_low": None,
            "model_sensitivity_high": None,
            "constraint_set_hash": constraint_set_hash,
        }
        if status in PUBLISHED_STATUSES:
            value = float(row["employment_value"])
            interval = {
                f"{stem}_{end}": value for _level, stem in LEVELS for end in ("low", "high")
            }
            rows.append(
                base
                | interval
                | {
                    "posterior_mean": value,
                    "posterior_median": value,
                    "observed_or_imputed": "observed",
                    "reconciliation_status": "observed",
                    "source_vintage_set": json.dumps([str(row["release_vintage"])]),
                }
            )
        elif status == PREDICTION_STATUS:
            if identifier not in column_of:
                raise ConceptViolationError(
                    f"suppressed cell {identifier} has no reconciled draws; §7.11 would release "
                    "it without a posterior"
                )
            j = column_of[identifier]
            used.add(identifier)
            interval = {}
            for position, (_level, stem) in enumerate(LEVELS):
                interval[f"{stem}_low"] = float(q[1 + 2 * position, j])
                interval[f"{stem}_high"] = float(q[2 + 2 * position, j])
            rows.append(
                base
                | interval
                | {
                    "posterior_mean": float(means[j]),
                    "posterior_median": float(q[0, j]),
                    "observed_or_imputed": "imputed",
                    "reconciliation_status": "anchored_and_reconciled",
                    "source_vintage_set": json.dumps(training),
                }
            )
        else:
            raise ConceptViolationError(
                f"state cell {identifier} has observation_status {status!r}; §7.11 has no row "
                "shape for it"
            )
    stray = sorted(set(draws.cell_ids) - used)
    if stray:
        raise ConceptViolationError(
            f"{len(stray)} reconciled cell(s) match no suppressed row, first {stray[:5]}"
        )
    frame = pl.DataFrame(rows, schema=POSTERIOR_SUMMARY_SCHEMA)
    validate_frame(frame, POSTERIOR_SUMMARY_SCHEMA, "posterior_summary")
    assert_declared_provenance(frame)
    return frame
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `uv run pytest tests/unit/test_posterior_summary.py tests/unit/test_contracts_validation.py -q`

Expected: all pass (`test_posterior_summary.py` observed: `9 passed`).

- [ ] **Step 6: Gates**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
uv run pytest -q -p no:cacheprovider -m "not slow"
```

Expected: gates clean. Non-slow suite: Task 5's count **+ 9 passed**.

- [ ] **Step 7: Commit**

```bash
git add src/logging_employment/contracts.py src/logging_employment/models/summary.py tests/unit/test_posterior_summary.py
git commit -m "feat(models): §7.11's posterior_summary from reconciled draws, intervals kept apart (INV-008)"
```

---

### Task 7: §15.4's store: the reconciled joint draws, in an ArviZ-shaped netCDF

**Implements:**
- §15.4 ("Joint draws SHOULD be stored in an ArviZ-compatible format … preserve draw, chain, state, month, and size indexes"; there is no size index until Stage 6);
- REQ-019 and INV-013 (joint draws retained);
- the roadmap's "reconciled joint draws in an ArviZ-compatible store under `runs/<run_id>/posterior/`";
- §11.14's R-hat and ESS statistics, the one other thing this module owns.

`read_store` rebuilds each month's `Anchor` from the store's recorded residual and basis. Since `D-122` landed on main, every one of them re-checks its basis as it is built, so a store whose basis was altered on disk is refused on read. `D-122`'s guard now has a live path in `src/`, not only a future one.

**Files:**
- Create: `src/logging_employment/models/arviz_io.py`
- Test: `tests/unit/test_arviz_store.py` (new; 4 tests)

**Interfaces:**
- Consumes: Task 5's `ReconciledDraws`, Task 3's `ModelData` and `StateModelFit`, `reconcile.anchor.Anchor`.
- Produces:
  - `rank_rhat(samples) -> np.ndarray` and `ess_bulk_tail(samples) -> tuple[np.ndarray, np.ndarray]`, over the trailing elements of a `(chain, draw, ...)` array;
  - `write_store(path, draws, fit, data, *, attrs: Mapping[str, str]) -> str`, which returns `draws_digest(draws)`;
  - `read_store(path) -> ReconciledDraws`;
  - `store_digest(path) -> str`, the `draws_sha256` root attribute;
  - `draws_digest(draws) -> str`, the sha256 of the reconciled values' float64 bytes, their shape and their cell ids, never the file;
  - `STORE_ENGINE = "h5netcdf"`, `PARAMETER_DIMS`, `TAIL_PROBABILITIES = (0.05, 0.95)`.
- The store has four groups. `posterior` holds the monitored parameters. `posterior_predictive` holds `reconciled_state_total`, `(chain, draw, cell)`, with `state_fips` and `reference_month` coordinates on `cell`. `sample_stats` holds `diverging`. `constant_data` holds `residual` and `anchor_basis` by month, and `deterministic_lower`, `deterministic_upper` and `raw_mean` by cell.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_arviz_store.py`:

```python
"""§15.4's store: the reconciled joint draws in ArviZ's layout, read back exactly."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from logging_employment.errors import ConceptViolationError
from logging_employment.models.arviz_io import draws_digest, read_store, store_digest, write_store
from logging_employment.models.data import build_model_data
from logging_employment.models.reconciliation import reconcile_fit
from logging_employment.reconcile.scaling import Bounds

SEED = sum(map(ord, "tests/arviz-store"))


@pytest.fixture()
def stored(tmp_path: Path, harmonized_toy, appendix_a_config, make_state_fit):
    """The toy's three suppressed cells over two chains of five draws, written to a store."""
    monthly = harmonized_toy.qcew_monthly
    data = build_model_data(monthly)
    cells = data.predict_cell_ids
    rng = np.random.default_rng(SEED)
    parameters = {
        "alpha": rng.normal(size=(2, 5)),
        "u": rng.normal(size=(2, 5, len(data.states))),
        "gamma": rng.normal(size=(2, 5, 12)),
    }
    fit = make_state_fit(rng.gamma(2.0, 20.0, size=(10, len(cells))), cells, parameters=parameters)
    bounds = Bounds(lower=dict.fromkeys(cells, 0.0), upper=dict.fromkeys(cells))
    draws = reconcile_fit(fit, monthly, bounds, appendix_a_config)
    path = tmp_path / "posterior" / "state_total_draws.nc"
    digest = write_store(path, draws, fit, data, attrs={"run_id": "run0"})
    return path, digest, draws


def test_the_store_reads_back_the_reconciled_draws_exactly(stored) -> None:
    path, _digest, draws = stored
    back = read_store(path)
    np.testing.assert_array_equal(back.values, draws.values)
    assert back.cell_ids == draws.cell_ids
    assert back.state_fips == draws.state_fips
    assert back.reference_months == draws.reference_months
    np.testing.assert_array_equal(back.lower, draws.lower)
    np.testing.assert_array_equal(back.upper, draws.upper)
    assert np.isinf(back.upper).all()
    assert back.anchors == draws.anchors


def test_the_digest_covers_the_draws_not_the_file(stored) -> None:
    path, digest, draws = stored
    assert digest == draws_digest(read_store(path)) == store_digest(path)
    moved = replace(draws, values=draws.values.copy())
    moved.values[0, 0] += 1e-9
    assert draws_digest(moved) != digest


def test_the_store_keeps_chain_draw_state_and_month_indexes(stored) -> None:
    """§15.4: "The storage must preserve draw, chain, state, and month indexes"."""
    path, _digest, _draws = stored
    tree = xr.open_datatree(path, engine="h5netcdf")
    try:
        predictive = tree["posterior_predictive"].to_dataset()
        assert predictive["reconciled_state_total"].dims == ("chain", "draw", "cell")
        assert {"state_fips", "reference_month"} <= set(predictive.coords)
        assert tree["posterior"].to_dataset()["u"].dims == ("chain", "draw", "state")
        assert tree["sample_stats"].to_dataset()["diverging"].dims == ("chain", "draw")
    finally:
        tree.close()


def test_draws_out_of_chain_major_order_are_refused(stored, harmonized_toy, make_state_fit) -> None:
    path, _digest, draws = stored
    data = build_model_data(harmonized_toy.qcew_monthly)
    fit = make_state_fit(draws.values, draws.cell_ids)
    with pytest.raises(ConceptViolationError, match="chain-major"):
        write_store(
            path.with_name("other.nc"),
            replace(draws, chain=draws.chain[::-1].copy()),
            fit,
            data,
            attrs={},
        )
```

- [ ] **Step 2: Run them to watch them fail**

Run: `uv run pytest tests/unit/test_arviz_store.py -q`

Expected (observed):

```text
E   ModuleNotFoundError: No module named 'logging_employment.models.arviz_io'
ERROR tests/unit/test_arviz_store.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error
```

- [ ] **Step 3: Write the store and the statistics**

`src/logging_employment/models/arviz_io.py`:

```python
"""The only module that imports ArviZ or xarray: §11.14's statistics and §15.4's store.

It is kept to one module so that an ArviZ API change is a one-file change, and so
`tests/unit/test_arviz_api_probe.py` can pin every call this package makes against the installed
versions. `arviz_stats.rhat` and `arviz_stats.ess` take an array shaped `(chain, draw, ...)` with
`chain_axis=0, draw_axis=1`. Plan 16 read that in the arviz-stats v1.3.3 source and ran it on
1.3.2, which is how it found that tail ESS needs its quantiles passed (`TAIL_PROBABILITIES`).
`method="rank"` is Vehtari et al.'s (2021) rank-normalized, folded split R-hat, the one §11.14's
1.01 threshold is calibrated for.

THE STORE IS AN ARVIZ-SHAPED `DataTree` IN NETCDF (§15.4: "Joint draws SHOULD be stored in an
ArviZ-compatible format ... The storage must preserve draw, chain, state, month, and size
indexes"). It has four groups:

* `posterior`: the monitored parameters, `(chain, draw, ...)`;
* `posterior_predictive`: `reconciled_state_total`, `(chain, draw, cell)`, the reconciled joint
  draws §12.7 summarises. `cell` carries `state_fips` and `reference_month` coordinates, which
  are §15.4's state and month indexes. There is no size index until Stage 6;
* `sample_stats`: `diverging`, `(chain, draw)`;
* `constant_data`: each month's residual and anchor basis, and each cell's deterministic bounds
  and mean raw score. With these, `reconcile` can re-verify INV-012 from the file alone.

Raw scores are NOT stored draw by draw. §12.7 summarises reconciled draws only, and 39 MB of D1
raw draws would buy an audit that `raw_mean` and the parameter trace already support.

WHAT IS DIGESTED IS THE DRAWS, NOT THE FILE. `draws_digest` hashes the reconciled array's float64
bytes, its shape and its cell ids. The netCDF bytes also carry HDF5 and ArviZ metadata that no one
here promises to hold still, so an idempotence check compares digests, never file hashes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xarray as xr
from arviz_stats import ess, rhat

from ..errors import ConceptViolationError
from ..reconcile.anchor import Anchor
from .interfaces import ModelData, StateModelFit
from .reconciliation import ReconciledDraws

STORE_ENGINE = "h5netcdf"
# The named dimension each vector parameter carries after (chain, draw).
PARAMETER_DIMS: dict[str, tuple[str, ...]] = {
    "u": ("state",),
    "rho": ("state",),
    "sigma_eta_state": ("state",),
    "v": ("division",),
    "gamma": ("month_of_year",),
    "delta": ("year",),
}


def rank_rhat(samples: np.ndarray) -> np.ndarray:
    """Rank-normalized split R-hat for every trailing element of a `(chain, draw, ...)` array."""
    return np.asarray(
        rhat(np.asarray(samples, dtype=np.float64), method="rank", chain_axis=0, draw_axis=1),
        dtype=np.float64,
    )


# Vehtari et al. (2021)'s tail ESS is the smaller of the 5% and 95% quantiles' ESS. The array
# interface REQUIRES `prob` for `method="tail"` and raises TypeError without it (run on
# arviz-stats 1.3.2 while plan 16 was written); the xarray interface would default it from the
# global `rcParams["stats.ci_prob"]`, which a gate should not depend on.
TAIL_PROBABILITIES = (0.05, 0.95)


def ess_bulk_tail(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bulk and tail effective sample size for every trailing element of `(chain, draw, ...)`."""
    values = np.asarray(samples, dtype=np.float64)
    bulk = ess(values, method="bulk", chain_axis=0, draw_axis=1)
    tail = ess(values, method="tail", prob=TAIL_PROBABILITIES, chain_axis=0, draw_axis=1)
    return np.asarray(bulk, dtype=np.float64), np.asarray(tail, dtype=np.float64)


def draws_digest(draws: ReconciledDraws) -> str:
    """The sha256 of the reconciled draws' float64 bytes, their shape and their cell ids."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps({"cells": list(draws.cell_ids), "shape": list(draws.values.shape)}).encode(
            "utf-8"
        )
    )
    digest.update(np.ascontiguousarray(draws.values, dtype=np.float64).tobytes())
    return digest.hexdigest()


def write_store(
    path: Path,
    draws: ReconciledDraws,
    fit: StateModelFit,
    data: ModelData,
    *,
    attrs: Mapping[str, str],
) -> str:
    """Write §15.4's store to `path` and return `draws_digest(draws)`.

    `attrs` go on the root group. Every value is a string, which netCDF can hold whatever the
    caller passes.
    """
    chains = fit.chains
    per_chain = draws.values.shape[0] // chains
    if draws.chain is not None and not np.array_equal(
        draws.chain, np.repeat(np.arange(chains), per_chain)
    ):
        raise ConceptViolationError(
            "reconciled draws are not in chain-major order, so the store would mislabel them"
        )
    shared = {"chain": np.arange(chains), "draw": np.arange(per_chain)}
    posterior = xr.Dataset(
        {
            name: (("chain", "draw", *PARAMETER_DIMS.get(name, ())), np.asarray(values))
            for name, values in fit.parameters.items()
        },
        coords={
            **shared,
            "state": list(data.states),
            "division": list(data.divisions),
            "month_of_year": np.arange(1, 13),
            "year": list(data.years),
        },
    )
    predictive = xr.Dataset(
        {
            "reconciled_state_total": (
                ("chain", "draw", "cell"),
                draws.values.reshape(chains, per_chain, -1),
            )
        },
        coords={
            **shared,
            "cell": list(draws.cell_ids),
            "state_fips": ("cell", list(draws.state_fips)),
            "reference_month": ("cell", list(draws.reference_months)),
        },
    )
    stats = xr.Dataset({"diverging": (("chain", "draw"), fit.diverging)}, coords=shared)
    months = sorted(draws.anchors)
    constant = xr.Dataset(
        {
            "residual": (("month",), np.array([draws.anchors[m].residual for m in months])),
            "anchor_basis": (("month",), [draws.anchors[m].anchor_basis for m in months]),
            "deterministic_lower": (("cell",), draws.lower),
            "deterministic_upper": (("cell",), draws.upper),
            "raw_mean": (("cell",), draws.raw_mean),
        },
        coords={"month": months, "cell": list(draws.cell_ids)},
    )
    digest = draws_digest(draws)
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={**dict(attrs), "draws_sha256": digest}),
            "posterior": posterior,
            "posterior_predictive": predictive,
            "sample_stats": stats,
            "constant_data": constant,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.to_netcdf(path, engine=STORE_ENGINE)
    return digest


def store_digest(path: Path) -> str:
    """The `draws_sha256` a store was written with, read from its root group's attributes."""
    tree = xr.open_datatree(path, engine=STORE_ENGINE)
    try:
        return str(tree.attrs["draws_sha256"])
    finally:
        tree.close()


def read_store(path: Path) -> ReconciledDraws:
    """The reconciled draws, bounds and anchors a store holds, read back without the fit.

    `reconcile` calls this to re-verify INV-012 against the persisted artifact rather than the
    in-memory draws `fit-state-model` checked. A defect in writing would otherwise pass unseen.
    Each `Anchor` rebuilt here checks its recorded basis against `contracts.ANCHOR_BASES` as it is
    built (`D-122`), so a store whose basis was altered on disk is refused on read.
    """
    tree = xr.open_datatree(path, engine=STORE_ENGINE)
    try:
        predictive = tree["posterior_predictive"].to_dataset().load()
        constant = tree["constant_data"].to_dataset().load()
    finally:
        tree.close()
    values = np.asarray(predictive["reconciled_state_total"].values, dtype=np.float64)
    chains, per_chain, cells = values.shape
    states = tuple(str(state) for state in predictive["state_fips"].values)
    months = tuple(str(month) for month in predictive["reference_month"].values)
    anchors = {
        str(month): Anchor(
            reference_month=str(month),
            residual=float(residual),
            missing_cells=tuple(s for s, m in zip(states, months, strict=True) if m == str(month)),
            anchor_basis=str(basis),
        )
        for month, residual, basis in zip(
            constant["month"].values,
            constant["residual"].values,
            constant["anchor_basis"].values,
            strict=True,
        )
    }
    return ReconciledDraws(
        cell_ids=tuple(str(cell) for cell in predictive["cell"].values),
        state_fips=states,
        reference_months=months,
        values=values.reshape(chains * per_chain, cells),
        chain=np.repeat(np.arange(chains), per_chain),
        draw=np.tile(np.arange(per_chain), chains),
        raw_mean=np.asarray(constant["raw_mean"].values, dtype=np.float64),
        lower=np.asarray(constant["deterministic_lower"].values, dtype=np.float64),
        upper=np.asarray(constant["deterministic_upper"].values, dtype=np.float64),
        anchors=anchors,
    )
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/unit/test_arviz_store.py tests/unit/test_arviz_api_probe.py -q`

Expected: `7 passed`. **These four store tests were never executed while this plan was written**: no local Python 3.14 environment had h5netcdf or h5py. The xarray and arviz-stats calls they rest on ran in the probes of Task 1. If a store test fails, first check whether the round-trip probe passes too. A failure in both points at the netCDF layer, not at this module.

- [ ] **Step 5: Gates**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
uv run pytest -q -p no:cacheprovider -m "not slow"
```

Expected: gates clean. Non-slow suite: Task 6's count **+ 4 passed**.

- [ ] **Step 6: Commit**

```bash
git add src/logging_employment/models/arviz_io.py tests/unit/test_arviz_store.py
git commit -m "feat(models): the ArviZ-shaped draws store and the R-hat/ESS statistics (§15.4, §11.14)"
```

---

### Task 8: §11.14's diagnostic gate, enforced in code

**Implements:**
- §11.14's six "Promotion requires" bullets, with Decision 5's two scopes;
- the roadmap Exit's "the §11.14 diagnostic gate is enforced in code, not documented";
- REQ-029's model half (fail closed on a model error), as `ModelDiagnosticsError`.

**Files:**
- Modify: `src/logging_employment/errors.py` (`ModelDiagnosticsError`)
- Create: `src/logging_employment/models/diagnostics.py`
- Test: `tests/unit/test_model_diagnostics.py` (new; 12 tests)

**Interfaces:**
- Consumes: Task 7's `rank_rhat` and `ess_bulk_tail`; Task 5's `ReconciledDraws` and `DrawCheck`; Task 2's `StateModelDiagnostics`; Task 3's `StateModelFit`.
- Produces:
  - `evaluate_gate(fit, draws, check, thresholds, *, scope: str) -> GateReport`, with `scope` in `SCOPES = ("production", "replicate")`.
  - `GateReport`: `scope`, `checks: tuple[GateCheck, ...]`, `cells_constant`, and the properties `passed` and `failures`. `to_json()` gives `{"scope", "passed", "failures", "cells_constant", "checks": {name: {"value", "threshold", "passed", "gating"}}}`, with NaN and inf written as null.
  - `GateCheck(name, value, threshold, passed, gating)`. The ten check names, in order: `divergences`, `parameter_rhat_max`, `parameter_ess_bulk_min`, `parameter_ess_tail_min`, `cell_rhat_max`, `cell_ess_bulk_min`, `cell_ess_tail_min`, `ppc_coverage_90`, `reconciliation_anchor_drift_max`, `reconciliation_bound_violations`. `ppc_coverage_90` reads the one-step-ahead check Task 4 computes (Decision 5).
  - `assert_gate_passes(report) -> None`, which raises `errors.ModelDiagnosticsError` naming every failed gating check.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_model_diagnostics.py`:

```python
"""§11.14's gate in code: what each check reads, which scope it gates, and how it fails."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from logging_employment.config import StateModelDiagnostics
from logging_employment.errors import ConceptViolationError, ModelDiagnosticsError
from logging_employment.models.diagnostics import assert_gate_passes, evaluate_gate
from logging_employment.models.reconciliation import DrawCheck, ReconciledDraws

SEED = sum(map(ord, "tests/model-diagnostics"))
CHAINS, PER_CHAIN = 4, 500
THRESHOLDS = StateModelDiagnostics()
PASSING = DrawCheck(
    draws_checked=CHAINS * PER_CHAIN,
    months_checked=1,
    cells_checked=4,
    max_anchor_drift=0.0,
    bound_violations=0,
    tolerance=1e-9,
)


def _draws(values: np.ndarray) -> ReconciledDraws:
    """Reconciled draws with only the fields the gate reads filled in meaningfully."""
    cells = values.shape[1]
    return ReconciledDraws(
        cell_ids=tuple(f"cell{j}" for j in range(cells)),
        state_fips=("01",) * cells,
        reference_months=("2023-01",) * cells,
        values=values,
        chain=np.repeat(np.arange(CHAINS), PER_CHAIN),
        draw=np.tile(np.arange(PER_CHAIN), CHAINS),
        raw_mean=values.mean(axis=0),
        lower=np.zeros(cells),
        upper=np.full(cells, math.inf),
        anchors={},
    )


def _mixed(shape: tuple[int, ...], offset_last_chain: float = 0.0) -> np.ndarray:
    values = np.random.default_rng(SEED).normal(size=shape)
    values[CHAINS - 1] += offset_last_chain
    return values


def _inputs(make_state_fit, *, stuck_parameter=0.0, stuck_cell=0.0, divergences=0, ppc=0.9):
    """A fit that mixes unless told otherwise, and its draws: three varying cells, one pinned."""
    cells = _mixed((CHAINS, PER_CHAIN, 3), stuck_cell).reshape(CHAINS * PER_CHAIN, 3) + 50.0
    values = np.column_stack([cells, np.full(CHAINS * PER_CHAIN, 20.0)])
    diverging = np.zeros((CHAINS, PER_CHAIN), dtype=bool)
    diverging.flat[:divergences] = True
    fit = make_state_fit(
        values,
        tuple(f"cell{j}" for j in range(4)),
        chains=CHAINS,
        parameters={
            "alpha": _mixed((CHAINS, PER_CHAIN), stuck_parameter),
            "u": _mixed((CHAINS, PER_CHAIN, 3)),
        },
        diverging=diverging,
        ppc_coverage_90=ppc,
    )
    return fit, _draws(values)


def test_a_well_mixed_fit_passes_the_production_gate(make_state_fit) -> None:
    fit, draws = _inputs(make_state_fit)
    report = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="production")
    assert report.passed, report.failures
    assert report.cells_constant == 1


def test_the_ess_floor_is_one_hundred_per_chain(make_state_fit) -> None:
    fit, draws = _inputs(make_state_fit)
    report = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="production")
    floors = {check.name: check.threshold for check in report.checks if "ess" in check.name}
    assert set(floors.values()) == {100.0 * CHAINS}


@pytest.mark.parametrize("scope", ["production", "replicate"])
def test_one_divergence_fails_either_scope(make_state_fit, scope: str) -> None:
    fit, draws = _inputs(make_state_fit, divergences=1)
    report = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope=scope)
    assert report.failures == ("divergences",)


@pytest.mark.parametrize("scope", ["production", "replicate"])
def test_a_stuck_chain_fails_parameter_rhat_in_either_scope(make_state_fit, scope: str) -> None:
    fit, draws = _inputs(make_state_fit, stuck_parameter=3.0)
    report = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope=scope)
    assert "parameter_rhat_max" in report.failures


def test_a_replicate_records_cell_diagnostics_without_gating_on_them(make_state_fit) -> None:
    """Decision 5: 27 replicates of ~1,400 cells would let one cell at 1.011 decide promotion."""
    fit, draws = _inputs(make_state_fit, stuck_cell=3.0)
    production = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="production")
    replicate = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="replicate")
    assert "cell_rhat_max" in production.failures
    assert replicate.passed
    cell_check = next(check for check in replicate.checks if check.name == "cell_rhat_max")
    assert (cell_check.passed, cell_check.gating) == (False, False)


def test_an_unmeasurable_check_fails_rather_than_passes(make_state_fit) -> None:
    """A NaN compares False: a predictive check that could not run is a failed check."""
    fit, draws = _inputs(make_state_fit, ppc=math.nan)
    report = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="production")
    assert report.failures == ("ppc_coverage_90",)


def test_a_reconciliation_failure_fails_production(make_state_fit) -> None:
    fit, draws = _inputs(make_state_fit)
    broken = DrawCheck(
        draws_checked=CHAINS * PER_CHAIN,
        months_checked=1,
        cells_checked=4,
        max_anchor_drift=1.0,
        bound_violations=2,
        tolerance=1e-9,
    )
    report = evaluate_gate(fit, draws, broken, THRESHOLDS, scope="production")
    assert report.failures == (
        "reconciliation_anchor_drift_max",
        "reconciliation_bound_violations",
    )


def test_the_report_is_json_without_nan(make_state_fit) -> None:
    fit, draws = _inputs(make_state_fit, ppc=math.nan)
    report = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="production")
    payload = json.loads(json.dumps(report.to_json(), allow_nan=False))
    assert payload["checks"]["ppc_coverage_90"]["value"] is None
    assert payload["passed"] is False


def test_a_failed_gate_raises_naming_each_failed_check(make_state_fit) -> None:
    fit, draws = _inputs(make_state_fit, divergences=2)
    report = evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="production")
    with pytest.raises(ModelDiagnosticsError, match="divergences=2.0"):
        assert_gate_passes(report)


def test_an_unknown_scope_is_refused(make_state_fit) -> None:
    fit, draws = _inputs(make_state_fit)
    with pytest.raises(ConceptViolationError, match="scope"):
        evaluate_gate(fit, draws, PASSING, THRESHOLDS, scope="exploratory")
```

- [ ] **Step 2: Run them to watch them fail**

Run: `uv run pytest tests/unit/test_model_diagnostics.py -q`

Expected (observed):

```text
E   ImportError: cannot import name 'ModelDiagnosticsError' from 'logging_employment.errors' (…/src/logging_employment/errors.py)
ERROR tests/unit/test_model_diagnostics.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error
```

- [ ] **Step 3: Add the error**

`src/logging_employment/errors.py`:

```diff
diff --git a/src/logging_employment/errors.py b/src/logging_employment/errors.py
index 309cb5c..26560f5 100644
--- a/src/logging_employment/errors.py
+++ b/src/logging_employment/errors.py
@@ -165,3 +165,14 @@ class ConstraintDataError(LoggingEmploymentError):
     emitted would describe a system that is wrong about the data. Distinct from
     `BoundViolationError`, which is an ESTIMATE outside its interval; this is the TRUTH outside it.
     """
+
+
+class ModelDiagnosticsError(LoggingEmploymentError):
+    """A production fit failed §11.14's diagnostic gate, so its draws may not be released.
+
+    Raised by `models/diagnostics.py::assert_gate_passes` AFTER `posterior/diagnostics.json` is
+    written, so the evidence of the failure survives the refusal. §11.14 lists what "Promotion
+    requires", and the roadmap's Stage 5 exit asks that the gate be "enforced in code, not
+    documented". This is the enforcement. Replicate fits inside §13's harness record their gate and
+    do not raise; §13.10's convergence gate reads those reports instead.
+    """
```

- [ ] **Step 4: Write the gate**

`src/logging_employment/models/diagnostics.py`:

```python
"""§11.14's diagnostic gate, enforced in code rather than documented.

§11.14 lists what "Promotion requires": no unresolved divergent transitions, R-hat at or below
1.01 for monitored parameters, adequate ESS for all release-relevant summaries, stable posterior
summaries across independent seeds and chains, posterior predictive checks on observed cells, and
exact reconciliation checks after draw transformation. The roadmap's exit criterion is that the
gate is "enforced in code, not documented". `fit-state-model` writes this report and then raises
`ModelDiagnosticsError` on a failure, and `validate-state-model` will not score a model whose
production fit failed.

WHAT EACH CHECK READS (plan 16, Decision 5):

* `divergences`: post-warmup `diverging`, summed over chains.
* `parameter_rhat_max`: rank R-hat, max over every element of `state_total.MONITORED`.
* `cell_rhat_max`: rank R-hat over the reconciled draws of every imputed cell. This is how
  "stable posterior summaries across independent seeds/chains" is enforced. Each chain starts from
  its own key, so between-chain agreement on the released quantity is the stability §11.14 asks
  for.
* `cell_ess_bulk_min` / `cell_ess_tail_min`: min over imputed cells, against
  `min_ess_per_chain * chains`. §11.14 scopes ESS to "release-relevant summaries", and the released
  summaries are these cells' intervals. Parameter ESS is recorded and does not gate.
* `ppc_coverage_90`: `StateModelFit.ppc_coverage_90` against `min_ppc_coverage_90`. It is
  one-step-ahead coverage of the training cells (`state_total._ppc_coverage_90`), because an exact
  training cell's in-sample replicate is the cell itself.
* `reconciliation_*`: `DrawCheck`, which is INV-012 measured on the draws.

A cell whose reconciled draws never vary has no R-hat and no ESS. That happens in a month whose
missing set is one cell, pinned to R_t in every draw. It is counted in `cells_constant` and left
out, because the constraints determine it and nothing sampled it. A NaN from any other quantity
fails its check, because a comparison with NaN is false and an unverifiable check must not pass.

TWO SCOPES. `production` gates on every check above. `replicate` covers the 27 fits §13's harness
runs, which record everything but gate only on divergences and parameter R-hat. Over about 1,400
quantities and 27 fits, a max-statistic gate would let one cell at R-hat 1.011 in one replicate
decide promotion. `validate/promotion.py` reads the replicate reports for §13.10's convergence
gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import StateModelDiagnostics
from ..errors import ConceptViolationError, ModelDiagnosticsError
from .arviz_io import ess_bulk_tail, rank_rhat
from .interfaces import StateModelFit
from .reconciliation import DrawCheck, ReconciledDraws

SCOPES: tuple[str, ...] = ("production", "replicate")


def _plain(value: float) -> float | None:
    """A float JSON can carry: NaN and the infinities become None rather than a non-JSON token."""
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class GateCheck:
    """One §11.14 check: what was measured, against what, and whether it gates this scope."""

    name: str
    value: float
    threshold: float
    passed: bool
    gating: bool


@dataclass(frozen=True)
class GateReport:
    """Every §11.14 check for one fit, and the verdict of the ones that gate its scope."""

    scope: str
    checks: tuple[GateCheck, ...]
    cells_constant: int

    @property
    def passed(self) -> bool:
        """True when every gating check passed; recorded-only checks cannot fail a fit."""
        return all(check.passed for check in self.checks if check.gating)

    @property
    def failures(self) -> tuple[str, ...]:
        """The names of the gating checks that failed, in check order."""
        return tuple(check.name for check in self.checks if check.gating and not check.passed)

    def to_json(self) -> dict[str, object]:
        """The report as `diagnostics.json` and the harness manifest record it."""
        return {
            "scope": self.scope,
            "passed": self.passed,
            "failures": list(self.failures),
            "cells_constant": self.cells_constant,
            "checks": {
                check.name: {
                    "value": _plain(float(check.value)),
                    "threshold": float(check.threshold),
                    "passed": check.passed,
                    "gating": check.gating,
                }
                for check in self.checks
            },
        }


def _parameter_statistics(fit: StateModelFit) -> tuple[float, float, float]:
    """Max rank R-hat and min bulk and tail ESS over every monitored parameter element."""
    rhats, bulks, tails = [], [], []
    for values in fit.parameters.values():
        per_element = np.asarray(values, dtype=np.float64).reshape(
            values.shape[0], values.shape[1], -1
        )
        rhats.append(rank_rhat(per_element).ravel())
        bulk, tail = ess_bulk_tail(per_element)
        bulks.append(bulk.ravel())
        tails.append(tail.ravel())
    if not rhats:
        return math.nan, math.nan, math.nan
    return (
        float(np.max(np.concatenate(rhats))),
        float(np.min(np.concatenate(bulks))),
        float(np.min(np.concatenate(tails))),
    )


def evaluate_gate(
    fit: StateModelFit,
    draws: ReconciledDraws,
    check: DrawCheck,
    thresholds: StateModelDiagnostics,
    *,
    scope: str,
) -> GateReport:
    """§11.14 for one fit and its reconciled draws, gating according to `scope`."""
    if scope not in SCOPES:
        raise ConceptViolationError(f"gate scope {scope!r} is not one of {list(SCOPES)}")
    production = scope == "production"
    chains = fit.chains
    per_chain = draws.values.shape[0] // chains
    parameter_rhat, parameter_bulk, parameter_tail = _parameter_statistics(fit)
    varying = np.ptp(draws.values, axis=0) > 0.0
    if varying.any():
        cells = draws.values[:, varying].reshape(chains, per_chain, -1)
        cell_rhat = float(np.max(rank_rhat(cells)))
        bulk, tail = ess_bulk_tail(cells)
        cell_bulk, cell_tail = float(np.min(bulk)), float(np.min(tail))
    else:
        # Every imputed cell is pinned by the constraints: nothing released was sampled.
        cell_rhat, cell_bulk, cell_tail = 1.0, math.inf, math.inf
    floor = float(thresholds.min_ess_per_chain * chains)
    checks = (
        GateCheck(
            "divergences",
            float(fit.divergences),
            float(thresholds.max_divergences),
            fit.divergences <= thresholds.max_divergences,
            True,
        ),
        GateCheck(
            "parameter_rhat_max",
            parameter_rhat,
            thresholds.max_rhat,
            parameter_rhat <= thresholds.max_rhat,
            True,
        ),
        GateCheck("parameter_ess_bulk_min", parameter_bulk, floor, parameter_bulk >= floor, False),
        GateCheck("parameter_ess_tail_min", parameter_tail, floor, parameter_tail >= floor, False),
        GateCheck(
            "cell_rhat_max",
            cell_rhat,
            thresholds.max_rhat,
            cell_rhat <= thresholds.max_rhat,
            production,
        ),
        GateCheck("cell_ess_bulk_min", cell_bulk, floor, cell_bulk >= floor, production),
        GateCheck("cell_ess_tail_min", cell_tail, floor, cell_tail >= floor, production),
        GateCheck(
            "ppc_coverage_90",
            fit.ppc_coverage_90,
            thresholds.min_ppc_coverage_90,
            fit.ppc_coverage_90 >= thresholds.min_ppc_coverage_90,
            production,
        ),
        GateCheck(
            "reconciliation_anchor_drift_max",
            check.max_anchor_drift,
            check.tolerance,
            check.max_anchor_drift <= check.tolerance,
            production,
        ),
        GateCheck(
            "reconciliation_bound_violations",
            float(check.bound_violations),
            0.0,
            check.bound_violations == 0,
            production,
        ),
    )
    return GateReport(scope=scope, checks=checks, cells_constant=int(np.count_nonzero(~varying)))


def assert_gate_passes(report: GateReport) -> None:
    """Raise `ModelDiagnosticsError` naming every gating check that failed."""
    if not report.passed:
        detail = "; ".join(
            f"{check.name}={check.value} against {check.threshold}"
            for check in report.checks
            if check.gating and not check.passed
        )
        raise ModelDiagnosticsError(f"§11.14's {report.scope} gate failed: {detail}")
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `uv run pytest tests/unit/test_model_diagnostics.py -q`

Expected: `12 passed` (observed: `12 passed`).

- [ ] **Step 6: Gates**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
uv run pytest -q -p no:cacheprovider -m "not slow"
```

Expected: gates clean. Non-slow suite: Task 7's count **+ 12 passed**.

- [ ] **Step 7: Commit**

```bash
git add src/logging_employment/errors.py src/logging_employment/models/diagnostics.py tests/unit/test_model_diagnostics.py
git commit -m "feat(models): §11.14's diagnostic gate in code, production and replicate scopes"
```

---

### Task 9: `fit-state-model`, and `reconcile` re-checking the stored draws

**Implements:**
- §16.1's `fit-state-model`, one of the five commands `--help` lacks. It writes a machine-readable manifest and is idempotent for the same inputs.
- §17.4 row 6, "reconcile posterior draws", through the CLI on the committed fixture. Row 5's synthetic-data fit is Task 10's.
- REQ-018 and INV-012 re-measured on the PERSISTED draws: `reconcile` reads the store back and checks the draws and their digest.
- REQ-029 and §18.3 for a model failure, as Decision 9 states it. A failed gate exits 1, keeps `posterior/diagnostics.json`, writes nothing else, and leaves no earlier success behind it.

**Files:**
- Modify: `src/logging_employment/cli.py` (`fit_state_model_command`, new; `reconcile_command`, extended)
- Modify: `tests/integration/conftest.py` (`build_staged_repo` with config overrides; `make_staged_repo`)
- Test: `tests/integration/test_cli_state_model.py` (new; 8 tests, 5 marked `slow`)

**Interfaces:**
- Consumes: every earlier `models/` task. From Task 3, `STORE_PATH = "posterior/state_total_draws.nc"`, `MODEL_ID` and `MODEL_VERSION`. From Task 7, `write_store`, `read_store`, `store_digest` and `draws_digest`. From Task 8, `evaluate_gate` and `assert_gate_passes`. Also `baselines.runner.state_total_bounds`, and `cli._write_manifest`, `cli._input_digests`.
- Produces, under `runs/<run_id>/`:
  - `posterior/diagnostics.json`: `GateReport.to_json()` plus `run_id`. It is written on every fit, passing or not. Task 13 reads `passed` from it.
  - `posterior/state_total_draws.nc`: the store. It is written only when the gate passes.
  - `posterior_summary.parquet`: §7.11. It is written only when the gate passes.
  - `state_model_manifest.json`: `run_id`, `model_id`, `model_version`, `constraint_set_hash`, `sampler`, `draws_sha256`, `store`, `posterior_summary_sha256`, `posterior_summary_schema`, `anchor_bases`, `training_cells`, `predicted_cells`, `ppc_coverage_90`, `ppc_cells`, `posterior_medians`, and `reconciliation`, plus `code_commit` and `uv_lock_sha256` from `_write_manifest`.
  - `reconcile_manifest.json` gains `state_model`, with `draws_checked`, `months_checked`, `cells_checked`, `max_anchor_drift`, `bound_violations`, `draws_sha256_matches` and `passed`, when a store exists. The key is absent otherwise.
  - The fixtures:
    - `build_staged_repo(tmp_path, *, overrides=None) -> StagedRepo`;
    - `make_staged_repo`, a module-scoped fixture returning `(overrides=None) -> StagedRepo`;
    - `staged_repo`, unchanged for every earlier test.

The CLI tests shrink the sampler to 2 chains of 60 draws after 60 warmup. They loosen the gate so it passes whatever a fit that size measures. These tests are about what the commands write, refuse and reproduce. Convergence belongs to Task 10 and to the D1 fit.

- [ ] **Step 1: Give the integration fixture config overrides, and write the failing tests**

`tests/integration/conftest.py`:

```diff
diff --git a/tests/integration/conftest.py b/tests/integration/conftest.py
index 5f9bbb2..6f269ac 100644
--- a/tests/integration/conftest.py
+++ b/tests/integration/conftest.py
@@ -12,6 +12,7 @@ and is not frozen in any useful sense.
 
 from __future__ import annotations
 
+from collections.abc import Callable, Mapping
 from dataclasses import dataclass
 from pathlib import Path
 
@@ -43,9 +44,16 @@ class StagedRepo:
     run_dir: Path
 
 
-@pytest.fixture()
-def staged_repo(tmp_path: Path) -> StagedRepo:
-    """A tmp repo holding the frozen Stage 3 staged layer and a built constraint system."""
+def build_staged_repo(
+    tmp_path: Path, *, overrides: Mapping[str, Mapping[str, object]] | None = None
+) -> StagedRepo:
+    """A tmp repo holding the frozen Stage 3 staged layer and a built constraint system.
+
+    `overrides` merges into the shipped config one block at a time, and one level deeper for a
+    nested mapping, so a test can shrink the sampler (`{"model": {"draws": 60}}`) without restating
+    a block. Plan 16 made this a function so its model tests can ask for a small sampler and a
+    loose gate. `staged_repo` is the no-override call every earlier test makes.
+    """
     staged = tmp_path / "staged"
     staged.mkdir()
     for name in STAGED_TABLES:
@@ -57,6 +65,12 @@ def staged_repo(tmp_path: Path) -> StagedRepo:
     raw["storage"]["raw_uri"] = str(tmp_path / "raw")
     raw["storage"]["output_uri"] = str(tmp_path / "runs")
     raw["storage"]["constraints_uri"] = str(tmp_path / "constraints")
+    for block, values in (overrides or {}).items():
+        for key, value in values.items():
+            if isinstance(value, Mapping):
+                raw[block][key] = {**raw[block][key], **value}
+            else:
+                raw[block][key] = value
     config_path = tmp_path / "config.yaml"
     config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
 
@@ -70,3 +84,24 @@ def staged_repo(tmp_path: Path) -> StagedRepo:
     return StagedRepo(
         config_path=config_path, run_dir=run_dir(cfg, run_id(cfg, _input_digests(cfg)))
     )
+
+
+@pytest.fixture()
+def staged_repo(tmp_path: Path) -> StagedRepo:
+    """`build_staged_repo` with the shipped config unchanged."""
+    return build_staged_repo(tmp_path)
+
+
+@pytest.fixture(scope="module")
+def make_staged_repo(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., StagedRepo]:
+    """`build_staged_repo` in a fresh directory per call, for tests that override the config.
+
+    A fixture rather than an import, so a test module never imports a conftest by path. It is
+    module-scoped so a module-scoped fixture can build its repository once and share it; each call
+    still gets its own directory from `tmp_path_factory`.
+    """
+
+    def _build(overrides: Mapping[str, Mapping[str, object]] | None = None) -> StagedRepo:
+        return build_staged_repo(tmp_path_factory.mktemp("repo"), overrides=overrides)
+
+    return _build
```

`tests/integration/test_cli_state_model.py`:

```python
"""`fit-state-model`, and `reconcile`'s check of the stored draws, on the committed fixture.

§17.4 row 6, "reconcile posterior draws", through the CLI; row 5's fit "on synthetic data" is
`test_state_total_recovery.py`'s. A two-chain sampler of 60 draws and a gate loosened to pass
whatever it measures: these tests are about what the commands write, refuse and reproduce, not about
convergence, which the recovery test and the D1 run own.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from logging_employment.cli import app
from logging_employment.models.arviz_io import draws_digest, read_store
from logging_employment.models.interfaces import STORE_PATH
from logging_employment.models.reconciliation import check_reconciled

REPO = Path(__file__).resolve().parents[2]
FIXTURE_MONTHLY = REPO / "tests" / "fixtures" / "baselines" / "qcew_monthly.parquet"
SMALL_SAMPLER = {"chains": 2, "warmup": 60, "draws": 60}
LOOSE_GATE = {
    "diagnostics": {
        "max_rhat": 100.0,
        "min_ess_per_chain": 1,
        "max_divergences": 1_000_000,
        "min_ppc_coverage_90": 0.0,
    }
}
# No R-hat is below 0.5, so this gate fails every fit, deterministically.
FAILING_GATE = {"diagnostics": {"max_rhat": 0.5}}


def _invoke(command: str, config: Path):
    return CliRunner().invoke(app, [command, "--config", str(config)])


@pytest.fixture()
def fitted(make_staged_repo):
    repo = make_staged_repo({"model": {**SMALL_SAMPLER, **LOOSE_GATE}})
    result = _invoke("fit-state-model", repo.config_path)
    assert result.exit_code == 0, result.output
    return repo


@pytest.mark.slow
def test_the_fit_writes_its_store_its_summary_and_its_manifest(fitted) -> None:
    run = fitted.run_dir
    states = pl.read_parquet(FIXTURE_MONTHLY).filter(pl.col("area_type") == "state")
    summary = pl.read_parquet(run / "posterior_summary.parquet")
    assert summary.height == states.height
    imputed = int((summary["observed_or_imputed"] == "imputed").sum())
    assert imputed == int((states["observation_status"] == "suppressed").sum())
    manifest = json.loads((run / "state_model_manifest.json").read_text())
    written = hashlib.sha256((run / "posterior_summary.parquet").read_bytes()).hexdigest()
    assert manifest["posterior_summary_sha256"] == written
    assert manifest["draws_sha256"] == draws_digest(read_store(run / STORE_PATH))
    assert manifest["anchor_bases"] == ["declared_national_total"]
    assert manifest["code_commit"]
    report = json.loads((run / "posterior" / "diagnostics.json").read_text())
    assert (report["scope"], report["passed"]) == ("production", True)


@pytest.mark.slow
def test_every_stored_draw_satisfies_its_hard_constraints(fitted) -> None:
    """INV-012 on the persisted draws, after transformation, never on a summary of them."""
    check = check_reconciled(read_store(fitted.run_dir / STORE_PATH), tolerance=1e-9)
    assert check.passed
    assert check.draws_checked == 120


@pytest.mark.slow
def test_the_fit_is_idempotent(fitted) -> None:
    """§16.1: same inputs, same draws, same summary bytes."""
    manifest = fitted.run_dir / "state_model_manifest.json"
    first = json.loads(manifest.read_text())
    result = _invoke("fit-state-model", fitted.config_path)
    assert result.exit_code == 0, result.output
    second = json.loads(manifest.read_text())
    for key in ("draws_sha256", "posterior_summary_sha256"):
        assert second[key] == first[key]


@pytest.mark.slow
def test_reconcile_re_verifies_the_draws_on_disk(fitted) -> None:
    for command in ("run-baselines", "reconcile"):
        result = _invoke(command, fitted.config_path)
        assert result.exit_code == 0, result.output
    state_model = json.loads((fitted.run_dir / "reconcile_manifest.json").read_text())[
        "state_model"
    ]
    assert state_model["passed"] is True
    assert state_model["draws_sha256_matches"] is True
    assert state_model["bound_violations"] == 0


@pytest.mark.slow
def test_a_failed_gate_exits_1_and_leaves_only_its_report(make_staged_repo) -> None:
    """Stale artifacts are planted first: a failed re-fit must not leave an old success behind."""
    repo = make_staged_repo({"model": {**SMALL_SAMPLER, **FAILING_GATE}})
    run = repo.run_dir
    stale = (run / STORE_PATH, run / "posterior_summary.parquet", run / "state_model_manifest.json")
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("from an earlier fit")
    result = _invoke("fit-state-model", repo.config_path)
    assert result.exit_code == 1
    report = json.loads((run / "posterior" / "diagnostics.json").read_text())
    assert report["passed"] is False
    assert "parameter_rhat_max" in report["failures"]
    assert not any(path.exists() for path in stale)


def test_the_fit_requires_solved_bounds(make_staged_repo) -> None:
    repo = make_staged_repo({"model": SMALL_SAMPLER})
    (repo.run_dir / "deterministic_bounds.parquet").unlink()
    result = _invoke("fit-state-model", repo.config_path)
    assert result.exit_code != 0
    # Short tokens only, as `test_baseline_cli.py` explains: Typer boxes and hard-wraps the message.
    assert "solve-bounds" in result.output
    assert "INV-012" in result.output


def test_a_baseline_only_reconcile_writes_the_keys_it_always_wrote(staged_repo) -> None:
    """No store, no `state_model` key: omitted rather than null, as `run_id`'s optional keys are."""
    for command in ("run-baselines", "reconcile"):
        result = _invoke(command, staged_repo.config_path)
        assert result.exit_code == 0, result.output
    manifest = json.loads((staged_repo.run_dir / "reconcile_manifest.json").read_text())
    assert "state_model" not in manifest


def test_the_cli_starts_without_a_ppl() -> None:
    """The import discipline, kept: `--help` and every baseline command start without JAX.

    `cli.py` imports each `models/` module inside the command that uses it. A subprocess, because
    this test process may already hold `jax` from another test, which would make the check vacuous.
    """
    ppl = ("jax", "numpyro", "arviz_base", "arviz_stats", "xarray", "h5netcdf")
    probe = f"import sys, logging_employment.cli; print(sorted(set({ppl!r}) & set(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
```

- [ ] **Step 2: Run them to watch them fail**

Run: `uv run pytest tests/integration/test_cli_state_model.py -q`

Expected (observed):

```text
E         │ No such command 'fit-state-model'.                                           │
E         ╰──────────────────────────────────────────────────────────────────────────────╯
E       assert 2 == 0
E        +  where 2 = <Result SystemExit(2)>.exit_code
E       assert 2 == 1
E       assert 'solve-bounds' in "Usage: root [OPTIONS] COMMAND [ARGS]...\nTry 'root --help' for help.\n╭─ Error ──────────────────────────────────────...                                 │\n╰──────────────────────────────────────────────────────────────────────────────╯\n"
E        +  where "Usage: root [OPTIONS] COMMAND [ARGS]...\nTry 'root --help' for help.\n╭─ Error ──────────────────────────────────────...                                 │\n╰──────────────────────────────────────────────────────────────────────────────╯\n" = <Result SystemExit(2)>.output
FAILED tests/integration/test_cli_state_model.py::test_a_failed_gate_exits_1_and_leaves_only_its_report
FAILED tests/integration/test_cli_state_model.py::test_the_fit_requires_solved_bounds
ERROR tests/integration/test_cli_state_model.py::test_the_fit_writes_its_store_its_summary_and_its_manifest
ERROR tests/integration/test_cli_state_model.py::test_every_stored_draw_satisfies_its_hard_constraints
ERROR tests/integration/test_cli_state_model.py::test_the_fit_is_idempotent
ERROR tests/integration/test_cli_state_model.py::test_reconcile_re_verifies_the_draws_on_disk
2 failed, 2 passed, 4 errors
```

Two tests pass here, and that is their job: they are guards on behaviour Step 3 must not change. `test_a_baseline_only_reconcile_writes_the_keys_it_always_wrote` guards `reconcile`'s existing manifest. `test_the_cli_starts_without_a_ppl` guards the Global Constraints' import discipline, which Step 3 is the first to put at risk.

- [ ] **Step 3: Add the command, and `reconcile`'s second check**

`src/logging_employment/cli.py`:

```diff
diff --git a/src/logging_employment/cli.py b/src/logging_employment/cli.py
index 07d380f..d42d359 100644
--- a/src/logging_employment/cli.py
+++ b/src/logging_employment/cli.py
@@ -382,6 +382,114 @@ def run_baselines_command(
             typer.echo(f"declined {estimator_id} {kind} {count}")
 
 
+@app.command("fit-state-model")
+def fit_state_model_command(
+    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
+) -> None:
+    """Fit §11's state-total model, reconcile every draw, gate it (§11.14) and summarise it (§7.11).
+
+    Preconditions are `build-constraints` and `solve-bounds`: every draw is reconciled into §9's
+    per-cell interval, so a run without `deterministic_bounds.parquet` has nothing to hold the draws
+    to. A RE-FIT REPLACES, NEVER MERGES. The previous fit's artifacts are deleted before sampling,
+    so a failed gate cannot leave an earlier success's summary beside its own failed report.
+    `run_id` does not cover source code, so a re-fit with new code lands in the same directory.
+
+    `posterior/diagnostics.json` is written BEFORE the gate is enforced, so a failure keeps its
+    evidence. The command then exits 1 and writes no store, no summary and no manifest.
+    `validate-state-model` reads the report and records the model as not beaten without scoring it.
+    """
+    import json
+    import shutil
+
+    import polars as pl
+
+    from .baselines.runner import state_total_bounds
+    from .build import write_parquet_deterministic
+    from .contracts import POSTERIOR_SUMMARY_SCHEMA, HarmonizedData, schema_fingerprint
+    from .errors import ModelDiagnosticsError
+    from .models.arviz_io import write_store
+    from .models.data import build_model_data
+    from .models.diagnostics import assert_gate_passes, evaluate_gate
+    from .models.interfaces import MODEL_ID, MODEL_VERSION, STORE_PATH, StateModelConfig
+    from .models.reconciliation import check_reconciled, reconcile_fit
+    from .models.state_total import fit_state_total_model
+    from .models.summary import posterior_summary
+    from .runs import run_dir, run_id
+
+    cfg = load_config(config)
+    data = HarmonizedData.load(Path(cfg.storage.staged_uri))
+    rid = run_id(cfg, _input_digests(cfg))
+    run = run_dir(cfg, rid)
+    manifest_path = run / "schema_manifest.json"
+    bounds_path = run / "deterministic_bounds.parquet"
+    for path, command in ((manifest_path, "build-constraints"), (bounds_path, "solve-bounds")):
+        if not path.exists():
+            raise typer.BadParameter(
+                f"{path} is missing: every draw is reconciled into this run's deterministic "
+                f"bounds (INV-012), and no run matches the staged inputs. Run `{command}` first"
+            )
+    posterior = run / "posterior"
+    shutil.rmtree(posterior, ignore_errors=True)
+    for stale in (run / "posterior_summary.parquet", run / "state_model_manifest.json"):
+        stale.unlink(missing_ok=True)
+    posterior.mkdir(parents=True)
+
+    constraint_set_hash = json.loads(manifest_path.read_text())["constraint_set_hash"]
+    bounds = pl.read_parquet(bounds_path)
+    monthly = data.qcew_monthly
+    model_data = build_model_data(monthly)
+    fit = fit_state_total_model(model_data, StateModelConfig.from_config(cfg.model))
+    draws = reconcile_fit(fit, monthly, state_total_bounds(bounds), cfg)
+    check = check_reconciled(draws, tolerance=cfg.reconciliation.tolerance)
+    report = evaluate_gate(fit, draws, check, cfg.model.diagnostics, scope="production")
+    _write_manifest(posterior / "diagnostics.json", {**report.to_json(), "run_id": rid})
+    try:
+        assert_gate_passes(report)
+    except ModelDiagnosticsError as error:
+        typer.echo(str(error), err=True)
+        raise typer.Exit(code=1) from error
+
+    attrs = {
+        "run_id": rid,
+        "model_id": MODEL_ID,
+        "model_version": MODEL_VERSION,
+        "constraint_set_hash": constraint_set_hash,
+    }
+    digest = write_store(run / STORE_PATH, draws, fit, model_data, attrs=attrs)
+    summary = posterior_summary(
+        draws, monthly, bounds, run_id=rid, constraint_set_hash=constraint_set_hash
+    )
+    _write_manifest(
+        run / "state_model_manifest.json",
+        {
+            **attrs,
+            "sampler": fit.sampler,
+            "draws_sha256": digest,
+            "store": STORE_PATH,
+            "posterior_summary_sha256": write_parquet_deterministic(
+                summary, run / "posterior_summary.parquet"
+            ),
+            "posterior_summary_schema": schema_fingerprint(POSTERIOR_SUMMARY_SCHEMA),
+            # §12.2 (since `D-120`): every row allocated against R_t carries its anchor basis.
+            # §7.11's table has no such column, so the fit's manifest and the store carry it.
+            "anchor_bases": sorted({anchor.anchor_basis for anchor in draws.anchors.values()}),
+            "training_cells": int(model_data.train_y.size),
+            "predicted_cells": len(model_data.predict_cell_ids),
+            "ppc_coverage_90": fit.ppc_coverage_90,
+            "ppc_cells": fit.ppc_cells,
+            "posterior_medians": fit.posterior_medians,
+            "reconciliation": {
+                "draws_checked": check.draws_checked,
+                "months_checked": check.months_checked,
+                "max_anchor_drift": check.max_anchor_drift,
+                "bound_violations": check.bound_violations,
+                "tolerance": check.tolerance,
+            },
+        },
+    )
+    typer.echo(f"fit {fit.chains} chains; gate passed; {len(draws.cell_ids)} cells reconciled")
+
+
 @app.command("reconcile")
 def reconcile_command(
     config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
@@ -419,19 +527,46 @@ def reconcile_command(
     # §16.1: "Every command MUST write a machine-readable manifest and MUST be idempotent for the
     # same inputs." A verifier that only echoes leaves nothing for §18.1 to reproduce against, so
     # the verdict and the digest of what was checked are persisted beside the results.
-    _write_manifest(
-        run / "reconcile_manifest.json",
-        {
-            "checked_pairs": drift.height,
-            "max_residual_drift": worst,
-            "tolerance": cfg.reconciliation.tolerance,
-            "within_tolerance": within_tolerance,
-            "baseline_results_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
-        },
-    )
+    payload: dict[str, object] = {
+        "checked_pairs": drift.height,
+        "max_residual_drift": worst,
+        "tolerance": cfg.reconciliation.tolerance,
+        "within_tolerance": within_tolerance,
+        "baseline_results_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
+    }
+    # Plan 16: INV-012 re-measured on the PERSISTED draws once `fit-state-model` has written them.
+    # `fit-state-model` checked the draws it held in memory; this reads the file back, so a defect
+    # in writing it cannot pass unseen. The key is OMITTED, never null, before a fit exists, so a
+    # baseline-only run writes the keys it always wrote.
+    from .models.interfaces import STORE_PATH
+
+    state_model_passed = True
+    store = run / STORE_PATH
+    if store.exists():
+        from .models.arviz_io import draws_digest, read_store, store_digest
+        from .models.reconciliation import check_reconciled
+
+        draws = read_store(store)
+        check = check_reconciled(draws, tolerance=cfg.reconciliation.tolerance)
+        digest_matches = draws_digest(draws) == store_digest(store)
+        state_model_passed = check.passed and digest_matches
+        payload["state_model"] = {
+            "draws_checked": check.draws_checked,
+            "months_checked": check.months_checked,
+            "cells_checked": check.cells_checked,
+            "max_anchor_drift": check.max_anchor_drift,
+            "bound_violations": check.bound_violations,
+            "draws_sha256_matches": digest_matches,
+            "passed": state_model_passed,
+        }
+        typer.echo(
+            f"state-total draws: {check.draws_checked} x {check.cells_checked}, max anchor drift "
+            f"{check.max_anchor_drift:.3e}, {check.bound_violations} bound violation(s)"
+        )
+    _write_manifest(run / "reconcile_manifest.json", payload)
     typer.echo(f"checked {drift.height} (estimator, month) pairs")
     typer.echo(f"max residual drift {worst:.3e}")
-    if not within_tolerance:
+    if not within_tolerance or not state_model_passed:
         raise typer.Exit(code=1)
 
 
```

The `models` imports sit inside each command, as the Global Constraints require, and `test_the_cli_starts_without_a_ppl` holds them there.

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/integration/test_cli_state_model.py -q`

Expected: `8 passed`. **Four of the eight were never executed while this plan was written**: the four that take the `fitted` fixture, whose fit writes a store (Task 7's caveat). The other four were observed passing (`4 passed, 4 deselected`, the four deselected). If only the four fail, re-run Task 7's store tests first.

- [ ] **Step 5: See the command in `--help`**

Run: `uv run logging-estimates --help`

Expected: `fit-state-model` is listed beside the existing commands.

- [ ] **Step 6: Gates**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
uv run pytest -q -p no:cacheprovider -m "not slow"
```

Expected: gates clean. Non-slow suite: Task 8's count **+ 3 passed**. The five `slow` tests ran in Step 4.

- [ ] **Step 7: Commit**

```bash
git add src/logging_employment/cli.py tests/integration/conftest.py tests/integration/test_cli_state_model.py
git commit -m "feat(cli): fit-state-model, and reconcile re-checks the stored draws (INV-012)"
```

---

### Task 10: §17.5's state-total recovery rows, on a panel simulated at D1's regime

**Implements:**
- §17.4 row 5, "fit a reduced Bayesian model on synthetic data";
- §17.5's five state-total rows (state and region effects, through each state's level; seasonality; AR persistence; heavy-tailed shocks), with "parameter and predictive recovery within tolerances appropriate to sample size". The other four rows (size compositions, top-class tails, TPO/FIA measurements, missingness) belong to the size and harvest models of later stages.

**The panel is D1's regime, not an easier one** (Decision 15). This plan's first recovery test used a friendlier panel: every state trained, and observation noise was comparable to the innovations (σ_y 0.05 against σ_η 0.08). It passed there, and it said nothing about D1, where the model then failed. So the test reproduces what makes D1 hard:
- Training cells are exact, and innovation scales are D1's: σ_η 0.035, dispersed across states by τ 0.8.
- x moves between states and barely within one. It is constant within a quarter and steps by sd 0.03 at quarter boundaries. Its within-state slope is −1.2 and its between-state slope +0.4, the signs D1's posterior has.
- Suppression takes whole quarters. Two states have no training cell, two keep only two or three quarters, and the rest lose runs of one to four quarters.
- The panel has 18 states in 6 divisions over 96 months, two persistence groups (0.9 and 0.6, both under the cap) and three injected 8-sigma shocks.

Each assertion's docstring gives the arithmetic its bound comes from, so a reviewer can argue with a number rather than a feeling.

**There is no red step.** Every function this test calls exists after Task 4. It checks that the model Task 4 wrote recovers what it was simulated from, and a red run would need a broken model. Its falsifiability was measured, not argued:
- With one β for both slopes, which was the model before Decision 15's split, two of these tests failed. In `test_untrained_state_levels_fall_inside_their_wide_intervals`, an untrained state's true level sat 2.5 log points from its posterior mean, outside its 95% interval. In `test_hidden_cells_are_predicted_within_their_intervals`, 90% coverage was 0.74 over all hidden cells and 0.50 in the untrained states.
- A model with one shared persistence cannot separate the two groups by 0.1 (`test_ar_persistence_separates_the_two_groups`).

**Files:**
- Test: `tests/integration/test_state_total_recovery.py` (new; 10 tests, the module marked `slow`)

**Interfaces:**
- Consumes: Task 3's `ModelData`, `StateModelConfig` and `StateModelFit`; Task 4's `fit_state_total_model`; Task 7's `rank_rhat`; `config.ModelConfig`.
- Produces: nothing later tasks import.

- [ ] **Step 1: Write the test**

`tests/integration/test_state_total_recovery.py`:

```python
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
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/integration/test_state_total_recovery.py -q`

Expected: `10 passed`: one fit of 4 chains of 500 warmup and 500 draws. It was observed passing on the older local environment (`10 passed`). When observed, the module took 34 s, almost all of it the fit: 31 leapfrog steps a draw, no tree at the depth ceiling, no divergence, and one-step coverage 0.914 over its 1,287 training cells. The first version of this test, on the friendlier panel and §11's original model, took 391 s with every tree at the ceiling.

If one assertion fails, **do not widen its bound to pass.** Read the docstring's arithmetic, decide whether the bound or the model is wrong, and stop and report to your human partner with the observed value. A recovery test loosened until it passes is a smoke test with extra steps.

- [ ] **Step 3: Gates**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
uv run pytest -q -p no:cacheprovider -m "not slow"
```

Expected: gates clean. Non-slow suite: **unchanged from Task 9**, because the whole module is `slow`. The ten ran in Step 2.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_state_total_recovery.py
git commit -m "test(models): §17.5's state-total recovery rows on a panel at D1's regime"
```

---

### Task 11: The harness scores the model through its own loop (the producer seam)

**Implements:**
- §13's pseudo-suppression harness applied to the model, which is the input §13.10's gate reads (Task 12). The model is scored by `run_pseudo_suppression`'s own loop, never a copy of it. That loop owns the masks, the leakage guard, the masked bounds (`D-087`), §13.2 step 6's rejection, the primary-like precondition, and every §13.5–§13.8 emitter that scored the Stage 4 comparand.
- INV-014 and REQ-024 (applied): "beats the transparent baselines" means under the same harness, the same masks and the same metrics.
- Decision 11: the harness's point estimate is the mean of the reconciled draws.
- Decision 12: the model's intervals come from its own reconciled draws, and `interval_source` becomes an enforced closed set.
- §12.6's integers, released by the baselines' own rule. `release_integers` is extracted from `run_baselines` verbatim, so both paths integerize through one function.

**The baseline path must not move by one byte.** `validate`'s output is the §13.10 comparand, and Task 15 byte-compares a re-run of it. Every change here is either a new parameter defaulting to today's behaviour or a verbatim extraction:
- `estimators=None` means `REGISTRY`, as before;
- `BaselineProducer` is the old loop body;
- `Production.notes` is written only when non-empty;
- `release_integers` is the old block.

The float goldens are the check: `test_validation_golden.py` and `test_baseline_golden.py` pass unchanged against their committed fixtures. The only edit to `test_validation_golden.py` is the `INTERVAL_SOURCES` pin, which gains the model's value, and the assertion that no BASELINE row carries it.

**Files:**
- Modify: `src/logging_employment/baselines/runner.py` (`release_integers`, extracted; `Anchor` imported)
- Modify: `src/logging_employment/contracts.py` (`INTERVAL_SOURCES` gains `reconciled_posterior_draws`; `assert_declared_provenance` checks `interval_source`)
- Modify: `src/logging_employment/validate/harness.py` (`Production`, `Producer`, `BaselineProducer`; `run_pseudo_suppression(..., producer=None)`)
- Modify: `src/logging_employment/validate/metrics.py` (`draw_interval_metrics`; `_division_coverage_rows`, shared by both interval sources)
- Modify: `src/logging_employment/validate/intervals.py` (module docstring: the rolling version is still not built)
- Create: `src/logging_employment/models/validation.py` (`model_results`, `draw_ensembles`, `StateModelProducer`)
- Test: `tests/unit/test_validate_producer_seam.py` (new; 5), `tests/unit/test_validate_metrics_draws.py` (new; 4), `tests/unit/test_model_validation.py` (new; 3), `tests/unit/test_contracts_validation.py` (+2), `tests/integration/test_validation_golden.py` (one pin updated; count unchanged)

**Interfaces:**
- Consumes: Task 5's `reconcile_fit`, `check_reconciled`, `exact_column_means` and `ReconciledDraws`; Task 7's `draws_digest`; Task 8's `evaluate_gate`; Task 4's `fit_state_total_model`; Task 3's `build_model_data`, `MODEL_ID` and `StateModelConfig`.
- Produces:
  - `validate.harness.Production(results, interval_metrics, notes={})`;
  - `validate.harness.Producer`, a Protocol with `estimator_ids` and `__call__(masked, system, config) -> Production`;
  - `validate.harness.BaselineProducer(estimators)`;
  - `run_pseudo_suppression(data, estimators=None, config=None, *, producer=None)`. Passing both `estimators` and `producer` raises `ConceptViolationError`. Each regime's manifest entry gains `producer_notes: [{seed, **notes}, ...]` only when a producer returns notes.
  - `validate.metrics.draw_interval_metrics(scores, *, regime, seed, arm, ensembles: Mapping[str, np.ndarray]) -> pl.DataFrame`;
  - `baselines.runner.release_integers(allocated, anchor, bounds, *, cell_ids, estimator_id, config) -> dict[str, int | None]`;
  - `models.validation.model_results(draws, bounds, config, *, constraint_set_hash=None) -> pl.DataFrame` (`BASELINE_RESULT_SCHEMA`), `draw_ensembles(draws) -> dict[str, np.ndarray]`, and `StateModelProducer()`. Its notes are `{"gate": GateReport.to_json(), "draws_sha256": ...}`, and Task 12 reads `gate`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_validate_producer_seam.py`:

```python
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
```

`tests/unit/test_validate_metrics_draws.py`:

```python
"""`metrics.draw_interval_metrics`: §13.7's rows from a model's own reconciled draws."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from logging_employment.errors import ConceptViolationError
from logging_employment.validate.intervals import crps
from logging_employment.validate.metrics import draw_interval_metrics

MODEL = "state_total_model"


def _scored(estimates: list[float | None], truths: list[float], states: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "estimator_id": [MODEL] * len(truths),
            "cell_id": [f"c{index}" for index in range(len(truths))],
            "state_fips": states,
            "estimate": estimates,
            "truth": truths,
        },
        schema={
            "estimator_id": pl.String,
            "cell_id": pl.String,
            "state_fips": pl.String,
            "estimate": pl.Float64,
            "truth": pl.Float64,
        },
    )


def _rows(frame: pl.DataFrame, name: str, kind: str = "overall") -> pl.DataFrame:
    return frame.filter((pl.col("metric_name") == name) & (pl.col("stratum_kind") == kind))


def test_each_cell_is_scored_against_its_own_draws() -> None:
    """Cell 0's draws cover its truth and cell 1's miss: coverage 1/2, and a hand-computed CRPS."""
    ensembles = {"c0": np.arange(1.0, 101.0), "c1": np.arange(1.0, 101.0)}
    frame = draw_interval_metrics(
        _scored([50.5, 50.5], [50.0, 500.0], ["01", "06"]),
        regime="regional_blocks",
        seed=1024,
        arm="state_total",
        ensembles=ensembles,
    )
    coverage = _rows(frame, "coverage_0.90").row(0, named=True)
    assert coverage["value"] == 0.5
    assert coverage["calibration_sample_size"] == 2
    assert coverage["interval_source"] == "reconciled_posterior_draws"
    expected = (crps(ensembles["c0"], 50.0) + crps(ensembles["c1"], 500.0)) / 2
    assert _rows(frame, "crps")["value"].item() == pytest.approx(expected, rel=1e-12)
    # Every draw already lies inside its bounds, so nothing is clipped at zero.
    assert _rows(frame, "n_clipped_at_zero")["value"].item() == 0.0


def test_divisions_get_their_own_coverage_rows() -> None:
    frame = draw_interval_metrics(
        _scored([50.5, 50.5], [50.0, 500.0], ["01", "06"]),
        regime="regional_blocks",
        seed=1024,
        arm="state_total",
        ensembles={"c0": np.arange(1.0, 101.0), "c1": np.arange(1.0, 101.0)},
    )
    divisions = _rows(frame, "coverage_0.90", "census_division").sort("stratum_value")
    assert divisions["stratum_value"].to_list() == ["east_south_central", "pacific"]
    assert divisions["value"].to_list() == [1.0, 0.0]
    assert divisions["calibration_sample_size"].to_list() == [1, 1]


def test_a_scored_cell_with_no_draws_is_refused() -> None:
    with pytest.raises(ConceptViolationError, match="carry no draws"):
        draw_interval_metrics(
            _scored([50.5, 50.5], [50.0, 500.0], ["01", "06"]),
            regime="regional_blocks",
            seed=1024,
            arm="state_total",
            ensembles={"c0": np.arange(1.0, 101.0)},
        )


def test_an_estimator_that_scored_nothing_writes_one_null_row() -> None:
    frame = draw_interval_metrics(
        _scored([None], [50.0], ["01"]),
        regime="regional_blocks",
        seed=1024,
        arm="state_total",
        ensembles={},
    )
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert (row["metric_name"], row["value"], row["interval_source"]) == (
        "coverage_0.90",
        None,
        "none",
    )
```

`tests/unit/test_model_validation.py`:

```python
"""`models/validation.py::model_results`: reconciled draws as `baseline_results` rows."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from logging_employment.contracts import BASELINE_RESULT_SCHEMA
from logging_employment.models.data import state_cell_ids
from logging_employment.models.reconciliation import reconcile_fit
from logging_employment.models.validation import model_results
from logging_employment.reconcile.scaling import Bounds

SEED = sum(map(ord, "tests/model-validation"))


def _results(harmonized_toy, appendix_a_config, make_state_fit) -> tuple[pl.DataFrame, np.ndarray]:
    monthly = harmonized_toy.qcew_monthly
    suppressed = monthly.filter(
        (pl.col("area_type") == "state") & (pl.col("observation_status") == "suppressed")
    ).sort("reference_month", "state_fips")
    cells = state_cell_ids(suppressed)
    bounds = Bounds(lower=dict.fromkeys(cells, 0.0), upper=dict.fromkeys(cells))
    fit = make_state_fit(np.random.default_rng(SEED).gamma(2.0, 20.0, size=(8, 3)), cells)
    draws = reconcile_fit(fit, monthly, bounds, appendix_a_config)
    return model_results(draws, bounds, appendix_a_config), draws.values


def test_the_estimate_is_the_mean_of_the_reconciled_draws(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    results, values = _results(harmonized_toy, appendix_a_config, make_state_fit)
    assert list(results.schema.items()) == list(BASELINE_RESULT_SCHEMA.items())
    np.testing.assert_allclose(
        results["estimate"].to_numpy(), values.mean(axis=0), rtol=1e-12, atol=0.0
    )


def test_the_mean_adds_up_exactly_as_every_draw_does(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    """Why the mean and not the median: `constraint_metrics` reads adding-up off these rows."""
    results, _values = _results(harmonized_toy, appendix_a_config, make_state_fit)
    for (month,), group in results.group_by("reference_month"):
        residual = group["residual"][0]
        assert math.isclose(math.fsum(group["estimate"].to_list()), residual, abs_tol=1e-9)
        assert sum(group["estimate_integer"].to_list()) == round(residual), month


def test_every_row_declares_the_models_provenance(
    harmonized_toy, appendix_a_config, make_state_fit
) -> None:
    results, _values = _results(harmonized_toy, appendix_a_config, make_state_fit)
    assert set(results["estimator_id"]) == {"state_total_model"}
    assert set(results["weight_basis"]) == {"own_estimator"}
    assert set(results["anchor_basis"]) == {"declared_national_total"}
    assert set(results["reconciliation_status"]) == {"anchored_and_reconciled"}
    assert results["decline_reason"].null_count() == results.height
```

`tests/unit/test_contracts_validation.py` and `tests/integration/test_validation_golden.py`:

```diff
diff --git a/tests/integration/test_validation_golden.py b/tests/integration/test_validation_golden.py
index 9f1c93b..06bd98e 100644
--- a/tests/integration/test_validation_golden.py
+++ b/tests/integration/test_validation_golden.py
@@ -126,17 +126,24 @@ def test_the_interval_source_names_leave_one_out_rather_than_rolling(fixture_run
     pool is every OTHER scored residual in the same (regime, seed, arm, estimator) group, with no
     time ordering and no window. Nothing rolls. The old value `rolling_residual_ensemble` promised
     §13.10's coverage gate a time-ordered interval that the code never computed, and the gate
-    cannot tell the difference: `interval_source` is outside `assert_declared_provenance`'s six
-    closed-set checks, so `INTERVAL_SOURCES` is checked by nothing at runtime and this test IS the check.
+    cannot tell the difference. Since plan 16, `assert_declared_provenance` refuses a value outside
+    `INTERVAL_SOURCES` at runtime. Plan 16 also added the model's own source,
+    `reconciled_posterior_draws`, which no baseline row may carry, and this test pins that.
 
     Scope is the label. A time-ordered rolling interval is explicitly NOT built here
-    (`specs/completed/stage5-preconditions.md` §4).
+    (`specs/completed/stage5-preconditions.md` §4), and plan 16 did not build one either.
     """
-    assert INTERVAL_SOURCES == ("leave_one_out_residual_ensemble", "none")
+    assert INTERVAL_SOURCES == (
+        "leave_one_out_residual_ensemble",
+        "reconciled_posterior_draws",
+        "none",
+    )
     golden = pl.read_parquet(GOLDEN)
     for frame, origin in ((golden, "golden"), (fixture_run.metrics, "produced")):
         sources = set(frame["interval_source"].drop_nulls().to_list())
-        assert sources <= set(INTERVAL_SOURCES), f"{origin} carries {sorted(sources)}"
+        assert sources <= {"leave_one_out_residual_ensemble", "none"}, (
+            f"{origin} carries {sorted(sources)}"
+        )
         assert "leave_one_out_residual_ensemble" in sources, origin
 
 
diff --git a/tests/unit/test_contracts_validation.py b/tests/unit/test_contracts_validation.py
index ac73913..bf0ec86 100644
--- a/tests/unit/test_contracts_validation.py
+++ b/tests/unit/test_contracts_validation.py
@@ -139,7 +139,8 @@ def test_a_stratum_kind_outside_the_declared_set_is_refused():
     """removing `("stratum_kind", STRATUM_KINDS)` from the provenance loop must redden this.
 
     `validate/harness.py` runs the gate over the assembled metrics frame; without a refusal test
-    the entry could be deleted with the suite green, which is `INTERVAL_SOURCES`' situation.
+    the entry could be deleted with the suite green, which was `INTERVAL_SOURCES`' situation until
+    plan 16 gave it one (below).
     """
     frame = pl.DataFrame({"stratum_kind": ["overall", "by_state"]})
     with pytest.raises(ConceptViolationError, match="stratum_kind"):
@@ -150,3 +151,16 @@ def test_every_declared_stratum_kind_passes():
     contracts.assert_declared_provenance(
         pl.DataFrame({"stratum_kind": list(contracts.STRATUM_KINDS)})
     )
+
+
+def test_an_interval_source_outside_the_declared_set_is_refused():
+    """Plan 16: the harness gates its assembled metrics, so a producer cannot invent a source."""
+    frame = pl.DataFrame({"interval_source": ["reconciled_posterior_draws", "rolling_window"]})
+    with pytest.raises(ConceptViolationError, match="interval_source"):
+        contracts.assert_declared_provenance(frame)
+
+
+def test_every_declared_interval_source_passes():
+    contracts.assert_declared_provenance(
+        pl.DataFrame({"interval_source": list(contracts.INTERVAL_SOURCES)})
+    )
```

- [ ] **Step 2: Run them to watch them fail**

Run: `uv run pytest tests/unit/test_validate_producer_seam.py tests/unit/test_validate_metrics_draws.py tests/unit/test_model_validation.py tests/unit/test_contracts_validation.py tests/integration/test_validation_golden.py -q`

Expected (observed):

```text
E   ImportError: cannot import name 'draw_interval_metrics' from 'logging_employment.validate.metrics' (…/src/logging_employment/validate/metrics.py)
E   ModuleNotFoundError: No module named 'logging_employment.models.validation'
E   ImportError: cannot import name 'BaselineProducer' from 'logging_employment.validate.harness' (…/src/logging_employment/validate/harness.py)
ERROR tests/unit/test_validate_metrics_draws.py
ERROR tests/unit/test_model_validation.py
ERROR tests/unit/test_validate_producer_seam.py
!!!!!!!!!!!!!!!!!!! Interrupted: 3 errors during collection !!!!!!!!!!!!!!!!!!!!
3 errors
```

- [ ] **Step 3: Extract `release_integers`, verbatim**

`src/logging_employment/baselines/runner.py`:

```diff
diff --git a/src/logging_employment/baselines/runner.py b/src/logging_employment/baselines/runner.py
index 16390f3..c2f1176 100644
--- a/src/logging_employment/baselines/runner.py
+++ b/src/logging_employment/baselines/runner.py
@@ -51,6 +51,7 @@ from ..errors import (
 )
 from ..reconcile.allocate import allocate
 from ..reconcile.anchor import (
+    Anchor,
     Partition,
     assert_universe_closes,
     closure_audit,
@@ -510,51 +511,14 @@ def run_baselines(
                     tolerance=config.reconciliation.tolerance,
                     quantity="estimate",
                 )
-            # The margin the integers must honour is the anchor's residual -- the published
-            # quantity being allocated -- so it is taken from the anchor rather than re-derived
-            # from the allocation. `allocate` and `scale_into_bounds` both make the values sum to
-            # R_t, so the two candidates cannot disagree; naming the anchor is the honest source.
-            integer_total = round(anchor.residual)
-            integers = (
-                integerize(
-                    allocated,
-                    total=integer_total,
-                    **(
-                        {}
-                        if scoped is None
-                        else integer_bounds(
-                            scoped, tolerance=config.constraints.feasibility_tolerance
-                        )
-                    ),
-                )
-                if config.reconciliation.integerize_release
-                else dict.fromkeys(allocated, None)
+            integers = release_integers(
+                allocated,
+                anchor,
+                bounds,
+                cell_ids=ids,
+                estimator_id=estimator.estimator_id,
+                config=config,
             )
-            if config.reconciliation.integerize_release and (
-                sum(integers.values()) != integer_total
-            ):
-                # §12.6 step 5, as defence in depth rather than an independent derivation:
-                # `integerize` already raises when it cannot place every unit, so this catches a
-                # regression in that contract. A bare `assert` would vanish under `python -O`.
-                raise InfeasibleResidualError(
-                    f"{month}: integerized estimates for {estimator.estimator_id} sum to "
-                    f"{sum(integers.values())}, not the required {integer_total}"
-                )
-            if bounds is not None and config.reconciliation.integerize_release:
-                # §12.6's integers are released too. `integerize` is now cut to the month's bounds,
-                # so this is defence in depth: checking only the float would leave the number
-                # actually published unchecked, which is the half INV-002 names. The tolerance is
-                # the one `integer_bounds` cut with -- the solver's -- so the check admits exactly
-                # what that cut was allowed to produce.
-                assert_within_bounds(
-                    {cell: float(value) for cell, value in integers.items() if value is not None},
-                    bounds,
-                    cell_ids=ids,
-                    estimator_id=estimator.estimator_id,
-                    reference_month=month,
-                    tolerance=config.constraints.feasibility_tolerance,
-                    quantity="estimate_integer",
-                )
             for cell in anchor.missing_cells:
                 rows.append(
                     {
@@ -580,6 +544,69 @@ def run_baselines(
     return results, audit
 
 
+def release_integers(
+    allocated: dict[str, float],
+    anchor: Anchor,
+    bounds: Bounds | None,
+    *,
+    cell_ids: Mapping[str, str],
+    estimator_id: str,
+    config: Config,
+) -> dict[str, int | None]:
+    """§12.6's integers for one month's allocation, cut to its bounds and checked against them.
+
+    Every value is None when `reconciliation.integerize_release` is off. Extracted from
+    `run_baselines` by plan 16 so the model's harness rows (`models/validation.py`) are
+    integerized by the same cut, the same tiebreak and the same two checks as every baseline's,
+    rather than by a copy that could drift from them.
+
+    The margin the integers must honour is the anchor's residual -- the published quantity being
+    allocated -- so it is taken from the anchor rather than re-derived from the allocation.
+    `allocate` and `scale_into_bounds` both make the values sum to R_t, so the two candidates
+    cannot disagree; naming the anchor is the honest source.
+    """
+    if not config.reconciliation.integerize_release:
+        return dict.fromkeys(allocated, None)
+    month = anchor.reference_month
+    tolerance = config.constraints.feasibility_tolerance
+    scoped = (
+        None
+        if bounds is None
+        else month_bounds(bounds, cell_ids=cell_ids, cells=anchor.missing_cells)
+    )
+    integer_total = round(anchor.residual)
+    integers: dict[str, int | None] = dict(
+        integerize(
+            allocated,
+            total=integer_total,
+            **({} if scoped is None else integer_bounds(scoped, tolerance=tolerance)),
+        )
+    )
+    if sum(integers.values()) != integer_total:
+        # §12.6 step 5, as defence in depth rather than an independent derivation: `integerize`
+        # already raises when it cannot place every unit, so this catches a regression in that
+        # contract. A bare `assert` would vanish under `python -O`.
+        raise InfeasibleResidualError(
+            f"{month}: integerized estimates for {estimator_id} sum to "
+            f"{sum(integers.values())}, not the required {integer_total}"
+        )
+    if bounds is not None:
+        # §12.6's integers are released too. `integerize` is cut to the month's bounds, so this is
+        # defence in depth: checking only the float would leave the number actually published
+        # unchecked, which is the half INV-002 names. The tolerance is the one `integer_bounds` cut
+        # with -- the solver's -- so the check admits exactly what that cut was allowed to produce.
+        assert_within_bounds(
+            {cell: float(value) for cell, value in integers.items() if value is not None},
+            bounds,
+            cell_ids=cell_ids,
+            estimator_id=estimator_id,
+            reference_month=month,
+            tolerance=tolerance,
+            quantity="estimate_integer",
+        )
+    return integers
+
+
 def _decline_rows(
     estimator_id: str,
     anchor,
```

Run: `uv run pytest tests/integration/test_baseline_golden.py -q`

Expected: passes, unchanged. That is the proof the extraction moved nothing. If it fails, the extraction is wrong: fix it, never the golden.

- [ ] **Step 4: Declare and enforce the interval source**

`src/logging_employment/contracts.py`:

```diff
diff --git a/src/logging_employment/contracts.py b/src/logging_employment/contracts.py
index b40b9f3..0d9848e 100644
--- a/src/logging_employment/contracts.py
+++ b/src/logging_employment/contracts.py
@@ -235,9 +235,9 @@ DECLINE_KINDS: tuple[str, ...] = ("by_design", "data_gap", "reconciliation_failu
 def assert_declared_provenance(frame: pl.DataFrame) -> None:
     """Refuse a provenance value outside its declared tuple.
 
-    The eight tuples this checks (`RECONCILIATION_STATUSES`, `WEIGHT_BASES`, `ANCHOR_BASES`,
-    `DECLINE_KINDS`, `SUPPRESSION_TYPES`, `STRATUM_KINDS`, `MASK_ARMS`, `OBSERVED_OR_IMPUTED`) are
-    the closed sets a row's provenance may draw from, but
+    The nine tuples this checks (`RECONCILIATION_STATUSES`, `WEIGHT_BASES`, `ANCHOR_BASES`,
+    `DECLINE_KINDS`, `SUPPRESSION_TYPES`, `STRATUM_KINDS`, `MASK_ARMS`, `INTERVAL_SOURCES`,
+    `OBSERVED_OR_IMPUTED`) are the closed sets a row's provenance may draw from, but
     `BASELINE_RESULT_SCHEMA` checks dtypes only -- `pl.String` accepts any string. `weight_basis`
     is the live exposure: `run_baselines` copies it from an estimator's own `outcome.basis`, so a
     third-party estimator's typo reached `baseline_results.parquet` and passed every test. Nulls
@@ -251,14 +251,17 @@ def assert_declared_provenance(frame: pl.DataFrame) -> None:
         ("suppression_type", SUPPRESSION_TYPES),
         # R-S5G-1. Added with a CALLER: `validate/harness.py` runs this over the assembled metrics
         # frame, and `tests/unit/test_contracts_validation.py` refuses an undeclared value.
-        # `INTERVAL_SOURCES` is the cautionary case: its own comment in this module records it as
-        # enforced by nothing at runtime, and a second declared-but-unenforced set is what this
-        # avoids.
+        # `INTERVAL_SOURCES` was the cautionary case, declared and enforced by nothing at runtime,
+        # until plan 16 added it below.
         ("stratum_kind", STRATUM_KINDS),
         # D-082. Since plan 12 `mask_arm` is PRODUCED from `MaskTarget.arm` rather than written as
         # a literal at the emit sites, so its value comes from data. Both harness calls pass a
         # frame that carries it: the scored frame and the assembled metrics.
         ("mask_arm", MASK_ARMS),
+        # Plan 16, with a caller on day one: the harness gates its assembled metrics here, and
+        # §13.10's promotion record refuses a model coverage row from any source but the model's
+        # own draws. Until plan 16 this set was enforced by nothing at runtime.
+        ("interval_source", INTERVAL_SOURCES),
         # Plan 16. §7.11 names the column and gives no values; `models/summary.py` writes two.
         ("observed_or_imputed", OBSERVED_OR_IMPUTED),
     ):
@@ -509,19 +512,22 @@ MASK_ARMS: tuple[str, ...] = ("state_total", "national_size")
 STRATUM_KINDS: tuple[str, ...] = ("overall", "census_division")
 
 # What produced a probabilistic row's interval, named for what the code computes. Until R-S5P-7
-# this value was named for a ROLLING window that `validate/metrics.py` has never computed: it
+# the first value was named for a ROLLING window that `validate/metrics.py` has never computed: it
 # builds each ensemble from `np.delete(residual_pool, position)` — every OTHER scored residual in
 # the same (regime, seed, arm, estimator) group, leave-one-out by INDEX, no time ordering, no
 # window. (The superseded string is spelled out in the test named below, so a reader who greps for
 # it lands on the reason; it is deliberately not repeated in `src/`.)
-# §13.10's coverage gate reads these intervals and cannot tell a time-ordered interval from this
-# one, so the name is the only thing carrying the distinction. Unlike WEIGHT_BASES and its five
-# siblings above (STRATUM_KINDS joined them in R-S5G-1), this tuple is enforced by nothing at
-# runtime — `assert_declared_provenance` does not cover `interval_source` — so
-# `tests/integration/test_validation_golden.py` is its only check.
-# Renaming the value here does not build the rolling version; that is Stage 5's, per
-# `specs/completed/stage5-preconditions.md` §4.
-INTERVAL_SOURCES: tuple[str, ...] = ("leave_one_out_residual_ensemble", "none")
+# `reconciled_posterior_draws` is plan 16's: `metrics.draw_interval_metrics` reads each cell's
+# interval off the state-total model's own reconciled joint draws (§12.7), pooling nothing across
+# cells. §13.10's coverage gate reads both kinds and cannot tell one from another, so the name is
+# the only thing carrying the distinction, and since plan 16 `assert_declared_provenance` enforces
+# it. The time-ordered rolling version of §10.7 is still not built; plan 16's model intervals do not
+# need one, and the baselines' leave-one-out stays what it is.
+INTERVAL_SOURCES: tuple[str, ...] = (
+    "leave_one_out_residual_ensemble",
+    "reconciled_posterior_draws",
+    "none",
+)
 
 # One row per (regime, seed, replicate, estimator, cell): the raw scored observations, including
 # the ones that were declined. Distinct in grain from VALIDATION_METRIC_SCHEMA below, and
```

- [ ] **Step 5: Open the harness to a producer, and score intervals from draws**

`src/logging_employment/validate/harness.py`:

```diff
diff --git a/src/logging_employment/validate/harness.py b/src/logging_employment/validate/harness.py
index 7fb2334..d4e094e 100644
--- a/src/logging_employment/validate/harness.py
+++ b/src/logging_employment/validate/harness.py
@@ -21,8 +21,9 @@ shipped three-seed, full-registry run is 11:03.
 
 from __future__ import annotations
 
-from collections.abc import Sequence
-from dataclasses import dataclass
+from collections.abc import Callable, Mapping, Sequence
+from dataclasses import dataclass, field
+from typing import Protocol
 
 import polars as pl
 
@@ -62,22 +63,89 @@ class ValidationResult:
     manifest: dict[str, object]
 
 
+@dataclass(frozen=True)
+class Production:
+    """One replicate's rows to score, and how to score their intervals.
+
+    `results` has `BASELINE_RESULT_SCHEMA`'s shape: every missing cell of every month the masked
+    frame leaves missing, estimate or decline, exactly as `run_baselines` returns it.
+    `interval_metrics` is called on the scored rows with `regime`, `seed` and `arm` and returns
+    §13.7's rows. `notes` rides into the regime's manifest entry only when it is non-empty, so the
+    baseline path writes the manifest it wrote before this seam existed.
+    """
+
+    results: pl.DataFrame
+    interval_metrics: Callable[..., pl.DataFrame]
+    notes: Mapping[str, object] = field(default_factory=dict)
+
+
+class Producer(Protocol):
+    """Whatever turns one masked frame into scoreable rows: §10's registry, or a model.
+
+    Plan 16 added this seam so the state-total model is scored by THIS loop. The masks, the
+    leakage guard, the masked bounds (`D-087`), step 6's rejection, the primary-like precondition
+    and every §13.5-§13.8 emitter are then the ones the baselines were scored under, rather than a
+    second copy that could drift from them.
+    """
+
+    estimator_ids: tuple[str, ...]
+
+    def __call__(self, masked: HarmonizedData, system: MaskedSystem, config: Config) -> Production:
+        """Rows for one replicate, from the masked frame and the MASKED system's bounds."""
+        ...
+
+
+@dataclass(frozen=True)
+class BaselineProducer:
+    """§10's registry as a `Producer`: `run_baselines` under the masked bounds (`D-087`)."""
+
+    estimators: tuple[Estimator, ...]
+
+    @property
+    def estimator_ids(self) -> tuple[str, ...]:
+        """The ids in the order given, which is the manifest's `estimators` list."""
+        return tuple(estimator.estimator_id for estimator in self.estimators)
+
+    def __call__(self, masked: HarmonizedData, system: MaskedSystem, config: Config) -> Production:
+        """§10's rows, with §13.7's leave-one-out intervals for them."""
+        # D-087: the MASKED bounds, never the run directory's, which still contain the truth
+        # this replicate hid (§13.4). `run_baselines` scales an estimate a finite bound binds on
+        # back into it (§12.3) exactly as production does, so the scoreboard ranks the
+        # estimates a release would publish rather than ones production would have rescaled.
+        results, _audit = run_baselines(
+            masked, config, estimators=self.estimators, bounds=state_total_bounds(system.bounds)
+        )
+        return Production(results=results, interval_metrics=probabilistic_metrics)
+
+
 def run_pseudo_suppression(
     data: HarmonizedData,
-    estimators: Sequence[Estimator] = REGISTRY,
+    estimators: Sequence[Estimator] | None = None,
     config: Config | None = None,
+    *,
+    producer: Producer | None = None,
 ) -> ValidationResult:
     """Every enabled regime, every seed. Refuses rather than skipping.
 
     `config` is keyword-optional only so the §16.2 argument ORDER survives; it is required in
     fact. A bare `assert` would vanish under `python -O`, and this package raises typed errors for
     caller mistakes everywhere else.
+
+    `estimators` (default: §10's whole `REGISTRY`) and `producer` are two ways to name what is
+    scored, and passing both is refused rather than letting one silently win.
     """
     if config is None:
         raise ConceptViolationError(
             "run_pseudo_suppression requires a Config: the harness needs `config.constraints` for "
             "solve_bounds and the full object for run_baselines"
         )
+    if producer is not None and estimators is not None:
+        raise ConceptViolationError(
+            "run_pseudo_suppression takes estimators OR a producer, not both: a producer decides "
+            "what is scored, and an estimator list beside it would be silently ignored"
+        )
+    if producer is None:
+        producer = BaselineProducer(tuple(REGISTRY if estimators is None else estimators))
     all_scores: list[pl.DataFrame] = []
     all_metrics: list[pl.DataFrame] = []
     regimes: dict[str, dict[str, object]] = {}
@@ -89,7 +157,7 @@ def run_pseudo_suppression(
             "n_scored": 0,
             "replicates": 0,
             "hashes": [],
-            "estimators": [e.estimator_id for e in estimators],
+            "estimators": list(producer.estimator_ids),
             "rejected_exactly_recoverable": 0,
         }
         # THE SWITCH IS CHECKED FIRST, AND THE FAIL-CLOSED REFUSAL SECOND. Order matters: an
@@ -160,13 +228,10 @@ def run_pseudo_suppression(
             masked, truth = apply_mask(data, targets)
             assert_no_retained_truth(masked, truth)
             system = mask_and_solve(data, targets, config)
-            # D-087: the MASKED bounds, never the run directory's, which still contain the truth
-            # this replicate hid (§13.4). `run_baselines` scales an estimate a finite bound binds on
-            # back into it (§12.3) exactly as production does, so the scoreboard ranks the
-            # estimates a release would publish rather than ones production would have rescaled.
-            results, _audit = run_baselines(
-                masked, config, estimators=estimators, bounds=state_total_bounds(system.bounds)
-            )
+            production = producer(masked, system, config)
+            results = production.results
+            if production.notes:
+                entry.setdefault("producer_notes", []).append({"seed": seed, **production.notes})
             joined = _join_truth(
                 results, truth, system, targets, regime=name, seed=seed, replicate=replicate
             )
@@ -197,7 +262,7 @@ def run_pseudo_suppression(
             # estimates, and stay on `scored`.
             all_metrics.append(bound_metrics(joined, regime=name, seed=seed, arm=arm))
             all_metrics.append(decline_and_basis_report(scored, regime=name, seed=seed, arm=arm))
-            all_metrics.append(probabilistic_metrics(scored, regime=name, seed=seed, arm=arm))
+            all_metrics.append(production.interval_metrics(scored, regime=name, seed=seed, arm=arm))
             all_metrics.append(
                 constraint_metrics(
                     scored,
@@ -237,9 +302,9 @@ def run_pseudo_suppression(
         # thirty-five (0, 0) frames is (0, 0), and the shaped branch is skipped. The emptiness
         # that matters is the RESULT's, not the accumulator's.
         metrics = pl.DataFrame(schema=VALIDATION_METRIC_SCHEMA)
-    # The metrics frame gets the same closed-set gate the scores frame has had (R-S5G-1).
-    # Declaring `STRATUM_KINDS` without a caller would repeat `INTERVAL_SOURCES`, which its own
-    # comment in `contracts.py` records as enforced by nothing at runtime.
+    # The metrics frame gets the same closed-set gate the scores frame has had (R-S5G-1). Since
+    # plan 16 that includes `interval_source`, so a producer's interval rows cannot name a source
+    # `contracts.INTERVAL_SOURCES` does not declare.
     assert_declared_provenance(metrics)
     board = build_scoreboard(metrics)
     return ValidationResult(scores, metrics, board, manifest)
```

`src/logging_employment/validate/metrics.py`:

```diff
diff --git a/src/logging_employment/validate/metrics.py b/src/logging_employment/validate/metrics.py
index 53b6bdb..25cf08a 100644
--- a/src/logging_employment/validate/metrics.py
+++ b/src/logging_employment/validate/metrics.py
@@ -10,6 +10,7 @@ what separates the two, and it is on every row for that reason.
 from __future__ import annotations
 
 import math
+from collections.abc import Mapping
 
 import numpy as np
 import polars as pl
@@ -391,28 +392,121 @@ def probabilistic_metrics(
         # fewer than two scored cells) emits only the overall null row and NO division rows, by
         # design, so the two families do NOT always share strata: a §13.10 gate reading both must
         # OUTER-join them on the division, or it silently drops point-only estimators and interval
-        # families with fewer than two scored cells. A division
-        # whose masked cells all declined, or whose scored cells never reached a leave-one-out
-        # ensemble, has `seen == 0` and a NULL value, never 0.0. Each row carries ITS division's base:
-        # masked rows as `denominator`, scored rows as `n_scored`, ensembled rows as
-        # `calibration_sample_size`. Inheriting the estimator-wide `n_scored` from `common` would
-        # report more scored cells than the division has masked cells, an impossible state.
-        divisions = sorted(set(group["census_division"].to_list())) if stratify else []
-        for division in divisions:
-            in_division = group.filter(pl.col("census_division") == division)
-            seen, hits = by_division.get(division, (0, 0))
+        # families with fewer than two scored cells.
+        if stratify:
+            rows.extend(_division_coverage_rows(group, by_division, common))
+    return pl.DataFrame(rows)
+
+
+def _division_coverage_rows(
+    group: pl.DataFrame, by_division: Mapping[str, list[int]], common: Mapping[str, object]
+) -> list[dict[str, object]]:
+    """One `coverage_0.90` row per division `group` has masked cells in, from the loop's tally.
+
+    A division whose masked cells all declined, or whose scored cells never reached an interval,
+    has `seen == 0` and a NULL value, never 0.0. Each row carries ITS division's base: masked rows
+    as `denominator`, scored rows as `n_scored`, interval-bearing rows as
+    `calibration_sample_size`. Inheriting the estimator-wide `n_scored` from `common` would report
+    more scored cells than the division has masked cells, an impossible state. Shared by both
+    interval sources so the two cannot stratify differently (plan 16).
+    """
+    rows: list[dict[str, object]] = []
+    for division in sorted(set(group["census_division"].to_list())):
+        in_division = group.filter(pl.col("census_division") == division)
+        seen, hits = by_division.get(division, (0, 0))
+        rows.append(
+            {
+                **common,
+                "stratum_kind": "census_division",
+                "stratum_value": division,
+                "metric_name": "coverage_0.90",
+                "value": hits / seen if seen else None,
+                "denominator": float(in_division.height),
+                "n_scored": in_division.filter(pl.col("estimate").is_not_null()).height,
+                "calibration_sample_size": seen,
+            }
+        )
+    return rows
+
+
+def draw_interval_metrics(
+    scores: pl.DataFrame,
+    *,
+    regime: str,
+    seed: int,
+    arm: str,
+    ensembles: Mapping[str, np.ndarray],
+) -> pl.DataFrame:
+    """§13.7's coverage, width and CRPS from a model's own predictive draws, keyed by `cell_id`.
+
+    The rows `probabilistic_metrics` writes, over the same levels and strata and through the same
+    `_division_coverage_rows`, with `interval_source = 'reconciled_posterior_draws'`. Each scored
+    cell's interval and CRPS come from its OWN reconciled joint draws (§12.7), so nothing is pooled
+    across cells: there is no leave-one-out to take and no minimum sample below which a cell goes
+    unscored. Nothing is clipped either, because every draw already lies inside the cell's
+    deterministic interval and no lower bound is negative. `n_clipped_at_zero` is written as 0.0
+    so both interval sources carry the same metric set.
+
+    A scored cell with no draws is refused rather than skipped: skipping would score the model on
+    the cells it happened to cover.
+    """
+    rows: list[dict[str, object]] = []
+    stratify = arm == "state_total"
+    frame = with_census_division(scores) if stratify else scores
+    for (estimator,), group in frame.group_by("estimator_id", maintain_order=True):
+        scored = group.filter(pl.col("estimate").is_not_null())
+        missing = sorted(set(scored["cell_id"].to_list()) - set(ensembles))
+        if missing:
+            raise ConceptViolationError(
+                f"{regime} seed {seed}: {len(missing)} scored cell(s) of {estimator} carry no "
+                f"draws, first {missing[:5]}; an interval metric over the rest would score the "
+                "model on the cells it happened to cover"
+            )
+        common = {
+            **_OVERALL,
+            "regime": regime,
+            "seed": seed,
+            "mask_arm": arm,
+            "estimator_id": str(estimator),
+            "metric_family": "probabilistic",
+            "denominator": float(group.height),
+            "denominator_basis": "masked_cell_rows",
+            "n_scored": scored.height,
+            "interval_source": "reconciled_posterior_draws" if scored.height else "none",
+            "calibration_sample_size": scored.height,
+        }
+        if scored.is_empty():
+            rows.append({**common, "metric_name": "coverage_0.90", "value": None})
+            continue
+        covered = dict.fromkeys(_LEVELS, 0)
+        widths: list[float] = []
+        crps_values: list[float] = []
+        by_division: dict[str, list[int]] = {}
+        for row in scored.iter_rows(named=True):
+            ensemble = np.asarray(ensembles[row["cell_id"]], dtype=float)
+            for level in _LEVELS:
+                lo, hi = empirical_interval(ensemble, level)
+                hit = lo <= row["truth"] <= hi
+                covered[level] += int(hit)
+                if level == 0.90:
+                    widths.append(hi - lo)
+                    if stratify:
+                        tally = by_division.setdefault(row["census_division"], [0, 0])
+                        tally[0] += 1
+                        tally[1] += int(hit)
+            crps_values.append(crps(ensemble, row["truth"]))
+        n = len(crps_values)
+        for level in _LEVELS:
             rows.append(
-                {
-                    **common,
-                    "stratum_kind": "census_division",
-                    "stratum_value": division,
-                    "metric_name": "coverage_0.90",
-                    "value": hits / seen if seen else None,
-                    "denominator": float(in_division.height),
-                    "n_scored": in_division.filter(pl.col("estimate").is_not_null()).height,
-                    "calibration_sample_size": seen,
-                }
+                {**common, "metric_name": f"coverage_{level:.2f}", "value": covered[level] / n}
             )
+        rows.append(
+            {**common, "metric_name": "mean_interval_width_0.90", "value": float(np.mean(widths))}
+        )
+        rows.append({**common, "metric_name": "crps", "value": float(np.mean(crps_values))})
+        rows.append({**common, "metric_name": "n_clipped_at_zero", "value": 0.0})
+        if stratify:
+            rows.extend(_division_coverage_rows(group, by_division, common))
     return pl.DataFrame(rows)
 
 
```

`src/logging_employment/validate/intervals.py`:

```diff
diff --git a/src/logging_employment/validate/intervals.py b/src/logging_employment/validate/intervals.py
index 648be65..9515940 100644
--- a/src/logging_employment/validate/intervals.py
+++ b/src/logging_employment/validate/intervals.py
@@ -6,8 +6,10 @@ compute that: `metrics.probabilistic_metrics` pools every OTHER scored residual
 cross-sectional leave-one-out within a replicate — no time ordering, no window. That gap is
 R-S5P-7, and `contracts.INTERVAL_SOURCES` now NAMES it rather than repeating the spec's word;
 the spec's word is quoted here so the divergence stays visible to a reader of this module and
-is not mistaken for a docstring that drifted. Building the time-ordered version is Stage 5's
-(`specs/completed/stage5-preconditions.md` §4), not this module's.
+is not mistaken for a docstring that drifted. Plan 16 did NOT build the time-ordered version:
+the state-total model's intervals come from its own reconciled draws
+(`metrics.draw_interval_metrics`), and §13.10 reads the baselines' intervals as this module
+computes them, under their own name.
 
 ONE object: the residual-shifted ensemble. Both the quantiles and the CRPS are derived from it, so
 an interval and a score can never disagree about the same predictive distribution.
```

- [ ] **Step 6: Write the model's producer**

`src/logging_employment/models/validation.py`:

```python
"""§11's state-total model as a `validate.harness.Producer`, so §13 scores it like a baseline.

ONE LOOP SCORES BOTH. `run_pseudo_suppression(data, config=config, producer=StateModelProducer())`
runs the model through the masks, the leakage guard, the masked bounds (`D-087`), §13.2 step 6's
rejection, the primary-like precondition and every §13.5-§13.8 emitter that scored the Stage 4
comparand. The only model-specific parts are the ones below: how a replicate's rows are produced,
and where their intervals come from.

PER REPLICATE (§13.2 steps 3-5 on the model's side):

1. `build_model_data(masked.qcew_monthly)`. The hidden targets are `suppressed` with a null value
   in the masked frame, so the model predicts them and never trains on them (§11.3, §13.4).
2. `fit_state_total_model`, with the production config's sampler settings and seed.
3. `reconcile_fit` against the MASKED system's bounds, then `check_reconciled`.
4. `evaluate_gate(..., scope="replicate")`. The report rides into the manifest's `producer_notes`
   for `validate/promotion.py` and fails nothing here (plan 16, Decision 5).

THE POINT ESTIMATE IS THE MEAN OF THE RECONCILED DRAWS, not their median. Every draw sums to its
month's residual and lies inside its cell's interval, so their mean does too, by linearity and
convexity. `constraint_metrics` then reads the model's adding-up at machine epsilon, as it reads
every baseline's. A per-cell median has neither property, and its adding-up residual would be
scored as the model violating a constraint it never broke. `posterior_summary` still publishes
both (§7.11).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import polars as pl

from ..baselines.runner import assert_within_bounds, release_integers, state_total_bounds
from ..config import Config
from ..contracts import BASELINE_RESULT_SCHEMA, HarmonizedData, assert_declared_provenance
from ..reconcile.scaling import Bounds
from ..validate.harness import Production
from ..validate.metrics import draw_interval_metrics
from ..validate.recover import MaskedSystem
from .arviz_io import draws_digest
from .data import build_model_data
from .diagnostics import evaluate_gate
from .interfaces import MODEL_ID, StateModelConfig
from .reconciliation import ReconciledDraws, check_reconciled, exact_column_means, reconcile_fit
from .state_total import fit_state_total_model


def model_results(
    draws: ReconciledDraws,
    bounds: Bounds,
    config: Config,
    *,
    constraint_set_hash: str | None = None,
) -> pl.DataFrame:
    """`BASELINE_RESULT_SCHEMA` rows for every reconciled cell, estimate = the draws' mean.

    `raw_weight` is the posterior-mean §11.5 score, the model's analogue of a baseline's raw
    weight. The integers come from `baselines.runner.release_integers`, the baselines' own cut, so
    both sources release integers by one rule. `constraint_set_hash` defaults to None because the
    harness path's `run_baselines` writes None there too (`contracts.VALIDATION_SCORE_SCHEMA`).
    """
    means = exact_column_means(draws.values)
    months = np.asarray(draws.reference_months)
    rows: list[dict[str, object]] = []
    for month, anchor in sorted(draws.anchors.items()):
        columns = np.flatnonzero(months == month)
        ids = {draws.state_fips[j]: draws.cell_ids[j] for j in columns}
        estimates = {draws.state_fips[j]: float(means[j]) for j in columns}
        assert_within_bounds(
            estimates,
            bounds,
            cell_ids=ids,
            estimator_id=MODEL_ID,
            reference_month=month,
            tolerance=config.reconciliation.tolerance,
            quantity="estimate",
        )
        integers = release_integers(
            estimates, anchor, bounds, cell_ids=ids, estimator_id=MODEL_ID, config=config
        )
        for j in columns:
            state = draws.state_fips[j]
            rows.append(
                {
                    "estimator_id": MODEL_ID,
                    "cell_id": draws.cell_ids[j],
                    "state_fips": state,
                    "reference_month": month,
                    "raw_weight": float(draws.raw_mean[j]),
                    "estimate": estimates[state],
                    "estimate_integer": integers[state],
                    "weight_basis": "own_estimator",
                    "anchor_basis": anchor.anchor_basis,
                    "reconciliation_status": "anchored_and_reconciled",
                    "decline_reason": None,
                    "decline_kind": None,
                    "residual": anchor.residual,
                    "missing_set_size": len(anchor.missing_cells),
                    "constraint_set_hash": constraint_set_hash,
                }
            )
    results = pl.DataFrame(rows, schema=BASELINE_RESULT_SCHEMA)
    assert_declared_provenance(results)
    return results


def draw_ensembles(draws: ReconciledDraws) -> dict[str, np.ndarray]:
    """Each reconciled cell's draws, keyed by `cell_id`, for `draw_interval_metrics`."""
    return {cell: draws.values[:, j] for j, cell in enumerate(draws.cell_ids)}


@dataclass(frozen=True)
class StateModelProducer:
    """The harness's `Producer` for §11's model: fit, reconcile, gate, and hand back rows."""

    estimator_ids: tuple[str, ...] = (MODEL_ID,)

    def __call__(self, masked: HarmonizedData, system: MaskedSystem, config: Config) -> Production:
        """One replicate's fit on the masked frame, reconciled into the MASKED system's bounds."""
        monthly = masked.qcew_monthly
        fit = fit_state_total_model(
            build_model_data(monthly), StateModelConfig.from_config(config.model)
        )
        bounds = state_total_bounds(system.bounds)
        draws = reconcile_fit(fit, monthly, bounds, config)
        check = check_reconciled(draws, tolerance=config.reconciliation.tolerance)
        report = evaluate_gate(fit, draws, check, config.model.diagnostics, scope="replicate")
        return Production(
            results=model_results(draws, bounds, config),
            interval_metrics=partial(draw_interval_metrics, ensembles=draw_ensembles(draws)),
            notes={"gate": report.to_json(), "draws_sha256": draws_digest(draws)},
        )
```

- [ ] **Step 7: Run the tests to see them pass**

Run the Step 2 command again.

Expected: all pass (`36 passed` observed). Then run the two float goldens by name. They must pass without a re-pin:

```bash
uv run pytest tests/integration/test_validation_golden.py tests/integration/test_baseline_golden.py -q
```

- [ ] **Step 8: Gates**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
uv run pytest -q -p no:cacheprovider -m "not slow"
```

Expected: gates clean. Non-slow suite: Task 10's count **+ 14 passed**.

- [ ] **Step 9: Commit**

```bash
git add src/logging_employment/baselines/runner.py src/logging_employment/contracts.py src/logging_employment/validate/harness.py src/logging_employment/validate/metrics.py src/logging_employment/validate/intervals.py src/logging_employment/models/validation.py tests/unit/test_validate_producer_seam.py tests/unit/test_validate_metrics_draws.py tests/unit/test_model_validation.py tests/unit/test_contracts_validation.py tests/integration/test_validation_golden.py
git commit -m "feat(validate): score the state-total model through the harness's own loop (producer seam)"
```

---

### Task 12: §13.10's promotion record, and `D-109` closed on its own done-when

**Implements:**
- §13.10's six gates, as Decision 4 reads them;
- REQ-024's "gate applied";
- INV-014 applied: the model is promoted only if it beats the preferred baseline;
- `D-109`'s done-when, in the order it states. The tripwire reddens naming the keys that moved (Step 4), then the coverage gate compares exactly, pinned by 17/20 against 0.90 ± 0.05 counting as within (Step 1's first test). `specs/deferred_items.md` is ticked at plan completion, not here (Global Constraints).

The record is `provisional: true` whatever the verdict. Stage 7 adds the harvest factor and re-runs this gate (Stage 7's Produces). Disclosure review is recorded as `pending_stage_8`, never as passed.

**Files:**
- Create: `src/logging_employment/validate/promotion.py`
- Modify: `tests/unit/test_config_validation_block.py` (the tripwire, converted rather than deleted)
- Modify: `src/logging_employment/config.py` (`PromotionConfig`'s docstring: who reads the keys now)
- Test: `tests/unit/test_validate_promotion.py` (new; 17 tests, from 14 functions with one parametrized four ways)

**Interfaces:**
- Consumes:
  - Task 11's metrics rows (`interval_source`, `calibration_sample_size`, per-division `coverage_0.90`) and each regime's `producer_notes`;
  - Stage 4's `validate.scoreboard.preferred_baseline`, which returns `None` for "no comparand";
  - `validate.regimes.DIVISION_OF`;
  - `config.PromotionConfig`'s four keys.
- Produces:
  - `evaluate_promotion(*, model_id, model_scores, model_metrics, comparand_scores, comparand_metrics, comparand_scoreboard, production_gate, production_store_check, replicate_gates, promotion) -> dict`, which is the record;
  - `production_failed_record(model_id, production_gate, promotion) -> dict`;
  - the exact helpers `coverage_hits(value, n) -> int`, `within_tolerance(hits, n, *, tolerance) -> bool`, `catastrophic(hits, n, *, alpha) -> bool`, `coverage_counts(metrics, *, estimator_id, stratum_kind)`, `matched_pairs(...)` and `assert_same_masks(model_scores, comparand_scores)`;
  - the constants `NOMINAL_COVERAGE = Fraction(90, 100)`, `FALLBACK_METHOD = "section_10_8_hierarchy"` and `DISCLOSURE_REVIEW = "pending_stage_8"`.
- The record's keys:
  - `verdict` (`beat` or `not_beaten`), `selected_method`, `provisional` and `disclosure_review`;
  - `gates`, holding `hard_constraints`, `convergence`, `coverage`, `improvement`, `stratum_degradation` and `disclosure_review`;
  - `thresholds`, where all four keys are echoed.
  Task 13 adds `run_id`, `model_version` and `comparand` around it.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_validate_promotion.py`:

```python
"""§13.10 applied (`validate/promotion.py`): exact coverage, the comparand, and the verdict.

Every scenario is two regimes of twenty masked cells, one seed each: ten cells in `01` (East South
Central) and ten in `06` (Pacific), every truth 100. A method's error on a cell is the percentage
point amount its estimate misses by, so a WAPE reads straight off the scenario.
"""

from __future__ import annotations

import json
from fractions import Fraction
from math import comb

import polars as pl
import pytest

from logging_employment.config import PromotionConfig
from logging_employment.contracts import (
    VALIDATION_METRIC_SCHEMA,
    VALIDATION_SCORE_SCHEMA,
    VALIDATION_SCOREBOARD_SCHEMA,
)
from logging_employment.errors import ConceptViolationError
from logging_employment.validate.promotion import (
    FALLBACK_METHOD,
    catastrophic,
    coverage_hits,
    evaluate_promotion,
    production_failed_record,
    within_tolerance,
)

MODEL = "state_total_model"
COMPARAND = "share_last_observed"
REGIMES = ("regional_blocks", "small_cell_biased")
SEED = 1024
STATES = ("01",) * 10 + ("06",) * 10
PASSING_CHECKS = {
    "reconciliation_anchor_drift_max": {"passed": True},
    "reconciliation_bound_violations": {"passed": True},
}
PASSING_GATE = {"passed": True, "failures": [], "checks": PASSING_CHECKS}


def _tail(hits: int, n: int) -> Fraction:
    """P(X <= hits) for X ~ Binomial(n, 9/10), in exact rational arithmetic: the test's oracle."""
    return sum(
        Fraction(comb(n, k)) * Fraction(9, 10) ** k * Fraction(1, 10) ** (n - k)
        for k in range(hits + 1)
    )


def _scores(
    estimator: str, errors: list[float | None], *, masked_hash: str = "masked"
) -> pl.DataFrame:
    """One scored row per masked cell and regime; a None error is a declined cell."""
    rows = []
    for regime in REGIMES:
        for index, (state, error) in enumerate(zip(STATES, errors, strict=True)):
            estimate = None if error is None else 100.0 + error
            rows.append(
                {
                    "regime": regime,
                    "seed": SEED,
                    "replicate": 0,
                    "mask_arm": "state_total",
                    "estimator_id": estimator,
                    "cell_id": f"state_total|{state}|2023-{index + 1:02d}|5|113310|NAICS 2022|ALL",
                    "state_fips": state,
                    "reference_month": f"2023-{index + 1:02d}",
                    "suppression_type": "primary_like",
                    "truth": 100.0,
                    "estimate": estimate,
                    "estimate_integer": None if estimate is None else round(estimate),
                    "weight_basis": "none" if estimate is None else "own_estimator",
                    "decline_kind": "data_gap" if estimate is None else None,
                    "bound_status": "unbounded",
                    "selected_lower": 0.0,
                    "selected_upper": None,
                    "masked_constraint_set_hash": masked_hash,
                    "lookback_months_masked": 1,
                    "missing_set_size": 10,
                    "raw_weight": estimate,
                    "anchor_basis": "declared_national_total",
                    "reconciliation_status": "declined"
                    if estimate is None
                    else "anchored_and_reconciled",
                    "decline_reason": "no history" if estimate is None else None,
                    "residual": 1000.0,
                    "constraint_set_hash": None,
                }
            )
    return pl.DataFrame(rows, schema=VALIDATION_SCORE_SCHEMA)


def _coverage(
    estimator: str,
    overall: tuple[int, int],
    divisions: dict[str, tuple[int, int]],
    *,
    source: str,
    crps: float = 1.0,
) -> list[dict[str, object]]:
    """Each regime's `coverage_0.90` and `crps` rows, overall and per division."""
    rows: list[dict[str, object]] = []
    for regime in REGIMES:
        base = {
            "regime": regime,
            "seed": SEED,
            "mask_arm": "state_total",
            "estimator_id": estimator,
            "metric_family": "probabilistic",
            "denominator_basis": "masked_cell_rows",
            "interval_source": source,
        }
        hits, n = overall
        overall_base = {
            **base,
            "denominator": float(n),
            "n_scored": n,
            "calibration_sample_size": n,
        }
        rows.append(
            {
                **overall_base,
                "stratum_kind": "overall",
                "stratum_value": "all",
                "metric_name": "coverage_0.90",
                "value": hits / n,
            }
        )
        rows.append(
            {
                **overall_base,
                "stratum_kind": "overall",
                "stratum_value": "all",
                "metric_name": "crps",
                "value": crps,
            }
        )
        for division, (division_hits, division_n) in divisions.items():
            rows.append(
                {
                    **base,
                    "denominator": float(division_n),
                    "n_scored": division_n,
                    "calibration_sample_size": division_n,
                    "stratum_kind": "census_division",
                    "stratum_value": division,
                    "metric_name": "coverage_0.90",
                    "value": division_hits / division_n,
                }
            )
    return rows


def _board(estimators: dict[str, float]) -> pl.DataFrame:
    """A scoreboard row per regime for each estimator, at the given WAPE."""
    rows = [
        {
            "regime": regime,
            "seed": SEED,
            "mask_arm": "state_total",
            "estimator_id": estimator,
            "wape": wape,
            "denominator": 20.0,
            "denominator_basis": "masked_cell_rows",
            "n_scored": 20,
            "n_declined_by_design": 0,
            "n_declined_data_gap": 0,
            "n_declined_reconciliation_failure": 0,
            "n_own_estimator": 20,
            "n_establishment_fallback": 0,
        }
        for regime in REGIMES
        for estimator, wape in estimators.items()
    ]
    return pl.DataFrame(rows, schema=VALIDATION_SCOREBOARD_SCHEMA)


def _evaluate(
    *,
    model_errors: list[float | None],
    comparand_errors: list[float | None],
    model_coverage: tuple[tuple[int, int], dict[str, tuple[int, int]]] = (
        (18, 20),
        {"east_south_central": (9, 10), "pacific": (9, 10)},
    ),
    comparand_coverage: tuple[int, int] = (18, 20),
    model_crps: float = 1.0,
    comparand_crps: float = 2.0,
    board: pl.DataFrame | None = None,
    replicate_gate: dict[str, object] = PASSING_GATE,
    comparand_hash: str = "masked",
    model_source: str = "reconciled_posterior_draws",
) -> dict[str, object]:
    overall, divisions = model_coverage
    metrics = pl.DataFrame(
        _coverage(MODEL, overall, divisions, source=model_source, crps=model_crps),
        schema=VALIDATION_METRIC_SCHEMA,
    )
    comparand_metrics = pl.DataFrame(
        _coverage(
            COMPARAND,
            comparand_coverage,
            {},
            source="leave_one_out_residual_ensemble",
            crps=comparand_crps,
        ),
        schema=VALIDATION_METRIC_SCHEMA,
    )
    return evaluate_promotion(
        model_id=MODEL,
        model_scores=_scores(MODEL, model_errors),
        model_metrics=metrics,
        comparand_scores=_scores(COMPARAND, comparand_errors, masked_hash=comparand_hash),
        comparand_metrics=comparand_metrics,
        comparand_scoreboard=_board({COMPARAND: 0.1}) if board is None else board,
        production_gate=PASSING_GATE,
        production_store_check={"passed": True, "draws_sha256_matches": True},
        replicate_gates={
            regime: [{"seed": SEED, "gate": replicate_gate, "draws_sha256": "d"}]
            for regime in REGIMES
        },
        promotion=PromotionConfig(),
    )


def test_seventeen_of_twenty_is_within_five_points_of_ninety() -> None:
    """`D-109`'s pin: exactly 17/20 against nominal 0.90 and tolerance 0.05 counts as within.

    The float comparison is the defect the item records: 44 of `runs/f03023ac9f3a`'s 170 groups sat
    at exactly 0.85 and a float `abs(c - 0.9) <= 0.05` excluded every one.
    """
    assert within_tolerance(17, 20, tolerance=0.05)
    assert not abs(17 / 20 - 0.9) <= 0.05
    assert not within_tolerance(16, 20, tolerance=0.05)


def test_coverage_hits_recovers_the_count_or_refuses_a_ratio() -> None:
    assert coverage_hits(17 / 20, 20) == 17
    with pytest.raises(ConceptViolationError, match="not an integer count"):
        coverage_hits(0.8123, 20)


@pytest.mark.parametrize(("hits", "n"), [(12, 20), (13, 20), (72, 100), (86, 100)])
def test_catastrophic_is_the_exact_binomial_lower_tail(hits: int, n: int) -> None:
    alpha = PromotionConfig().catastrophic_stratum_coverage_alpha
    assert catastrophic(hits, n, alpha=alpha) is (_tail(hits, n) < Fraction(repr(alpha)))


def test_a_model_that_beats_the_comparand_everywhere_is_selected() -> None:
    record = _evaluate(model_errors=[5.0] * 20, comparand_errors=[10.0] * 20)
    assert (record["verdict"], record["selected_method"]) == ("beat", MODEL)
    assert record["provisional"] is True
    assert record["disclosure_review"] == "pending_stage_8"
    regime = record["gates"]["improvement"]["per_regime"]["regional_blocks"]
    assert (regime["comparand"], regime["status"]) == (COMPARAND, "wape")
    assert (regime["n_matched"], regime["n_model_only"]) == (20, 0)
    assert regime["relative_improvement"] == pytest.approx(0.5)
    json.dumps(record, allow_nan=False)


def test_a_model_that_does_not_beat_it_deploys_the_simpler_method() -> None:
    """§13.10's last line: "If these gates are not met, deploy the simpler method"."""
    record = _evaluate(model_errors=[10.0] * 20, comparand_errors=[10.0] * 20)
    assert (record["verdict"], record["selected_method"]) == ("not_beaten", FALLBACK_METHOD)
    assert record["gates"]["improvement"]["per_regime"]["regional_blocks"]["status"] == "not_beaten"


def test_calibrated_uncertainty_can_carry_a_regime_the_wape_gate_does_not() -> None:
    """The comparand's coverage is catastrophic (5 of 20) and the model's CRPS is lower."""
    record = _evaluate(
        model_errors=[10.0] * 20, comparand_errors=[10.0] * 20, comparand_coverage=(5, 20)
    )
    regime = record["gates"]["improvement"]["per_regime"]["regional_blocks"]
    assert regime["status"] == "calibrated_uncertainty"
    assert record["verdict"] == "beat"
    worse = _evaluate(
        model_errors=[10.0] * 20,
        comparand_errors=[10.0] * 20,
        comparand_coverage=(5, 20),
        model_crps=3.0,
    )
    assert worse["gates"]["improvement"]["per_regime"]["regional_blocks"]["status"] == "not_beaten"


def test_cells_the_comparand_declined_are_counted_not_compared() -> None:
    record = _evaluate(model_errors=[5.0] * 20, comparand_errors=[None, None] + [10.0] * 18)
    regime = record["gates"]["improvement"]["per_regime"]["small_cell_biased"]
    assert (regime["n_matched"], regime["n_model_only"]) == (18, 2)
    assert regime["model_wape"] == pytest.approx(0.05)


def test_a_regime_with_no_comparand_is_neither_a_pass_nor_a_failure() -> None:
    """Stage 4's SHIPPED point (3): `preferred_baseline` returning None means "no comparand"."""
    no_hierarchy = _board({"equal_residual": 0.1})
    record = _evaluate(model_errors=[5.0] * 20, comparand_errors=[10.0] * 20, board=no_hierarchy)
    statuses = {
        regime: row["status"]
        for regime, row in record["gates"]["improvement"]["per_regime"].items()
    }
    assert statuses == dict.fromkeys(REGIMES, "not_applicable")
    # With no regime to compare in, nothing shows the model beats anything.
    assert record["gates"]["improvement"]["passed"] is False
    assert record["verdict"] == "not_beaten"


def test_a_division_the_model_degrades_fails_the_gate() -> None:
    """Better in every regime, worse in the Pacific: 0.11 against 0.10 is a 10% degradation."""
    record = _evaluate(model_errors=[0.0] * 10 + [11.0] * 10, comparand_errors=[10.0] * 20)
    assert record["gates"]["improvement"]["passed"] is True
    pacific = record["gates"]["stratum_degradation"]["per_division"]["pacific"]
    assert pacific["degraded"] is True
    assert pacific["relative_degradation"] == pytest.approx(0.1)
    assert record["verdict"] == "not_beaten"


def test_a_catastrophic_division_fails_coverage_when_the_pool_passes() -> None:
    """86% pooled is within five points; one division at 72 of 100 is not calibration's doing."""
    record = _evaluate(
        model_errors=[5.0] * 20,
        comparand_errors=[10.0] * 20,
        model_coverage=((86, 100), {"east_south_central": (50, 50), "pacific": (36, 50)}),
    )
    coverage = record["gates"]["coverage"]
    assert coverage["overall"]["within_tolerance"] is True
    assert [stratum["stratum_value"] for stratum in coverage["catastrophic_strata"]] == ["pacific"]
    assert record["verdict"] == "not_beaten"


def test_a_failed_replicate_fails_convergence_and_is_named() -> None:
    failed = {"passed": False, "failures": ["parameter_rhat_max"], "checks": PASSING_CHECKS}
    record = _evaluate(model_errors=[5.0] * 20, comparand_errors=[10.0] * 20, replicate_gate=failed)
    convergence = record["gates"]["convergence"]
    assert convergence["passed"] is False
    assert convergence["replicate_failures"][0]["failures"] == ["parameter_rhat_max"]
    assert record["verdict"] == "not_beaten"


def test_runs_that_masked_under_different_systems_are_refused() -> None:
    with pytest.raises(ConceptViolationError, match="different masked"):
        _evaluate(model_errors=[5.0] * 20, comparand_errors=[10.0] * 20, comparand_hash="other")


def test_the_models_coverage_must_come_from_its_own_draws() -> None:
    with pytest.raises(ConceptViolationError, match="reconciled draws"):
        _evaluate(
            model_errors=[5.0] * 20,
            comparand_errors=[10.0] * 20,
            model_source="leave_one_out_residual_ensemble",
        )


def test_a_failed_production_fit_is_recorded_without_being_scored() -> None:
    gate = {"passed": False, "failures": ["cell_rhat_max"], "checks": PASSING_CHECKS}
    record = production_failed_record(MODEL, gate, PromotionConfig())
    assert (record["verdict"], record["selected_method"]) == ("not_beaten", FALLBACK_METHOD)
    assert record["gates"]["convergence"]["production_failures"] == ["cell_rhat_max"]
    assert record["gates"]["coverage"]["status"] == "not_evaluated_production_fit_failed"
```

- [ ] **Step 2: Run them to watch them fail**

Run: `uv run pytest tests/unit/test_validate_promotion.py -q`

Expected (observed):

```text
E   ModuleNotFoundError: No module named 'logging_employment.validate.promotion'
ERROR tests/unit/test_validate_promotion.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error
```

- [ ] **Step 3: Write the promotion record**

`src/logging_employment/validate/promotion.py`:

```python
"""§13.10's promotion gates, applied to the state-total model against Stage 4's comparand.

THE RECORD IS PROVISIONAL AND SAYS SO. Stage 7 adds the harvest factor and §11.13's sensitivity
variants and re-runs this gate, and a Stage 5 decision stands only until that re-run records a
final one (roadmap, Stage 7's Produces). Disclosure review, §13.10's sixth gate, is Stage 8's and
is recorded as `pending_stage_8`, never as passed.

§13.10 states six gates and no procedure. The readings below are plan 16's, and its "Decisions this
plan makes" section flags them for review. Each is written down so a reader can disagree with a
named choice rather than reverse-engineer one:

* COVERAGE is pooled over every regime and seed: exact integer hits over exact integer samples,
  never a mean of ratios. Per-regime coverage is reported and does not gate. A perfectly
  calibrated model passes a ±5-point test in all nine regimes separately with probability 0.106 at
  D1's sample sizes, so a per-regime reading would reject calibration itself. The pool is not
  balanced: `whole_seasonal_blocks` supplies about 69% of it (`D-106`), and `pool_share` records
  each regime's part.
* "DOES NOT FAIL CATASTROPHICALLY IN ANY MAJOR STRATUM" is an exact binomial lower tail below
  `catastrophic_stratum_coverage_alpha`, tested per regime and per Census division pooled across
  regimes. That is 18 tests, and the family-wise false-alarm rate for a calibrated model is 0.0122
  at alpha = 0.001.
* WAPE is compared on MATCHED (seed, cell) pairs, per regime, against `preferred_baseline`'s
  choice. The model never declines while a comparand can, so pooled scoreboard WAPEs would compare
  two different cell sets. `n_model_only` counts what the matching drops. The improvement is
  relative: (comparand - model) / comparand.
* DEGRADATION is judged per Census division, pooled across regimes over the same matched pairs,
  relative, against `maximum_major_stratum_wape_degradation`. "Without documented substantive
  benefit" is not machine-evaluable, so a degraded division fails the gate and the record names it.
* "CLEARLY SUPERIOR CALIBRATED UNCERTAINTY" passes a regime whose WAPE gate failed only when three
  things hold. On the model side, the pooled coverage passes and the regime is not catastrophic. On
  the comparand side, it has no interval in any seed, or its pooled regime coverage is itself
  catastrophic. And the model's CRPS is no worse wherever the comparand has one.
* A regime `preferred_baseline` returns None for is `not_applicable`: neither a pass nor a
  failure (Stage 4's SHIPPED point (3)). At least one regime must have a comparand.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction

import polars as pl
from scipy.stats import binom

from ..config import PromotionConfig
from ..errors import ConceptViolationError
from .regimes import DIVISION_OF
from .scoreboard import preferred_baseline

# §13.10's "90% interval coverage", and the metric both interval sources write for it.
NOMINAL_COVERAGE = Fraction(90, 100)
COVERAGE_METRIC = "coverage_0.90"
MODEL_INTERVAL_SOURCE = "reconciled_posterior_draws"
FALLBACK_METHOD = "section_10_8_hierarchy"
DISCLOSURE_REVIEW = "pending_stage_8"


def coverage_hits(value: float, n: int) -> int:
    """The integer hit count behind a stored coverage value, recovered exactly or refused.

    Both emitters write `hits / n` as a Python float, so `round(value * n)` recovers `hits`.
    Re-doing the emitters' own division and demanding the same double proves it. A value that did
    not come from an integer count over this `n` is refused rather than rounded into one.
    """
    if n <= 0:
        raise ConceptViolationError(f"coverage {value} carries calibration_sample_size {n}")
    hits = round(value * n)
    if not 0 <= hits <= n or hits / n != value:
        raise ConceptViolationError(
            f"coverage {value!r} over {n} cells is not an integer count; the gate compares exact "
            "counts (D-109) and will not round a ratio into one"
        )
    return hits


def within_tolerance(hits: int, n: int, *, tolerance: float) -> bool:
    """|hits/n - 0.90| <= tolerance, in exact rational arithmetic (D-109).

    `tolerance` is read as the decimal the operator wrote, `Fraction(repr(tolerance))`, not as its
    binary double. So 0.05 is 1/20, and 17/20 against 0.90 sits exactly on the boundary and counts
    as within. A float `abs(0.85 - 0.9) <= 0.05` is `0.05000000000000004 <= 0.05`, which is False.
    That is how 44 of `runs/f03023ac9f3a`'s 170 interval-bearing groups were miscounted.
    """
    return abs(Fraction(hits, n) - NOMINAL_COVERAGE) <= Fraction(repr(tolerance))


def catastrophic(hits: int, n: int, *, alpha: float) -> bool:
    """P(X <= hits) < alpha for X ~ Binomial(n, 0.90): fewer hits than calibration can explain."""
    return n > 0 and float(binom.cdf(hits, n, float(NOMINAL_COVERAGE))) < alpha


def _fraction_json(hits: int, n: int) -> dict[str, object]:
    """A pooled coverage as its integers and their ratio, the integers being the record."""
    return {"hits": hits, "n": n, "coverage": hits / n if n else None}


def coverage_counts(metrics: pl.DataFrame, *, estimator_id: str, stratum_kind: str) -> pl.DataFrame:
    """The interval-bearing `coverage_0.90` rows of one estimator and stratum kind, with hits."""
    rows = metrics.filter(
        (pl.col("estimator_id") == estimator_id)
        & (pl.col("metric_family") == "probabilistic")
        & (pl.col("metric_name") == COVERAGE_METRIC)
        & (pl.col("stratum_kind") == stratum_kind)
        & pl.col("value").is_not_null()
        & (pl.col("calibration_sample_size") > 0)
    )
    hits = [
        coverage_hits(value, n)
        for value, n in zip(
            rows["value"].to_list(), rows["calibration_sample_size"].to_list(), strict=True
        )
    ]
    return rows.select(
        "regime",
        "seed",
        "stratum_value",
        "interval_source",
        pl.col("calibration_sample_size").alias("n"),
    ).with_columns(pl.Series("hits", hits, dtype=pl.Int64))


def _pooled(counts: pl.DataFrame, by: str) -> dict[str, tuple[int, int]]:
    """Integer hits and samples summed within each value of `by`."""
    grouped = counts.group_by(by).agg(pl.col("hits").sum(), pl.col("n").sum()).sort(by)
    return {
        str(key): (int(hits), int(n))
        for key, hits, n in zip(grouped[by], grouped["hits"], grouped["n"], strict=True)
    }


def _pooled_crps(metrics: pl.DataFrame, *, estimator_id: str, regime: str) -> float | None:
    """One regime's CRPS pooled across seeds, weighted by each seed's calibration sample."""
    rows = metrics.filter(
        (pl.col("estimator_id") == estimator_id)
        & (pl.col("regime") == regime)
        & (pl.col("metric_name") == "crps")
        & (pl.col("stratum_kind") == "overall")
        & pl.col("value").is_not_null()
        & (pl.col("calibration_sample_size") > 0)
    )
    weights = rows["calibration_sample_size"].to_list()
    if not weights:
        return None
    total = math.fsum(v * w for v, w in zip(rows["value"].to_list(), weights, strict=True))
    return total / sum(weights)


def assert_same_masks(model_scores: pl.DataFrame, comparand_scores: pl.DataFrame) -> None:
    """Refuse a comparison unless both runs masked the same cells under the same masked system.

    Checked per (regime, seed): the set of masked `cell_id`s and the `masked_constraint_set_hash`.
    A masked target's score depends on which bounds bind on the other cells of its month (plan
    15's final review), so a WAPE gap between runs with different masks or different
    identification sets would measure the runs, not the methods.
    """
    keys = ["regime", "seed"]

    def shape(scores: pl.DataFrame) -> dict[tuple[str, int], tuple[frozenset[str], str]]:
        """Each (regime, seed)'s masked cells and masked-system hash."""
        grouped = scores.group_by(keys).agg(
            pl.col("cell_id").unique().sort(), pl.col("masked_constraint_set_hash").unique()
        )
        out: dict[tuple[str, int], tuple[frozenset[str], str]] = {}
        for regime, seed, cells, hashes in grouped.iter_rows():
            if len(hashes) != 1:
                raise ConceptViolationError(f"{regime} seed {seed} carries hashes {hashes}")
            out[(regime, seed)] = (frozenset(cells), hashes[0])
        return out

    model, comparand = shape(model_scores), shape(comparand_scores)
    if model.keys() != comparand.keys():
        raise ConceptViolationError(
            f"the model scored (regime, seed) {sorted(model.keys() ^ comparand.keys())} that the "
            "comparand did not, or the reverse; the two runs did not mask the same replicates"
        )
    differing = sorted(key for key in model if model[key] != comparand[key])
    if differing:
        raise ConceptViolationError(
            f"{len(differing)} replicate(s) masked different cells or solved a different masked "
            f"system in the two runs, first {differing[:3]}"
        )


def matched_pairs(
    model_scores: pl.DataFrame,
    comparand_scores: pl.DataFrame,
    comparands: Mapping[str, str],
    *,
    model_id: str,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """One row per (regime, seed, cell) both methods estimated, and per-regime `n_model_only`.

    A cell the comparand scored and the model did not is refused. The model does not decline, so
    such a cell is a defect in the producer, not a property of the method.
    """
    frames: list[pl.DataFrame] = []
    model_only: dict[str, int] = {}
    for regime, comparand in sorted(comparands.items()):
        model = model_scores.filter(
            (pl.col("regime") == regime)
            & (pl.col("estimator_id") == model_id)
            & pl.col("estimate").is_not_null()
        ).select("seed", "cell_id", "state_fips", "truth", pl.col("estimate").alias("model"))
        other = comparand_scores.filter(
            (pl.col("regime") == regime)
            & (pl.col("estimator_id") == comparand)
            & pl.col("estimate").is_not_null()
        ).select(
            "seed",
            "cell_id",
            pl.col("truth").alias("comparand_truth"),
            pl.col("estimate").alias("comparand"),
        )
        orphans = other.join(model, on=["seed", "cell_id"], how="anti")
        if orphans.height:
            raise ConceptViolationError(
                f"{regime}: {comparand} estimated {orphans.height} cell(s) the model did not, "
                f"first {orphans['cell_id'].to_list()[:3]}"
            )
        joined = model.join(other, on=["seed", "cell_id"], how="inner")
        if (joined["truth"] != joined["comparand_truth"]).any():
            raise ConceptViolationError(f"{regime}: the two runs disagree on a masked cell's truth")
        if joined.is_empty():
            raise ConceptViolationError(
                f"{regime}: {comparand} is the comparand but shares no scored cell with the model"
            )
        model_only[regime] = model.height - joined.height
        frames.append(
            joined.drop("comparand_truth").with_columns(
                pl.lit(regime).alias("regime"),
                pl.col("state_fips").replace_strict(DIVISION_OF).alias("census_division"),
            )
        )
    if not frames:
        return pl.DataFrame(), model_only
    return pl.concat(frames, how="vertical"), model_only


def _wapes(pairs: pl.DataFrame) -> tuple[float, float, int]:
    """Matched-pair WAPE for the model and the comparand, exactly summed, and the pair count."""
    truth = math.fsum(abs(t) for t in pairs["truth"].to_list())
    if truth == 0.0:
        raise ConceptViolationError("matched pairs with zero total truth have no WAPE")
    model = math.fsum(
        abs(m - t) for m, t in zip(pairs["model"].to_list(), pairs["truth"].to_list(), strict=True)
    )
    comparand = math.fsum(
        abs(c - t)
        for c, t in zip(pairs["comparand"].to_list(), pairs["truth"].to_list(), strict=True)
    )
    return model / truth, comparand / truth, pairs.height


def _relative(model_wape: float, comparand_wape: float) -> float | None:
    """(comparand - model) / comparand, or None when the comparand's WAPE is zero."""
    return None if comparand_wape == 0.0 else (comparand_wape - model_wape) / comparand_wape


def evaluate_promotion(
    *,
    model_id: str,
    model_scores: pl.DataFrame,
    model_metrics: pl.DataFrame,
    comparand_scores: pl.DataFrame,
    comparand_metrics: pl.DataFrame,
    comparand_scoreboard: pl.DataFrame,
    production_gate: Mapping[str, object],
    production_store_check: Mapping[str, object],
    replicate_gates: Mapping[str, list[Mapping[str, object]]],
    promotion: PromotionConfig,
) -> dict[str, object]:
    """§13.10 applied: every gate's evidence, the verdict, and the method that ships.

    `production_gate` is `posterior/diagnostics.json`. `production_store_check` is INV-012
    re-measured on the persisted store (`models.reconciliation.check_reconciled` over
    `models.arviz_io.read_store`). `replicate_gates` is the harness manifest's `producer_notes`
    gate reports, keyed by regime. Hard constraints and convergence include every replicate fit,
    because a model that converged once on the full panel and failed on a masked one has not
    shown §13.10 that its convergence holds.
    """
    tolerance = promotion.nominal_coverage_tolerance
    alpha = promotion.catastrophic_stratum_coverage_alpha
    replicates = [
        {"regime": regime, **note}
        for regime, notes in sorted(replicate_gates.items())
        for note in notes
    ]

    # Gate 1: all hard constraints pass, on the production draws and on every replicate's.
    reconciliation_checks = ("reconciliation_anchor_drift_max", "reconciliation_bound_violations")
    replicate_constraint_failures = [
        {"regime": r["regime"], "seed": r["seed"], "check": name}
        for r in replicates
        for name in reconciliation_checks
        if not r["gate"]["checks"][name]["passed"]
    ]
    production_constraints = all(
        production_gate["checks"][name]["passed"] for name in reconciliation_checks
    ) and bool(production_store_check["passed"])
    hard_constraints = {
        "passed": production_constraints and not replicate_constraint_failures,
        "production_in_memory": {
            name: production_gate["checks"][name] for name in reconciliation_checks
        },
        "production_store": dict(production_store_check),
        "replicates_checked": len(replicates),
        "replicate_failures": replicate_constraint_failures,
    }

    # Gate 2: convergence diagnostics pass, in production and in every replicate (Decision 5).
    replicate_convergence_failures = [
        {"regime": r["regime"], "seed": r["seed"], "failures": r["gate"]["failures"]}
        for r in replicates
        if not r["gate"]["passed"]
    ]
    convergence = {
        "passed": bool(production_gate["passed"]) and not replicate_convergence_failures,
        "production_failures": list(production_gate["failures"]),
        "replicates_checked": len(replicates),
        "replicate_failures": replicate_convergence_failures,
    }

    # Gate 3: coverage, pooled overall, and no catastrophic regime or division.
    overall = coverage_counts(model_metrics, estimator_id=model_id, stratum_kind="overall")
    divisions = coverage_counts(
        model_metrics, estimator_id=model_id, stratum_kind="census_division"
    )
    foreign = sorted(set(overall["interval_source"].to_list()) - {MODEL_INTERVAL_SOURCE})
    if foreign:
        raise ConceptViolationError(
            f"the model's coverage rows carry interval_source {foreign}; §13.10 reads the "
            "model's own reconciled draws"
        )
    hits, n = int(overall["hits"].sum()), int(overall["n"].sum())
    if n == 0:
        raise ConceptViolationError("the model scored no interval anywhere; coverage is unmeasured")
    per_regime = _pooled(overall, "regime")
    per_division = _pooled(divisions, "stratum_value")
    catastrophic_strata = [
        {"stratum_kind": kind, "stratum_value": key, **_fraction_json(h, m)}
        for kind, pooled in (("regime", per_regime), ("census_division", per_division))
        for key, (h, m) in pooled.items()
        if catastrophic(h, m, alpha=alpha)
    ]
    pooled_within = within_tolerance(hits, n, tolerance=tolerance)
    coverage = {
        "passed": pooled_within and not catastrophic_strata,
        "overall": {**_fraction_json(hits, n), "within_tolerance": pooled_within},
        "per_regime": {
            regime: {
                **_fraction_json(h, m),
                "within_tolerance": within_tolerance(h, m, tolerance=tolerance),
                "catastrophic": catastrophic(h, m, alpha=alpha),
                "pool_share": m / n,
            }
            for regime, (h, m) in per_regime.items()
        },
        "per_division": {
            division: {**_fraction_json(h, m), "catastrophic": catastrophic(h, m, alpha=alpha)}
            for division, (h, m) in per_division.items()
        },
        "catastrophic_strata": catastrophic_strata,
    }

    # Gates 4 and 5 compare against §13.10's comparand, regime by regime.
    assert_same_masks(model_scores, comparand_scores)
    regimes = sorted(set(model_scores["regime"].to_list()))
    comparands = {
        regime: name
        for regime in regimes
        if (name := preferred_baseline(comparand_scoreboard, regime=regime)) is not None
    }
    pairs, model_only = matched_pairs(model_scores, comparand_scores, comparands, model_id=model_id)
    improvement_rows: dict[str, dict[str, object]] = {}
    for regime in regimes:
        if regime not in comparands:
            improvement_rows[regime] = {"comparand": None, "status": "not_applicable"}
            continue
        comparand = comparands[regime]
        model_wape, comparand_wape, n_matched = _wapes(pairs.filter(pl.col("regime") == regime))
        relative = _relative(model_wape, comparand_wape)
        wape_improved = relative is not None and relative >= promotion.minimum_wape_improvement
        alternative = _calibrated_alternative(
            regime,
            comparand,
            model_id=model_id,
            model_metrics=model_metrics,
            comparand_metrics=comparand_metrics,
            pooled_coverage_passes=pooled_within,
            model_catastrophic=bool(coverage["per_regime"].get(regime, {}).get("catastrophic")),
            alpha=alpha,
        )
        improvement_rows[regime] = {
            "comparand": comparand,
            "status": (
                "wape"
                if wape_improved
                else "calibrated_uncertainty"
                if alternative["passed"]
                else "not_beaten"
            ),
            "n_matched": n_matched,
            "n_model_only": model_only[regime],
            "model_wape": model_wape,
            "comparand_wape": comparand_wape,
            "relative_improvement": relative,
            "calibrated_alternative": alternative,
        }
    applicable = [row for row in improvement_rows.values() if row["status"] != "not_applicable"]
    improvement = {
        "passed": bool(applicable) and all(row["status"] != "not_beaten" for row in applicable),
        "regimes_with_comparand": len(applicable),
        "per_regime": improvement_rows,
    }

    degradation_rows: dict[str, dict[str, object]] = {}
    if not pairs.is_empty():
        for division in sorted(set(pairs["census_division"].to_list())):
            model_wape, comparand_wape, n_matched = _wapes(
                pairs.filter(pl.col("census_division") == division)
            )
            worse = (
                model_wape > 0.0
                if comparand_wape == 0.0
                else (model_wape - comparand_wape) / comparand_wape
                > promotion.maximum_major_stratum_wape_degradation
            )
            degradation_rows[division] = {
                "n_matched": n_matched,
                "model_wape": model_wape,
                "comparand_wape": comparand_wape,
                "relative_degradation": None
                if comparand_wape == 0.0
                else (model_wape - comparand_wape) / comparand_wape,
                "degraded": worse,
            }
    degradation = {
        "passed": not any(row["degraded"] for row in degradation_rows.values()),
        "per_division": degradation_rows,
    }

    gates = {
        "hard_constraints": hard_constraints,
        "convergence": convergence,
        "coverage": coverage,
        "improvement": improvement,
        "stratum_degradation": degradation,
        "disclosure_review": {"status": DISCLOSURE_REVIEW},
    }
    beat = all(
        gates[name]["passed"]
        for name in (
            "hard_constraints",
            "convergence",
            "coverage",
            "improvement",
            "stratum_degradation",
        )
    )
    return _record(model_id, beat, gates, promotion)


def _calibrated_alternative(
    regime: str,
    comparand: str,
    *,
    model_id: str,
    model_metrics: pl.DataFrame,
    comparand_metrics: pl.DataFrame,
    pooled_coverage_passes: bool,
    model_catastrophic: bool,
    alpha: float,
) -> dict[str, object]:
    """§13.10's "clearly superior calibrated uncertainty" for one regime, with its evidence."""
    counts = coverage_counts(comparand_metrics, estimator_id=comparand, stratum_kind="overall")
    counts = counts.filter(pl.col("regime") == regime)
    comparand_hits, comparand_n = int(counts["hits"].sum()), int(counts["n"].sum())
    comparand_has_interval = comparand_n > 0
    comparand_catastrophic = catastrophic(comparand_hits, comparand_n, alpha=alpha)
    model_crps = _pooled_crps(model_metrics, estimator_id=model_id, regime=regime)
    comparand_crps = _pooled_crps(comparand_metrics, estimator_id=comparand, regime=regime)
    crps_no_worse = comparand_crps is None or (
        model_crps is not None and model_crps <= comparand_crps
    )
    model_side = pooled_coverage_passes and not model_catastrophic
    comparand_side = not comparand_has_interval or comparand_catastrophic
    return {
        "passed": model_side and comparand_side and crps_no_worse,
        "model_side": model_side,
        "comparand_side": comparand_side,
        "comparand_coverage": _fraction_json(comparand_hits, comparand_n),
        "comparand_catastrophic": comparand_catastrophic,
        "model_crps": model_crps,
        "comparand_crps": comparand_crps,
        "crps_no_worse": crps_no_worse,
    }


def _record(
    model_id: str, beat: bool, gates: Mapping[str, object], promotion: PromotionConfig
) -> dict[str, object]:
    """The promotion record's envelope: verdict, selection, provisionality and thresholds."""
    return {
        "model_id": model_id,
        "verdict": "beat" if beat else "not_beaten",
        # §13.10's last line: "If these gates are not met, deploy the simpler method."
        "selected_method": model_id if beat else FALLBACK_METHOD,
        "provisional": True,
        "provisional_until": "Stage 7 re-runs §13.10 with the harvest factor and §11.13's variants",
        "disclosure_review": DISCLOSURE_REVIEW,
        "thresholds": {
            "nominal_coverage": str(NOMINAL_COVERAGE),
            "nominal_coverage_tolerance": promotion.nominal_coverage_tolerance,
            "minimum_wape_improvement": promotion.minimum_wape_improvement,
            "maximum_major_stratum_wape_degradation": (
                promotion.maximum_major_stratum_wape_degradation
            ),
            "catastrophic_stratum_coverage_alpha": promotion.catastrophic_stratum_coverage_alpha,
        },
        "gates": dict(gates),
    }


def production_failed_record(
    model_id: str, production_gate: Mapping[str, object], promotion: PromotionConfig
) -> dict[str, object]:
    """The record when the production fit failed §11.14: not beaten, and no gate beyond it scored.

    `validate-state-model` writes this without running the harness. 27 replicate fits of a model
    that did not converge on the full panel would be spent learning nothing §13.10 can use.
    """
    not_evaluated = {"passed": False, "status": "not_evaluated_production_fit_failed"}
    gates = {
        "hard_constraints": not_evaluated,
        "convergence": {
            "passed": False,
            "production_failures": list(production_gate["failures"]),
            "replicates_checked": 0,
            "replicate_failures": [],
        },
        "coverage": not_evaluated,
        "improvement": not_evaluated,
        "stratum_degradation": not_evaluated,
        "disclosure_review": {"status": DISCLOSURE_REVIEW},
    }
    return _record(model_id, False, gates, promotion)
```

- [ ] **Step 4: Watch `D-109`'s tripwire redden and name the keys**

This red is the evidence `D-109`'s done-when asks for. Run it BEFORE converting the test, and keep its output for the Task 15 log entry.

Run: `uv run pytest tests/unit/test_config_validation_block.py -q`

Expected (observed):

```text
E       AssertionError: now read by {'validate/promotion.py': ['catastrophic_stratum_coverage_alpha', 'maximum_major_stratum_wape_degradation', 'minimum_wape_improvement', 'nominal_coverage_tolerance']}; update PromotionConfig's docstring
E       assert {'validate/pr...e_tolerance']} == {}
E         Left contains 1 more item:
E         {'validate/promotion.py': ['catastrophic_stratum_coverage_alpha',
E                                    'maximum_major_stratum_wape_degradation',
E                                    'minimum_wape_improvement',
E                                    'nominal_coverage_tolerance']}
E         Use -v to get more diff
FAILED tests/unit/test_config_validation_block.py::test_the_promotion_keys_are_still_unread_and_the_docstring_still_says_so
1 failed, 5 passed
```

- [ ] **Step 5: Convert the tripwire, and rewrite the docstring it guards**

The test is converted, not deleted. It now pins `validate/promotion.py` as the ONLY reader, so a second module that could apply a gate differently reddens it again.

`tests/unit/test_config_validation_block.py`:

```diff
diff --git a/tests/unit/test_config_validation_block.py b/tests/unit/test_config_validation_block.py
index bb1c4c7..ee8dda8 100644
--- a/tests/unit/test_config_validation_block.py
+++ b/tests/unit/test_config_validation_block.py
@@ -116,16 +116,16 @@ def test_the_tripwire_detector_sees_reads_and_ignores_mentions():
     assert _named_keys("gate(**cfg.promotion.model_dump())") == set()
 
 
-def test_the_promotion_keys_are_still_unread_and_the_docstring_still_says_so():
-    """R-S5G-3: a TRIPWIRE, not a prohibition. It fails when `src/` code NAMES one of these keys.
-
-    `PromotionConfig`'s docstring records all three as inert. A docstring cannot notice when it
-    stops being true, and the failure mode is specific: the day a promotion path reads one of
-    these, the note becomes a false statement in the file a reader consults first. Derived from the
-    code's AST (`_named_keys`) rather than asserting a sentence exists, so it tracks code, not prose.
-    `config.py` is scanned too: its field declarations are annotated `Name` targets and do not count,
-    so a validator there that READS a key still trips this. When it reddens, the fix is to update the
-    docstring -- not to delete this test.
+def test_the_promotion_keys_are_read_only_by_the_promotion_record():
+    """R-S5G-3's tripwire, converted when §13.10's evaluator landed (plan 16, `D-109`).
+
+    Until plan 16 this was
+    `test_the_promotion_keys_are_still_unread_and_the_docstring_still_says_so`, and it asserted
+    that no `src/` module named a key. `validate/promotion.py` now reads all four, and this test
+    reddened naming them, as `D-109`'s done-when required. It now pins that module as the ONLY
+    reader, so a second reader -- one that could apply a gate differently -- reddens it again. The
+    fix then is to update `PromotionConfig`'s docstring, not to delete this test. Derived from the
+    code's AST (`_named_keys`), so it tracks code, not prose.
     """
     src = Path(__file__).resolve().parents[2] / "src" / "logging_employment"
     readers = {
@@ -133,4 +133,6 @@ def test_the_promotion_keys_are_still_unread_and_the_docstring_still_says_so():
         for path in sorted(src.rglob("*.py"))
         if (named := _named_keys(path.read_text(encoding="utf-8")))
     }
-    assert readers == {}, f"now read by {readers}; update PromotionConfig's docstring"
+    assert readers == {"validate/promotion.py": sorted(PROMOTION_KEYS)}, (
+        f"now read by {readers}; update PromotionConfig's docstring"
+    )
```

`src/logging_employment/config.py`:

```diff
diff --git a/src/logging_employment/config.py b/src/logging_employment/config.py
index f4706b6..f1f91cb 100644
--- a/src/logging_employment/config.py
+++ b/src/logging_employment/config.py
@@ -287,35 +287,23 @@ class ValidationConfig(_Strict):
 class PromotionConfig(_Strict):
     """§13.10's gates. Configurable engineering thresholds, not findings.
 
-    ALL FOUR KEYS ARE INERT TODAY, and this note is the record R-S5G-3 requires rather than a
-    disclaimer. Each reaches `config.resolved.yaml` and folds into `runs.run_id`, so an unread key
-    is a claim in a run's record that no code backs -- the defect `D-064` names for four other
-    keys. The convention there is to write inertness into the code (`constraints/matrix.py`'s "NO
-    REAL INPUT UNTIL STAGE 6", `errors.py::NoHarvestFactorError`), which is what this is.
-
-    They are inert for different reasons, and the distinction is the useful half:
-
-    - `maximum_major_stratum_wape_degradation` and `nominal_coverage_tolerance` have their INPUT as
-      of R-S5G-1. `validate/metrics.py` now emits a per-census-division WAPE and a per-division
-      90% coverage into `validation_metrics.parquet`, so both gates are evaluable from a shipped
-      artifact. The coverage values `nominal_coverage_tolerance` would gate on carried a
-      residual-sign defect until `D-112` (fixed 2026-09-12, `19fbdec`); one written before that fix
-      must be regenerated before any gate reads it. What is missing is the CANDIDATE to evaluate: §13.10 compares a model against the
-      preferred transparent baseline and Stage 5 produces the model.
-    - `minimum_wape_improvement` is missing both. Its comparison needs a second scoreboard, and
-      `validation_scoreboard.parquet` exists in one copy -- the baseline one.
-    - `catastrophic_stratum_coverage_alpha` is plan 16's. §13.10's "does not fail catastrophically
-      in any major stratum" states no number, and this is the exact binomial lower-tail level the
-      promotion record plan 16 builds will read. It is declared with the `model:` block, in one
-      commit, so `runs.run_id` moves once rather than twice.
-
-    NO EVALUATOR IS BUILT HERE, deliberately. A function whose primary argument is Stage 5's
-    not-yet-designed output would fix that signature by guessing it, and three of §13.10's six
-    gates (hard constraints on draws, convergence diagnostics, disclosure review) are Stage-5 and
-    Stage-8 concepts an evaluator written now could not represent at all. Stage 5 wires these;
-    `tests/unit/test_config_validation_block.py` fails the day `src/` code names one of the keys --
+    ALL FOUR KEYS ARE READ BY `validate/promotion.py`, AND BY NOTHING ELSE. `D-109` recorded the
+    three Appendix A keys as inert until §13.10 had a candidate to evaluate. Plan 16's state-total
+    model is that candidate: `validate-state-model` applies the gates and writes
+    `promotion_record.json`, and every key reaches that record's `thresholds` beside the evidence it
+    was applied to. `catastrophic_stratum_coverage_alpha` is plan 16's own. §13.10's "does not fail
+    catastrophically in any major stratum" states no number, and this is the exact binomial
+    lower-tail level below which a regime's or a Census division's 90% coverage counts as
+    catastrophic.
+
+    THE COVERAGE COMPARISON IS EXACT (`D-109`). Hits over `calibration_sample_size` are a
+    `Fraction`, compared against `Fraction(repr(nominal_coverage_tolerance))`. So 17/20 against
+    0.90 +/- 0.05 counts as within, where the float `abs(0.85 - 0.9) <= 0.05` does not: 44 of
+    `runs/f03023ac9f3a`'s 170 interval-bearing groups sat at exactly 0.85.
+
+    `tests/unit/test_config_validation_block.py` fails the day another `src/` module names a key --
     as an attribute, a string, a parameter or a keyword argument, `config.py` itself included -- so
-    this docstring cannot quietly outlive that truth. A reader that never spells a key exactly -- a
+    this note cannot quietly outlive that truth. A reader that never spells a key exactly -- a
     generic `model_dump()` loop, or a dotted path string such as `attrgetter("promotion.<key>")` --
     would not trip it; update this note by hand in that case.
     """
```

- [ ] **Step 6: Run the tests to see them pass**

Run: `uv run pytest tests/unit/test_validate_promotion.py tests/unit/test_config_validation_block.py -q`

Expected: all pass (`test_validate_promotion.py` observed: `17 passed`).

- [ ] **Step 7: Gates**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
uv run pytest -q -p no:cacheprovider -m "not slow"
```

Expected: gates clean. Non-slow suite: Task 11's count **+ 17 passed**. The converted tripwire keeps its count.

- [ ] **Step 8: Commit**

```bash
git add src/logging_employment/validate/promotion.py src/logging_employment/config.py tests/unit/test_validate_promotion.py tests/unit/test_config_validation_block.py
git commit -m "feat(validate): §13.10's promotion record; the promotion keys are read, exactly (D-109)"
```

---

### Task 13: `validate-state-model`, the command that writes the promotion record

**Implements:**
- §13.10 applied end to end: this run's own `validate` output is the comparand, and the model is scored through Task 11's seam and judged by Task 12's record;
- the roadmap Exit's "the promotion record states beat or not-beaten against the §13.10 comparand named in `Consumes` and the simpler method is selected when not beaten (§13.10 final line)";
- Decision 9: a new command, so `validate` stays byte-identical to Stage 4's, and a failed production fit is recorded as `not_beaten` without spending the replicate fits.

**Files:**
- Modify: `src/logging_employment/cli.py` (`validate_state_model_command`, new)
- Test: `tests/integration/test_cli_validate_state_model.py` (new; 6 tests, the module marked `slow`)

**Interfaces:**
- Consumes:
  - `validate`'s three tables in the run directory: `validation_scores.parquet`, `validation_metrics.parquet` and `validation_scoreboard.parquet`;
  - Task 9's `posterior/diagnostics.json` and store;
  - Task 11's `StateModelProducer` and `run_pseudo_suppression(..., producer=...)`;
  - Task 12's `evaluate_promotion` and `production_failed_record`.
- Produces, under `runs/<run_id>/`:
  - `state_model_validation/`, holding the model's `validation_scores.parquet`, `validation_metrics.parquet`, `validation_scoreboard.parquet` and `validation_manifest.json`;
  - `promotion_record.json`: Task 12's record, plus `run_id`, `model_version` and `comparand`. The comparand holds its `run_id` and the sha256 of each of the three tables. A scored run also carries `model_validation_hashes`.
  - The comparand's own files are read and never rewritten; a test pins their hashes. Either verdict exits 0.

The fixture validation runs one seed, so one replicate fit per regime; `replicates_per_regime: 3` sizes each mask, as the Stage 4 golden's config does. It uses Task 9's small sampler and loose gate. The verdict on this fixture means nothing, and these tests check what the record contains, not which way it went.

- [ ] **Step 1: Write the failing tests**

`tests/integration/test_cli_validate_state_model.py`:

```python
"""`validate-state-model`: §13's harness on the model and §13.10's record, on the committed fixture.

One seed, so one replicate fit per regime (`replicates_per_regime: 3` sizes each mask, as the Stage
4 golden's config does), and the small sampler and loose gate of `test_cli_state_model.py`. Seven
regimes score here, so a passing fit costs seven replicate fits. The verdict on this fixture carries
no meaning: these tests check what the record contains, not which way it went.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from logging_employment.cli import app

pytestmark = pytest.mark.slow

SMALL_SAMPLER = {"chains": 2, "warmup": 60, "draws": 60}
LOOSE_GATE = {
    "diagnostics": {
        "max_rhat": 100.0,
        "min_ess_per_chain": 1,
        "max_divergences": 1_000_000,
        "min_ppc_coverage_90": 0.0,
    }
}
FIXTURE_VALIDATION = {"replicates_per_regime": 3, "pseudo_suppression_seeds": [1024]}
COMPARAND_TABLES = ("validation_scores", "validation_metrics", "validation_scoreboard")


def _invoke(command: str, config: Path):
    return CliRunner().invoke(app, [command, "--config", str(config)])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def validated(make_staged_repo):
    """`validate`, `fit-state-model`, then `validate-state-model`, in the documented order."""
    repo = make_staged_repo(
        {"model": {**SMALL_SAMPLER, **LOOSE_GATE}, "validation": FIXTURE_VALIDATION}
    )
    for command in ("validate", "fit-state-model"):
        result = _invoke(command, repo.config_path)
        assert result.exit_code == 0, result.output
    before = {table: _sha256(repo.run_dir / f"{table}.parquet") for table in COMPARAND_TABLES}
    result = _invoke("validate-state-model", repo.config_path)
    assert result.exit_code == 0, result.output
    return repo, before


def test_the_record_states_a_verdict_and_what_it_was_measured_against(validated) -> None:
    repo, before = validated
    record = json.loads((repo.run_dir / "promotion_record.json").read_text())
    assert record["verdict"] in {"beat", "not_beaten"}
    assert record["selected_method"] == (
        "state_total_model" if record["verdict"] == "beat" else "section_10_8_hierarchy"
    )
    assert record["provisional"] is True
    assert record["disclosure_review"] == "pending_stage_8"
    assert record["comparand"] == {
        "run_id": record["run_id"],
        **{f"{t}_sha256": h for t, h in before.items()},
    }
    assert set(record["gates"]) == {
        "hard_constraints",
        "convergence",
        "coverage",
        "improvement",
        "stratum_degradation",
        "disclosure_review",
    }


def test_the_comparand_tables_are_read_never_rewritten(validated) -> None:
    repo, before = validated
    after = {table: _sha256(repo.run_dir / f"{table}.parquet") for table in COMPARAND_TABLES}
    assert after == before


def test_the_models_intervals_come_from_its_reconciled_draws(validated) -> None:
    repo, _before = validated
    metrics = pl.read_parquet(
        repo.run_dir / "state_model_validation" / "validation_metrics.parquet"
    )
    coverage = metrics.filter(pl.col("metric_name") == "coverage_0.90")
    assert set(coverage["estimator_id"]) == {"state_total_model"}
    assert set(coverage["interval_source"]) <= {"reconciled_posterior_draws", "none"}
    assert "reconciled_posterior_draws" in set(coverage["interval_source"])


def test_every_replicate_fit_left_a_gate_report(validated) -> None:
    """§13.10's convergence gate reads these: one per (regime, seed) that scored."""
    repo, _before = validated
    manifest = json.loads(
        (repo.run_dir / "state_model_validation" / "validation_manifest.json").read_text()
    )
    for entry in manifest["regimes"].values():
        notes = entry.get("producer_notes", [])
        assert len(notes) == entry["replicates"]
        assert all(note["gate"]["scope"] == "replicate" for note in notes)
    record = json.loads((repo.run_dir / "promotion_record.json").read_text())
    replicates = sum(entry["replicates"] for entry in manifest["regimes"].values())
    assert record["gates"]["convergence"]["replicates_checked"] == replicates


def test_a_failed_production_fit_is_recorded_without_scoring(make_staged_repo) -> None:
    repo = make_staged_repo(
        {
            "model": {**SMALL_SAMPLER, "diagnostics": {"max_rhat": 0.5}},
            "validation": FIXTURE_VALIDATION,
        }
    )
    assert _invoke("validate", repo.config_path).exit_code == 0
    assert _invoke("fit-state-model", repo.config_path).exit_code == 1
    result = _invoke("validate-state-model", repo.config_path)
    assert result.exit_code == 0, result.output
    record = json.loads((repo.run_dir / "promotion_record.json").read_text())
    assert (record["verdict"], record["selected_method"]) == (
        "not_beaten",
        "section_10_8_hierarchy",
    )
    assert record["gates"]["coverage"]["status"] == "not_evaluated_production_fit_failed"
    assert not (repo.run_dir / "state_model_validation" / "validation_scores.parquet").exists()


def test_the_command_requires_the_comparand(make_staged_repo) -> None:
    repo = make_staged_repo({"model": SMALL_SAMPLER, "validation": FIXTURE_VALIDATION})
    result = _invoke("validate-state-model", repo.config_path)
    assert result.exit_code != 0
    # Short tokens only, as `test_baseline_cli.py` explains: Typer boxes and hard-wraps the message.
    # Not "validate": Typer's "No such command 'validate-state-model'" would contain it too.
    assert "§13.10" in result.output
    assert "comparand" in result.output
```

- [ ] **Step 2: Run them to watch them fail**

Run: `uv run pytest tests/integration/test_cli_validate_state_model.py -q`

Expected: all six fail. Observed for the two whose fixtures write no store:

```text
E       AssertionError: Usage: root [OPTIONS] COMMAND [ARGS]...
E         Try 'root --help' for help.
E         ╭─ Error ──────────────────────────────────────────────────────────────────────╮
E         │ No such command 'validate-state-model'. Did you mean 'fit-state-model'?      │
E         ╰──────────────────────────────────────────────────────────────────────────────╯
E       assert 2 == 0
E        +  where 2 = <Result SystemExit(2)>.exit_code
E       assert '§13.10' in "Usage: root [OPTIONS] COMMAND [ARGS]...\nTry 'root --help' for help.\n╭─ Error ──────────────────────────────────────...you mean 'fit-state-model'?      │\n╰──────────────────────────────────────────────────────────────────────────────╯\n"
E        +  where "Usage: root [OPTIONS] COMMAND [ARGS]...\nTry 'root --help' for help.\n╭─ Error ──────────────────────────────────────...you mean 'fit-state-model'?      │\n╰──────────────────────────────────────────────────────────────────────────────╯\n" = <Result SystemExit(2)>.output
FAILED tests/integration/test_cli_validate_state_model.py::test_a_failed_production_fit_is_recorded_without_scoring
FAILED tests/integration/test_cli_validate_state_model.py::test_the_command_requires_the_comparand
2 failed, 4 deselected
```

The other four were deselected while this plan was written, because the `validated` fixture's fit writes a store (Task 7's caveat). Each errors at setup here, on `No such command 'validate-state-model'`.

- [ ] **Step 3: Add the command**

`src/logging_employment/cli.py`:

```diff
diff --git a/src/logging_employment/cli.py b/src/logging_employment/cli.py
index d42d359..c9c73b7 100644
--- a/src/logging_employment/cli.py
+++ b/src/logging_employment/cli.py
@@ -672,3 +672,132 @@ def validate_command(
     _write_manifest(run / "validation_manifest.json", manifest)
     for regime, entry in sorted(result.manifest["regimes"].items()):
         typer.echo(f"{regime} {entry['disposition']} scored={entry['n_scored']}")
+
+
+@app.command("validate-state-model")
+def validate_state_model_command(
+    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
+) -> None:
+    """Score §11's model through §13's harness and write §13.10's promotion record.
+
+    Not in §16.1's list: plan 16 adds it so `validate` stays the comparand's command, byte-identical
+    to Stage 4's. The comparand is this run's own `validate` output. The precondition is that
+    output, plus `fit-state-model`'s `posterior/diagnostics.json`. A production fit that failed
+    §11.14 is recorded as not beaten without running the harness. Otherwise the model is scored
+    through `run_pseudo_suppression`'s own loop (`StateModelProducer`), which is 27 fits on D1. Its
+    tables go to `state_model_validation/` and the verdict to `promotion_record.json`. Both verdicts
+    exit 0, because "deploy the simpler method" is an outcome and not an error.
+    """
+    import hashlib
+    import json
+    import shutil
+
+    import polars as pl
+
+    from .build import write_parquet_deterministic
+    from .contracts import (
+        VALIDATION_METRIC_SCHEMA,
+        VALIDATION_SCORE_SCHEMA,
+        VALIDATION_SCOREBOARD_SCHEMA,
+        HarmonizedData,
+        assert_required_columns_present,
+        validate_frame,
+    )
+    from .models.arviz_io import draws_digest, read_store, store_digest
+    from .models.interfaces import MODEL_ID, MODEL_VERSION, STORE_PATH
+    from .models.reconciliation import check_reconciled
+    from .models.validation import StateModelProducer
+    from .runs import run_dir, run_id
+    from .validate.harness import run_pseudo_suppression
+    from .validate.promotion import evaluate_promotion, production_failed_record
+
+    cfg = load_config(config)
+    rid = run_id(cfg, _input_digests(cfg))
+    run = run_dir(cfg, rid)
+    comparand = {
+        table: run / f"{table}.parquet"
+        for table in ("validation_scores", "validation_metrics", "validation_scoreboard")
+    }
+    diagnostics = run / "posterior" / "diagnostics.json"
+    for path, command in [
+        *((p, "validate") for p in comparand.values()),
+        (diagnostics, "fit-state-model"),
+    ]:
+        if not path.exists():
+            raise typer.BadParameter(
+                f"{path} is missing: §13.10 compares the model against this run's own comparand "
+                f"and reads its production gate. Run `{command}` first"
+            )
+    out = run / "state_model_validation"
+    shutil.rmtree(out, ignore_errors=True)
+    (run / "promotion_record.json").unlink(missing_ok=True)
+    out.mkdir(parents=True)
+    envelope = {
+        "run_id": rid,
+        "model_version": MODEL_VERSION,
+        "comparand": {
+            "run_id": rid,
+            **{
+                f"{table}_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
+                for table, path in comparand.items()
+            },
+        },
+    }
+    production_gate = json.loads(diagnostics.read_text())
+    if not production_gate["passed"]:
+        record = production_failed_record(MODEL_ID, production_gate, cfg.promotion)
+        _write_manifest(run / "promotion_record.json", {**record, **envelope})
+        typer.echo("production fit failed §11.14: not_beaten, section_10_8_hierarchy selected")
+        return
+
+    store = run / STORE_PATH
+    draws = read_store(store)
+    check = check_reconciled(draws, tolerance=cfg.reconciliation.tolerance)
+    store_check = {
+        "max_anchor_drift": check.max_anchor_drift,
+        "bound_violations": check.bound_violations,
+        "draws_sha256_matches": draws_digest(draws) == store_digest(store),
+    }
+    store_check["passed"] = check.passed and bool(store_check["draws_sha256_matches"])
+
+    data = HarmonizedData.load(Path(cfg.storage.staged_uri))
+    result = run_pseudo_suppression(data, config=cfg, producer=StateModelProducer())
+    for frame, schema, table in (
+        (result.scores, VALIDATION_SCORE_SCHEMA, "validation_scores"),
+        (result.metrics, VALIDATION_METRIC_SCHEMA, "validation_metrics"),
+        (result.scoreboard, VALIDATION_SCOREBOARD_SCHEMA, "validation_scoreboard"),
+    ):
+        validate_frame(frame, schema, table)
+        assert_required_columns_present(frame, table)
+    hashes = {
+        table: write_parquet_deterministic(frame, out / f"{table}.parquet")
+        for table, frame in (
+            ("validation_scores", result.scores),
+            ("validation_metrics", result.metrics),
+            ("validation_scoreboard", result.scoreboard),
+        )
+    }
+    _write_manifest(
+        out / "validation_manifest.json",
+        {**result.manifest, "estimators": [MODEL_ID], "output_hashes": hashes},
+    )
+    record = evaluate_promotion(
+        model_id=MODEL_ID,
+        model_scores=result.scores,
+        model_metrics=result.metrics,
+        comparand_scores=pl.read_parquet(comparand["validation_scores"]),
+        comparand_metrics=pl.read_parquet(comparand["validation_metrics"]),
+        comparand_scoreboard=pl.read_parquet(comparand["validation_scoreboard"]),
+        production_gate=production_gate,
+        production_store_check=store_check,
+        replicate_gates={
+            regime: list(entry["producer_notes"])
+            for regime, entry in result.manifest["regimes"].items()
+            if entry.get("producer_notes")
+        },
+        promotion=cfg.promotion,
+    )
+    _write_manifest(
+        run / "promotion_record.json", {**record, **envelope, "model_validation_hashes": hashes}
+    )
+    typer.echo(f"verdict {record['verdict']}; selected {record['selected_method']}")
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/integration/test_cli_validate_state_model.py -q`

Expected: `6 passed`. The module-scoped fixture runs `validate`, `fit-state-model` and then seven replicate fits, so allow several minutes. **Four of the six were never executed while this plan was written** (the `validated` fixture). The two that ran passed: `2 passed, 4 deselected`. If only the four fail, re-run Task 7's store tests and Task 9's store tests first.

- [ ] **Step 5: Gates**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
uv run pytest -q -p no:cacheprovider -m "not slow"
```

Expected: gates clean. Non-slow suite: **unchanged from Task 12**, because the module is `slow`. The six ran in Step 4.

- [ ] **Step 6: Commit**

```bash
git add src/logging_employment/cli.py tests/integration/test_cli_validate_state_model.py
git commit -m "feat(cli): validate-state-model scores the model and writes §13.10's promotion record"
```

---

### Task 14: Documentation, and the whole suite measured

**Implements:**
- the repo's convention that each package carries a `CLAUDE.md` that says what it owns, what it refuses and where its tests are;
- the root `CLAUDE.md`'s command list, module map, marker census, CI counts and gotchas, updated for a package that did not exist;
- Decision 13's whole-suite run, slow tier included, and the counts measured rather than asserted.

Nothing here claims Task 15's results. Task 15 writes what the D1 re-run and the fit showed after they have shown it.

**Files:**
- Create: `src/logging_employment/models/CLAUDE.md`
- Modify: `CLAUDE.md`, `.github/workflows/ci.yml` (its header comment only), `src/logging_employment/validate/CLAUDE.md`, `src/logging_employment/reconcile/CLAUDE.md`, `src/logging_employment/baselines/CLAUDE.md`

**Interfaces:** none. Documentation only.

- [ ] **Step 1: Write the package's own guide**

`src/logging_employment/models/CLAUDE.md`:

````markdown
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
  Every prior is a `config.StateModelPriors` field, so it is in `resolved_dict` and in `run_id`.

## The gate and its two scopes

`diagnostics.evaluate_gate` measures divergences, parameter R-hat, cell R-hat and bulk and tail
ESS over the imputed cells' reconciled draws, one-step-ahead 90% predictive coverage over the
training cells (they are exact, so an in-sample check would pass by construction), and the two
reconciliation checks. `production` gates on all of them and `fit-state-model` raises
`ModelDiagnosticsError` after writing `posterior/diagnostics.json`. `replicate`, the harness's 27
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
before it scores anything.

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
````

- [ ] **Step 2: Update the root guide, the CI comment, and the three neighbouring guides**

`CLAUDE.md` and `.github/workflows/ci.yml`:

````diff
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 8d85cbe..14b6729 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -6,12 +6,13 @@
 # gitignored path or a personal file outside the repo (D-055), skips here by design. A green run
 # is therefore no evidence about the D1 integration tests: run those where data/ lives.
 #
-# `-m "not slow and not network"` states the policy the markers declare. Measured 2026-09-13
-# without data/ on the author's Mac: the bare run gives 1408 passed, 72 skipped; this one 1408
-# passed, 45 skipped, 27 deselected. A runner reports 1407 passed, 46 skipped: one audit test skips
-# without a personal file outside the repo (D-055). The six `mark.slow` sites are mostly module-level, so they reach 27 tests, all of
-# which skip without data/, and no test carries `network`: the expression removes nothing that
-# would have run today. It keeps a future slow or live-endpoint test out of this tier instead.
+# `-m "not slow and not network"` states the policy the markers declare. Measured at plan 16's
+# Task 14 without data/ on the author's Mac: the bare run gives 1633 passed, 72 skipped; this one
+# 1612 passed, 45 skipped, 48 deselected. A runner reports one more skip: one audit test skips
+# without a personal file outside the repo (D-055). Of the 48 deselected, 27 are data-bound and
+# would skip here anyway. The other 21 are plan 16's slow state-total model tests, which need no
+# data/ and WOULD run: the expression keeps their minutes of NUTS sampling out of this tier on
+# purpose, and they run locally with `pytest -m slow`. No test carries `network`.
 name: CI
 
 on:
diff --git a/CLAUDE.md b/CLAUDE.md
index af0104e..06fe565 100644
--- a/CLAUDE.md
+++ b/CLAUDE.md
@@ -3,8 +3,10 @@
 Monthly state Logging (NAICS 113310, private ownership, states + DC, 2017-01..2024-12) employment
 by establishment size class, for the cells BLS suppresses for disclosure. Identification precedes
 imputation: sharp LP/MILP bounds from public accounting facts (§9), then transparent baselines
-(§10), exact reconciliation (§12), then a pseudo-suppression harness (§13). Sources are QCEW,
-QCEW-by-size and CBP. The §11 Bayesian model is not built yet.
+(§10), exact reconciliation (§12), a pseudo-suppression harness (§13), and §11's state-total model,
+scored by that same harness (plan 16). Sources are QCEW, QCEW-by-size and CBP. §11's
+size-composition model is Stage 6's, and the state model's §13.10 promotion record stays
+provisional until Stage 7 re-runs the gate.
 
 **The spec is authoritative and this codebase cites it constantly.**
 `specs/logging-employment-spec.md` — §3 estimand, §4 invariants (`INV-001`..`INV-016`), §6.2
@@ -28,33 +30,47 @@ logging-estimates fetch --source qcew --config config.yaml   # or qcew_parent /
                                                              # CENSUS_API_KEY from ./.env
 # then, in order:
 build-harmonized → build-constraints → solve-bounds → run-baselines → reconcile → validate
+# §11's state-total model (plan 16), on the same run directory:
+fit-state-model   # after solve-bounds; `reconcile` then also re-verifies the stored draws
+validate-state-model   # after validate and fit-state-model; writes promotion_record.json
 ```
 
 Every command takes `--config config.yaml` and must be idempotent for the same inputs (§16.1);
 each gates on the previous one's artifact and refuses with the missing path
-(`cli.py::{solve_bounds_command, run_baselines_command, reconcile_command}` — symbols, not line
-numbers, because plan 13 drifted the old `:250` / `:349` pins by inserting above them).
+(`cli.py::{solve_bounds_command, run_baselines_command, reconcile_command,
+fit_state_model_command, validate_state_model_command}` — symbols, not line numbers, because plan
+13 drifted the old `:250` / `:349` pins by inserting above them).
 **`run-baselines` gates on TWO** as of plan 13 (R-S5P-3): `schema_manifest.json` *and*
 `deterministic_bounds.parquet`, so `solve-bounds` is a hard precondition rather than an advisory
 step in the documented order — every estimate is checked against its §9 interval (INV-002's
 per-cell half) instead of the check being skipped when its input is absent. `validate --estimators a,b` scores a subset in its own run dir. `--help` is the real
-command list — five of §16.1's fifteen do not exist yet (`fit-state-model`, `fit-size-model`,
-`disclosure-review`, `publish`, `run-all`).
-
-Markers are declared but never applied by `addopts`: `slow` is on one unit test and FIVE
-integration modules (measured 2026-09-10: six `mark.slow` sites) and nothing excludes it locally (pass `-m "not slow"` yourself), and `network` is
-declared — its help text even says "excluded from the default run" — but no test carries it.
+command list — four of §16.1's fifteen do not exist yet (`fit-size-model`, `disclosure-review`,
+`publish`, `run-all`). `validate-state-model` is not one of the fifteen: plan 16 added it so that
+`validate` stays the §13.10 comparand's command, byte for byte. `fit-state-model` gates on
+`schema_manifest.json` and `deterministic_bounds.parquet`; `validate-state-model` on the three
+`validation_*` tables and `posterior/diagnostics.json`.
+
+Markers are declared but never applied by `addopts`: `slow` is on one unit test and EIGHT
+integration modules (thirteen `mark.slow` sites since plan 16, which added the three model
+modules: every NUTS fit bigger than `test_state_total_model.py`'s toy) and nothing excludes it
+locally (pass `-m "not slow"` yourself), and `network` is declared — its help text even says
+"excluded from the default run" — but no test carries it.
 
 **CI** (`.github/workflows/ci.yml`, added 2026-09-13) runs the four gates above on every push to
 `main` and every PR, after `uv sync --locked`, with `pytest -m "not slow and not network"`. It is
 the hermetic tier only: no runner has `data/`, so every data-bound test skips there by design, and
 a green check says nothing about the D1 integration tests — those still need a local run where
-`data/` lives. Measured 2026-09-13 without `data/`: the expression deselects 27 tests (the six
-`slow` sites are mostly module-level) and passed stays at 1408, so it removes nothing that would
-have run. Nothing else is deselected. Those counts are THIS MAC's: ubuntu-latest reports 1407
-passed, 46 skipped, 27 deselected (run 34765933053), because
+`data/` lives. Since plan 16 the expression DOES remove tests that would have run: the 21 slow
+model tests (5 in `test_cli_state_model.py`, 6 in `test_cli_validate_state_model.py`, 10 in
+`test_state_total_recovery.py`) need no `data/`, only minutes of NUTS. Collected at plan 16's
+Task 14 without `data/`: a bare run is 1633 passed, 72 skipped; the hermetic tier 1612 passed, 45
+skipped, 48 deselected (27 data-bound, 21 slow model tests). Those counts are THIS MAC's: on
+ubuntu-latest one more test skips (1407 passed, 46 skipped, 27 deselected before plan 16, run
+34765933053), because
 `tests/audit/test_qcew_codes.py::test_period_basis_quotes_the_reference_verbatim_where_the_reference_is_readable`
-skips where the personal `~/.claude/skills/bls-data-context/` reference is absent (D-055).
+skips where the personal `~/.claude/skills/bls-data-context/` reference is absent (D-055). `uv sync
+--locked` installs JAX, NumPyro, ArviZ and h5netcdf there since plan 16, and the hermetic tier runs
+the API probes and the toy fits.
 
 **Float goldens compare through `tests/golden_compare.py`, not `.equals`** (2026-09-13). The two
 float goldens (`test_validation_golden.py::test_the_metrics_match_the_golden`,
@@ -75,7 +91,9 @@ diff old against new by join before re-pinning one (§17.6).
 five harmonized Parquet tables (`contracts.HarmonizedData`), never an endpoint. `build-constraints`
 turns those into a cell/row/coefficient system, `solve-bounds` bounds each component,
 `run-baselines` produces weights that `reconcile/` turns into estimates, `validate` re-runs it under
-synthetic masks. Outputs land in `runs/<run_id>/` beside one JSON manifest per command.
+synthetic masks. `fit-state-model` fits §11's state-total model and passes every draw through the
+same `reconcile/` layer, and `validate-state-model` scores it through the same harness. Outputs land
+in `runs/<run_id>/` beside one JSON manifest per command.
 
 | Module | Owns |
 |---|---|
@@ -94,6 +112,7 @@ synthetic masks. Outputs land in `runs/<run_id>/` beside one JSON manifest per c
 | `reconcile/` | §12 exact reconciliation → see `reconcile/CLAUDE.md` |
 | `baselines/` | §10 transparent baselines → see `baselines/CLAUDE.md` |
 | `validate/` | §13 pseudo-suppression harness → see `validate/CLAUDE.md` |
+| `models/` | §11's state-total model: `ModelData`, the NumPyro fit, per-draw reconciliation, the §11.14 gate, §7.11's `posterior_summary`, the ArviZ store → see `models/CLAUDE.md` |
 | `registry/` | §7.1 source registry: row model, `registry/sources.yaml`, `registry verify`'s checks |
 | `disclosure/` | §9.8 flags only — `exact_reconstruction_flag`, `narrow_feasible_interval_flag`, on suppressed cells only, thresholds from config |
 
@@ -177,6 +196,10 @@ synthetic masks. Outputs land in `runs/<run_id>/` beside one JSON manifest per c
   fields to `SourcesConfig` and moved no id, because each is `Field(default=None, exclude=True)`
   and so reaches no dump — that is the second remedy, for a key nothing outside `config.py` reads. For a CLI-only choice use `run_id`'s `overrides`,
   omitting the key when unset (`runs.py` docstring), so existing runs keep their id.
+  **Plan 16 re-identified every run once, on purpose**: `model:` stays in the dump because every
+  key in it changes the draws, and `promotion.catastrophic_stratum_coverage_alpha` was added in the
+  same commit so the id moved once. The config-only canary moved `39d1d0859838` → `14352bb8e56e`
+  and the staged pin `4cf47a918dd8` → `dd7337e89047`.
 - **A run directory can be stale w.r.t. your code.** `run_id` ignores source, so editing an
   estimator and re-running overwrites the same `runs/<id>/`. The one cross-stage check that does
   fire is `constraint_set_hash` (see `constraints/CLAUDE.md`).
@@ -184,6 +207,11 @@ synthetic masks. Outputs land in `runs/<run_id>/` beside one JSON manifest per c
   re-fetch stores another object for the same year. `build.snapshot_paths` raises
   `AmbiguousSnapshotError` (`build.py::snapshot_paths`) rather than stacking two snapshots of one key; pass the
   run manifest.
+- **`models/state_total.py` switches JAX to float64 when it is imported** (`numpyro.enable_x64()`),
+  and `_fit_numpyro` refuses to sample in float32, because `reconcile_draws` checks adding-up to
+  1e-9. Import it before any other JAX work. `cli.py` imports every model module inside the command
+  that needs it, so `--help`, `validate` and the baseline commands never initialise JAX. Chains run
+  `vectorized` in one process, so `numpyro.set_host_device_count` is never called.
 - **Government APIs answer 200 with an error body.** `ingest/base.HttpFetcher` returns
   non-200 responses instead of raising, exactly so the caller classifies on content, not status.
 - **`scripts/audit/` is destructive-first.** Standalone PEP 723 files (`uv run --no-project`),
````

`src/logging_employment/validate/CLAUDE.md`, `src/logging_employment/reconcile/CLAUDE.md` and `src/logging_employment/baselines/CLAUDE.md`:

````diff
diff --git a/src/logging_employment/baselines/CLAUDE.md b/src/logging_employment/baselines/CLAUDE.md
index 6d43794..13be053 100644
--- a/src/logging_employment/baselines/CLAUDE.md
+++ b/src/logging_employment/baselines/CLAUDE.md
@@ -79,11 +79,18 @@ after scaling — a defect, never a `Decline` row. `cli.py` passes the run direc
 `validate/CLAUDE.md`).
 
 Consumers outside this package: `validate/harness.py` imports `Estimator` and
-`runner.{REGISTRY, run_baselines}`; `validate/scoreboard.py` imports
+`runner.{REGISTRY, run_baselines, state_total_bounds}`; `validate/scoreboard.py` imports
 `runner.{FALLBACK_RUNGS, PREFERRABLE, RUNG_OF}`; `cli.py` imports `runner.{REGISTRY,
 run_baselines, resolve_estimators, preferred_estimator, preferred_estimator_by_month,
-state_total_bounds}`. Nothing
-outside imports an estimator class directly.
+state_total_bounds}`. Since plan 16 `models/` imports the runner's bound and identifier helpers so
+the model is held to the baselines' rules rather than a copy of them: `models/reconciliation.py`
+takes `runner.{assert_bounds_cover, missing_cell_ids, month_bounds}`, `models/summary.py`
+`runner.state_total_bounds`, and `models/validation.py` `runner.{assert_within_bounds,
+release_integers, state_total_bounds}`. `missing_cell_ids` (formerly the private `_cell_ids`) and
+`release_integers` (§12.6's integer cut and its two checks, extracted verbatim from
+`run_baselines`) became public for that reason; `run_baselines`' output is unchanged, which
+`tests/integration/test_baseline_golden.py` pins. Nothing outside imports an estimator class
+directly.
 
 ## Adding an estimator
 
diff --git a/src/logging_employment/reconcile/CLAUDE.md b/src/logging_employment/reconcile/CLAUDE.md
index f03d98d..8c55da0 100644
--- a/src/logging_employment/reconcile/CLAUDE.md
+++ b/src/logging_employment/reconcile/CLAUDE.md
@@ -9,7 +9,8 @@ post-hoc cosmetic adjustment" — there is no config key to skip it.
 
 ## Read this first: most of this package has no production caller yet
 
-Nothing outside `baselines/` calls into this package. `baselines/runner.py::run_baselines` is the
+Two production paths call into this package: `baselines/` since Stage 3, and `models/` since plan
+16. `baselines/runner.py::run_baselines` is the
 production path and calls, in order: `observed_partition` → `closure_audit` →
 `assert_universe_closes` → `national_residual` → `allocate` → `integerize`, and (since R-S5P-3)
 reads `scaling.Bounds` so `runner.assert_within_bounds` can enforce INV-002's per-cell half. Since
@@ -18,10 +19,16 @@ plan 15 it also calls `scaling.scale_into_bounds` wherever `allocate` leaves a f
 context's partition and refuse an anchor that disagrees. The other six `baselines/` modules import
 `Weights` / `Anchor` / `Partition` as types only.
 
+`models/reconciliation.py::reconcile_fit` (plan 16) runs the same gate, `observed_partition` →
+`closure_audit` → `assert_universe_closes`, then calls `national_residual` and
+**`draws.reconcile_draws`** once per month, with a `ReconciliationInputs` built from that month's
+anchor and `deterministic_bounds` projected by `baselines.runner.month_bounds`. Every draw of a
+state-total fit passes through it, and nothing reconciles a draw any other way. And
+`models/arviz_io.py::read_store` rebuilds `Anchor`s from a store's recorded residuals and bases,
+and each one re-checks its basis as it is built (`D-122`, below).
+
 Everything else is implemented, tested, and **dead until a later stage wires it**:
 
-- `draws.reconcile_draws` — no caller; Stage 5 builds `ReconciliationInputs` from
-  `deterministic_bounds` plus the anchor.
 - `matrix.reconcile_matrix` / `projection.kl_project` — no caller; Stage 6, when `target_cell` first
   carries a state×size cell. `matrix.py`'s docstring says so explicitly: "NO REAL INPUT UNTIL STAGE
   6."
@@ -35,7 +42,9 @@ Do not "fix" the empty call sites by inventing one. Do not delete them as dead c
 
 **Name collision:** the `logging-estimates reconcile` CLI command (`cli.py`) imports nothing
 from this package. It re-sums persisted `baseline_results.parquet` estimates against each month's
-recorded residual and reports drift — a verifier, not a producer.
+recorded residual and reports drift, and since plan 16 it also re-checks a stored state-total fit
+(`models.reconciliation.check_reconciled` over `models.arviz_io.read_store`) — a verifier, not a
+producer, either way.
 
 ## The anchor is a modeling assumption, not a constraint
 
@@ -61,6 +70,9 @@ The load-bearing points:
   `E_{s,t} <= R_t` **must never be written back into `deterministic_bounds`**.
 - Honest caveat already in the docstring: `qtrly_establishments` is constant within a quarter, so
   the per-month gate is quarterly-resolution evidence in a monthly shape.
+- §12.2 says every row allocated against R_t carries its basis. §7.11's `posterior_summary` has no
+  such column, so plan 16 records it in `state_model_manifest.json` (`anchor_bases`) and in the
+  store's `constant_data`, where `models/arviz_io.py::read_store` reads it back into each `Anchor`.
 
 ## Contracts a fresh agent gets wrong
 
@@ -72,7 +84,8 @@ The load-bearing points:
 - **An `Anchor` checks its own basis.** `Anchor.__post_init__` refuses an `anchor_basis` outside
   `contracts.ANCHOR_BASES` with `ConceptViolationError` (`D-122`). The frame guard
   `assert_declared_provenance` cannot cover it: `reconcile_draws` takes the `Anchor` itself, and
-  §7.11's `posterior_summary` carries no `anchor_basis` column.
+  §7.11's `posterior_summary` carries no `anchor_basis` column. Since plan 16 the guard also runs on
+  every anchor `models/arviz_io.py::read_store` rebuilds from a store on disk.
 - **`closure_audit` iterates the months in `monthly`, not the keys of `partitions`**
   (`anchor.py::closure_audit`-171`). A month with a national row and no state rows — the shape a truncated
   ingest produces — would otherwise be skipped in silence.
diff --git a/src/logging_employment/validate/CLAUDE.md b/src/logging_employment/validate/CLAUDE.md
index 97c13a5..0893c03 100644
--- a/src/logging_employment/validate/CLAUDE.md
+++ b/src/logging_employment/validate/CLAUDE.md
@@ -3,8 +3,9 @@
 Hides cells QCEW actually published, re-runs the §10 baselines and rebuilds and re-solves the §9
 constraint system on the masked frame, then scores the estimates against the withheld truth. Spec
 map: §13.2 → `propensity` + `regimes`, §13.3 → `regimes`, §13.4 → `leakage`, §13.5-13.8 →
-`metrics`, §13.10 → `scoreboard`; also §10.7 → `intervals`, §10.8 → `scoreboard`'s hierarchy
-restriction, §16.2 → `harness.run_pseudo_suppression`.
+`metrics`, §13.10 → `scoreboard` (the comparand) and `promotion` (the record, plan 16); also §10.7
+→ `intervals`, §10.8 → `scoreboard`'s hierarchy restriction, §16.2 →
+`harness.run_pseudo_suppression`.
 
 ## Read this first: the config knobs do not do what they look like
 
@@ -42,11 +43,25 @@ restriction, §16.2 → `harness.run_pseudo_suppression`.
 
 ## The boundary
 
-One entry point: `run_pseudo_suppression(data, estimators, config) -> ValidationResult(scores,
-metrics, scoreboard, manifest)`. The only production caller is `cli.py::validate_command`
-(`logging-estimates validate`), which writes `validation_scores.parquet`,
-`validation_metrics.parquet`, `validation_scoreboard.parquet` and `validation_manifest.json` into
-`runs/<id>/`.
+One entry point: `run_pseudo_suppression(data, estimators=None, config=None, *, producer=None) ->
+ValidationResult(scores, metrics, scoreboard, manifest)`. Two production callers.
+`cli.py::validate_command` (`logging-estimates validate`) scores §10's registry and writes
+`validation_scores.parquet`, `validation_metrics.parquet`, `validation_scoreboard.parquet` and
+`validation_manifest.json` into `runs/<id>/`. Since plan 16, `cli.py::validate_state_model_command`
+scores §11's model through `producer=StateModelProducer()` and writes the same three tables into
+`runs/<id>/state_model_validation/`, then §13.10's record to `runs/<id>/promotion_record.json`.
+
+- **The producer seam (plan 16).** A `harness.Producer` turns one masked frame and its
+  `MaskedSystem` into `Production(results, interval_metrics, notes)`. `BaselineProducer` wraps
+  `run_baselines` under the masked bounds (`D-087`) with `probabilistic_metrics`, and it is what
+  `estimators=` builds, so `validate`'s output is byte-identical to Stage 4's.
+  `models/validation.py::StateModelProducer` fits the model on the masked frame, reconciles every
+  draw into the MASKED bounds, and scores each cell's interval from its own reconciled draws
+  (`metrics.draw_interval_metrics`, `interval_source = 'reconciled_posterior_draws'`). Everything
+  else is shared, which is the point of the seam: the masks, the leakage guard, §13.5's halt, step
+  6's rejection and every emitter. Passing both `estimators` and `producer` is refused. `notes`
+  reach the regime's manifest entry as `producer_notes` only when non-empty, which is how each
+  replicate fit's §11.14 gate report reaches §13.10's convergence gate.
 
 - **§16.2's signature is deviated from deliberately.** The spec writes `config: ValidationConfig`;
   the implementation takes the whole `Config` (it needs `config.constraints` for `solve_bounds` and
@@ -220,6 +235,34 @@ metrics, scoreboard, manifest)`. The only production caller is `cli.py::validate
 - `intervals` offers CRPS and refuses log score on purpose: an empirical ensemble gives -inf
   whenever the truth falls outside its range. §13.7 permits either.
 
+## §13.10's promotion record (plan 16)
+
+`promotion.evaluate_promotion` is the ONLY reader of `config.promotion`'s four keys, and
+`tests/unit/test_config_validation_block.py::test_the_promotion_keys_are_read_only_by_the_promotion_record`
+reddens on a second reader. It compares the model's `state_model_validation/` tables against the
+SAME run's `validate` tables, whose digests the record carries. The readings §13.10 leaves open are
+plan 16's Decision 4, and each is written into the record beside its evidence:
+
+- **Coverage is pooled** over every regime and seed, and per-regime values are reported but gate
+  nothing, because `whole_seasonal` is about 69% of the pool (`D-106`). The comparison is EXACT
+  (`D-109`): hits over `calibration_sample_size` as a `Fraction` against
+  `Fraction(repr(tolerance))`, so 17/20 is within 0.90 ± 0.05.
+- **Catastrophic** means the exact binomial lower tail at 0.90 falls below
+  `catastrophic_stratum_coverage_alpha` (0.001), per regime and per Census division pooled.
+- **WAPE improvement** is per regime, on matched (seed, cell) pairs, relative, at least
+  `minimum_wape_improvement`. A regime whose comparand is `None` (`scoreboard.preferred_baseline`)
+  is `not_applicable`, and the gate fails when no regime has a comparand at all.
+- **Degradation** is per division pooled across regimes, relative, above
+  `maximum_major_stratum_wape_degradation`.
+- The hard-constraint and convergence gates read the production fit AND every replicate fit's
+  report (`producer_notes`). Hard constraints read each replicate's reconciliation checks, which
+  the `replicate` scope records without gating; convergence reads each replicate's own verdict,
+  which gates on divergences and parameter R-hat only (`models/diagnostics.py`).
+- The verdict is `beat` or `not_beaten`, `selected_method` is the model or
+  `section_10_8_hierarchy`, `provisional` is always true (Stage 7 re-runs the gate with the harvest
+  factor), and disclosure review is `pending_stage_8`. A production fit that failed §11.14 gets a
+  `not_beaten` record without being scored.
+
 ## Tests, commands, and the environment
 
 ```bash
@@ -228,7 +271,10 @@ uv run pytest tests/unit/test_validate_scoreboard.py tests/unit/test_validate_in
   tests/unit/test_validate_metrics_bounds.py tests/unit/test_validate_metrics_constraint.py
   # 62 passed (measured 2026-09-12 at `19fbdec`, +3 for D-112's sign tests; 59 after plan 14's review fixes, 40 over the five modules before plan 14) — no data/ needed
 uv run pytest tests/integration/test_validation_golden.py   # 7 passed, 11s — in-git fixtures
+uv run pytest tests/unit/test_validate_producer_seam.py tests/unit/test_validate_metrics_draws.py \
+  tests/unit/test_validate_promotion.py   # plan 16's seam, draw intervals and record — no data/
 uv run logging-estimates validate --config config.yaml [--estimators id,id]
+uv run logging-estimates validate-state-model --config config.yaml   # 27 fits on D1 (plan 16)
 ```
 
 - **Every `tests/unit/test_validate_*.py` module that loads `data/staged` is now guarded**
````

- [ ] **Step 3: Re-measure the marker census the root guide now states**

Run: `grep -rn "mark\.slow" tests | wc -l`

Expected: `13`. That is one unit test, plus eight integration modules, with `test_cli_state_model.py` marked per test (five sites) because three of its tests are hermetic.

- [ ] **Step 4: Run the whole suite, slow tier included, with `data/` linked**

```bash
uv run pytest -q -p no:cacheprovider
```

Expected: Task 0's bare count **+ 135 passed**, 0 failed, skipped unchanged. Most of the run is Task 0's slow D1 tests, about 12 minutes. Task 10 took 34 s when observed. Tasks 9 and 13 add their store tests, which never ran while this plan was written. Their fixtures fit at 2 chains of 60 warmup and 60 draws, and the three of their slow tests that did run took 29 s together, so expect minutes, not tens of minutes.

- [ ] **Step 5: Measure the two counts the guides quote, without `data/`**

The symlink is moved aside and put back. Nothing under the main checkout is touched.

```bash
mv data data.unlinked
uv run pytest -q -p no:cacheprovider
uv run pytest -q -p no:cacheprovider -m "not slow and not network"
mv data.unlinked data && ls data/staged
```

Expected: `1633 passed, 72 skipped` for the bare run, and `1612 passed, 45 skipped, 48 deselected` for the CI expression. Those are the numbers Step 2 wrote into `CLAUDE.md` and `ci.yml`. **If either differs, write the measured numbers into both files, and report the difference in the Task 15 log entry.** Never adjust a test to meet a written count. The arithmetic is the check: skipped stays at 72 and 45, because every test this plan adds runs without `data/`.

- [ ] **Step 6: Gates**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
```

Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md .github/workflows/ci.yml src/logging_employment/models/CLAUDE.md src/logging_employment/validate/CLAUDE.md src/logging_employment/reconcile/CLAUDE.md src/logging_employment/baselines/CLAUDE.md
git commit -m "docs: the models package, its commands, and the suite counts plan 16 measured"
```

---

### Task 15: D1: the comparand re-run under the new id, the production fit, and the record

**Implements:**
- Decision 1's re-run of the §13.10 comparand under `dd7337e89047`, and its byte-for-byte check against `runs/4cf47a918dd8`;
- the production fit on D1 and §11.14's gate on it;
- `validate-state-model`'s 27 replicate fits, and the promotion record against the comparand the brief names;
- the stage's findings log.

**Decision 15 measured this task's fit, and it passed there.** On the older local environment the production gate passed on D1 on two seeds, with parameter R-hat 1.0047 and 1.0086 against 1.01. Execution samples on the locked jax and numpyro, so the draws will differ, and the second seed's margin was 0.0014. Both branches of Step 5 stand. The promotion verdict was never measured: it is whatever Step 6 records. **Every step below that says STOP means stop and report**, with the output that triggered it.

**Files:**
- Modify: `specs/findings/stage-5-log.md` (one dated entry, appended)
- Modify: `CLAUDE.md` and `src/logging_employment/validate/CLAUDE.md`: the comparand re-run, stated only once it has happened
- Modify: `src/logging_employment/models/CLAUDE.md`: the measured gate result and verdict, as a dated witness
- Writes, untracked: `runs/dd7337e89047/`. Rewrites `data/constraints/` with bytes Step 4 proves identical.

**Interfaces:** consumes every command this plan built. Produces nothing later tasks import.

- [ ] **Step 1: Preconditions**

```bash
git log --oneline -1
uv run python -c "from pathlib import Path; from logging_employment.config import load_config; from logging_employment.runs import run_id; from logging_employment.cli import _input_digests; c = load_config(Path('config.yaml')); print(run_id(c, {}), run_id(c, _input_digests(c)))"
test ! -e runs/dd7337e89047 && echo "runs/dd7337e89047 is free"
ls runs/4cf47a918dd8
```

Expected: Task 14's commit; `14352bb8e56e dd7337e89047`; `runs/dd7337e89047 is free`; and the comparand's 14 entries, `baseline_results/` among them. **STOP if `runs/dd7337e89047` exists.** Something already wrote under the new id, and a byte comparison against a directory of unknown history proves nothing.

- [ ] **Step 2: Keep a copy of the shared constraint tables**

`build-constraints` writes `data/constraints/`, which is the MAIN checkout's, through the Task 0 symlink. The re-run must rewrite it with identical bytes, and this copy is what Step 4 proves that against.

```bash
rm -rf /tmp/plan16-constraints-before && mkdir /tmp/plan16-constraints-before
cp -p data/constraints/*.parquet /tmp/plan16-constraints-before/
ls /tmp/plan16-constraints-before
```

Expected: `constraint_coefficient.parquet`, `constraint_row.parquet` and `target_cell.parquet`.

- [ ] **Step 3: Re-run the comparand's chain under the new id**

```bash
for command in build-constraints solve-bounds run-baselines reconcile validate; do
  uv run logging-estimates "$command" --config config.yaml || { echo "STOP: $command failed"; break; }
done
```

Expected: every command exits 0. `validate` takes about 12.5 minutes on D1, measured on the comparand. **STOP on any failure.** The comparand's inputs and config are unchanged except `model:`, which none of these five commands reads.

- [ ] **Step 4: Prove the re-run byte-identical to the comparand**

```bash
uv run python - <<'EOF'
import hashlib
import json
from pathlib import Path

import yaml

old, new = Path("runs/4cf47a918dd8"), Path("runs/dd7337e89047")
STAMP = {"code_commit", "uv_lock_sha256"}  # beside the id by design (runs.code_provenance)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(root):
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())


problems = []
if files(old) != files(new):
    problems.append(f"file sets differ: {sorted(set(files(old)) ^ set(files(new)))}")
for rel in files(old):
    a, b = old / rel, new / rel
    if not b.exists():
        continue
    if rel.suffix == ".parquet":
        if sha(a) != sha(b):
            problems.append(f"{rel}: bytes differ")
    elif rel.suffix == ".json":
        strip = lambda d: {k: v for k, v in d.items() if k not in STAMP}
        if strip(json.loads(a.read_text())) != strip(json.loads(b.read_text())):
            problems.append(f"{rel}: differs beyond {sorted(STAMP)}")
    elif rel.name == "config.resolved.yaml":
        before, after = yaml.safe_load(a.read_text()), yaml.safe_load(b.read_text())
        after.pop("model")
        after["promotion"].pop("catastrophic_stratum_coverage_alpha")
        if before != after:
            problems.append(f"{rel}: differs beyond model: and the new promotion key")
    else:
        problems.append(f"{rel}: a file type this check does not cover")
for table in sorted(Path("/tmp/plan16-constraints-before").glob("*.parquet")):
    if sha(table) != sha(Path("data/constraints") / table.name):
        problems.append(f"data/constraints/{table.name}: bytes differ")
print("\n".join(problems) or f"{len(files(old))} run files and 3 constraint tables: identical")
raise SystemExit(1 if problems else 0)
EOF
```

Expected: `… identical`, exit 0. Every Parquet file matches by sha256. Every JSON manifest matches except `code_commit` and `uv_lock_sha256`: the commit is new, and so is the lock (Task 1). No manifest in the comparand names its own run id, which was checked while this plan was written. `config.resolved.yaml` differs only by `model:` and `promotion.catastrophic_stratum_coverage_alpha`. **STOP on any difference.** If the constraint tables differ, first restore them from `/tmp/plan16-constraints-before/`, because the main checkout's session reads them. Then report. Do not fit a model against a comparand this plan changed.

- [ ] **Step 5: Fit the production model**

```bash
time uv run logging-estimates fit-state-model --config config.yaml; echo "exit $?"
cat runs/dd7337e89047/posterior/diagnostics.json
```

Expected, per Decision 15: `exit 0`, and `diagnostics.json` shows `"passed": true`. On the older environment, at the config's seed 3645, the gating checks read 0 divergences, parameter R-hat 1.0047, cell R-hat 1.0042, cell ESS bulk 975 and tail 1,176, and one-step coverage 0.904. The fit took 130 s and reconciliation 38 s. **Record the whole file, and the wall time, in Step 8 either way.** Then branch:

- **The gate passed:** run `uv run logging-estimates reconcile --config config.yaml`. Expected: exit 0, with `state_model.passed` true in `reconcile_manifest.json`. Then apply the timing guard. **If `fit-state-model` took more than 10 minutes, run Step 5a, then STOP before Step 6** and report its time. That is about three times the measured wall time, and Step 6 fits 27 more.
- **The gate failed, which Decision 15 did not observe: STOP.** Report `diagnostics.json` in full. Nothing else is written, by design (Task 9). A failure here contradicts Decision 15's measurement. Your human partner decides whether to record it, since Step 6 would write `not_beaten` without spending replicate fits (Decision 9), or to investigate the locked versions first.

- [ ] **Step 5a: Count the imputed cells whose interval is a bound**

Decision 15 found 192 imputed cells at the config's seed whose 90% interval had zero width on their §9 upper bound, 177 of them in both seeds it ran. In most of them the raw model overshoots the bound. This step measures the same thing on the executed fit. It gates nothing, and its count goes into Step 8's entry.

```bash
uv run python - <<'EOF'
import polars as pl

summary = pl.read_parquet("runs/dd7337e89047/posterior_summary.parquet")
imputed = summary.filter(pl.col("observed_or_imputed") == "imputed")
flat = imputed.filter(pl.col("ci90_low") == pl.col("ci90_high"))
on_upper = flat.filter(pl.col("ci90_high") == pl.col("deterministic_upper"))
print(f"{imputed.height} imputed, {flat.height} zero-width, {on_upper.height} of them on the upper bound")
state = pl.col("cell_id").str.split("|").list.get(1).alias("state_fips")
print(on_upper.group_by(state).len().sort("len", descending=True))
EOF
```

Expected: `1227 imputed`, and about 190 zero-width intervals, every one on the upper bound. The older environment gave 192: VT (50) 65, NM (35) 29, NH (33) 22, CO (08) 21, UT (49) 16, and 39 across seven more states. The locked versions' draws can move the count. Whatever it is, record it.

- [ ] **Step 6: Write the promotion record**

Run it in the background. It fits 27 replicates, about 77 minutes at Decision 15's measured speed.

```bash
time uv run logging-estimates validate-state-model --config config.yaml; echo "exit $?"
python3 -m json.tool runs/dd7337e89047/promotion_record.json
```

Expected: `exit 0`, whichever the verdict.
- `verdict` is `beat` or `not_beaten`, and `selected_method` is `state_total_model` or `section_10_8_hierarchy` to match.
- `provisional` is true, and `disclosure_review` is `pending_stage_8`.
- `gates.convergence.replicates_checked` and `gates.hard_constraints.replicates_checked` are both 27.
- `comparand` carries `dd7337e89047` and the sha256 of its three validation tables. Those must equal `shasum -a 256 runs/4cf47a918dd8/validation_{scores,metrics,scoreboard}.parquet`, the files Step 4 proved identical.

**The verdict was not measured while this plan was written, and whichever one this records is the result.** Do not re-run with other seeds, settings or regimes to change it. If a replicate's gate failed, `gates.convergence.replicate_failures` names it, and Decision 4 makes the verdict `not_beaten`. **If it has not finished after 4 hours, stop it and STOP**, reporting how far it got. That is about three times the measured estimate.

- [ ] **Step 7: State what is now true in the guides**

`CLAUDE.md`, in the gotcha Task 14 wrote about the re-id, replace:

```text
  and the staged pin `4cf47a918dd8` → `dd7337e89047`.
```

with:

```text
  and the staged pin `4cf47a918dd8` → `dd7337e89047`. The §13.10 comparand was re-run under the
  new id and matched `runs/4cf47a918dd8` byte for byte, manifests aside from `code_commit` and
  `uv_lock_sha256` (`specs/findings/stage-5-log.md`).
```

`src/logging_employment/validate/CLAUDE.md`, in the bullet on dated D1 numbers, replace:

```text
  by `runs/4cf47a918dd8`) live in the docstrings and in `tests/unit/test_validate_regimes.py`.
```

with:

```text
  by `runs/4cf47a918dd8`, re-run byte for byte by plan 16 as `runs/dd7337e89047` when the `model:`
  block re-identified every run) live in the docstrings and in `tests/unit/test_validate_regimes.py`.
```

`src/logging_employment/models/CLAUDE.md`: append a bullet to its read-first list that states Step 5's gate result and Step 6's verdict as a dated witness. Include the date, the run id, `passed`, each gating check's value, the fit's wall time, `mean_leapfrog_steps` and `tree_depth_saturation_share`, Step 5a's count, the verdict, and a pointer to `specs/findings/stage-5-log.md`.

- [ ] **Step 8: Append the stage log entry**

Append one entry to `specs/findings/stage-5-log.md`, in its own format (`## YYYY-MM-DD — …`, then prose). It records:
- the base commit;
- Task 14's measured counts, and any difference from those written;
- Step 4's result;
- Step 5's `diagnostics.json` in full, with the command's wall time and the `sampler` block;
- Step 5a's count, by state;
- Step 6's verdict, each gate's `passed`, and the command's wall time;
- Task 12 Step 4's tripwire output, the evidence `D-109`'s done-when asks for;
- a sentence saying what stays open for the Plan Completion Protocol: Decision 7's four flagged readings, Decision 15's open points (the cells whose interval is a bound, and the unstable means in NV and RI), and `D-116`'s and `D-115`'s dispositions (Decision 1).

- [ ] **Step 9: Gates and commit**

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run interrogate src
git add CLAUDE.md src/logging_employment/validate/CLAUDE.md src/logging_employment/models/CLAUDE.md specs/findings/stage-5-log.md
git commit -m "docs(findings): plan 16's D1 run: the comparand re-run byte for byte, the fit, the record"
```

- [ ] **Step 10: STOP for your human partner**

Report:
- Step 4's line;
- the gate report and the fit's wall time;
- Step 5a's count;
- the verdict, and which gates decided it;
- whether §19 Phase 3's "diagnostics pass" holds.

The Plan Completion Protocol comes next. Ticking Stage 5 in the roadmap, filing Decision 7's flagged readings and Decision 15's open points as deferred items, and ticking `D-109` all belong to it, not to this task.

---

## Spec coverage

Every item on the brief's `Spec:`, `Gap closed:` and `Exit:` lines, and where this plan meets it. "Not met" is stated as such.

| Brief item | Where | Note |
|---|---|---|
| §11 intro: "separable interfaces" | Task 3 (`models/interfaces.py`) | Decision 8: `StateModelFit` carries the `PosteriorDraws` §16.2 names |
| §11.1 mean, §11.2 Student-t AR(1) | Task 4 | H omitted until Stage 7 (Decision 7). X split into within-state and between-state parts, **flagged** (Decision 7, reading 4) |
| §11.3 likelihood on disclosed training cells; "Suppressed cells have no pseudo-observation" | Tasks 3–4 | Training cells are exact: the likelihood is the AR(1) density of the innovations they pin, over training cells alone. The "practical response model" is dropped, **flagged** (Decision 7, reading 1) |
| §11.5 score | Task 4, `raw_scores` | Literal, q = A·exp(μ) (Decision 6) |
| §11.11's QCEW rows | Task 3 | Disclosed final: trained on. Non-final training row: refused. Suppressed: predicted only |
| §11.12 priors | Tasks 2 and 4 | Every prior in config; df 5 as §11.12 states. Persistence 0.95 · Beta(8, 2), prior mean 0.76, **flagged** (Decision 7, reading 2) |
| §11.14 | Task 4 (vectorized; non-centred where no training cell pins a state); Task 8 (the gate in code) | Two scopes and a one-step predictive check (Decision 5). Trained states set the non-centred SHOULD aside, **flagged** (Decision 7, reading 3). §11's model as written failed the gate on D1, and the revised model passed it on two seeds (Decision 15) |
| §12.7 | Tasks 5, 6, 11 | Summaries and harness estimates from reconciled joint draws only |
| §13.10, applied | Tasks 11–13 | Readings flagged in Decision 4 |
| §15.4 | Task 7 | No size index until Stage 6 |
| §16.1 `fit-state-model` | Task 9 | `validate-state-model` is added, not §16.1's (Decision 9) |
| §16.2 `fit_state_total_model` | Task 4 | Returns `StateModelFit` (Decision 8) |
| §17.4 rows 5–6 | Task 10 (row 5); Tasks 5 and 9 (row 6) | |
| §17.5 state-total rows | Task 10 | The other four rows are later stages' |
| §19 Phase 3 deliverables | Tasks 4, 5, 11–13 | |
| §19 Phase 3 acceptance: "diagnostics pass" | Task 15 | Met on D1 as measured on the older environment, two seeds (Decision 15). Task 15 measures it on the locked versions |
| §19 Phase 3 acceptance: "constraints pass on every draw"; "baselines are beaten or the simpler model is retained" | Tasks 5, 9; Tasks 12–13, 15 | |
| §21 PPL row | Task 1, Decision 10 | NumPyro, with a CmdStanPy slot (`BACKENDS`). §21's benchmark "before final selection" is not run: no CmdStanPy is installed |
| Appendix A `model:` block | Task 2 | Into `resolved_dict`, one re-id (Decision 1) |
| REQ-012 (posterior half), INV-008 | Task 6 | Distinct columns, never relabelled |
| REQ-014 | Task 4 | |
| REQ-018 (every draw), INV-012 | Task 5 (`check_reconciled`); Task 9 (`reconcile` re-reads the store) | After transformation, never on summaries |
| REQ-019, INV-013 | Tasks 5, 7 | The draw axis is never reduced |
| REQ-024 (gate applied), INV-014 (applied) | Tasks 11–13 | Same harness, same masks, same metrics as the comparand |
| REQ-029 (model diagnostics) | Tasks 8–9 | `ModelDiagnosticsError`; `fit-state-model` exits 1 and keeps its report |
| INV-006 (measurement models) | Tasks 4–6 | Modeled values never become equations. Every draw is reconciled into §9's bounds and the declared anchor, and nothing the model writes feeds `build-constraints`. Published QCEW enters the likelihood. CBP, TPO, FIA and CES measurement models belong to later stages |
| Exit: §17.5 recovery within stated tolerances | Task 10 | On a panel simulated at D1's regime (Decision 15). Recovery at D1's own mask also passed during the design pass |
| Exit: every retained draw satisfies all hard constraints, after transformation | Tasks 5, 9 | Measured on D1 in Decision 15's fits: drift 4.5e-13, 0 violations, on both seeds |
| Exit: §11.14 gate enforced in code, not documented | Tasks 8–9 | |
| Exit: suppressed cells receive no pseudo-observation | Tasks 3–4 | |
| Exit: the record states beat or not-beaten, and the simpler method is selected when not beaten | Tasks 12, 13, 15 | |
| REQ-022 | none | **Not closed, by ruling.** `D-071` (2026-09-10) declared four regimes unscored by decision, and the spec's stamp says "Stage 5 should not infer a design for the other four". §13.10 is applied over nine regimes |

The brief's three flags are Decisions 1, 2 and 3. Its open items: `D-109` closes in Task 12; `D-115` and `D-116` are declined for this re-run (Decision 1). `D-120` to `D-124` all landed before this plan was saved. `D-125` (design, open) concerns `run-baselines`' audit and names no Stage 5 work.
