"""Typed configuration, loaded from YAML, with credentials kept out of the resolved form."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

# Named here so the secret guard and the resolved-config writer share one list (D3).
SECRET_ENV_VARS = ("CENSUS_API_KEY", "BLS_API_KEY", "BEA_API_KEY", "FRED_API_KEY")


class _Strict(BaseModel):
    """Base model that rejects unknown keys, so a typo in config.yaml halts rather than defaults."""

    model_config = ConfigDict(extra="forbid")


class ProjectConfig(_Strict):
    """The estimand: industry, ownership, geography universe, window, and mode."""

    name: str
    industry_code_supplied: str
    industry_code_used: str
    industry_title: str
    ownership: Literal["private"]
    geography_universe: Literal["states_dc"]
    start_month: str
    end_month: str
    analysis_mode: Literal["retrospective_final", "realtime_asof"]
    size_concept: Literal["march_reference", "contemporaneous_modeled"]

    @field_validator("start_month", "end_month")
    @classmethod
    def _is_a_month(cls, value: str) -> str:
        """Reject anything that is not a zero-padded `YYYY-MM`."""
        if not _MONTH.match(value):
            raise ValueError(f"expected YYYY-MM, got {value!r}")
        return value


class StorageConfig(_Strict):
    """Where raw bytes, staged frames, constraint tables, and run outputs live.

    Appendix A's `storage:` block supplies four keys and no constraints location, so
    `constraints_uri` is originated here; its VALUE is §6.2's `data/constraints/`. Defaulted
    rather than required so a config written before the key existed stays valid under
    `extra="forbid"`, and the default is the exact path the location was previously *derived*
    to be. Configured rather than inferred: deriving it from `staged_uri`'s parent moved the
    constraint tables whenever the staged root moved.
    """

    raw_uri: str
    staged_uri: str
    output_uri: str
    immutable_raw: bool
    constraints_uri: str = "data/constraints"


class QcewSourceConfig(_Strict):
    """QCEW quarterly acquisition settings."""

    enabled: bool
    release_status: Literal["final", "preliminary"] = "final"


class QcewSizeSourceConfig(_Strict):
    """QCEW by-size acquisition settings."""

    enabled: bool


class CbpSourceConfig(_Strict):
    """CBP acquisition settings, including the fail-closed disclosure switch."""

    enabled: bool
    api_key_env: str
    fail_on_unknown_disclosure_regime: bool


class InactiveSourceConfig(_Strict):
    """An Appendix A source entry that carries nothing but `enabled: false`.

    `enabled` is `Literal[False]`, not `bool`: this package has no ingest module, no registry row
    (`registry/sources.yaml` lists three) and no `fetch --source` branch for any of these seven, so
    `enabled: true` names a source nothing can acquire. Accepting it would defer the failure to a
    `fetch` that reports an unknown source; refusing it at load is §18.3's fail-closed rule, and
    `validate-config` is where the operator is already looking.
    """

    enabled: Literal[False] = False


class SourcesConfig(_Strict):
    """The configured sources Stage 1 ingests, plus Appendix A's seven declared-inactive entries.

    Plan 15's `qcew_parent`, the private `113` state series, has no entry and needs none: it is the
    same QCEW product on the same route, so `fetching.py` fetches it with the `qcew` client and its
    own industry constant, and it adds nothing to `resolved_dict`.

    Appendix A's `sources:` block lists ten. Seven of them -- `tpo`, `fia`, `ces`, `susb`, `bds`,
    `nonemployer`, `bea` -- carry `enabled: false` and belong to Stages 4-8; before they were
    declared here, `extra="forbid"` made the spec's own reference configuration unloadable, which
    is `R-S5P-6`'s defect (`specs/completed/stage5-preconditions.md`) and what this fixes. `R-S5P-6` scoped
    only these seven. The fence's three MISSING keys were a spec gap, which `D-121` closed in
    Appendix A rather than with defaults here, and its `model:` block loads since plan 16 gave it
    `ModelConfig`. So Appendix A's fence now loads whole, and `tests/unit/test_config.py` pins that.

    They are declared as fields rather than admitted by `extra="allow"`
    so that a MISSPELLED source name is still a load-time error and so that `enabled: true` on an
    unimplemented source is refused (`InactiveSourceConfig`); `extra="allow"` would have accepted
    both silently.

    EVERY ONE OF THE SEVEN IS `exclude=True`, so none of them reaches `resolved_dict` and none can
    move a run id. That is deliberate and load-bearing, not a serialization detail: `runs.run_id`
    hashes `resolved_dict`, and a source that is `enabled: false` contributes no bytes to any
    stage -- nothing outside this module reads these fields. Emitting them would re-identify every
    run directory on disk (orphaning `runs/f03023ac9f3a`) in exchange for a key that cannot change
    what a run computes. It is the same omitted-not-null rule `runs.run_id` applies to `overrides`.
    """

    qcew: QcewSourceConfig
    qcew_size: QcewSizeSourceConfig
    cbp: CbpSourceConfig
    tpo: InactiveSourceConfig | None = Field(default=None, exclude=True)
    fia: InactiveSourceConfig | None = Field(default=None, exclude=True)
    ces: InactiveSourceConfig | None = Field(default=None, exclude=True)
    susb: InactiveSourceConfig | None = Field(default=None, exclude=True)
    bds: InactiveSourceConfig | None = Field(default=None, exclude=True)
    nonemployer: InactiveSourceConfig | None = Field(default=None, exclude=True)
    bea: InactiveSourceConfig | None = Field(default=None, exclude=True)


class ConstraintsConfig(_Strict):
    """The deterministic engine's solver settings (Appendix A `constraints:`).

    `use_milp_when_lp_interval_width_below` is a *performance* switch: it decides when an integer
    re-solve is worth its cost, never whether a cell is disclosive. The disclosure thresholds live
    in `DisclosureConfig` and are a governance decision (§21).

    ALL THREE FLOATS REFUSE NaN AND ±inf AT LOAD (`allow_inf_nan=False`): no reader refuses one.
    HiGHS takes a NaN or +inf feasibility tolerance and answers `kOk` (measured 2026-09-27, on all
    three options `bounds.configured_highs` sets), so the solver ran at it before
    `classify_bound_status`'s `math.ceil` raised a bare `ValueError` or `OverflowError` on the
    first bounded integer cell: closed, but unnamed and one solve late (§18.3). A NaN
    `rank_tolerance` records every numerical rank as zero, and a NaN MILP width silently means
    "never re-solve".

    The two tolerances are also `gt=0.0`; the switch is not. At or below zero HiGHS refuses the
    feasibility tolerance with `kError` and keeps its own default, a status `configured_highs` does
    not read, so `solver_tolerance` would record a tolerance the solver never ran at. A negative one
    also turns §9.6's `ceil(lower - tolerance) == floor(upper + tolerance)` inside out, so the exact
    point [5, 5] reads as partially identified and §9.8's `exact_reconstruction_flag` misses it.
    `matrix_rank`
    counts singular values strictly above `rank_tolerance`, so at zero it counts float residue as
    rank (a 2e-17 singular value, measured) and CON-003's recorded nullity understates the free
    directions. At zero the switch re-solves nothing -- `enforce_integrality: false`'s outcome,
    spelled as a width -- and Appendix A gives it no bound to enforce.
    """

    enforce_integrality: bool
    use_milp_when_lp_interval_width_below: float = Field(..., allow_inf_nan=False)
    solver: Literal["highs"]
    feasibility_tolerance: float = Field(..., gt=0.0, allow_inf_nan=False)
    rank_tolerance: float = Field(..., gt=0.0, allow_inf_nan=False)


class ReconciliationConfig(_Strict):
    """The §12 reconciliation layer's settings (Appendix A `reconciliation:`, plus this
    package's own numerical decisions).

    Appendix A supplies exactly three keys. §12 specifies no tolerance, no iteration cap, and no
    convergence criterion anywhere, so the remaining five are originated here rather than
    inherited. `feasibility_tolerance` under `constraints:` belongs to the LP/MILP bound solver
    and is deliberately not reused: a bound solved to 1e-7 and a residual reconciled to 1e-9 are
    different obligations, and coupling them would make a solver tuning change silently move a
    published total.

    `tolerance` and `zero_seed_floor` refuse NaN and ±inf at load and are `gt=0.0`. `tolerance` is
    the margin in `baselines.runner.leaves_its_interval`, in the float estimate's
    `assert_within_bounds` (INV-002's per-cell half; the integer release uses the solver's), in
    `scale_into_bounds`'s §12.3 infeasibility and drift tests, and in `cli.py::reconcile_command`'s
    gate. +inf passed all of them and NaN all but the last, because `value > upper + tolerance` is
    false for every value: INV-002's per-cell check passed vacuously, and
    `math.isclose(abs_tol=inf)` sent `scale_into_bounds` to every cell's lower bound whatever the
    residual. At or below zero it fails the other way and is no tolerance at all: `math.isclose`
    raises on a negative `abs_tol`, and zero reinstates the exact comparison `scale_into_bounds`
    documents rejecting a feasible month. §12.4's "small positive floor" is `zero_seed_floor`'s
    bound, and a non-finite value is no floor: `max(0.0, nan)` is 0.0.
    """

    single_margin_method: Literal["bounded_proportional_scaling"]
    general_method: Literal["kl_projection", "weighted_quadratic"]
    integerize_release: bool
    # Originated here. Tighter than the solver's 1e-7 because a residual is an adding-up identity
    # over at most 15 cells, not an optimum over a polytope.
    tolerance: float = Field(default=1.0e-9, gt=0.0, allow_inf_nan=False)
    max_bisection_iterations: int = 200
    max_projection_iterations: int = 1000
    # §12.4 requires "a small positive floor for zero raw seeds" and gives no value. A seed of
    # exactly 0 makes the KL objective undefined, so this is a hard numerical requirement.
    zero_seed_floor: float = Field(default=1.0e-12, gt=0.0, allow_inf_nan=False)
    # §12.6 step 3. MUST stay deterministic or §16.1's idempotence requirement breaks.
    integerization_tiebreak: Literal["largest_remainder"] = "largest_remainder"


class BaselinesConfig(_Strict):
    """Which §10 baselines run, and the declared-composite policy they run under.

    `allow_declared_composite` is not a convenience switch. With it false, §10.3 and §10.4 decline
    in every month of the D1 window — six states have no observed employment history and two have
    no CBP row at all, and every month's missing set contains at least one of them — which would
    leave §10.8's rungs 1 and 3 permanently empty. See the plan's coverage table.
    """

    allow_declared_composite: bool = True
    composite_fallback: Literal["establishment_proportional"] = "establishment_proportional"
    historical_lookback_months: int = 24
    # §10.3 requires "classification-consistent periods". The window breaks at 2022-01.
    historical_may_cross_naics_vintage: bool = False
    # §10.6. scipy is already a declared dependency; sklearn is not and is not added.
    regression_ridge_penalty: float = 1.0


class DisclosureConfig(_Strict):
    """Disclosure actions and the narrowness thresholds (Appendix A `disclosure:`, §21).

    The two width keys resolve §21's "Disclosure thresholds" row, which the spec leaves to the
    governance owner. A cell is narrow when its feasible width is at most
    `narrow_interval_absolute_width` employees, or when width divided by midpoint is at most
    `narrow_interval_relative_width`. §14.2 asks for both an absolute and a relative test, so both
    are configured and either one alone is sufficient to route a cell to review.

    Both widths refuse NaN and ±inf at load, and neither is sign-bounded. `disclosure.flags`
    compares them in polars, which orders NaN above every number (measured 2026-09-27: `width <=
    NaN` is true for every width), so a NaN width flagged every finite suppressed interval where
    an IEEE comparison flags none -- a threshold whose meaning depends on the library comparing it
    is no policy -- and ±inf say "every interval" and "none", which a finite width also says. At
    zero an arm flags nothing wider than a point. That is an "off" position, the kind
    `tests/integration/test_d1_acceptance.py` sets each arm to in turn and calls "a width a config
    file could legally carry"; where narrow begins is the owner's call (§21), not a bound this
    module may invent.
    """

    exact_reconstruction_action: Literal["withhold", "manual_review", "release"]
    narrow_interval_action: Literal["withhold", "manual_review", "release"]
    publish_label_required: bool
    narrow_interval_absolute_width: float = Field(..., allow_inf_nan=False)
    narrow_interval_relative_width: float = Field(..., allow_inf_nan=False)


class ValidationConfig(_Strict):
    """§13's harness settings, adapted from Appendix A.

    THE SEVEN `include_*` SWITCHES ARE THREE DIFFERENT KINDS OF THING, declared in
    `contracts.VALIDATION_SWITCH_KINDS` and summarised here because this is where a reader meets
    them. Four are REGIME SWITCHES and gate their regime in `validate/harness.py` (wired in the
    commit after the one that declared these kinds; before it the harness read exactly one of the
    seven, and read it to RAISE); two name INV-009 mask LABELS (§13.2 steps 3 and 8) and gate no
    regime; one names a design with no implementation anywhere in the package and is an operand of
    `_refuse_a_random_mask_only_design` below. Nine of the thirteen regimes have no switch at all,
    so the set was never a partition.

    NO FIELD MAY BE ADDED OR REMOVED. `runs.run_id` hashes `resolved_dict`, which is
    `model_dump(mode="json")` — the whole model — so either direction re-identifies every run
    directory and orphans `runs/f03023ac9f3a`. Changing a DEFAULT is free: `config.yaml` pins all
    fourteen keys, so no default reaches the resolved config.

    `include_vintage_comparison` defaults False against Appendix A's True: measured 2026-09-07, no
    period in any staged table carries a second snapshot, and `release_vintage` is the reference
    quarter lowercased rather than a publication vintage. Turning it on is a fail-closed error in
    `regimes.py`, not a silent empty partition.
    """

    pseudo_suppression_seeds: list[int] = [1024, 2048, 4096]
    include_random_mask_sanity_check: bool = True
    include_primary_like: bool = True
    include_complementary_like: bool = True
    include_long_runs: bool = True
    include_rolling_origin: bool = True
    include_retrospective_smoothing: bool = False
    include_vintage_comparison: bool = False
    replicates_per_regime: int = 20
    minimum_unmasked_lookback_months: int = 6
    minimum_missing_set_size: int = 2
    # DECLARED, not detected. A per-cell break detector inside a validation stage is a research
    # project, and the national series does not separate COVID from seasonality cleanly enough to
    # justify one. The second window overlaps `naics_transition`; score a month under ONE label.
    structural_break_windows: list[tuple[str, str]] = [
        ("2020-03", "2020-06"),
        ("2021-10", "2022-03"),
    ]
    naics_seam_month: str = "2022-01"
    naics_seam_halfwidth_months: int = 3

    @model_validator(mode="after")
    def _refuse_a_random_mask_only_design(self) -> ValidationConfig:
        """§13.2 forbids random masking as the ONLY validation design; refuse that config.

        A validator rather than a check inside the harness, because the harness would discover it
        one full pass late. `include_random_mask_sanity_check` is deliberately absent from the
        disjunction: it names the design this rule exists to exclude, not one of the designed
        regimes that satisfies it.
        """
        designed = (
            self.include_primary_like
            or self.include_complementary_like
            or self.include_long_runs
            or self.include_rolling_origin
            or self.include_retrospective_smoothing
            or self.include_vintage_comparison
        )
        if not designed:
            raise ValueError(
                "random_mask_only: §13.2 prohibits random masking as the ONLY validation design. "
                "Enable at least one designed regime beside the random sanity check."
            )
        return self


class PromotionConfig(_Strict):
    """§13.10's gates. Configurable engineering thresholds, not findings.

    ALL FOUR KEYS ARE INERT TODAY, and this note is the record R-S5G-3 requires rather than a
    disclaimer. Each reaches `config.resolved.yaml` and folds into `runs.run_id`, so an unread key
    is a claim in a run's record that no code backs -- the defect `D-064` names for four other
    keys. The convention there is to write inertness into the code (`constraints/matrix.py`'s "NO
    REAL INPUT UNTIL STAGE 6", `errors.py::NoHarvestFactorError`), which is what this is.

    They are inert for different reasons, and the distinction is the useful half:

    - `maximum_major_stratum_wape_degradation` and `nominal_coverage_tolerance` have their INPUT as
      of R-S5G-1. `validate/metrics.py` now emits a per-census-division WAPE and a per-division
      90% coverage into `validation_metrics.parquet`, so both gates are evaluable from a shipped
      artifact. The coverage values `nominal_coverage_tolerance` would gate on carried a
      residual-sign defect until `D-112` (fixed 2026-09-12, `19fbdec`); one written before that fix
      must be regenerated before any gate reads it. What is missing is the CANDIDATE to evaluate: §13.10 compares a model against the
      preferred transparent baseline and Stage 5 produces the model.
    - `minimum_wape_improvement` is missing both. Its comparison needs a second scoreboard, and
      `validation_scoreboard.parquet` exists in one copy -- the baseline one.
    - `catastrophic_stratum_coverage_alpha` is plan 16's. §13.10's "does not fail catastrophically
      in any major stratum" states no number, and this is the exact binomial lower-tail level the
      promotion record plan 16 builds will read. It is declared with the `model:` block, in one
      commit, so `runs.run_id` moves once rather than twice.

    NO EVALUATOR IS BUILT HERE, deliberately. A function whose primary argument is Stage 5's
    not-yet-designed output would fix that signature by guessing it, and three of §13.10's six
    gates (hard constraints on draws, convergence diagnostics, disclosure review) are Stage-5 and
    Stage-8 concepts an evaluator written now could not represent at all. Stage 5 wires these;
    `tests/unit/test_config_validation_block.py` fails the day `src/` code names one of the keys --
    as an attribute, a string, a parameter or a keyword argument, `config.py` itself included -- so
    this docstring cannot quietly outlive that truth. A reader that never spells a key exactly -- a
    generic `model_dump()` loop, or a dotted path string such as `attrgetter("promotion.<key>")` --
    would not trip it; update this note by hand in that case.
    """

    # Each threshold is finite at load (plan 16). The promotion record compares against each, and a
    # NaN or infinite one either stops a gate flagging anything or fails only after the replicate
    # fits. The alpha is also a probability: at or below 0, `cdf < alpha` can never hold.
    minimum_wape_improvement: FiniteFloat = 0.05
    maximum_major_stratum_wape_degradation: FiniteFloat = 0.02
    nominal_coverage_tolerance: FiniteFloat = 0.05
    # Originated by plan 16. §13.10 says "does not fail catastrophically" and gives no number.
    catastrophic_stratum_coverage_alpha: float = Field(default=0.001, gt=0.0, lt=1.0)


# A prior's scale or concentration. NumPyro would take a non-positive or non-finite one without
# complaint and fail later as a backend error or a non-finite density, not as a refusal naming the
# value (§18.3), so it is refused here, at load.
PositiveFinite = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]


class StateModelPriors(_Strict):
    """§11.12's starting priors for the state-total model, each one a configured value.

    §11.12: "Every prior must be exposed in resolved configuration." These are in `resolved_dict`
    and so in `runs.run_id`, deliberately: a prior changes every draw the model writes. The values
    are weakly informative on the log employees-per-establishment scale, where D1's observed cells
    have mean 1.46 and SD 0.48 (measured 2026-09-26), so `intercept_mean` centres the intercept
    there. The innovations' Student-t degrees of freedom are fixed at 5, §11.12's "fixed near 5 for
    the first implementation".

    EVERY PRIOR IS CHECKED AT LOAD, the slopes' scale included, which sits at the block's top level
    as Appendix A's own `standardized_beta_sd`. Scales and concentrations are positive and finite
    (`PositiveFinite`), the intercept's mean is finite, and the degrees of freedom exceed 2: eta's
    first month starts at the variance-matched scale sigma / sqrt(1 - rho^2), which needs the
    innovations to have a variance.

    PERSISTENCE IS `persistence_max * Beta(8, 2)`: capped at 0.95, mean 0.76, §11.12's "transformed
    Beta prior centered near 0.8". §11.2 prefers the AR(1) because Logging intensity is "plausibly
    mean-reverting". Uncapped, the largest state's posterior-mean persistence on D1 was 0.978, and
    a production fit failed §11.14's R-hat on one of two seeds; capped, both passed (plan 16,
    Decision 15). 1.0 removes the cap.

    THERE IS NO OBSERVATION SCALE. Training cells are exact (§11.3), so §11.3's `sigma_y` and its
    degrees of freedom are not parameters of the model.
    """

    intercept_mean: FiniteFloat = 1.5
    intercept_sd: PositiveFinite = 1.0
    state_scale_sd: PositiveFinite = 0.5
    region_scale_sd: PositiveFinite = 0.5
    month_scale_sd: PositiveFinite = 0.25
    year_scale_sd: PositiveFinite = 0.25
    persistence_concentration1: PositiveFinite = 8.0
    persistence_concentration0: PositiveFinite = 2.0
    persistence_max: float = Field(default=0.95, gt=0.0, le=1.0)
    innovation_scale_sd: PositiveFinite = 0.1
    innovation_dispersion_sd: PositiveFinite = 0.5
    innovation_df: float = Field(default=5.0, gt=2.0, allow_inf_nan=False)


class StateModelDiagnostics(_Strict):
    """§11.14's diagnostic gate as thresholds, enforced by `models/diagnostics.py`.

    §11.14 gives R-hat "at or below 1.01" and asks for "adequate effective sample size" without a
    number. `min_ess_per_chain` fills that gap with the convention of 100 per chain, so the floor is
    400 at Appendix A's four chains. `min_ppc_coverage_90` is a floor on the share of training cells
    inside their 90% ONE-STEP-AHEAD predictive interval, predicted from the month before. Training
    cells are exact (§11.3), so an in-sample replicate of a training cell is y itself and could not
    fail. It is a floor only: coverage above nominal means the innovation law is wider than the
    data's month-to-month moves, which widens intervals without understating them, while a fit
    that misses more than 15% of its training cells' moves understates them. `max_divergences` is 0
    because §11.14 says "no unresolved divergent transitions".

    THE THRESHOLDS ARE FINITE AT LOAD. An infinite `max_rhat` would pass every fit's R-hat check,
    and a NaN one would fail every fit, but only after it had sampled. `min_ppc_coverage_90` is a
    share, so it lies in [0, 1].
    """

    max_rhat: FiniteFloat = 1.01
    min_ess_per_chain: int = 100
    max_divergences: int = 0
    min_ppc_coverage_90: float = Field(default=0.85, ge=0.0, le=1.0)


class ModelConfig(_Strict):
    """Appendix A's `model:` block (§11), plus the keys a fit needs that Appendix A omits.

    IN `resolved_dict`, NOT `exclude=True`. The seven inactive sources are excluded because they
    contribute no bytes to any stage. Every key here changes the draws `fit-state-model` writes, so
    `runs.run_id` must see it. Adding the block re-identifies every run directory ONCE. Plan 16
    re-runs the §13.10 comparand `runs/4cf47a918dd8` under the new id and checks the re-run byte for
    byte against it (`specs/findings/stage-5-log.md`; the plan's Decision 1).

    Appendix A's eleven keys are declared verbatim, and each default is Appendix A's value. Three
    are originated here: `seed`, `priors` (§11.12) and `diagnostics` (§11.14).

    EVERY FLOAT IN THE BLOCK IS FINITE AT LOAD, because YAML reads `.inf` and `.nan` as floats and
    a plain `float` admits both. `standardized_beta_sd` is both slopes' prior scale, so it is
    `PositiveFinite`, as the scales in `priors` are.

    THE THREE `include_*` SWITCHES ARE `Literal[False]`, refused at load like an inactive source.
    Nothing implements them. §11.2's change points "MAY be added only if validation shows material
    gains". §11.4's harvest factor and §11.11's CES row are Stage 7's. Accepting `true` would
    record a model component in the run's config that no code fits.

    `backend` admits one value. `models/state_total.py::BACKENDS` is §2.2's CmdStanPy slot: a
    second backend adds a value here and an entry there together.

    `suppressed_variance_multipliers` is §11.13's sensitivity list. This stage fits the 1.0 model
    only, and the other values are Stage 7's sensitivity runs, so the validator requires 1.0 to be
    present and refuses a multiplier below 1.0 or a repeated one. Each multiplier is a
    `FiniteFloat` because that check cannot see a NaN: `min()` returns a leading one, and
    `nan < 1.0` is false. The key is otherwise inert here, and that is recorded rather than hidden:
    `models/state_total.py` never reads it.

    `seed` is `sum(map(ord, "logging-employment/state-total-model"))`, a descriptive seed rather
    than a bare constant. It is in the run id because a different seed writes different draws.
    """

    backend: Literal["numpyro"] = "numpyro"
    chains: int = Field(default=4, ge=2)
    warmup: int = Field(default=1000, ge=1)
    draws: int = Field(default=1000, ge=1)
    target_accept: float = Field(default=0.9, gt=0.0, lt=1.0)
    state_dynamic: Literal["student_t_ar1"] = "student_t_ar1"
    include_change_points: Literal[False] = False
    include_harvest_factor: Literal[False] = False
    include_ces: Literal[False] = False
    standardized_beta_sd: PositiveFinite = 0.5
    suppressed_variance_multipliers: list[FiniteFloat] = [1.0, 1.5, 2.0]
    seed: int = 3645
    priors: StateModelPriors = StateModelPriors()
    diagnostics: StateModelDiagnostics = StateModelDiagnostics()

    @field_validator("suppressed_variance_multipliers")
    @classmethod
    def _the_fitted_multiplier_is_present(cls, value: list[float]) -> list[float]:
        """Refuse a list without 1.0, with a multiplier below 1.0, or with a repeat (§11.13)."""
        if 1.0 not in value or min(value) < 1.0 or len(set(value)) != len(value):
            raise ValueError(
                f"suppressed_variance_multipliers {value}: §11.13 inflates the suppressed-cell "
                "process variance, so every multiplier must be >= 1.0, the fitted model's 1.0 must "
                "be present, and no multiplier may repeat"
            )
        return value


class Config(_Strict):
    """The whole resolved configuration."""

    project: ProjectConfig
    storage: StorageConfig
    sources: SourcesConfig
    constraints: ConstraintsConfig
    reconciliation: ReconciliationConfig
    baselines: BaselinesConfig
    disclosure: DisclosureConfig
    model: ModelConfig = ModelConfig()
    validation: ValidationConfig = ValidationConfig()
    promotion: PromotionConfig = PromotionConfig()


def load_config(path: Path) -> Config:
    """Parse and validate a config file. Raises pydantic.ValidationError on any violation."""
    return Config.model_validate(yaml.safe_load(path.read_text()))


def resolved_dict(cfg: Config) -> dict[str, object]:
    """The config as it is written to a run directory.

    Only `api_key_env` -- the *name* of an environment variable -- survives; no value is read
    here, so no key can leak into a manifest through this path (§7.2, D3).
    """
    return cfg.model_dump(mode="json")


def credentials(env_path: Path | None = None) -> dict[str, str]:
    """Credentials from the repo-root `.env`, falling back to the process environment.

    Returns only the keys D3 names. The caller passes values to a request; nothing here writes
    them anywhere.
    """
    values: dict[str, str] = {}
    if env_path is not None and env_path.exists():
        values.update({k: v for k, v in dotenv_values(env_path).items() if v is not None})
    for name in (*SECRET_ENV_VARS, "BLS_CONTACT_EMAIL"):
        from_env = os.environ.get(name)
        if from_env:
            values[name] = from_env
    return values
