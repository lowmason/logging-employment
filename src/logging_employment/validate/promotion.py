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
