"""Named fail-closed exceptions (§18.3).

Each carries the offending value, so a caller's log line names what halted the run rather than
only that something did.
"""

from __future__ import annotations


class LoggingEmploymentError(Exception):
    """Base class for every fail-closed condition in this package."""


class UnknownDisclosureCodeError(LoggingEmploymentError):
    """A QCEW disclosure code outside the measured allowlist reached the parser."""


class UnknownDisclosureRegimeError(LoggingEmploymentError):
    """A CBP reference year carries no established disclosure regime."""


class UnknownSizeCodeError(LoggingEmploymentError):
    """A QCEW establishment-size code reached the parser with no published bounds."""


class UnsupportedReferenceYearError(LoggingEmploymentError):
    """A reference year lies outside the NAICS-vintage range this package can process.

    Deliberately not an "unknown vintage": BLS publishes a vintage for every year back to 1990,
    and `harmonize/naics.py` quotes the table. What is missing is downstream support, not the
    classification -- the vendored 113310 crosswalk carries only the 2017 and 2022 vintages,
    `validate/regimes.py` documents a single vintage seam and derives its targets from it, and
    `baselines/historical.py` filters its lookback on vintage equality -- that filter is the
    enforcing one; the seam is prose. Emitting a third vintage
    string would compose it into `cell_id` and five `contracts.py` schemas, so the year is
    refused rather than labelled with a vintage nothing else in the package can consume.
    """


class SourceFetchError(LoggingEmploymentError):
    """A source request answered a status or a body `fetch` cannot record, undeclared.

    Raised rather than skipped (R-S5P-4, §18.3). `ingest/base.HttpFetcher` deliberately RETURNS a
    non-200 so the caller classifies on content instead of status; this class is that
    classification's refusing branch, and it lives in the caller so the fetcher's contract is
    unchanged. Skipping is the behaviour it replaces, and skipping is unrecoverable downstream: a
    dropped quarter narrows the window, `runs.run_id` hashes the inputs so the shortened run takes
    a NEW id rather than colliding with the full one, and no manifest carries the fact that a
    quarter is missing -- so every later stage proceeds on the shorter window and nothing says so.

    The exceptions are declared in `fetching.DECLARED_ABSENCES`, keyed by (source, reference
    year), so a genuine hole in the published record is a line of code carrying its measurement
    rather than an incidental pass through the same branch a transient 500 takes.
    """


class SchemaMismatchError(LoggingEmploymentError):
    """A file's columns, or a row bound for a persisted table, do not match the declared schema.

    The parsers raise it for a fetched file whose columns are not the ones they declare;
    `ingest/qcew_size.py::read_by_size_zip` for a by-size archive that does not hold exactly one
    CSV, so there is no file to read the columns of; and `harmonize/bridge.py::bridge_frame` for a
    §8.6 bridge row missing a declared field, which Polars would otherwise persist as a null
    `verification_status` or `uncertainty_treatment` -- the one thing §8.6 asks a bridge row to
    say. Each carries the offending value: the columns, the members, or the fields (`D-140`).
    """


class MissingCrossTabulationError(LoggingEmploymentError):
    """A requested simultaneous cross-tabulation is absent from the source file."""


class ConceptViolationError(LoggingEmploymentError):
    """A value would cross a concept boundary the spec forbids crossing."""


class AmbiguousSnapshotError(LoggingEmploymentError):
    """A reference key has more than one stored snapshot and nothing says which one to build."""


class IncompatibleMarginError(LoggingEmploymentError):
    """Two source margins failed the §5.5 compatibility gate and must not be stacked (INV-007)."""


class InfeasibleComponentError(LoggingEmploymentError):
    """A constraint component has no feasible point; §9.6 forbids silently relaxing it."""


class HardConstraintClassError(LoggingEmploymentError):
    """A caller asked for `is_hard=true` on a restriction that is not eligible for it."""


class SolverError(LoggingEmploymentError):
    """The solver returned a status that is neither an optimum nor a recognised refusal."""


class SolverOptionError(LoggingEmploymentError):
    """HiGHS would not take a setting the engine gave it, so nothing is solved (REQ-029, §18.3).

    HiGHS does not raise on an option it refuses: `setOptionValue` answers `kError` and keeps the
    value HiGHS already held. Unread, that answer let a `feasibility_tolerance` outside HiGHS's
    option range run every solve at HiGHS's own default while `deterministic_bounds` recorded the
    configured value as `solver_tolerance`, and `constraints.bounds.classify_bound_status`,
    `baselines.runner.integer_bounds` and `validate.recover.assert_truth_within_bounds` all applied
    that recorded value to bounds solved at another (`D-126`).

    Distinct from `SolverError`, which is a solve's OUTCOME that is neither an optimum nor an
    unbounded direction. This is a SETTING refused before any model exists -- it fires inside
    `constraints.diagnostics.diagnose` too, which records no bound -- and the remedy is the config,
    not the model.
    """


class UniverseClosureError(LoggingEmploymentError):
    """The national row and the state rows do not close.

    `reconcile.anchor.assert_universe_closes` raises it in three shapes: the establishment
    universes differ, a residual is negative, or a month with no suppressed state cell leaves a
    nonzero residual, the one month where §12.2's employment identity is testable (D-123).
    §18.3 requires the pipeline to fail rather than guess when source universes cannot be
    reconciled. This is a whole-run halt, not a per-month decline: a nonzero gap means the
    published national row contains something the state table does not, and every month's
    residual is then suspect, not just the failing one's.
    """


class InfeasibleResidualError(LoggingEmploymentError):
    """§12.3's summed bounds exclude the residual, so no feasible scaling exists.

    `reconcile.integerize` raises it for §12.6's three integer refusals (`D-119`): a lower bound
    above its cap, lower bounds summing past the total, and caps summing short of it. Each leaves no
    integer allocation inside the bounds that sums to the total (`D-139`). `reconcile.scaling.Bounds`
    raises it one layer earlier, as it is built from `deterministic_bounds`, for a cell whose lower
    bound sits above its upper bound (`D-096`): the same "lower above its cap" shape, refused before
    either clipping site can settle it in the cap's favour. `solve-bounds` refuses an infeasible
    component before it writes a bound, so on a bounds file it wrote the shape is unreachable; it is
    named all the same because the bounds are read off a file, and a file is the run's own state
    (`D-140`).
    """


class WeightDomainError(LoggingEmploymentError):
    """A weight vector's domain is not the missing set, or it carries a null or non-positive weight.

    Normalizing a weight vector defined on a strict subset of the missing set silently reallocates
    the absent cells' share onto the cells that happen to have inputs. That is fabricating an
    allocation, so it is refused rather than normalized.
    """


class NoHarvestFactorError(LoggingEmploymentError):
    """The harvest-proportional baseline has no harvest-origin volume and no latent factor.

    Reserved for Stage 7 and deliberately unraised today. §10.5 states no precondition, so
    Stage 3's `HarvestProportional` returns a `Decline` rather than raising -- a per-month
    refusal the runner records, not a whole-run halt -- and that is the correct Stage 3
    behaviour. Stage 7 replaces that declining stub with a live baseline and owns the decision
    of when an absent factor is a halt instead; this class is held for that path.
    """


class LeakageError(LoggingEmploymentError):
    """§13.4's leakage controls found the thing they exist to find.

    A typed raise rather than a bare `assert`, because `python -O` strips assert statements and
    both guards run on the live path: `assert_no_retained_truth` from
    `validate/harness.py::run_pseudo_suppression` inside the scoring loop, and
    `assert_no_future_rows` once per rolling origin. Measured 2026-09-08 before this class existed,
    the same call raised under `python` and returned silently under `python -O`. The package states
    this convention in `run_pseudo_suppression`'s own docstring and these two functions were the
    only places in it that violated the convention.
    """


class BoundViolationError(LoggingEmploymentError):
    """A released point estimate falls outside its own `deterministic_bounds` interval (INV-002).

    INV-002 has two halves. The adding-up half -- every estimate sums to the anchor's residual --
    has been enforced on the baseline path since Stage 3 by `allocate` and
    `InfeasibleResidualError`. The per-cell half was not: §9's LP/MILP interval is a hard public
    accounting fact, and nothing in `baselines/` read it, so an estimate above a solved upper
    bound shipped as `anchored_and_reconciled` with no signal at all.

    A RAISE, NOT A `Decline` ROW, and that is a deliberate break with the runner's other
    post-allocation failure. `WeightDomainError` out of `allocate` becomes a
    `reconciliation_failure` row because it is a DATA gap -- one cell with no usable input must
    not abort ten estimators across 96 months. This is not a data gap: the bound and the estimate
    are both this pipeline's own output, and INV-002 admits no per-month refusal. Filing it as a
    decline would let the run ship, with a violated accounting fact recorded as an estimator's
    considered opinion. §18.3's "fail rather than guess" governs, and R-S5P-3 says so in words:
    "a violation MUST raise a named error from the `LoggingEmploymentError` hierarchy".
    """


class ConstraintDataError(LoggingEmploymentError):
    """A known value lies outside the deterministic bounds solved for its cell (§13.5).

    §13.5: "A known pseudo-hidden truth outside the deterministic bounds is a constraint-data bug
    until proven otherwise." A RAISE, because the harness would otherwise score estimates against a
    constraint system that provably excludes the value being scored, and every bound metric it
    emitted would describe a system that is wrong about the data. Distinct from
    `BoundViolationError`, which is an ESTIMATE outside its interval; this is the TRUTH outside it.
    """


class ModelDiagnosticsError(LoggingEmploymentError):
    """A production fit failed §11.14's diagnostic gate, so its draws may not be released.

    Raised by `models/diagnostics.py::assert_gate_passes` AFTER `posterior/diagnostics.json` is
    written, so the evidence of the failure survives the refusal. §11.14 lists what "Promotion
    requires", and the roadmap's Stage 5 exit asks that the gate be "enforced in code, not
    documented". This is the enforcement. Replicate fits inside §13's harness record their gate and
    do not raise; §13.10's convergence gate reads those reports instead.
    """


class UnsettledPublishError(LoggingEmploymentError):
    """A run directory holds leftovers of a state-model validation publish that no publish leaves.

    `cli.py::_settle_state_model_validation` reads a publish's progress from its leftovers and
    finishes or undoes it. One state it cannot read: a staged record with the last tables at
    `.old` beside other tables. Renaming `.old` over them would be a guess, and `os.replace` onto
    an empty directory succeeds without a word, so it is refused, naming both, for a human to
    settle. Only two publishes on one run at once can leave it, and `validate-state-model` refuses
    to run beside another on the same run (`RunInUseError`), so only a publish run outside that
    command, or a directory edited by hand, can.
    """


class RunInUseError(LoggingEmploymentError):
    """Another `validate-state-model` holds the run's lock (`D-138`).

    `cli.py::_settle_state_model_validation` reads a publish's progress from its leftovers, and
    those look the same whether the publish was killed or is still running. So a settle beside a
    publish in flight undoes it under it, and in one measured interleaving the publish then
    returned without an error, having committed a record beside tables it had lost. The command
    holds an `flock` on `validate_state_model.lock` from before its first settle to the end of its
    publish, and a second invocation is refused with this before it touches anything. The kernel
    releases the lock when its holder exits, killed or not, so no crash leaves the run locked.
    """


class StoredObjectMismatchError(LoggingEmploymentError):
    """A raw-store object's bytes do not hash to the digest its path claims (§6.2, `D-134`).

    `store.RawStore.put` addresses every object by its content's sha256 and writes it once, so an
    object that exists was taken as the bytes its path names. `write_bytes` truncates before it
    writes, so a kill or a full disk mid-write left one cut short, and every later `put` of the
    same response left it alone and reported it present. `put` now writes whole and hashes an
    existing object before trusting it; this is the refusal, naming the path, the digest found and
    the digest claimed. The remedy is to remove the object and fetch again. Nothing repairs it in
    place, because the store's promise is that a stored object is never rewritten.
    """


class ClassificationContinuityError(LoggingEmploymentError):
    """113310 does not survive the D1 window's two NAICS vintages unchanged (§3.1, `D-102`).

    `harmonize/naics.py::assert_113310_survives_the_window` raises it for any of its four premises:
    a window vintage missing from the vendored crosswalk, a title other than Logging, a non-empty
    structure-file change indicator, or a 2017 -> 2022 link that is not one-to-one. §3.1: "The ETL
    MUST verify the 113310 mapping mechanically", and `build.build_harmonized` runs the check
    before it writes a table, so a re-vendored crosswalk halts the build by name. Distinct from
    `UnsupportedReferenceYearError`, which refuses a YEAR outside the vintages this package
    handles; this refuses the CROSSWALK for the years it does.
    """


class FallbackExhaustedError(LoggingEmploymentError):
    """No rung of §10.8's fallback hierarchy produced an estimate, so nothing can be preferred.

    `baselines/runner.py::preferred_estimator` raises it. Unreachable on D1 -- §10.2's inputs are
    complete on every suppressed cell, so rung 4 always produces estimates -- but reachable from a
    §13 mask that empties every month's missing set, which is why it is a raise rather than a
    sentinel: a `baseline_manifest.json` recording `preferred_estimator: null` would read as a
    considered choice, and `validate/scoreboard.py` ranks over what ran.
    """


class SecretInPayloadError(LoggingEmploymentError):
    """A credential's value reached bytes bound for an artifact (§7.2, D3).

    `store.assert_no_secret` raises it before a `source_snapshot` row is recorded, scanning the
    payload for every non-empty value of `config.SECRET_ENV_VARS`. It is the last of three guards:
    `fetching._without_credentials` strips the `key` parameter before a response is recorded, and
    `config.resolved_dict` keeps only an env var's NAME. This one catches a branch that forgot
    either. Alone among the classes here it carries no offending value, because the value is the
    secret.
    """
