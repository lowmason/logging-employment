import ast
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from logging_employment.config import PromotionConfig, ValidationConfig


def test_a_random_mask_only_design_is_refused_at_construction():
    """§13.2's opening line is a config-time refusal, not a runtime warning."""
    with pytest.raises(ValidationError, match="random_mask_only"):
        ValidationConfig(
            pseudo_suppression_seeds=[1024],
            include_random_mask_sanity_check=True,
            include_primary_like=False,
            include_complementary_like=False,
            include_long_runs=False,
            include_rolling_origin=False,
            include_retrospective_smoothing=False,
            include_vintage_comparison=False,
        )


def test_vintage_comparison_defaults_off_because_d1_carries_one_vintage():
    cfg = ValidationConfig(pseudo_suppression_seeds=[1024], include_primary_like=True)
    assert cfg.include_vintage_comparison is False


def test_promotion_gates_carry_appendix_a_defaults():
    p = PromotionConfig()
    assert p.minimum_wape_improvement == 0.05
    assert p.maximum_major_stratum_wape_degradation == 0.02
    assert p.nominal_coverage_tolerance == 0.05


def test_the_catastrophic_coverage_alpha_is_one_in_a_thousand():
    """Plan 16's origination: §13.10 says "does not fail catastrophically" and states no number.

    At 0.001 over 18 strata (9 regimes, 9 divisions), a calibrated model is flagged somewhere with
    probability 0.0122 (the plan's Decision 4).
    """
    assert PromotionConfig().catastrophic_stratum_coverage_alpha == 0.001


def test_every_promotion_threshold_is_finite_and_the_alpha_is_a_probability():
    """Plan 16's promotion record is the first code to read these, and a non-finite one breaks it
    silently or late. A NaN degradation ceiling or alpha makes its comparison false, so the gate
    stops flagging degraded divisions or catastrophic strata. `-inf` counts every regime's WAPE as
    improved. A NaN or infinite tolerance raises inside `Fraction`, but only after the replicate
    fits. So each is refused at load, naming the field (§18.3). Read off the model, so a threshold
    added later is covered, and the count fails first. An alpha at or below 0 is the same silence
    as a NaN one, because `cdf < alpha` can never hold."""
    floats = [
        name for name, field in PromotionConfig.model_fields.items() if field.annotation is float
    ]
    assert len(floats) == 4
    for name in floats:
        for value in (math.inf, -math.inf, math.nan):
            with pytest.raises(ValidationError, match=name):
                PromotionConfig.model_validate({name: value})
    for alpha in (0.0, -0.1, 1.0):
        with pytest.raises(ValidationError, match="catastrophic_stratum_coverage_alpha"):
            PromotionConfig.model_validate({"catastrophic_stratum_coverage_alpha": alpha})


PROMOTION_KEYS = frozenset(
    {
        "minimum_wape_improvement",
        "maximum_major_stratum_wape_degradation",
        "nominal_coverage_tolerance",
        # Plan 16's, and watched by the same tripwire from the day it was declared.
        "catastrophic_stratum_coverage_alpha",
    }
)


def _named_keys(source: str) -> set[str]:
    """The promotion keys a module NAMES in code, ignoring comments and docstrings.

    Counted: an attribute access (`cfg.promotion.nominal_coverage_tolerance`), a string constant equal
    to a key (what `getattr(p, "...")` and `model_dump()["..."]` spell), a parameter name
    (`def gate(*, nominal_coverage_tolerance)`, which `**cfg.promotion.model_dump()` would fill), and a
    keyword argument (`PromotionConfig(minimum_wape_improvement=...)`). A comment is not in the AST
    and a docstring is excluded by position, so a cross-reference that merely MENTIONS a key cannot
    trip the tripwire. Not counted: a reader that never spells a key -- a generic
    `model_dump().items()` loop -- and a field DECLARATION, which is an annotated `Name` target.
    """
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    named: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            spelled = node.attr
        elif isinstance(node, (ast.arg, ast.keyword)):
            spelled = node.arg
        elif isinstance(node, ast.Constant) and id(node) not in docstrings:
            spelled = node.value
        else:
            continue
        if isinstance(spelled, str) and spelled in PROMOTION_KEYS:
            named.add(spelled)
    return named


def test_the_tripwire_detector_sees_reads_and_ignores_mentions():
    """Without this the tripwire below could pass vacuously -- a detector that finds nothing is green."""
    assert _named_keys("x = cfg.promotion.nominal_coverage_tolerance") == {
        "nominal_coverage_tolerance"
    }
    assert _named_keys('v = getattr(p, "minimum_wape_improvement")') == {"minimum_wape_improvement"}
    assert _named_keys('v = p.model_dump()["maximum_major_stratum_wape_degradation"]') == {
        "maximum_major_stratum_wape_degradation"
    }
    assert _named_keys("def gate(s, *, nominal_coverage_tolerance): ...") == {
        "nominal_coverage_tolerance"
    }
    assert _named_keys("c = PromotionConfig(minimum_wape_improvement=0.1)") == {
        "minimum_wape_improvement"
    }
    mentions = (
        '"""Feeds nominal_coverage_tolerance."""\n'
        "# see minimum_wape_improvement\n"
        "def f():\n"
        '    """Reads maximum_major_stratum_wape_degradation later."""\n'
        "    return 1\n"
    )
    assert _named_keys(mentions) == set()
    # The recorded blind spot, pinned so a change to it is deliberate.
    assert _named_keys("gate(**cfg.promotion.model_dump())") == set()


def test_the_promotion_keys_are_still_unread_and_the_docstring_still_says_so():
    """R-S5G-3: a TRIPWIRE, not a prohibition. It fails when `src/` code NAMES one of these keys.

    `PromotionConfig`'s docstring records all three as inert. A docstring cannot notice when it
    stops being true, and the failure mode is specific: the day a promotion path reads one of
    these, the note becomes a false statement in the file a reader consults first. Derived from the
    code's AST (`_named_keys`) rather than asserting a sentence exists, so it tracks code, not prose.
    `config.py` is scanned too: its field declarations are annotated `Name` targets and do not count,
    so a validator there that READS a key still trips this. When it reddens, the fix is to update the
    docstring -- not to delete this test.
    """
    src = Path(__file__).resolve().parents[2] / "src" / "logging_employment"
    readers = {
        path.relative_to(src).as_posix(): sorted(named)
        for path in sorted(src.rglob("*.py"))
        if (named := _named_keys(path.read_text(encoding="utf-8")))
    }
    assert readers == {}, f"now read by {readers}; update PromotionConfig's docstring"
