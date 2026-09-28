"""§12.6 balanced integerization: the margin survives rounding, and rounding is reproducible."""

from __future__ import annotations

import math

import pytest

from logging_employment.errors import InfeasibleResidualError
from logging_employment.reconcile.integerize import integerize


def test_the_total_survives_rounding() -> None:
    """Independent rounding would give 9, not 10. §12.6 forbids exactly that."""
    out = integerize({"01": 3.4, "02": 3.3, "04": 3.3}, total=10)
    assert sum(out.values()) == 10


def test_units_go_to_the_largest_fractional_remainders() -> None:
    out = integerize({"01": 1.9, "02": 1.6, "04": 1.5}, total=5)
    assert sum(out.values()) == 5
    assert out["01"] == 2


def test_the_tiebreak_is_deterministic_and_repeatable() -> None:
    """§16.1 requires idempotence; a random tie-break would break it.

    Identical fractional parts across every cell force the tie-break to decide alone.
    """
    values = {"04": 1.5, "01": 1.5, "02": 1.5}
    first = integerize(values, total=5)
    for _ in range(20):
        assert integerize(values, total=5) == first
    # Ties break by cell_id ascending, so the lowest ids take the extra units.
    assert first["01"] == 2
    assert first["02"] == 2
    assert first["04"] == 1


def test_integer_lower_and_upper_bounds_are_respected() -> None:
    out = integerize(
        {"01": 5.5, "02": 2.5, "04": 2.0}, total=10, upper={"01": 4, "02": None, "04": None}
    )
    assert sum(out.values()) == 10
    assert out["01"] <= 4


def test_a_lower_bound_seats_a_cell_its_raw_value_would_round_below() -> None:
    """The test above is named for lower AND upper bounds and passes no `lower=` at all, so the
    seat floor (`integerize`'s `max(math.floor(value), lower.get(cell, 0))`) was pinned by nothing:
    dropping it leaves every unit test green.
    Both results here sum to 11, so a totals-only assertion cannot tell them apart -- the point is
    WHICH cell holds the units, not how many were placed. Unfloored, "01" comes back at 1, below
    the lower bound its caller declared."""
    out = integerize({"01": 0.4, "02": 9.6}, total=11, lower={"01": 2})
    assert sum(out.values()) == 11
    assert out["01"] == 2


def test_a_zero_total_yields_all_zeros() -> None:
    assert integerize({"01": 0.0, "02": 0.0}, total=0) == {"01": 0, "02": 0}


def test_an_empty_input_returns_empty() -> None:
    assert integerize({}, total=0) == {}


@pytest.mark.parametrize(
    ("total", "refusal"),
    [
        (5, r"5 unit\(s\) could not be placed"),
        (-3, "summed integer lower bounds 0 exceed the required total -3"),
    ],
)
def test_no_cells_cannot_hold_a_nonzero_total(total: int, refusal: str) -> None:
    """Zero cells hold exactly zero, so any other total is infeasible and falls to the refusal that
    names it: caps summing short of a positive total, lower bounds past a negative one. An early
    return used to hand back `{}` for both, a result that does not sum to `total` (D-139's review)."""
    with pytest.raises(InfeasibleResidualError, match=refusal):
        integerize({}, total=total)


def test_a_total_below_the_summed_lower_bounds_raises() -> None:
    with pytest.raises(InfeasibleResidualError, match="2 exceed the required total 1"):
        integerize({"01": 1.0, "02": 1.0}, total=1, lower={"01": 1, "02": 1})


def test_a_remainder_the_caps_cannot_absorb_is_refused_by_name() -> None:
    """No lower bound is set, so the summed-lower-bounds check passes; the cap then leaves room for
    one unit of the four still to place. The third of `integerize`'s refusals, and until D-119 the
    only one no test reached."""
    with pytest.raises(InfeasibleResidualError, match=r"3 unit\(s\) could not be placed"):
        integerize({"a": 1.0}, total=5, upper={"a": 2})


def test_a_capped_cell_does_not_starve_an_uncapped_one() -> None:
    """The termination bound must not shrink as units are placed.

    Computing `len(order) * (remaining + 1)` inside the loop condition re-evaluates it against a
    `remaining` that falls with every placement, so the ceiling drops toward the index and the
    loop exits early. Here cells 01 and 02 are capped at 0 and every index spent skipping them
    eats budget: the bound falls 12 -> 9 -> 6 while the index climbs to 6, and one unit is left
    unplaced even though cell 03 is uncapped and can absorb it.
    """
    out = integerize({"01": 0.0, "02": 0.0, "03": 10.0}, total=13, upper={"01": 0, "02": 0})
    assert out == {"01": 0, "02": 0, "03": 13}


def test_every_feasible_bounded_input_places_all_its_units() -> None:
    """A randomised sweep over inputs that are feasible by independent construction.

    Feasibility is decided outside the function -- summed floors at or below the total, and enough
    headroom under the caps to absorb the remainder -- so any exception here is the algorithm's,
    not the fixture's.
    """
    import random

    rng = random.Random(20260905)
    for _ in range(400):
        n = rng.randint(2, 6)
        cells = [f"{i:02d}" for i in range(n)]
        values = {c: rng.uniform(0.0, 12.0) for c in cells}
        floors = {c: math.floor(v) for c, v in values.items()}
        caps: dict[str, int | None] = {}
        for c in cells:
            caps[c] = floors[c] + rng.randint(0, 4) if rng.random() < 0.5 else None
        capacity = sum((caps[c] - floors[c]) if caps[c] is not None else 10_000 for c in cells)
        base = sum(floors.values())
        total = base + rng.randint(0, min(capacity, 8))
        out = integerize(values, total=total, upper=caps)
        assert sum(out.values()) == total
        for c in cells:
            if caps[c] is not None:
                assert out[c] <= caps[c]


def test_a_cell_seated_at_its_floor_gives_a_unit_back_rather_than_refusing() -> None:
    """D-139. Seating each cell at `max(floor, lower)` gives 2, 2 and 5 -- one past the total -- and
    the old base check refused. A floor is not a bound: `c` can give its unit back, and 2, 2 and 4
    satisfies every bound."""
    out = integerize({"a": 1.01, "b": 1.01, "c": 5.98}, total=8, lower={"a": 2, "b": 2})
    assert out == {"a": 2, "b": 2, "c": 4}


def test_a_unit_is_handed_back_in_the_reverse_of_placement_order() -> None:
    """`a` and `b` tie on remainder, so placement would give `a` the unit first; handing back runs the
    same order backwards, so `b` gives its unit up and `a` keeps its own. `c` and `d` sit at their
    lower bounds and have nothing to give."""
    out = integerize({"a": 1.45, "b": 1.45, "c": 0.05, "d": 0.05}, total=3, lower={"c": 1, "d": 1})
    assert out == {"a": 1, "b": 0, "c": 1, "d": 1}


def test_the_smallest_remainder_gives_its_unit_back_first() -> None:
    """Remainder order and `cell_id` order disagree here, which the tie above cannot show: `b`'s
    remainder (.8) outranks `a`'s (.1), so placement serves `b` first and handing back takes from
    `a` first. A hand-back by descending `cell_id` would take from `b` instead."""
    out = integerize({"a": 1.1, "b": 1.8, "c": 0.05, "d": 0.05}, total=3, lower={"c": 1, "d": 1})
    assert out == {"a": 0, "b": 1, "c": 1, "d": 1}


def test_one_cell_gives_back_several_units_over_several_passes() -> None:
    """Three lower bounds seat 1 each over values of .01, so the seats overshoot by two and only `c`
    can give. The loop visits `c` once per pass, so this needs the second pass the step limit
    allows; a single pass would return a sum of 4 without raising."""
    out = integerize(
        {"a": 0.01, "b": 0.01, "c": 2.97, "d": 0.01}, total=3, lower={"a": 1, "b": 1, "d": 1}
    )
    assert out == {"a": 1, "b": 1, "c": 0, "d": 1}


def test_the_refusal_reports_the_summed_lower_bounds_not_the_seats() -> None:
    """The lower bounds sum to 2 against a total of 1, so this is refused. The old check summed the
    seats instead and reported 7, a number that is no bound of anything."""
    with pytest.raises(
        InfeasibleResidualError, match="summed integer lower bounds 2 exceed the required total 1"
    ):
        integerize({"a": 5.0, "b": 0.0}, total=1, lower={"b": 2})


def test_every_feasible_input_cut_from_float_bounds_places_all_its_units() -> None:
    """The runner's call shape, swept. Each cell gets a float interval, cut to integer bounds the way
    `baselines.runner.integer_bounds` cuts it (ceil the lower, floor the upper), and its value lies
    inside the FLOAT interval -- so a value may sit below its integer lower bound, which is what makes
    a seat exceed its floor. The values sum to the total, as a reconciled allocation does. Feasibility
    is decided outside the function: summed integer lower bounds at or below the total, and the caps
    at or above it, so any exception is the algorithm's. The sweep above draws caps only, which is
    why D-139's shape went unwitnessed there.

    Seats overshoot the total only when values sit well below their integer lower bounds, which an
    even spread rarely produces (1 case in 400). So half the draws pin some cells at a float lower
    bound just above an integer and spread the rest of the total over the others, and the sweep
    asserts it reached the hand-back path, including hand-backs of two or more units.
    """
    import random

    def spread(cells: list[str], amount: float) -> dict[str, float] | None:
        """`amount` shared across `cells` at one fraction of each cell's float range, if it fits."""
        floor_sum = sum(low[c] for c in cells)
        reach_sum = sum(reach[c] for c in cells)
        if not cells or reach_sum == floor_sum or not floor_sum <= amount <= reach_sum:
            return None
        share = (amount - floor_sum) / (reach_sum - floor_sum)
        return {c: low[c] + share * (reach[c] - low[c]) for c in cells}

    rng = random.Random(20260928)
    checked = handed_back = handed_back_twice = 0
    while checked < 400:
        n = rng.randint(2, 6)
        cells = [f"{i:02d}" for i in range(n)]
        tight = [c for c in cells if rng.random() < 0.5] if rng.random() < 0.5 else []
        low = {
            c: rng.randint(0, 5) + rng.uniform(0.01, 0.2)
            if c in tight
            else rng.choice([0.0, float(rng.randint(0, 6)), rng.uniform(0.0, 6.0)])
            for c in cells
        }
        high: dict[str, float | None] = {
            c: None if rng.random() < 0.4 else low[c] + rng.uniform(0.0, 5.0) for c in cells
        }
        lower = {c: math.ceil(low[c]) for c in cells}
        upper = {c: None if high[c] is None else math.floor(high[c]) for c in cells}
        if any(upper[c] is not None and lower[c] > upper[c] for c in cells):
            continue  # an interval holding no integer is the contradictory-pair refusal, not this one
        floor_total = sum(lower.values())
        cap_total = sum(u if u is not None else floor_total + 20 for u in upper.values())
        total = rng.randint(floor_total, min(cap_total, floor_total + 20))
        reach = {c: high[c] if high[c] is not None else low[c] + total + 1.0 for c in cells}
        values = None
        if tight:
            rest = spread([c for c in cells if c not in tight], total - sum(low[c] for c in tight))
            values = None if rest is None else {**{c: low[c] for c in tight}, **rest}
        values = values or spread(cells, total)
        if values is None:
            continue
        seats = sum(
            max(math.floor(values[c]), lower[c])
            if upper[c] is None
            else min(max(math.floor(values[c]), lower[c]), upper[c])
            for c in cells
        )
        handed_back += seats > total
        handed_back_twice += seats - total >= 2
        out = integerize(values, total=total, lower=lower, upper=upper)
        assert sum(out.values()) == total
        for c in cells:
            assert out[c] >= lower[c]
            if upper[c] is not None:
                assert out[c] <= upper[c]
        checked += 1
    assert handed_back >= 40, f"the sweep reached the hand-back path only {handed_back} times"
    assert handed_back_twice >= 10, f"only {handed_back_twice} cases handed back two or more units"


def test_a_contradictory_bound_pair_is_refused() -> None:
    """`floors` took the lower bound, then the cap loop overwrote it with the upper bound.

    Nothing compared the two, so `lower=5, upper=3` returned 3 -- below a bound the caller
    declared. It cannot arise on D1, where every LP, MILP and selected bound endpoint is an integer
    and no lower sits above its upper (measured 2026-09-28); it takes a float interval holding no
    integer, which `baselines.runner.integer_bounds` cuts to a lower above its upper.
    """
    with pytest.raises(InfeasibleResidualError, match="lower bound 5 above its upper bound 3"):
        integerize({"a": 4.0}, 3, lower={"a": 5}, upper={"a": 3})


def test_a_cap_for_a_cell_with_no_value_does_not_inject_a_phantom_entry() -> None:
    """The cap loop iterated `upper`, not `values`, so it could create a cell out of nothing.

    A negative cap made `floors.get(cell, 0) > cap` true for a cell that was never passed in.
    That entry lowered `base`, which raised `remaining`, so the surviving real cell also came
    back wrong: `{'a': 2, 'ghost': -1}` for a total of 1.
    """
    assert integerize({"a": 1.0}, 1, upper={"a": None, "ghost": -1}) == {"a": 1}


def test_the_remainder_order_is_taken_above_the_effective_floor() -> None:
    """Largest-remainder is only largest-remainder if the remainder is measured from the floor used.

    `a`'s floor is raised to its lower bound of 2, which already consumes its 0.9 fraction, so it
    has no claim on the spare unit -- but the raw fractional part still sorted it first and it
    took the unit anyway, leaving `b` at 0 despite a 0.8 remainder.
    """
    assert integerize({"a": 0.9, "b": 0.8}, 3, lower={"a": 2}) == {"a": 2, "b": 1}
