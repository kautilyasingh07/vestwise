"""Unit tests for app/tools/vesting.py (spec §8.4, §12). No database needed."""

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from app.tools.vesting import compute_vesting, full_months_between, vest_date

WINDOW = 90
AS_OF = date(2026, 10, 3)  # the date spec §8.4's expected table is computed for


def make_grant(grant_date: date, options: int = 4800, cliff: int = 12, total: int = 48,
               exercised: int = 0, termination_date: date | None = None) -> dict[str, Any]:
    """A grant dict in the `grants` collection shape."""
    return {
        "_id": "g_test", "grant_date": grant_date, "options": options,
        "vesting": {"cliff_months": cliff, "total_months": total, "frequency": "monthly"},
        "exercised": exercised, "lapsed": 0, "termination_date": termination_date,
    }


PRIYA = make_grant(date(2025, 1, 1))
RAHUL = make_grant(date(2026, 3, 15), options=2400)
KIRAN = make_grant(date(2023, 6, 1), options=3600, exercised=600, termination_date=date(2026, 8, 31))


# ---- spec §8.4 expected outputs (as of 2026-10-03) ----

def test_priya_spec_row() -> None:
    v = compute_vesting(PRIYA, AS_OF, WINDOW)
    assert (v["months_elapsed"], v["vested"], v["exercised"], v["exercisable"]) == (21, 2100, 0, 2100)
    assert v["unvested"] == 2700
    assert v["next_vest_date"] == date(2026, 11, 1)
    assert v["next_vest_options"] == 100
    assert v["is_terminated"] is False and v["exercise_deadline"] is None and v["lapsed"] == 0


def test_rahul_spec_row() -> None:
    v = compute_vesting(RAHUL, AS_OF, WINDOW)
    assert (v["months_elapsed"], v["vested"], v["exercised"], v["exercisable"]) == (6, 0, 0, 0)
    assert v["next_vest_date"] == date(2027, 3, 15)  # the cliff
    assert v["next_vest_options"] == 600


def test_kiran_spec_row() -> None:
    v = compute_vesting(KIRAN, AS_OF, WINDOW)
    assert (v["months_elapsed"], v["vested"], v["exercised"], v["exercisable"]) == (38, 2850, 600, 2250)
    assert v["lapsed"] == 750
    assert v["exercise_deadline"] == date(2026, 11, 29)
    assert v["is_terminated"] is True and v["next_vest_date"] is None


# ---- cliff and end of schedule ----

def test_day_before_cliff_is_zero() -> None:
    v = compute_vesting(PRIYA, date(2025, 12, 31), WINDOW)
    assert v["months_elapsed"] == 11 and v["vested"] == 0
    assert v["next_vest_date"] == date(2026, 1, 1)


def test_exactly_on_cliff_is_25_percent() -> None:
    v = compute_vesting(PRIYA, date(2026, 1, 1), WINDOW)
    assert v["vested"] == 1200 == PRIYA["options"] // 4
    assert v["next_vest_date"] == date(2026, 2, 1)


def test_exactly_48_months_is_fully_vested() -> None:
    v = compute_vesting(PRIYA, date(2029, 1, 1), WINDOW)
    assert v["vested"] == 4800 and v["unvested"] == 0
    assert v["next_vest_date"] is None and v["next_vest_options"] == 0


def test_day_before_48_months_is_not_fully_vested() -> None:
    v = compute_vesting(PRIYA, date(2028, 12, 31), WINDOW)
    assert v["months_elapsed"] == 47 and v["vested"] == 4700
    assert v["next_vest_date"] == date(2029, 1, 1)


def test_60_months_is_still_fully_vested() -> None:
    v = compute_vesting(PRIYA, date(2030, 1, 1), WINDOW)
    assert v["months_elapsed"] == 60 and v["vested"] == 4800 and v["unvested"] == 0


def test_before_grant_date_nothing_vested() -> None:
    v = compute_vesting(PRIYA, date(2024, 6, 1), WINDOW)
    assert v["months_elapsed"] == 0 and v["vested"] == 0
    assert v["next_vest_date"] == date(2026, 1, 1)


def test_uneven_options_never_lose_fractions() -> None:
    grant = make_grant(date(2025, 1, 1), options=1000)
    amounts = [compute_vesting(grant, vest_date(date(2025, 1, 1), m), WINDOW)["vested"] for m in range(49)]
    assert amounts[12] == 250 and amounts[13] == 270  # floor(1000 * 13 / 48) = 270
    assert amounts == sorted(amounts)                  # never decreases
    assert amounts[48] == 1000                         # fractions caught up by the last month


# ---- termination ----

def test_terminated_grant_stops_vesting() -> None:
    at_termination = compute_vesting(KIRAN, date(2026, 8, 31), WINDOW)
    later = compute_vesting(KIRAN, date(2026, 10, 31), WINDOW)
    assert at_termination["vested"] == later["vested"] == 2850
    assert later["months_elapsed"] == 38


def test_before_termination_date_grant_is_active() -> None:
    v = compute_vesting(KIRAN, date(2025, 6, 1), WINDOW)
    assert v["is_terminated"] is False and v["months_elapsed"] == 24 and v["vested"] == 1800
    assert v["lapsed"] == 0 and v["exercise_deadline"] is None


def test_last_day_of_exercise_window_still_exercisable() -> None:
    v = compute_vesting(KIRAN, date(2026, 11, 29), WINDOW)
    assert v["exercisable"] == 2250 and v["lapsed"] == 750


def test_after_exercise_window_vested_options_lapse() -> None:
    v = compute_vesting(KIRAN, date(2026, 11, 30), WINDOW)
    assert v["exercisable"] == 0
    assert v["lapsed"] == 3000  # 750 unvested + 2,250 vested but never exercised
    assert v["vested"] == 2850 and v["exercised"] == 600  # history is unchanged


def test_window_comes_from_the_argument() -> None:
    v = compute_vesting(KIRAN, AS_OF, exercise_window_days=30)
    assert v["exercise_deadline"] == date(2026, 9, 30)
    assert v["exercisable"] == 0 and v["lapsed"] == 3000


# ---- exercisable bounds ----

def test_exercisable_never_negative() -> None:
    # Bad data: more exercised than vested. Clamp rather than report negative options.
    grant = make_grant(date(2025, 1, 1), exercised=3000)
    assert compute_vesting(grant, AS_OF, WINDOW)["exercisable"] == 0


@pytest.mark.parametrize("grant", [PRIYA, RAHUL, KIRAN], ids=["priya", "rahul", "kiran"])
def test_exercisable_within_bounds_every_week(grant: dict[str, Any]) -> None:
    day = date(2023, 1, 1)
    while day <= date(2031, 1, 1):
        v = compute_vesting(grant, day, WINDOW)
        assert 0 <= v["exercisable"] <= v["vested"] <= v["granted"]
        assert v["vested"] + v["unvested"] == v["granted"]
        day += timedelta(days=7)


# ---- month arithmetic ----

def test_full_months_basic() -> None:
    assert full_months_between(date(2025, 1, 1), date(2025, 12, 31)) == 11
    assert full_months_between(date(2025, 1, 1), date(2026, 1, 1)) == 12
    assert full_months_between(date(2026, 5, 1), date(2026, 3, 1)) == 0  # end before start


def test_grant_on_31st_uses_month_end_anniversaries() -> None:
    grant = make_grant(date(2025, 1, 31))
    assert compute_vesting(grant, date(2026, 1, 30), WINDOW)["vested"] == 0
    on_cliff = compute_vesting(grant, date(2026, 1, 31), WINDOW)
    assert on_cliff["vested"] == 1200
    assert on_cliff["next_vest_date"] == date(2026, 2, 28)          # Feb has no 31st
    assert compute_vesting(grant, date(2026, 2, 27), WINDOW)["vested"] == 1200
    feb = compute_vesting(grant, date(2026, 2, 28), WINDOW)
    assert feb["vested"] == 1300
    assert feb["next_vest_date"] == date(2026, 3, 31)               # back to the 31st, no drift
    assert compute_vesting(grant, date(2026, 3, 30), WINDOW)["vested"] == 1300
    assert compute_vesting(grant, date(2028, 2, 29), WINDOW)["months_elapsed"] == 37  # leap year


def test_vest_dates_and_month_count_agree_for_month_end_grant() -> None:
    # Month k completes exactly on vest_date(k), not a day earlier, for 5 years.
    start = date(2025, 1, 31)
    for k in range(1, 61):
        anniversary = vest_date(start, k)
        assert full_months_between(start, anniversary) == k
        assert full_months_between(start, anniversary - timedelta(days=1)) == k - 1


# ---- inputs ----

def test_accepts_mongo_datetimes() -> None:
    grant = dict(KIRAN, grant_date=datetime(2023, 6, 1), termination_date=datetime(2026, 8, 31))
    assert compute_vesting(grant, AS_OF, WINDOW) == compute_vesting(KIRAN, AS_OF, WINDOW)


def test_does_not_modify_the_grant() -> None:
    grant = make_grant(date(2023, 6, 1), options=3600, exercised=600, termination_date=date(2026, 8, 31))
    snapshot = {**grant, "vesting": dict(grant["vesting"])}
    compute_vesting(grant, date(2027, 1, 1), WINDOW)
    assert grant == snapshot


@pytest.mark.parametrize("vesting", [
    {"cliff_months": 12, "total_months": 48, "frequency": "quarterly"},
    {"cliff_months": 60, "total_months": 48, "frequency": "monthly"},
    {"cliff_months": 0, "total_months": 0, "frequency": "monthly"},
])
def test_rejects_invalid_schedules(vesting: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        compute_vesting(dict(PRIYA, vesting=vesting), AS_OF, WINDOW)


def test_computed_lapses_match_recorded_seed_data(seed: dict[str, Any]) -> None:
    # The stored `lapsed` ledger field must agree with what the rules compute for the seed date.
    for grant in seed["grants"]:
        g = dict(grant, grant_date=date.fromisoformat(grant["grant_date"]),
                 termination_date=date.fromisoformat(grant["termination_date"]) if grant["termination_date"] else None)
        assert compute_vesting(g, AS_OF, WINDOW)["lapsed"] == grant["lapsed"], grant["_id"]
