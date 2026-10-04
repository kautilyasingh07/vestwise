"""Vesting calculations (spec §8.4). Pure functions: no database, no I/O, no clock.

Everything is a function of its arguments, so the same grant and date always
give the same answer, and tests need no fixtures beyond a dict.
"""

from datetime import date, datetime, timedelta
from typing import Any

from dateutil.relativedelta import relativedelta


def to_date(value: date | datetime) -> date:
    """Return a plain date. Mongo returns datetimes; comparing those with dates raises TypeError."""
    return value.date() if isinstance(value, datetime) else value


def full_months_between(start: date, end: date) -> int:
    """Count whole months from start to end; 0 if end is before start.

    A month counts once its anniversary is reached: 2025-01-01 -> 2025-12-31 is 11.
    Anniversaries in short months fall on the month's last day, so
    2025-01-31 -> 2025-02-28 is 1 and 2025-01-31 -> 2025-03-30 is still 1.

    >>> full_months_between(date(2025, 1, 1), date(2026, 10, 3))
    21
    """
    delta = relativedelta(end, start)
    return max(0, delta.years * 12 + delta.months)


def vest_date(grant_date: date, month: int) -> date:
    """Date on which month `month` of the schedule completes.

    Always offset from the grant date, never from the previous vest date:
    chaining (Jan 31 -> Feb 28 -> Mar 28) would drift off the month end.
    """
    return grant_date + relativedelta(months=month)


def vested_at(options: int, months: int, cliff_months: int, total_months: int) -> int:
    """Options vested after `months` full months (spec §8.4 formula).

    Integer floor division of the cumulative amount, so fractions are never
    lost: they surface in later months and the last month vests exactly `options`.
    """
    if months < cliff_months:
        return 0
    return options * min(months, total_months) // total_months


def _validate(grant: dict[str, Any]) -> None:
    """Reject schedules this module can't compute correctly."""
    schedule = grant["vesting"]
    if schedule.get("frequency", "monthly") != "monthly":
        raise ValueError(f"unsupported vesting frequency: {schedule['frequency']!r}")
    if not 0 <= schedule["cliff_months"] <= schedule["total_months"] or schedule["total_months"] <= 0:
        raise ValueError(f"invalid schedule: {schedule}")
    if grant["options"] < 0 or grant.get("exercised", 0) < 0:
        raise ValueError("options and exercised must be non-negative")


def compute_vesting(grant: dict[str, Any], as_of: date, exercise_window_days: int) -> dict[str, Any]:
    """Vesting status of one grant on `as_of` (spec §8.4). Never modifies `grant`.

    `grant` has the `grants` collection shape: grant_date, options, vesting
    {cliff_months, total_months, frequency}, exercised, termination_date (or None).
    Dates may be date or datetime.

    Rules:
    - months are counted up to as_of, or up to the termination date if the holder
      left on or before as_of; nothing vests after termination;
    - on termination, unvested options lapse;
    - after the exercise deadline (termination + exercise_window_days), vested
      options that were not exercised lapse too (ESOP policy clause 6.2);
    - exercisable = vested - exercised, never negative.

    Example (Priya, as of 2026-10-03): months_elapsed 21, vested 2100,
    exercisable 2100, next_vest_date 2026-11-01, next_vest_options 100.
    """
    _validate(grant)
    as_of = to_date(as_of)
    grant_date = to_date(grant["grant_date"])
    options: int = grant["options"]
    exercised: int = grant.get("exercised", 0)
    cliff: int = grant["vesting"]["cliff_months"]
    total: int = grant["vesting"]["total_months"]

    termination = grant.get("termination_date")
    termination = to_date(termination) if termination else None
    is_terminated = termination is not None and termination <= as_of
    end = termination if is_terminated else as_of

    months = full_months_between(grant_date, end)
    vested = vested_at(options, months, cliff, total)
    unvested = options - vested
    exercisable = max(0, vested - exercised)

    lapsed = 0
    exercise_deadline: date | None = None
    if is_terminated:
        exercise_deadline = termination + timedelta(days=exercise_window_days)
        lapsed = unvested
        if as_of > exercise_deadline:
            lapsed += exercisable
            exercisable = 0

    next_vest_date: date | None = None
    next_vest_options = 0
    if not is_terminated and months < total:
        next_month = max(months + 1, cliff)
        next_vest_date = vest_date(grant_date, next_month)
        next_vest_options = vested_at(options, next_month, cliff, total) - vested

    return {
        "grant_id": grant.get("_id"),
        "as_of": as_of,
        "granted": options,
        "vested": vested,
        "unvested": unvested,
        "exercised": exercised,
        "exercisable": exercisable,
        "lapsed": lapsed,
        "months_elapsed": months,
        "next_vest_date": next_vest_date,
        "next_vest_options": next_vest_options,
        "is_terminated": is_terminated,
        "exercise_deadline": exercise_deadline,
    }
