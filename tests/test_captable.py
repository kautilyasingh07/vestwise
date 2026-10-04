"""Unit tests for app/tools/captable.py (spec §2, §7 pool accounting, §8.5). No database needed."""

from typing import Any

import pytest

from app.tools.captable import POOL_ROW_NAME, cap_table, pool_status, simulate_dilution

TOLERANCE = 0.05  # rounding each row to 2 dp can make the sum drift by a few hundredths

SPEC_HOLDINGS = [  # spec §2 worked dilution example
    {"stakeholder_id": "founder", "share_class": "equity", "shares": 8_000_000},
    {"stakeholder_id": "investor_a", "share_class": "preference", "shares": 1_500_000},
]


def row(table: dict[str, Any], stakeholder_id: str | None) -> dict[str, Any]:
    """The row for one stakeholder (None = the unallocated pool row)."""
    return next(r for r in table["rows"] if r["stakeholder_id"] == stakeholder_id)


def names(seed: dict[str, Any]) -> dict[str, str]:
    """stakeholder_id -> name from the seed."""
    return {s["_id"]: s["name"] for s in seed["stakeholders"]}


# ---- spec §2 example ----

def test_spec_dilution_example() -> None:
    result = simulate_dilution(SPEC_HOLDINGS, [], 1_000_000, 2_000_000, "Series A")
    founder = next(r for r in result["rows"] if r["stakeholder_id"] == "founder")
    assert founder["fully_diluted_pct_before"] == 76.19
    assert founder["fully_diluted_pct_after"] == 64.00
    assert result["total_fully_diluted_before"] == 10_500_000
    assert result["total_fully_diluted_after"] == 12_500_000


# ---- seed data ----

def test_seed_pool_status(seed: dict[str, Any]) -> None:
    pool = pool_status(seed["grants"], seed["company"]["esop_pool_size"])
    assert pool == {"pool_size": 1_000_000, "outstanding": 9_450, "exercised": 600,
                    "lapsed": 750, "unallocated": 989_950}


def test_seed_cap_table_totals(seed: dict[str, Any]) -> None:
    table = cap_table(seed["holdings"], seed["grants"], seed["company"]["esop_pool_size"], names(seed))
    assert table["total_issued"] == 9_500_600
    assert table["total_fully_diluted"] == 10_500_000


def test_seed_cap_table_rows(seed: dict[str, Any]) -> None:
    table = cap_table(seed["holdings"], seed["grants"], seed["company"]["esop_pool_size"], names(seed))
    arjun = row(table, "sh_arjun")
    assert arjun["name"] == "Arjun Mehta"
    assert arjun["issued_pct"] == 63.15
    assert arjun["fully_diluted_pct"] == 57.14
    kiran = row(table, "sh_kiran")
    assert (kiran["shares"], kiran["outstanding_options"], kiran["fully_diluted"]) == (600, 2_250, 2_850)
    priya = row(table, "sh_priya")
    assert (priya["shares"], priya["outstanding_options"], priya["issued_pct"]) == (0, 4_800, 0.0)
    pool = row(table, None)
    assert pool["name"] == POOL_ROW_NAME and pool["fully_diluted"] == 989_950


def test_seed_percentages_sum_to_100(seed: dict[str, Any]) -> None:
    table = cap_table(seed["holdings"], seed["grants"], seed["company"]["esop_pool_size"])
    assert sum(r["issued_pct"] for r in table["rows"]) == pytest.approx(100, abs=TOLERANCE)
    assert sum(r["fully_diluted_pct"] for r in table["rows"]) == pytest.approx(100, abs=TOLERANCE)
    assert sum(r["fully_diluted"] for r in table["rows"]) == table["total_fully_diluted"]


def test_seed_dilution(seed: dict[str, Any]) -> None:
    result = simulate_dilution(seed["holdings"], seed["grants"], seed["company"]["esop_pool_size"],
                               2_000_000, "Series A Fund", names(seed))
    arjun = next(r for r in result["rows"] if r["stakeholder_id"] == "sh_arjun")
    assert arjun["shares"] == 6_000_000  # share count unchanged, only the percentage moves
    assert (arjun["fully_diluted_pct_before"], arjun["fully_diluted_pct_after"]) == (57.14, 48.00)
    assert (arjun["issued_pct_before"], arjun["issued_pct_after"]) == (63.15, 52.17)
    investor = next(r for r in result["rows"] if r["name"] == "Series A Fund")
    assert investor["fully_diluted_pct_before"] == 0.0 and investor["fully_diluted_pct_after"] == 16.0
    for basis in ("issued_pct_before", "issued_pct_after", "fully_diluted_pct_before", "fully_diluted_pct_after"):
        assert sum(r[basis] for r in result["rows"]) == pytest.approx(100, abs=TOLERANCE), basis


def test_dilution_does_not_modify_inputs(seed: dict[str, Any]) -> None:
    holdings = [dict(h) for h in seed["holdings"]]
    simulate_dilution(holdings, seed["grants"], 1_000_000, 2_000_000, "X")
    assert holdings == seed["holdings"]


# ---- error cases ----

def test_dilution_rejects_non_positive_shares() -> None:
    with pytest.raises(ValueError):
        simulate_dilution(SPEC_HOLDINGS, [], 1_000_000, 0, "Nobody")


def test_inconsistent_grant_raises() -> None:
    bad = [{"_id": "g", "stakeholder_id": "s", "options": 100, "exercised": 80, "lapsed": 30}]
    with pytest.raises(ValueError):
        pool_status(bad, 1_000_000)


def test_over_allocated_pool_raises() -> None:
    big = [{"_id": "g", "stakeholder_id": "s", "options": 1_200_000, "exercised": 0, "lapsed": 0}]
    with pytest.raises(ValueError):
        pool_status(big, 1_000_000)


def test_fully_lapsed_leaver_without_shares_has_no_row() -> None:
    gone = [{"_id": "g", "stakeholder_id": "leaver", "options": 100, "exercised": 0, "lapsed": 100}]
    table = cap_table(SPEC_HOLDINGS, gone, 1_000_000)
    assert all(r["stakeholder_id"] != "leaver" for r in table["rows"])
    assert table["pool"]["unallocated"] == 1_000_000
