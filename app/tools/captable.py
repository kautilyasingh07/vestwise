"""Cap table, pool accounting and dilution (spec §7 pool accounting, §8.5). Pure functions.

Terms (spec §7):
- outstanding options = options - exercised - lapsed
- unallocated pool    = pool_size - outstanding - exercised
- fully diluted total = holdings + outstanding + unallocated

Uses the *recorded* `exercised` and `lapsed` fields on each grant: this module
reports the ledger as it stands, it doesn't recompute lapses from dates.
"""

from collections.abc import Mapping, Sequence
from typing import Any

POOL_ROW_NAME = "Unallocated ESOP pool"
NEW_INVESTOR_ID = "new_investor"


def outstanding_options(grant: Mapping[str, Any]) -> int:
    """Options in a grant that are neither exercised nor lapsed."""
    return grant["options"] - grant.get("exercised", 0) - grant.get("lapsed", 0)


def pool_status(grants: Sequence[Mapping[str, Any]], pool_size: int) -> dict[str, int]:
    """Split the pool into outstanding, exercised and unallocated (lapsed shown for reference).

    Raises ValueError if the ledger is inconsistent (a negative bucket), since every
    ownership figure would be wrong.
    """
    for grant in grants:
        if outstanding_options(grant) < 0:
            raise ValueError(f"grant {grant.get('_id')}: exercised + lapsed exceeds options")
    outstanding = sum(outstanding_options(g) for g in grants)
    exercised = sum(g.get("exercised", 0) for g in grants)
    lapsed = sum(g.get("lapsed", 0) for g in grants)
    unallocated = pool_size - outstanding - exercised
    if unallocated < 0:
        raise ValueError(f"pool over-allocated by {-unallocated} options")
    return {"pool_size": pool_size, "outstanding": outstanding, "exercised": exercised,
            "lapsed": lapsed, "unallocated": unallocated}


def _pct(part: int, whole: int) -> float:
    """Percentage rounded to 2 dp for display; 0.0 when the whole is 0."""
    return round(100 * part / whole, 2) if whole else 0.0


def cap_table(
    holdings: Sequence[Mapping[str, Any]],
    grants: Sequence[Mapping[str, Any]],
    pool_size: int,
    names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Ownership per stakeholder on an issued and a fully diluted basis.

    One row per stakeholder with shares or outstanding options, plus a final row
    for the unallocated pool. `names` maps stakeholder_id -> display name
    (defaults to the id). Percentages are rounded only here, at output.

    Returns {"rows": [...], "total_issued", "total_fully_diluted", "pool": pool_status}.
    """
    names = names or {}
    pool = pool_status(grants, pool_size)

    shares: dict[str, int] = {}
    for h in holdings:
        shares[h["stakeholder_id"]] = shares.get(h["stakeholder_id"], 0) + h["shares"]
    options: dict[str, int] = {}
    for g in grants:
        options[g["stakeholder_id"]] = options.get(g["stakeholder_id"], 0) + outstanding_options(g)

    total_issued = sum(shares.values())
    total_fd = total_issued + pool["outstanding"] + pool["unallocated"]

    rows = []
    for sid in sorted(shares.keys() | options.keys(), key=lambda s: (-shares.get(s, 0), -options.get(s, 0), s)):
        held, opts = shares.get(sid, 0), options.get(sid, 0)
        if held == 0 and opts == 0:
            continue  # e.g. a leaver whose options all lapsed and who never exercised
        rows.append({
            "stakeholder_id": sid, "name": names.get(sid, sid),
            "shares": held, "outstanding_options": opts, "fully_diluted": held + opts,
            "issued_pct": _pct(held, total_issued), "fully_diluted_pct": _pct(held + opts, total_fd),
        })
    rows.append({
        "stakeholder_id": None, "name": POOL_ROW_NAME,
        "shares": 0, "outstanding_options": 0, "fully_diluted": pool["unallocated"],
        "issued_pct": 0.0, "fully_diluted_pct": _pct(pool["unallocated"], total_fd),
    })
    return {"rows": rows, "total_issued": total_issued, "total_fully_diluted": total_fd, "pool": pool}


def simulate_dilution(
    holdings: Sequence[Mapping[str, Any]],
    grants: Sequence[Mapping[str, Any]],
    pool_size: int,
    new_shares: int,
    investor_name: str,
    names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Before/after ownership if `new_shares` are issued to a new investor (spec §8.5).

    ownership_after = shares_holder / (total_before + new_shares), on both bases.
    Inputs are not modified; the new investor is a hypothetical extra holding.

    Example (spec §2): a founder with 8,000,000 of 10,500,000 fully diluted
    (76.19%) falls to 64.00% after 2,000,000 new shares.
    """
    if new_shares <= 0:
        raise ValueError("new_shares must be positive")
    names = dict(names or {})
    names[NEW_INVESTOR_ID] = investor_name
    before = cap_table(holdings, grants, pool_size, names)
    new_holding = {"stakeholder_id": NEW_INVESTOR_ID, "share_class": "preference", "shares": new_shares}
    after = cap_table([*holdings, new_holding], grants, pool_size, names)

    before_by_id = {r["stakeholder_id"]: r for r in before["rows"]}
    rows = []
    for r in after["rows"]:
        b = before_by_id.get(r["stakeholder_id"], {})
        rows.append({
            "stakeholder_id": r["stakeholder_id"], "name": r["name"],
            "shares": r["shares"], "outstanding_options": r["outstanding_options"],
            "issued_pct_before": b.get("issued_pct", 0.0), "issued_pct_after": r["issued_pct"],
            "fully_diluted_pct_before": b.get("fully_diluted_pct", 0.0),
            "fully_diluted_pct_after": r["fully_diluted_pct"],
        })
    return {
        "investor_name": investor_name, "new_shares": new_shares, "rows": rows,
        "total_issued_before": before["total_issued"], "total_issued_after": after["total_issued"],
        "total_fully_diluted_before": before["total_fully_diluted"],
        "total_fully_diluted_after": after["total_fully_diluted"],
    }
