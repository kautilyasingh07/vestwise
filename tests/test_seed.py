"""Consistency checks on data/seed.json (spec §7). No database needed.

Pool accounting terms used throughout:
- outstanding options = options - exercised - lapsed   (still exercisable or yet to vest)
- unallocated pool    = pool_size - outstanding - exercised   (lapsed options return to the pool)
- fully diluted total = holdings + outstanding + unallocated
"""

from typing import Any

# The `seed` fixture lives in conftest.py.


def stakeholder_ids(seed: dict[str, Any]) -> set[str]:
    """All stakeholder _ids in the seed."""
    return {s["_id"] for s in seed["stakeholders"]}


def outstanding(grant: dict[str, Any]) -> int:
    """Options in a grant that are neither exercised nor lapsed."""
    return grant["options"] - grant["exercised"] - grant["lapsed"]


def total_outstanding(seed: dict[str, Any]) -> int:
    """Outstanding options across all grants."""
    return sum(outstanding(g) for g in seed["grants"])


def total_exercised(seed: dict[str, Any]) -> int:
    """Exercised options across all grants (these are now shares)."""
    return sum(g["exercised"] for g in seed["grants"])


def unallocated_pool(seed: dict[str, Any]) -> int:
    """Pool capacity still available for new grants."""
    return seed["company"]["esop_pool_size"] - total_outstanding(seed) - total_exercised(seed)


def total_holdings(seed: dict[str, Any]) -> int:
    """Issued shares across all holdings."""
    return sum(h["shares"] for h in seed["holdings"])


def test_grant_buckets_add_up(seed: dict[str, Any]) -> None:
    # Every granted option is in exactly one bucket: outstanding, exercised or lapsed.
    for grant in seed["grants"]:
        assert outstanding(grant) >= 0, grant["_id"]
        assert outstanding(grant) + grant["exercised"] + grant["lapsed"] == grant["options"], grant["_id"]


def test_kiran_outstanding(seed: dict[str, Any]) -> None:
    kiran = next(g for g in seed["grants"] if g["stakeholder_id"] == "sh_kiran")
    assert outstanding(kiran) == 3_600 - 600 - 750 == 2_250


def test_total_outstanding_and_exercised(seed: dict[str, Any]) -> None:
    assert total_outstanding(seed) == 9_450
    assert total_exercised(seed) == 600


def test_unallocated_pool_matches_spec(seed: dict[str, Any]) -> None:
    # Spec §8.7 relies on this number for the Neha "exceeds pool" test letter.
    assert unallocated_pool(seed) == 989_950


def test_pool_buckets_sum_to_pool_size(seed: dict[str, Any]) -> None:
    buckets = total_outstanding(seed) + total_exercised(seed) + unallocated_pool(seed)
    assert unallocated_pool(seed) >= 0
    assert buckets == seed["company"]["esop_pool_size"] == 1_000_000


def test_fully_diluted_total(seed: dict[str, Any]) -> None:
    assert total_holdings(seed) == 9_500_600
    assert total_holdings(seed) + total_outstanding(seed) + unallocated_pool(seed) == 10_500_000


def test_exercised_options_are_held_as_equity(seed: dict[str, Any]) -> None:
    # Exercise turns options into shares, so each exerciser must hold at least that much equity.
    for grant in seed["grants"]:
        if grant["exercised"]:
            equity = sum(h["shares"] for h in seed["holdings"]
                         if h["stakeholder_id"] == grant["stakeholder_id"] and h["share_class"] == "equity")
            assert equity >= grant["exercised"], grant["_id"]


def test_every_grant_stakeholder_exists(seed: dict[str, Any]) -> None:
    ids = stakeholder_ids(seed)
    for grant in seed["grants"]:
        assert grant["stakeholder_id"] in ids, grant["_id"]


def test_every_holding_stakeholder_exists(seed: dict[str, Any]) -> None:
    ids = stakeholder_ids(seed)
    for holding in seed["holdings"]:
        assert holding["stakeholder_id"] in ids, holding["_id"]


def test_every_user_maps_to_stakeholder(seed: dict[str, Any]) -> None:
    ids = stakeholder_ids(seed)
    for user in seed["users"]:
        assert user["stakeholder_id"] in ids, user["_id"]


def test_everything_belongs_to_the_company(seed: dict[str, Any]) -> None:
    company_id = seed["company"]["_id"]
    for key in ("users", "stakeholders", "holdings", "grants"):
        for doc in seed[key]:
            assert doc["company_id"] == company_id, doc["_id"]


def test_ids_unique_per_collection(seed: dict[str, Any]) -> None:
    for key in ("users", "stakeholders", "holdings", "grants"):
        ids = [doc["_id"] for doc in seed[key]]
        assert len(ids) == len(set(ids)), key


def test_grant_status_matches_termination_date(seed: dict[str, Any]) -> None:
    for grant in seed["grants"]:
        assert (grant["status"] == "terminated") == (grant["termination_date"] is not None), grant["_id"]
