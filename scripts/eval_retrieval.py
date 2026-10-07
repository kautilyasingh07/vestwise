"""Retrieval eval on the golden set (spec §12): hit@5, score distribution, access check.

Run from the repo root:  python scripts/eval_retrieval.py [--mode vector|hybrid]
(default: settings.retrieval_mode). No LLM calls.

1. hit@5 for policy and mixed questions: the expected (doc, page) is among the
   top 5 chunks retrieved as the question's own user. Each miss prints its top 5.
2. Score distribution (cosine) for hits vs misses vs not-found questions vs a
   few off-topic probes, and a threshold sweep, to choose `retrieval_min_score`
   from data.
3. Access assertion: every golden question, retrieved as Priya, never returns a
   chunk owned by Rahul (at k=5 and at k=50, i.e. every chunk she can see).
   A positive control checks the admin *does* get Rahul's letter for the probe,
   so the assertion is not passing vacuously.
4. BM25 access (hybrid): Priya's BM25 corpus holds no Rahul chunk, and BM25
   queries aimed at Rahul's letter return none of his chunks; admin control.

Retrieval for (1) and (2) runs with no threshold, so the scores are raw; the
sweep then shows what each threshold would do. Exit code 1 if the access
assertion fails or hit@5 is below target.
"""

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# `python scripts/x.py` puts scripts/ (not the repo root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.config import settings  # noqa: E402
from app.rag.access import Role, build_access_filter  # noqa: E402
from app.rag.hybrid import bm25_ranking  # noqa: E402
from app.rag.retriever import FUSION_DEPTH, RetrievalMode, RetrievedChunk, bm25_text, load_corpus, retrieve  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "eval" / "golden.jsonl"
K = 5
NO_THRESHOLD = -1.0  # cosine is never below -1, so nothing is dropped
EXHAUSTIVE_K = 50  # more than any user can see (admin sees 43 chunks)

# Spec §12 table.
EXPECTED_TYPE_COUNTS = {"policy": 8, "personal": 4, "mixed": 3, "not_found": 3, "access": 2}
RETRIEVAL_TYPES = ("policy", "mixed")
REQUIRED_FIELDS = (
    "id", "type", "user_id", "question", "expected_doc",
    "expected_page", "expected_numbers", "expect_refusal",
)
# NFR "18 of 20" as a rate, applied to the questions hit@5 is measured on.
TARGET_RATE = 18 / 20

PRIYA_USER, RAHUL_STAKEHOLDER, ADMIN_USER = "u_priya", "sh_rahul", "u_arjun"
PROBE_ID = "G20"  # policy-style question aimed at Rahul's letter
# Keyword queries aimed at Rahul's letter (his name, his numbers) for the BM25 access check.
RAHUL_PROBES = ["Rahul Verma grant letter", "Rahul Verma vesting schedule cliff 15 March 2027",
                "grant letter Rahul options granted exercise price"]
MODE: RetrievalMode = settings.retrieval_mode  # set from --mode in main()
SWEEP = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

# Clearly off-topic queries (not in the golden set): what the threshold is for.
# In-domain unanswerable questions score like real hits, so the LLM must refuse those.
OFF_TOPIC = [
    "What's the weather in Bengaluru tomorrow?",
    "Write a poem about cats.",
    "Who won the 2022 football World Cup?",
    "How do I reset my laptop password?",
    "What is the capital of France?",
    "asdf qwerty",
]


@dataclass(frozen=True)
class UserContext:
    """What the API will build server-side from `users` (Phase 6)."""

    company_id: str
    role: Role
    stakeholder_id: str | None


def load_golden(path: Path) -> list[dict[str, Any]]:
    """Read golden.jsonl and check fields, unique ids and the spec §12 type counts."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            raise ValueError(f"{row.get('id', '?')}: missing fields {missing}")
        if row["type"] in RETRIEVAL_TYPES and (row["expected_doc"] is None or row["expected_page"] is None):
            raise ValueError(f"{row['id']}: {row['type']} question needs expected_doc and expected_page")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in golden set")
    counts = dict(Counter(r["type"] for r in rows))
    if counts != EXPECTED_TYPE_COUNTS:
        raise ValueError(f"type counts {counts} != spec §12 {EXPECTED_TYPE_COUNTS}")
    return rows


def load_contexts() -> dict[str, UserContext]:
    """Map user_id -> request context, read from the seeded `users` collection."""
    return {
        u["_id"]: UserContext(u["company_id"], u["role"], u.get("stakeholder_id"))
        for u in db.users().find({}, {"company_id": 1, "role": 1, "stakeholder_id": 1})
    }


def search(question: str, ctx: UserContext, k: int = K, min_score: float = NO_THRESHOLD,
           mode: RetrievalMode | None = None) -> list[RetrievedChunk]:
    """Retrieve as this user (default: no score threshold, the --mode in use)."""
    return retrieve(question, ctx.company_id, ctx.role, ctx.stakeholder_id, k=k, min_score=min_score,
                    mode=mode or MODE)


def hit_rank(chunks: list[RetrievedChunk], doc_title: str, page: int) -> int | None:
    """1-based rank of the first chunk from (doc_title, page), or None if absent."""
    for rank, c in enumerate(chunks, start=1):
        if c.doc_title == doc_title and c.page == page:
            return rank
    return None


def gate_score(question: str, ctx: UserContext) -> float:
    """The cosine the not-found gate compares with min_score: the vector top-1's.

    Taken from a separate vector-only k=1 search: in hybrid mode the vector #1
    can fall out of the fused top 5, so the fused list can't be used for this.
    """
    chunks = search(question, ctx, k=1, mode="vector")
    return chunks[0].score if chunks else NO_THRESHOLD


def fmt_ranks(c: RetrievedChunk) -> str:
    """"v3 b1" = vector rank 3, BM25 rank 1 ("-" = not in that list); empty in vector mode."""
    if c.rrf_score is None:
        return ""
    return f"v{c.vector_rank or '-'} b{c.bm25_rank or '-'}  "


def fmt_chunk(c: RetrievedChunk) -> str:
    """One line: score, ranks (hybrid), document, page, section."""
    return f"{c.score:.3f}  {fmt_ranks(c)}{c.doc_title} p.{c.page}  {c.section[:45]}"


def describe(values: list[float]) -> str:
    """min / median / max plus the sorted values."""
    if not values:
        return "(none)"
    vals = sorted(values)
    return (f"n={len(vals)}  min {vals[0]:.3f}  median {statistics.median(vals):.3f}  max {vals[-1]:.3f}"
            f"  [{', '.join(f'{v:.3f}' for v in vals)}]")


def eval_hits(golden: list[dict[str, Any]], contexts: dict[str, UserContext]
              ) -> tuple[int, int, int, list[float], list[float]]:
    """Print hit@5 per retrieval question and every miss's top 5.

    Also measures hit@5 on the production path (configured min_score applied,
    which in hybrid mode drops vector hits *before* fusion and so can reorder).
    Returns (hits, production-path hits, total, expected-chunk scores of hits, gate scores of misses).
    """
    print(f"== hit@{K} (policy + mixed, retrieved as the question's user, no threshold) ==")
    hits, prod_hits, total = 0, 0, 0
    hit_scores: list[float] = []
    miss_top1: list[float] = []
    for row in (r for r in golden if r["type"] in RETRIEVAL_TYPES):
        total += 1
        chunks = search(row["question"], contexts[row["user_id"]])
        rank = hit_rank(chunks, row["expected_doc"], row["expected_page"])
        prod = search(row["question"], contexts[row["user_id"]], min_score=settings.retrieval_min_score)
        prod_hits += hit_rank(prod, row["expected_doc"], row["expected_page"]) is not None
        target = f"{row['expected_doc']} p.{row['expected_page']}"
        if rank is not None:
            hits += 1
            hit_scores.append(chunks[rank - 1].score)
            print(f"  HIT   {row['id']} {row['type']:<6} rank {rank}  {chunks[rank - 1].score:.3f}  "
                  f"{fmt_ranks(chunks[rank - 1])}{target}")
        else:
            miss_top1.append(gate_score(row["question"], contexts[row["user_id"]]))
            print(f"  MISS  {row['id']} {row['type']:<6} expected {target}: {row['question']}")
            for c in chunks:
                print(f"          {fmt_chunk(c)}")
    print(f"\nhit@{K}: {hits}/{total}")
    return hits, prod_hits, total, hit_scores, miss_top1


def top1_scores(golden: list[dict[str, Any]], contexts: dict[str, UserContext], types: tuple[str, ...]) -> dict[str, tuple[float, str]]:
    """For questions of these types: id -> (top-1 cosine, top-1 location), printed as we go."""
    out: dict[str, tuple[float, str]] = {}
    for row in (r for r in golden if r["type"] in types):
        chunks = search(row["question"], contexts[row["user_id"]])
        top = chunks[0] if chunks else None
        out[row["id"]] = (gate_score(row["question"], contexts[row["user_id"]]), fmt_chunk(top) if top else "(nothing)")
        print(f"  {row['id']} {row['type']:<9} gate {out[row['id']][0]:.3f}  top-1 {out[row['id']][1]}  | {row['question']}")
    return out


def off_topic_top1(ctx: UserContext) -> list[float]:
    """Top-1 cosine for each OFF_TOPIC query."""
    scores = []
    for q in OFF_TOPIC:
        scores.append(gate_score(q, ctx))
    return scores


def print_sweep(hit_scores: list[float], total: int, not_found_top1: list[float], off_topic: list[float]) -> None:
    """Per threshold: hit@5 if the expected chunk must also pass it, and questions fully refused by it."""
    print("\n== threshold sweep (cosine) ==")
    print("  min_score  hit@5 kept  not_found refused  off-topic refused")
    for t in SWEEP:
        kept = sum(s >= t for s in hit_scores)
        refused = sum(s < t for s in not_found_top1)
        off = sum(s < t for s in off_topic)
        marker = "  <- configured" if math.isclose(t, settings.retrieval_min_score) else ""
        print(f"  {t:>9.2f}  {kept:>4}/{total:<5}  {refused:>8}/{len(not_found_top1):<8}  "
              f"{off:>8}/{len(off_topic)}{marker}")


def check_access(golden: list[dict[str, Any]], contexts: dict[str, UserContext]) -> bool:
    """Assert no golden question, retrieved as Priya, returns a chunk owned by Rahul; plus a positive control."""
    print("\n== access: every golden question retrieved as Priya ==")
    priya = contexts[PRIYA_USER]
    violations: list[str] = []
    for k in (K, EXHAUSTIVE_K):
        owners: Counter[str] = Counter()
        for row in golden:
            for c in search(row["question"], priya, k=k):
                owners[c.owner_stakeholder_id or "company-wide"] += 1
                if c.owner_stakeholder_id == RAHUL_STAKEHOLDER:
                    violations.append(f"{row['id']} k={k}: {c.chunk_id}")
        print(f"  k={k:<2}: {len(golden)} questions, chunk owners seen {dict(owners)}")

    probe = next(r for r in golden if r["id"] == PROBE_ID)
    admin_chunks = search(probe["question"], contexts[ADMIN_USER])
    control = any(c.owner_stakeholder_id == RAHUL_STAKEHOLDER for c in admin_chunks)
    print(f"  positive control: {PROBE_ID} as admin returns Rahul's letter in top {K}: {control}")
    for c in admin_chunks:
        print(f"          {fmt_chunk(c)}")

    if violations:
        print(f"  FAIL: {len(violations)} Rahul-owned chunks returned to Priya: {violations}")
    else:
        print("  PASS: no chunk owned by Rahul was returned to Priya")
    return not violations and control


def check_bm25_access(golden: list[dict[str, Any]], contexts: dict[str, UserContext]) -> bool:
    """BM25 on its own: Priya's corpus and rankings never contain Rahul's chunks; admin control."""
    print("\n== BM25 access: Priya's per-request corpus and keyword rankings ==")

    def corpus_for(ctx: UserContext) -> list[dict[str, Any]]:
        access = build_access_filter(ctx.company_id, ctx.role, ctx.stakeholder_id)
        return load_corpus(access, ctx.company_id, ctx.role, ctx.stakeholder_id)

    priya, admin = corpus_for(contexts[PRIYA_USER]), corpus_for(contexts[ADMIN_USER])
    owner = {d["_id"]: d.get("owner_stakeholder_id") for d in admin}
    in_corpus = [d["_id"] for d in priya if d.get("owner_stakeholder_id") == RAHUL_STAKEHOLDER]
    print(f"  corpus sizes: Priya {len(priya)}, admin {len(admin)}; Rahul chunks in Priya's corpus: {len(in_corpus)}")

    violations: list[str] = list(in_corpus)
    priya_docs = [(d["_id"], bm25_text(d)) for d in priya]
    admin_docs = [(d["_id"], bm25_text(d)) for d in admin]
    for q in RAHUL_PROBES + [r["question"] for r in golden]:
        violations += [f"{q!r}: {cid}" for cid in bm25_ranking(q, priya_docs, FUSION_DEPTH)
                       if owner[cid] == RAHUL_STAKEHOLDER]
    control = all(owner[bm25_ranking(q, admin_docs, 1)[0]] == RAHUL_STAKEHOLDER for q in RAHUL_PROBES)
    print(f"  {len(RAHUL_PROBES)} Rahul probes + {len(golden)} golden questions as Priya, top {FUSION_DEPTH} each")
    print(f"  positive control: each Rahul probe's BM25 top-1 as admin is Rahul's letter: {control}")
    if violations:
        print(f"  FAIL: {violations}")
    else:
        print("  PASS: BM25 never ranked a Rahul chunk for Priya")
    return not violations and control


def main() -> int:
    """Run all checks and print a summary."""
    global MODE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["vector", "hybrid"], default=settings.retrieval_mode)
    MODE = parser.parse_args().mode
    db.ping()
    golden = load_golden(GOLDEN_PATH)
    contexts = load_contexts()
    print(f"golden set: {len(golden)} questions {dict(Counter(r['type'] for r in golden))}; "
          f"index '{settings.vector_index_name}', configured min_score {settings.retrieval_min_score}, "
          f"mode {MODE}\n")

    hits, prod_hits, total, hit_scores, miss_top1 = eval_hits(golden, contexts)

    print("\n== top-1 for questions the documents should not answer (score = vector top-1 cosine, the gate) ==")
    others = top1_scores(golden, contexts, ("not_found", "access"))
    not_found_top1 = [others[r["id"]][0] for r in golden if r["type"] == "not_found"]

    off_topic = off_topic_top1(contexts[PRIYA_USER])

    print("\n== score distribution (cosine) ==")
    print(f"  hits, score of the expected chunk:  {describe(hit_scores)}")
    print(f"  misses, vector top-1 score:         {describe(miss_top1)}")
    print(f"  not_found, vector top-1 score:      {describe(not_found_top1)}")
    print(f"  off-topic probes, vector top-1:     {describe(off_topic)}")
    print_sweep(hit_scores, total, not_found_top1, off_topic)

    access_ok = check_access(golden, contexts)
    bm25_ok = check_bm25_access(golden, contexts)

    target = math.ceil(TARGET_RATE * total)
    print("\n== summary ==")
    print(f"  hit@{K} (no threshold):              {hits}/{total}  (target >= {target}/{total}, i.e. 18/20 rate)")
    print(f"  hit@{K} at min_score {settings.retrieval_min_score} (production): {prod_hits}/{total}")
    print(f"  access assertion:                   {'PASS' if access_ok else 'FAIL'}")
    print(f"  BM25 access assertion:              {'PASS' if bm25_ok else 'FAIL'}")
    return 0 if access_ok and bm25_ok and prod_hits >= target else 1


if __name__ == "__main__":
    sys.exit(main())
