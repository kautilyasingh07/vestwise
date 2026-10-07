"""Keyword (BM25) ranking and reciprocal rank fusion for hybrid retrieval (spec §8.2, "Should").

Pure functions only: no Mongo, no model. `retriever.retrieve` builds the BM25
corpus per request from the chunks the user may see and passes it in here, so
these functions never see a forbidden chunk.
"""

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

RRF_K = 60  # standard RRF constant (Cormack et al., 2009)

# Words that carry no meaning for ESOP retrieval; BM25's IDF already discounts
# common words, but dropping these stops a question's phrasing from dominating.
STOPWORDS = frozenset("""
a an and are as at be been but by can could do does for from had has have how i if in into is it its
me my of on or our so than that the their them then there these they this to was we were what when
where which who whom why will with would you your about after before any all also am
""".split())

# Clause numbers ("8.2", "17(2)(vi)" -> "17", "2", "vi") stay whole: a token is
# letters/digits, optionally followed by ".digits" groups.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")
SUFFIXES = ("ing", "ed", "s")


def stem(token: str) -> str:
    """Light suffix stripping so "acquired", "acquires" and "acquiring" share a stem.

    Strips one of -ing/-ed/-s (keeping a stem of at least 3 letters and leaving
    "-ss" words alone), then a final "e". Numbers and clause numbers are unchanged.
    Deliberately tiny: no new dependency, and predictable enough to test.
    """
    if not token.isalpha():
        return token
    for suffix in SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3 and not token.endswith("ss"):
            token = token[: -len(suffix)]
            break
    if token.endswith("e") and len(token) > 3:
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, split into word/clause-number tokens, drop stopwords, stem."""
    return [stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def bm25_ranking(query: str, docs: Sequence[tuple[str, str]], n: int) -> list[str]:
    """Return the ids of the top n docs by BM25 score for `query`, best first.

    `docs` is (id, text) pairs: the caller's already access-filtered corpus.
    Docs with score <= 0 (no useful term overlap) are left out, so a query with
    no matching words returns []. Ties keep corpus order (deterministic).
    """
    query_tokens = tokenize(query)
    if not docs or not query_tokens:
        return []
    tokenized = [tokenize(text) for _, text in docs]
    if not any(tokenized):
        return []
    scores = BM25Okapi(tokenized).get_scores(query_tokens)
    order = sorted(range(len(docs)), key=lambda i: -scores[i])  # sorted() is stable
    return [docs[i][0] for i in order[:n] if scores[i] > 0]


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank of d), rank 1-based.

    Returns (id, fused score) best first. Ties break by first appearance across
    the rankings in order, so the first ranking (vector) wins ties.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])  # dicts keep insertion order
