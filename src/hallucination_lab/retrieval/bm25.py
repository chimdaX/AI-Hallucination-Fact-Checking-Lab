"""Okapi BM25, implemented directly for transparency.

    score(q, d) = sum over unique query terms t present in d of
                  idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

The ``1 +`` inside the logarithm (Lucene's variant) keeps IDF positive even for
terms that appear in most passages. The classic formula can go negative and
then *penalise* a passage for matching.

The score measures **lexical overlap** between the query and a passage,
weighted by how rare the shared terms are. It is not a probability, it is not
calibrated, and it says nothing about whether the claim is true. A passage that
contradicts a claim can score just as high as one that supports it.

Determinism: query terms are processed in sorted order and ties are broken by
passage position, so identical inputs always give identical rankings.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class BM25Params:
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if not 0 <= self.k1 <= 10:
            raise ValueError("k1 must be between 0 and 10")
        if not 0 <= self.b <= 1:
            raise ValueError("b must be between 0 and 1")


@dataclass(frozen=True)
class ScoredPassage:
    index: int  # position in the indexed passage list
    score: float
    term_contributions: dict[str, float]  # query term -> its share of the score


class BM25Index:
    def __init__(self, tokenized_passages: list[list[str]], params: BM25Params | None = None):
        if not tokenized_passages:
            raise ValueError("cannot build a BM25 index over zero passages")
        self.params = params or BM25Params()
        self.doc_count = len(tokenized_passages)
        self.doc_lengths = [len(tokens) for tokens in tokenized_passages]
        total = sum(self.doc_lengths)
        self.avg_length = total / self.doc_count if total else 1.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for idx, tokens in enumerate(tokenized_passages):
            for term, tf in sorted(Counter(tokens).items()):
                self.postings.setdefault(term, []).append((idx, tf))

    def document_frequency(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def idf(self, term: str) -> float:
        df = self.document_frequency(term)
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def score(
        self, query_terms: list[str], params: BM25Params | None = None
    ) -> list[ScoredPassage]:
        """Score every passage sharing at least one term with the query (score > 0)."""
        params = params or self.params
        k1, b = params.k1, params.b
        contributions: dict[int, dict[str, float]] = {}
        for term in sorted(set(query_terms)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for idx, tf in postings:
                norm = k1 * (1 - b + b * self.doc_lengths[idx] / self.avg_length)
                contributions.setdefault(idx, {})[term] = idf * tf * (k1 + 1) / (tf + norm)
        scored = [
            ScoredPassage(idx, sum(terms.values()), terms) for idx, terms in contributions.items()
        ]
        scored.sort(key=lambda s: (-s.score, s.index))
        return scored
