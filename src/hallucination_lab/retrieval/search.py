"""Top-k passage search over a processed collection.

Results are **retrieval candidates**: passages that share rare words with the
query. A high BM25 score means strong word overlap. It does not mean the
passage supports the claim, and it is not a confidence value.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from hallucination_lab.errors import QueryError
from hallucination_lab.models import ScalarValue
from hallucination_lab.retrieval.bm25 import BM25Index, BM25Params
from hallucination_lab.retrieval.models import Passage, ProcessedCollection, ProcessedDocument
from hallucination_lab.retrieval.tokenizer import tokenize

SCORE_DECIMALS = 6  # rounding keeps stored scores identical across platforms


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    max_per_source: int = 3
    k1: float = 1.5
    b: float = 0.75
    min_score: float = 0.0

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if not 1 <= self.max_per_source <= 50:
            raise ValueError("max_per_source must be between 1 and 50")
        if self.min_score < 0:
            raise ValueError("min_score must be >= 0")
        BM25Params(self.k1, self.b)  # validates ranges

    @property
    def bm25(self) -> BM25Params:
        return BM25Params(self.k1, self.b)


@dataclass(frozen=True)
class SearchHit:
    rank: int
    score: float
    passage: Passage
    document: ProcessedDocument
    term_contributions: dict[str, float]

    @property
    def matched_terms(self) -> list[str]:
        return sorted(self.term_contributions, key=lambda t: (-self.term_contributions[t], t))


@dataclass(frozen=True)
class SearchResult:
    query: str
    query_terms: list[str]
    hits: list[SearchHit]
    parameters: dict[str, ScalarValue]
    suppressed: int = 0  # candidates skipped as duplicates/overlaps/per-source cap
    notes: list[str] = field(default_factory=list)


def _substantially_overlaps(a: Passage, b: Passage) -> bool:
    """True if two passages of the same source share more than half of the shorter one.

    Neighbouring passages share ``overlap_sentences`` sentences by design, so a
    small overlap does not make them duplicates.
    """
    if a.source_id != b.source_id:
        return False
    shared = min(a.char_end, b.char_end) - max(a.char_start, b.char_start)
    shorter = min(a.char_end - a.char_start, b.char_end - b.char_start)
    return shared > shorter / 2


class CollectionSearcher:
    def __init__(self, collection: ProcessedCollection) -> None:
        if not collection.passages:
            raise QueryError(f"collection {collection.collection_name!r} has no passages")
        self.collection = collection
        self._documents = {d.entry.source_id: d for d in collection.documents}
        self._index = BM25Index([tokenize(p.text) for p in collection.passages])

    def parameters(self, config: RetrievalConfig) -> dict[str, ScalarValue]:
        c = self.collection
        return {
            "retrieval_method": "bm25",
            "k1": config.k1,
            "b": config.b,
            "top_k": config.top_k,
            "max_per_source": config.max_per_source,
            "min_score": config.min_score,
            "tokenizer_version": c.tokenizer_version,
            "chunk_max_chars": c.chunking.max_chars,
            "chunk_overlap_sentences": c.chunking.overlap_sentences,
            "collection_name": c.collection_name,
            "collection_fingerprint": c.fingerprint,
        }

    def search(self, query: str, config: RetrievalConfig | None = None) -> SearchResult:
        config = config or RetrievalConfig()
        if not query or not query.strip():
            raise QueryError("query is empty")
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            raise QueryError(
                "query has no searchable terms after removing stopwords and punctuation"
            )

        passages = self.collection.passages
        selected: list[SearchHit] = []
        seen_text: set[str] = set()
        per_source: Counter[str] = Counter()
        suppressed = 0
        for scored in self._index.score(terms, config.bm25):
            if scored.score <= config.min_score:
                break
            passage = passages[scored.index]
            if (
                passage.text_sha256 in seen_text
                or per_source[passage.source_id] >= config.max_per_source
                or any(_substantially_overlaps(passage, hit.passage) for hit in selected)
            ):
                suppressed += 1
                continue
            seen_text.add(passage.text_sha256)
            per_source[passage.source_id] += 1
            selected.append(
                SearchHit(
                    rank=len(selected) + 1,
                    score=round(scored.score, SCORE_DECIMALS),
                    passage=passage,
                    document=self._documents[passage.source_id],
                    term_contributions={
                        t: round(v, SCORE_DECIMALS) for t, v in scored.term_contributions.items()
                    },
                )
            )
            if len(selected) == config.top_k:
                break

        notes = []
        unmatched = [t for t in terms if self._index.document_frequency(t) == 0]
        if unmatched:
            notes.append(f"terms not found anywhere in the collection: {', '.join(unmatched)}")
        if not selected:
            notes.append("no passage shares any searchable term with the query")
        return SearchResult(
            query=query,
            query_terms=terms,
            hits=selected,
            parameters=self.parameters(config),
            suppressed=suppressed,
            notes=notes,
        )
