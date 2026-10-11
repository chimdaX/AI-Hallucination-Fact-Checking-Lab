"""Retrieval evaluation against labelled relevance criteria.

How relevance is defined: each positive query lists one or more evidence
criteria. A criterion is a (document file, exact phrase) pair. A retrieved
passage is relevant if it comes from that file and contains the phrase
(whitespace-insensitive). Labels are tied to document text, not to passage
IDs, so they stay valid when chunking settings change, and they are written
without looking at retriever output.

Metrics, computed over positive queries only:

* Recall@k: fraction of a query's criteria satisfied by at least one top-k
  passage, averaged over queries.
* Precision@k: relevant passages in the top k divided by k (missing results
  count as non-relevant), averaged over queries.
* MRR: mean over queries of 1 / rank of the first relevant passage (0 if none
  in the returned list).

Negative queries (no relevant passage exists) are reported separately. BM25
almost always returns *something*, so the useful observation is how their
scores compare with positive queries. These numbers measure **retrieval**
quality only. They are not hallucination-detection accuracy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from pydantic import Field, model_validator

from hallucination_lab.models import NonBlankStr, StrictModel
from hallucination_lab.retrieval.search import CollectionSearcher, RetrievalConfig, SearchHit

DEFAULT_EVAL_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "evaluation" / "retrieval_relevance_demo.json"
)


class RelevanceCriterion(StrictModel):
    local_filename: NonBlankStr
    phrase: NonBlankStr


class RelevanceQuery(StrictModel):
    query_id: NonBlankStr
    query: NonBlankStr
    category: NonBlankStr
    relevant: list[RelevanceCriterion] = Field(default_factory=list)  # empty = negative query
    notes: str | None = None

    @property
    def is_negative(self) -> bool:
        return not self.relevant


class RelevanceDataset(StrictModel):
    dataset: NonBlankStr
    collection: NonBlankStr
    labelling_method: NonBlankStr
    labelled_by: NonBlankStr
    review_status: NonBlankStr
    queries: list[RelevanceQuery] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> RelevanceDataset:
        ids = [q.query_id for q in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate query_id")
        return self


def load_relevance_dataset(path: Path = DEFAULT_EVAL_PATH) -> RelevanceDataset:
    return RelevanceDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _squash(text: str) -> str:
    return " ".join(text.split())


def _satisfies(hit: SearchHit, criterion: RelevanceCriterion) -> bool:
    return hit.document.entry.local_filename == criterion.local_filename and _squash(
        criterion.phrase
    ) in _squash(hit.passage.text)


def validate_criteria(dataset: RelevanceDataset, searcher: CollectionSearcher) -> list[str]:
    """Problems with labels: criteria whose phrase appears in no passage of the collection."""
    problems = []
    passages_by_file: dict[str, list[str]] = {}
    for p in searcher.collection.passages:
        name = searcher.collection.document(p.source_id).entry.local_filename
        passages_by_file.setdefault(name, []).append(_squash(p.text))
    for q in dataset.queries:
        for c in q.relevant:
            texts = passages_by_file.get(c.local_filename)
            if texts is None:
                problems.append(f"{q.query_id}: unknown file {c.local_filename}")
            elif not any(_squash(c.phrase) in t for t in texts):
                problems.append(f"{q.query_id}: phrase not found in any passage: {c.phrase!r}")
    return problems


@dataclass(frozen=True)
class QueryOutcome:
    query_id: str
    category: str
    negative: bool
    first_relevant_rank: int | None
    criteria_found: dict[int, float]  # k -> fraction of criteria satisfied in top k
    relevant_in_top: dict[int, int]  # k -> number of relevant passages in top k
    top_score: float | None
    hit_count: int


@dataclass(frozen=True)
class EvaluationReport:
    dataset: str
    collection: str
    k_values: tuple[int, ...]
    parameters: dict
    outcomes: list[QueryOutcome]

    @property
    def positives(self) -> list[QueryOutcome]:
        return [o for o in self.outcomes if not o.negative]

    @property
    def negatives(self) -> list[QueryOutcome]:
        return [o for o in self.outcomes if o.negative]

    def recall_at(self, k: int) -> float:
        return mean(o.criteria_found[k] for o in self.positives)

    def precision_at(self, k: int) -> float:
        return mean(o.relevant_in_top[k] / k for o in self.positives)

    def mrr(self) -> float:
        return mean(
            (1 / o.first_relevant_rank) if o.first_relevant_rank else 0.0 for o in self.positives
        )

    def summary(self) -> dict:
        def top_scores(group: list[QueryOutcome]) -> list[float]:
            return [o.top_score for o in group if o.top_score is not None]

        pos, neg = top_scores(self.positives), top_scores(self.negatives)
        return {
            "positive_queries": len(self.positives),
            "negative_queries": len(self.negatives),
            **{f"recall@{k}": round(self.recall_at(k), 3) for k in self.k_values},
            **{f"precision@{k}": round(self.precision_at(k), 3) for k in self.k_values},
            "mrr": round(self.mrr(), 3),
            "negatives_with_zero_candidates": sum(o.hit_count == 0 for o in self.negatives),
            "mean_top_score_positive": round(mean(pos), 3) if pos else None,
            "mean_top_score_negative": round(mean(neg), 3) if neg else None,
            "max_top_score_negative": round(max(neg), 3) if neg else None,
        }


def evaluate(
    searcher: CollectionSearcher,
    dataset: RelevanceDataset,
    k_values: tuple[int, ...] = (1, 3, 5),
    config: RetrievalConfig | None = None,
) -> EvaluationReport:
    if not dataset.queries or all(q.is_negative for q in dataset.queries):
        raise ValueError("evaluation requires at least one query with relevance labels")
    k_values = tuple(sorted(set(k_values)))
    base = config or RetrievalConfig()
    config = RetrievalConfig(
        top_k=max(max(k_values), base.top_k),
        max_per_source=base.max_per_source,
        k1=base.k1,
        b=base.b,
        min_score=base.min_score,
    )
    outcomes = []
    for q in dataset.queries:
        hits = searcher.search(q.query, config).hits
        relevant_flags = [any(_satisfies(h, c) for c in q.relevant) for h in hits]
        first = next((i + 1 for i, flag in enumerate(relevant_flags) if flag), None)
        criteria_found = {
            k: (
                sum(any(_satisfies(h, c) for h in hits[:k]) for c in q.relevant) / len(q.relevant)
                if q.relevant
                else 0.0
            )
            for k in k_values
        }
        outcomes.append(
            QueryOutcome(
                query_id=q.query_id,
                category=q.category,
                negative=q.is_negative,
                first_relevant_rank=first,
                criteria_found=criteria_found,
                relevant_in_top={k: sum(relevant_flags[:k]) for k in k_values},
                top_score=hits[0].score if hits else None,
                hit_count=len(hits),
            )
        )
    return EvaluationReport(
        dataset=dataset.dataset,
        collection=dataset.collection,
        k_values=k_values,
        parameters=searcher.parameters(config),
        outcomes=outcomes,
    )
