"""Retrieval quality on benchmark claims, reported separately from verdict accuracy.

For items whose evidence includes passages from a local reference collection
(for example the packaged ``demo`` collection), the claim text is used as a
query, and each stored evidence passage becomes a relevance criterion
(``file + exact passage``). Retrieval metrics come from the Milestone 2
evaluator (Recall@k, MRR).

A passage is *relevant* whether it supports or contradicts the claim.
Retrieval is judged on finding the evidence, never on whether the claim is
true. Items are reported by the evidence's proposed relation, so you can see
that contradicting evidence is retrieved as well as supporting evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from hallucination_lab.benchmark.models import BenchmarkItem, CaptureStatus, ReviewStatus
from hallucination_lab.benchmark.store import BenchmarkStore
from hallucination_lab.retrieval.evaluation import (
    EvaluationReport,
    RelevanceDataset,
    RelevanceQuery,
    evaluate,
    validate_criteria,
)
from hallucination_lab.retrieval.search import CollectionSearcher, RetrievalConfig


@dataclass
class RetrievalAnalysis:
    collection: str
    scope: str
    items_considered: int
    items_with_local_evidence: int
    evidence_coverage: float | None
    report: EvaluationReport | None
    by_relation: dict[str, dict]
    label_problems: list[str]
    notice: str = (
        "Retrieval relevance is independent of claim truth: supporting and "
        "contradicting evidence both count as relevant. These are not verification "
        "accuracy figures."
    )


def build_relevance_dataset(
    items: list[BenchmarkItem], refs: dict, collection: str, scope: str
) -> tuple[RelevanceDataset | None, dict[str, str]]:
    queries = []
    relation_of: dict[str, str] = {}
    for item in items:
        criteria = []
        relations = set()
        for e in item.evidence:
            ref = refs.get(e.reference_id)
            if (
                ref is None
                or ref.local_collection != collection
                or e.passage_text is None
                or e.capture_status is not CaptureStatus.CAPTURED
            ):
                continue
            criteria.append({"local_filename": ref.local_filename, "phrase": e.passage_text})
            relations.add(e.proposed_relation.value if e.proposed_relation else "unspecified")
        if criteria:
            queries.append(
                RelevanceQuery(
                    query_id=item.item_id,
                    query=item.claim_text,
                    category=item.domain.value,
                    relevant=criteria,
                )
            )
            relation_of[item.item_id] = "+".join(sorted(relations))
    if not queries:
        return None, relation_of
    return RelevanceDataset(
        dataset=f"benchmark-evidence-{scope}",
        collection=collection,
        labelling_method="Relevance criteria are the benchmark's captured evidence passages "
        "(exact text) for each claim.",
        labelled_by="benchmark evidence records"
        + (
            " (human-approved items)"
            if scope == "approved"
            else " (provisional: candidate evidence)"
        ),
        review_status="approved" if scope == "approved" else "provisional",
        queries=queries,
    ), relation_of


def analyse_retrieval(
    store: BenchmarkStore,
    searcher: CollectionSearcher,
    *,
    approved_only: bool = False,
    k_values=(1, 3, 5),
    config: RetrievalConfig | None = None,
) -> RetrievalAnalysis:
    items, _ = store.load_items()
    items = [i for i in items if i.status is not ReviewStatus.EXCLUDED]
    if approved_only:
        items = [i for i in items if i.status is ReviewStatus.APPROVED]
    scope = "approved" if approved_only else "candidates"
    collection = searcher.collection.collection_name
    dataset, relation_of = build_relevance_dataset(items, store.references(), collection, scope)
    with_evidence = [i for i in items if i.evidence]
    if dataset is None:
        return RetrievalAnalysis(collection, scope, len(items), 0, None, None, {}, [])
    problems = validate_criteria(dataset, searcher)
    usable = (
        RelevanceDataset.model_validate(
            {
                **dataset.model_dump(),
                "queries": [
                    q
                    for q in dataset.queries
                    if not any(p.startswith(f"{q.query_id}:") for p in problems)
                ],
            }
        )
        if problems
        else dataset
    )
    report = evaluate(searcher, usable, tuple(k_values), config) if usable.queries else None
    by_relation: dict[str, dict] = {}
    if report:
        for outcome in report.outcomes:
            group = by_relation.setdefault(
                relation_of[outcome.query_id], {"items": 0, "found_in_top_5": 0}
            )
            group["items"] += 1
            group["found_in_top_5"] += int(outcome.criteria_found.get(max(report.k_values), 0) > 0)
    return RetrievalAnalysis(
        collection=collection,
        scope=scope,
        items_considered=len(items),
        items_with_local_evidence=len(dataset.queries),
        evidence_coverage=round(len(dataset.queries) / len(with_evidence), 3)
        if with_evidence
        else None,
        report=report,
        by_relation=by_relation,
        label_problems=problems,
    )
