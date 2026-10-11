"""Retrieval metrics and the packaged demo collection."""

import pytest

from hallucination_lab.models import SourceOrigin
from hallucination_lab.retrieval.catalog import CollectionCatalog
from hallucination_lab.retrieval.evaluation import (
    RelevanceDataset,
    evaluate,
    load_relevance_dataset,
    validate_criteria,
)
from hallucination_lab.retrieval.ingest import check_documents
from hallucination_lab.retrieval.models import MetadataStatus
from hallucination_lab.retrieval.search import RetrievalConfig


def _dataset(queries):
    return RelevanceDataset(
        dataset="t",
        collection="testrefs",
        labelling_method="hand-written for tests",
        labelled_by="test",
        review_status="test",
        queries=queries,
    )


def test_metrics_match_hand_computation(searcher):
    dataset = _dataset(
        [
            {
                "query_id": "a",
                "query": "Osterland Railway opened 1902",
                "category": "x",
                "relevant": [
                    {"local_filename": "osterland_railway.txt", "phrase": "opened in 1902"}
                ],
            },
            {
                "query_id": "b",
                "query": "Mara Quill bridge engineer",
                "category": "x",
                "relevant": [
                    {"local_filename": "lake_verenna.md", "phrase": "Mara Quill"},
                    {"local_filename": "lake_verenna.md", "phrase": "never appears"},
                ],
            },
            {"query_id": "n", "query": "volcano eruption lava", "category": "neg", "relevant": []},
        ]
    )
    report = evaluate(searcher, dataset, k_values=(1, 3), config=RetrievalConfig(top_k=3))
    a, b, n = report.outcomes
    assert a.first_relevant_rank == 1 and b.first_relevant_rank == 1
    assert b.criteria_found[3] == 0.5  # one of two criteria satisfiable
    assert report.recall_at(1) == pytest.approx((1.0 + 0.5) / 2)
    assert report.mrr() == 1.0
    assert report.precision_at(1) == 1.0
    assert n.negative and n.hit_count == 0
    assert report.summary()["negatives_with_zero_candidates"] == 1


def test_validate_criteria_flags_bad_labels(searcher):
    dataset = _dataset(
        [
            {
                "query_id": "a",
                "query": "x y z",
                "category": "x",
                "relevant": [
                    {"local_filename": "lake_verenna.md", "phrase": "not in the text"},
                    {"local_filename": "missing.txt", "phrase": "anything"},
                ],
            }
        ]
    )
    problems = validate_criteria(dataset, searcher)
    assert len(problems) == 2


def test_metrics_require_labels(searcher):
    dataset = _dataset([{"query_id": "n", "query": "volcano", "category": "neg"}])
    with pytest.raises(ValueError, match="relevance labels"):
        evaluate(searcher, dataset)


# --------------------------------------------------------------- packaged demo data


@pytest.fixture(scope="module")
def demo_searcher(tmp_path_factory):
    from hallucination_lab.config import Settings

    catalog = CollectionCatalog(Settings(data_dir=tmp_path_factory.mktemp("lab")))
    return catalog, catalog.searcher("demo")[0]


def test_demo_manifest_is_traceable_and_intact(demo_searcher):
    catalog, _ = demo_searcher
    manifest = catalog.load_manifest("demo")
    assert manifest.origin is SourceOrigin.PACKAGED_DEMO
    assert check_documents(manifest, catalog.documents_dir("demo")) == {}
    for doc in manifest.documents:
        m = doc.metadata
        assert doc.origin is SourceOrigin.PACKAGED_DEMO
        assert m.url and m.license and m.accessed_on and m.source_version
        assert m.status is MetadataStatus.CAPTURED_AT_DOWNLOAD and m.status_note


def test_demo_works_offline_and_finds_topic(demo_searcher):
    _, searcher = demo_searcher
    hit = searcher.search("The Eiffel Tower was completed in 1899.").hits[0]
    assert hit.document.entry.local_filename == "wikipedia_eiffel_tower.txt"


def test_demo_relevance_labels_match_collection(demo_searcher):
    _, searcher = demo_searcher
    dataset = load_relevance_dataset()
    assert validate_criteria(dataset, searcher) == []
    assert sum(q.is_negative for q in dataset.queries) >= 3
    report = evaluate(searcher, dataset)
    assert len(report.positives) == len([q for q in dataset.queries if not q.is_negative])
