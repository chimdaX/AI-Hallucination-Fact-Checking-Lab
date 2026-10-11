"""BM25 search behaviour on the synthetic collection."""

import pytest

from hallucination_lab.errors import QueryError
from hallucination_lab.retrieval.search import RetrievalConfig
from tests.conftest import synthetic_metadata


def _files(result):
    return [h.document.entry.local_filename for h in result.hits]


def test_relevant_passage_ranked_first(searcher):
    result = searcher.search("The Osterland Railway opened in 1902.")
    top = result.hits[0]
    assert top.document.entry.local_filename == "osterland_railway.txt"
    assert "1902" in top.passage.text
    assert "1902" in top.matched_terms and "osterland" in top.matched_terms


def test_numeric_and_date_queries(searcher):
    hit = searcher.search("Lake Verenna is 312 metres deep").hits[0]
    assert "312 m" in hit.passage.text and "metre" in hit.matched_terms
    hit = searcher.search("A second bridge opened in May 1921").hits[0]
    assert "3 May 1921" in hit.passage.text


def test_wrong_number_still_retrieves_topic_passage(searcher):
    # Retrieval finds the passage needed to check the claim; it does not judge it.
    hit = searcher.search("The first bridge across Lake Verenna was completed in 1880.").hits[0]
    assert "completed in 1874" in hit.passage.text


@pytest.mark.parametrize("query", ["", "   ", "the and of it was"])
def test_empty_or_stopword_queries_rejected(searcher, query):
    with pytest.raises(QueryError):
        searcher.search(query)


def test_no_match_returns_empty_with_note(searcher):
    result = searcher.search("Quantum chromodynamics gluon")
    assert result.hits == []
    assert any("not found" in n for n in result.notes)


def test_hits_preserve_ids_and_provenance(searcher):
    hit = searcher.search("Mara Quill engineer").hits[0]
    assert hit.passage.passage_id.startswith("psg_")
    assert hit.passage.source_id == hit.document.entry.source_id
    assert hit.document.system.content_sha256 == hit.document.entry.content_sha256


def test_parameters_recorded(searcher):
    params = searcher.search("Brindle station", RetrievalConfig(top_k=2, k1=1.2)).parameters
    assert params["retrieval_method"] == "bm25" and params["k1"] == 1.2 and params["top_k"] == 2
    assert params["collection_fingerprint"] == searcher.collection.fingerprint
    assert params["tokenizer_version"]


def test_top_k_limit_and_determinism(searcher):
    a = searcher.search("Lake Verenna Brindle bridge", RetrievalConfig(top_k=1))
    assert len(a.hits) == 1
    b = searcher.search("Lake Verenna Brindle bridge", RetrievalConfig(top_k=1))
    assert [(h.passage.passage_id, h.score) for h in a.hits] == [
        (h.passage.passage_id, h.score) for h in b.hits
    ]


def test_scores_are_raw_bm25_not_probabilities(searcher):
    hits = searcher.search("Lake Verenna Osterland Brindle northern shore").hits
    assert hits and any(h.score > 1 for h in hits)  # unbounded, not a 0-1 confidence
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_identical_passages_deduplicated(catalog, synthetic_files, tmp_path):
    shared = "The Brindle lighthouse was painted red in 1955."
    for i in (1, 2):
        path = tmp_path / f"copy{i}.txt"
        # The shared paragraph sits in its own section, so both documents contain an
        # identical passage (e.g. copied boilerplate).
        path.write_text(f"# Note {i}\nSynthetic note {i} for testing.\n\n# Shared\n{shared}\n")
        catalog.add_document("dups", path, synthetic_metadata(f"Copy {i}"))
    searcher, _ = catalog.searcher("dups")
    result = searcher.search("Brindle lighthouse painted red 1955", RetrievalConfig(top_k=5))
    texts = [h.passage.text for h in result.hits]
    assert sum(shared in t for t in texts) == 1
    assert result.suppressed >= 1


def test_per_source_cap(catalog, tmp_path):
    path = tmp_path / "long.txt"
    path.write_text("\n\n".join(f"Brindle harbour fact number {i} for tests." for i in range(30)))
    catalog.add_document("cap", path, synthetic_metadata("Long"))
    searcher, _ = catalog.searcher("cap")
    result = searcher.search("Brindle harbour fact", RetrievalConfig(top_k=10, max_per_source=2))
    assert len(result.hits) == 2


def test_min_score_filters(searcher):
    assert searcher.search("Brindle", RetrievalConfig(min_score=1000)).hits == []


@pytest.mark.parametrize(
    "kwargs", [{"top_k": 0}, {"max_per_source": 0}, {"min_score": -1}, {"b": 2}]
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        RetrievalConfig(**kwargs)
