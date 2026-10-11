"""Integration of retrieval with sessions: provenance, no overwrites, no verdicts."""

import json
from datetime import timedelta

import pytest
from pydantic import ValidationError

from hallucination_lab.errors import HallucinationLabError, SessionLoadError
from hallucination_lab.extraction.editing import edit_claim, remove_claim
from hallucination_lab.ids import SESSION, make_id
from hallucination_lab.models import (
    SCHEMA_VERSION,
    AIResponse,
    ResponseProvenance,
    RetrievalOutcome,
    SourceOrigin,
    Verdict,
    VerificationSession,
    utc_now,
)
from hallucination_lab.retrieval.session_retrieval import retrieve_for_session
from hallucination_lab.sessions import new_session
from hallucination_lab.storage import JsonSessionStore
from tests.conftest import make_verification

TEXT = (
    "The first bridge across Lake Verenna was completed in 1880. "
    "The Osterland Railway is 48 km long. "
    "Do you like trains? "
    "Quantum gluons dominate chromodynamics."
)


@pytest.fixture
def lake_session():
    response = AIResponse.from_text(TEXT, ResponseProvenance.SYNTHETIC_FIXTURE)
    return new_session(response, title="Lake test", session_id=make_id(SESSION, "lake"))


def test_retrieval_attaches_traceable_candidates(lake_session, searcher):
    updated, run = retrieve_for_session(lake_session, searcher)
    assert updated.schema_version == SCHEMA_VERSION
    assert updated.verifications == []  # retrieval never produces verdicts
    bridge = lake_session.claims[0].claim_id
    evidence = updated.evidence_for(bridge)
    assert evidence and evidence[0].retrieval_rank == 1
    ev = evidence[0]
    assert "1874" in ev.quoted_passage
    assert ev.passage_id and ev.passage_sha256 and ev.source_content_sha256
    assert ev.source_origin is SourceOrigin.USER_ADDED
    assert ev.document_reference.startswith("testrefs/lake_verenna.md#L")
    assert ev.passage_location.line_start and ev.passage_location.section
    assert ev.retrieval_parameters["retrieval_method"] == "bm25"
    assert ev.retrieval_run_id == run.run_id
    assert updated.human_cited_evidence_ids() == set()  # retrieved != verified


def test_outcomes_distinguish_not_found_from_not_searched(lake_session, searcher):
    updated, run = retrieve_for_session(lake_session, searcher)
    by_claim = {r.claim_id: r for r in run.results}
    question, gluon = lake_session.claims[2], lake_session.claims[3]
    assert question.claim_id not in by_claim  # non-checkable claims skipped by default
    assert updated.latest_retrieval(question.claim_id) is None
    assert by_claim[gluon.claim_id].outcome is RetrievalOutcome.NO_CANDIDATES
    assert updated.evidence_for(gluon.claim_id) == []


def test_no_searchable_terms_outcome(searcher):
    response = AIResponse.from_text("It was what it was and so it is.")
    session = new_session(response, session_id=make_id(SESSION, "stop"))
    cid = session.claims[0].claim_id
    _, run = retrieve_for_session(session, searcher, claim_ids=[cid])
    assert run.results[0].outcome is RetrievalOutcome.NO_SEARCHABLE_TERMS


def test_include_non_checkable(lake_session, searcher):
    _, run = retrieve_for_session(lake_session, searcher, include_non_checkable=True)
    assert len(run.results) == 4


def test_rerun_reuses_evidence_without_overwriting(lake_session, searcher):
    first, run1 = retrieve_for_session(lake_session, searcher)
    later = utc_now() + timedelta(minutes=5)
    second, run2 = retrieve_for_session(first, searcher, now=later)
    assert len(second.evidence) == len(first.evidence)  # no duplicates
    assert second.evidence == first.evidence  # original records untouched
    assert [r.run_id for r in second.retrieval_runs] == [run1.run_id, run2.run_id]
    cid = lake_session.claims[0].claim_id
    assert second.latest_retrieval(cid) is run2.results[0] or (
        second.latest_retrieval(cid) == run2.results[0]
    )


def test_reviewer_edits_and_verdicts_preserved(lake_session, searcher):
    cid = lake_session.claims[0].claim_id
    claims = edit_claim(
        lake_session.claims, cid, "Lake Verenna's first bridge opened in 1874.", "reviewer-1"
    )
    claims = remove_claim(claims, lake_session.claims[1].claim_id, "reviewer-1")
    session = lake_session.model_copy(update={"claims": claims})
    updated, run = retrieve_for_session(session, searcher)
    assert updated.claims == session.claims
    assert run.results[0].query_text == "Lake Verenna's first bridge opened in 1874."
    assert lake_session.claims[1].claim_id not in {r.claim_id for r in run.results}

    ev = updated.evidence_for(cid)[0]
    verdict = make_verification(cid, Verdict.SUPPORTED, supporting=[ev.evidence_id])
    with_verdict = VerificationSession.model_validate(
        updated.model_copy(update={"verifications": [verdict]}).model_dump()
    )
    again, _ = retrieve_for_session(with_verdict, searcher, now=utc_now() + timedelta(minutes=1))
    assert again.verifications == [verdict]
    assert again.human_cited_evidence_ids() == {ev.evidence_id}


def test_claim_selection_errors(lake_session, searcher):
    with pytest.raises(HallucinationLabError, match="not in session"):
        retrieve_for_session(lake_session, searcher, claim_ids=["clm_ffffffffffffffff"])
    cid = lake_session.claims[0].claim_id
    removed = lake_session.model_copy(
        update={"claims": remove_claim(lake_session.claims, cid, "r")}
    )
    with pytest.raises(HallucinationLabError, match="inactive"):
        retrieve_for_session(removed, searcher, claim_ids=[cid])


def test_no_eligible_claims(searcher):
    session = new_session(
        AIResponse.from_text("Is this a question?"), session_id=make_id(SESSION, "q")
    )
    with pytest.raises(HallucinationLabError, match="no active claims"):
        retrieve_for_session(session, searcher)


def test_saved_and_reloaded(lake_session, searcher, tmp_path):
    store = JsonSessionStore(tmp_path / "sessions")
    updated, _ = retrieve_for_session(lake_session, searcher)
    store.save(updated)
    assert store.load(updated.session_id) == updated


def test_altered_passage_in_saved_session_rejected(lake_session, searcher, tmp_path):
    store = JsonSessionStore(tmp_path / "sessions")
    updated, _ = retrieve_for_session(lake_session, searcher)
    path = store.save(updated)
    data = json.loads(path.read_text())
    data["evidence"][0]["quoted_passage"] = data["evidence"][0]["quoted_passage"].replace(
        "1874", "1880"
    )
    path.write_text(json.dumps(data))
    with pytest.raises(SessionLoadError, match="must not be edited"):
        store.load(updated.session_id)


def test_integrity_rules_for_runs(lake_session, searcher):
    updated, _ = retrieve_for_session(lake_session, searcher)
    data = updated.model_dump()
    data["retrieval_runs"] = []  # evidence would point at a missing run
    with pytest.raises(ValidationError, match="unknown retrieval run"):
        VerificationSession.model_validate(data)
    data = updated.model_dump()
    data["retrieval_runs"][0]["results"][0]["evidence_ids"].append("evd_ffffffffffffffff")
    data["retrieval_runs"][0]["results"][0]["scores"].append(1.0)
    with pytest.raises(ValidationError, match="missing or belongs to a different claim"):
        VerificationSession.model_validate(data)
