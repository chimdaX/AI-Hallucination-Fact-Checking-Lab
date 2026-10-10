from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hallucination_lab.ids import EVIDENCE, SESSION, VERIFICATION, make_id
from hallucination_lab.models import (
    AIResponse,
    Evidence,
    PassageLocation,
    Provenance,
    ResponseProvenance,
    RetrievalMethod,
    Verdict,
    Verification,
    VerificationMethod,
    VerificationSession,
)
from hallucination_lab.sessions import new_session

FIXED_TIME = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)

SAMPLE_TEXT = (
    "The Eiffel Tower was completed in 1889 for the World's Fair in Paris. "
    "A leap year has 365 days. I think Paris is the most beautiful city."
)


@pytest.fixture
def response() -> AIResponse:
    return AIResponse.from_text(SAMPLE_TEXT, ResponseProvenance.SYNTHETIC_FIXTURE)


@pytest.fixture
def session(response: AIResponse) -> VerificationSession:
    return new_session(response, title="Sample", session_id=make_id(SESSION, "test"))


def make_evidence(claim_id: str, passage: str = "The tower was completed in 1889.") -> Evidence:
    return Evidence(
        evidence_id=make_id(EVIDENCE, claim_id, passage),
        claim_id=claim_id,
        source_id=make_id("src", "eiffel-reference"),
        source_title="Eiffel Tower reference note",
        document_reference="reference/eiffel_tower.txt",
        retrieved_at=FIXED_TIME,
        quoted_passage=passage,
        passage_location=PassageLocation(char_start=0, char_end=len(passage)),
        retrieval_method=RetrievalMethod.BM25,
        retrieval_rank=1,
        retrieval_score=4.2,
        retrieval_parameters={"k1": 1.5, "b": 0.75, "top_k": 3},
    )


def make_verification(
    claim_id: str,
    verdict: Verdict = Verdict.SUPPORTED,
    supporting: list[str] | None = None,
    contradicting: list[str] | None = None,
    method: VerificationMethod = VerificationMethod.HUMAN,
    reviewer_id: str | None = "reviewer-1",
    timestamp: datetime = FIXED_TIME,
) -> Verification:
    return Verification(
        verification_id=make_id(VERIFICATION, claim_id, method.value, timestamp.isoformat()),
        claim_id=claim_id,
        verdict=verdict,
        reasoning="The reference passage states the completion year directly.",
        supporting_evidence_ids=supporting or [],
        contradicting_evidence_ids=contradicting or [],
        verification_method=method,
        reviewer_id=reviewer_id,
        verification_timestamp=timestamp,
        provenance=Provenance(method_detail="manual review of retrieved passage"),
    )
