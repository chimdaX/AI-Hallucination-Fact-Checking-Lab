"""Attach retrieval candidates to a verification session.

Rules:

* Existing claims, evidence and verifications are never modified or removed.
* Evidence IDs are deterministic (claim + passage + passage hash). If a passage
  was already retrieved for a claim, the existing Evidence record is kept as it
  is and the new run simply references it.
* Every run is recorded, including claims that returned no candidates, so
  "searched, nothing found" can be told apart from "never searched".
* No verdicts are produced here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from hallucination_lab.errors import HallucinationLabError, QueryError
from hallucination_lab.ids import EVIDENCE, RETRIEVAL_RUN, make_id
from hallucination_lab.models import (
    SCHEMA_VERSION,
    Claim,
    ClaimRetrievalResult,
    Evidence,
    PassageLocation,
    RetrievalMethod,
    RetrievalOutcome,
    RetrievalRun,
    VerificationSession,
    utc_now,
)
from hallucination_lab.retrieval.search import CollectionSearcher, RetrievalConfig, SearchHit

logger = logging.getLogger(__name__)


def select_claims(
    session: VerificationSession,
    claim_ids: list[str] | None = None,
    include_non_checkable: bool = False,
) -> list[Claim]:
    if claim_ids:
        chosen = []
        for cid in claim_ids:
            try:
                claim = session.get_claim(cid)
            except KeyError as exc:
                raise HallucinationLabError(
                    f"claim {cid} is not in session {session.session_id}"
                ) from exc
            if not claim.is_active:
                raise HallucinationLabError(f"claim {cid} is inactive (removed, split or merged)")
            chosen.append(claim)
        return chosen
    return [c for c in session.active_claims if c.requires_verification or include_non_checkable]


def _evidence_from_hit(
    claim: Claim,
    hit: SearchHit,
    searcher: CollectionSearcher,
    run_id: str,
    parameters: dict,
    now: datetime,
) -> Evidence:
    passage, doc = hit.passage, hit.document
    meta = doc.entry.metadata
    collection = searcher.collection.collection_name
    return Evidence(
        evidence_id=make_id(EVIDENCE, claim.claim_id, passage.passage_id, passage.text_sha256),
        claim_id=claim.claim_id,
        source_id=passage.source_id,
        source_title=meta.title,
        source_url=meta.url,
        document_reference=(
            f"{collection}/{doc.entry.local_filename}#L{passage.line_start}-L{passage.line_end}"
        ),
        retrieved_at=now,
        quoted_passage=passage.text,
        passage_location=PassageLocation(
            char_start=passage.char_start,
            char_end=passage.char_end,
            line_start=passage.line_start,
            line_end=passage.line_end,
            section=passage.section,
        ),
        document_version=meta.source_version,
        publication_date=meta.publication_date,
        retrieval_method=RetrievalMethod.BM25,
        retrieval_rank=hit.rank,
        retrieval_score=hit.score,
        retrieval_parameters=parameters,
        passage_id=passage.passage_id,
        passage_sha256=passage.text_sha256,
        collection_name=collection,
        source_content_sha256=doc.entry.content_sha256,
        source_origin=doc.entry.origin,
        retrieval_run_id=run_id,
        matched_terms=hit.matched_terms,
    )


def retrieve_for_session(
    session: VerificationSession,
    searcher: CollectionSearcher,
    config: RetrievalConfig | None = None,
    *,
    claim_ids: list[str] | None = None,
    include_non_checkable: bool = False,
    now: datetime | None = None,
) -> tuple[VerificationSession, RetrievalRun]:
    """Search each selected claim and return an updated (validated) session and the run."""
    config = config or RetrievalConfig()
    now = now or utc_now()
    claims = select_claims(session, claim_ids, include_non_checkable)
    if not claims:
        raise HallucinationLabError(
            "no active claims flagged for verification; use --include-non-checkable or "
            "--claim-id to search specific claims"
        )
    parameters = searcher.parameters(config)
    run_id = make_id(
        RETRIEVAL_RUN,
        session.session_id,
        now.isoformat(),
        json.dumps(parameters, sort_keys=True),
        [c.claim_id for c in claims],
    )
    existing = {e.evidence_id for e in session.evidence}
    new_evidence: list[Evidence] = []
    results: list[ClaimRetrievalResult] = []
    for claim in claims:
        query = claim.normalized_claim
        try:
            result = searcher.search(query, config)
        except QueryError:
            results.append(
                ClaimRetrievalResult(
                    claim_id=claim.claim_id,
                    query_text=query,
                    outcome=RetrievalOutcome.NO_SEARCHABLE_TERMS,
                )
            )
            continue
        ids, scores = [], []
        for hit in result.hits:
            evidence = _evidence_from_hit(claim, hit, searcher, run_id, parameters, now)
            if evidence.evidence_id not in existing:
                new_evidence.append(evidence)
                existing.add(evidence.evidence_id)
            ids.append(evidence.evidence_id)
            scores.append(hit.score)
        results.append(
            ClaimRetrievalResult(
                claim_id=claim.claim_id,
                query_text=query,
                query_terms=result.query_terms,
                outcome=(
                    RetrievalOutcome.CANDIDATES_FOUND if ids else RetrievalOutcome.NO_CANDIDATES
                ),
                evidence_ids=ids,
                scores=scores,
            )
        )
    run = RetrievalRun(
        run_id=run_id,
        collection_name=searcher.collection.collection_name,
        collection_fingerprint=searcher.collection.fingerprint,
        retrieval_method=RetrievalMethod.BM25,
        parameters=parameters,
        started_at=now,
        results=results,
    )
    updated = session.model_copy(
        update={
            "schema_version": SCHEMA_VERSION,
            "evidence": [*session.evidence, *new_evidence],
            "retrieval_runs": [*session.retrieval_runs, run],
            "updated_at": max(now, session.created_at),
        }
    )
    logger.info(
        "retrieval run %s: %d claims, %d new evidence records",
        run_id,
        len(claims),
        len(new_evidence),
    )
    return VerificationSession.model_validate(updated.model_dump()), run
