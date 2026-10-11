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


# ------------------------------------------------------------------ retrieval fixtures
# Invented text about a fictional place, used only to test software behaviour.

LAKE_DOC = """# Lake Verenna (synthetic test fixture)

This document is invented for software tests and describes a fictional place.

## Geography

Lake Verenna lies in the fictional province of Osterland. The lake has a surface area of
1,250 square kilometres and a maximum depth of 312 m.

The town of Brindle sits on its northern shore.

## History

The first bridge across the lake was completed in 1874. It was designed by the engineer
Mara Quill. A second bridge opened on 3 May 1921.
"""

RAILWAY_DOC = """== Osterland Railway (synthetic test fixture) ==
This document is invented for software tests.
The Osterland Railway opened in 1902 and is 48 km long.
Its main station is in Brindle, near the northern shore of Lake Verenna.
"""


@pytest.fixture
def synthetic_files(tmp_path):
    folder = tmp_path / "incoming"
    folder.mkdir()
    lake = folder / "lake_verenna.md"
    lake.write_text(LAKE_DOC, encoding="utf-8")
    railway = folder / "osterland_railway.txt"
    railway.write_text(RAILWAY_DOC, encoding="utf-8")
    return {"lake": lake, "railway": railway, "dir": folder}


@pytest.fixture
def settings(tmp_path):
    from hallucination_lab.config import Settings

    return Settings(data_dir=tmp_path / "lab")


@pytest.fixture
def catalog(settings):
    from hallucination_lab.retrieval.catalog import CollectionCatalog

    return CollectionCatalog(settings)


def synthetic_metadata(title: str):
    from hallucination_lab.retrieval.models import DocumentType, SourceMetadata

    return SourceMetadata(
        title=title, document_type=DocumentType.SYNTHETIC_FIXTURE, notes="synthetic test fixture"
    )


@pytest.fixture
def test_collection(catalog, synthetic_files):
    catalog.add_document("testrefs", synthetic_files["lake"], synthetic_metadata("Lake Verenna"))
    catalog.add_document(
        "testrefs", synthetic_files["railway"], synthetic_metadata("Osterland Railway")
    )
    return "testrefs"


@pytest.fixture
def searcher(catalog, test_collection):
    return catalog.searcher(test_collection)[0]


# --------------------------------------------------------------- verification fixtures

LAKE_RESPONSE = (
    "The first bridge across Lake Verenna was completed in 1880. "
    "The Osterland Railway is 48 km long. "
    "Brindle has a population of 90,000 people."
)


class ScriptedPrompter:
    """Feeds scripted answers to the interactive workflow; EOF when exhausted."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.output: list[str] = []
        self.prompts: list[str] = []

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    def say(self, text: str = "") -> None:
        self.output.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.output + self.prompts)


@pytest.fixture
def verification_env(settings, catalog, searcher):
    """A saved session (bridge / railway / population claims) with retrieval results."""
    from hallucination_lab.retrieval.session_retrieval import retrieve_for_session
    from hallucination_lab.storage import JsonSessionStore

    response = AIResponse.from_text(LAKE_RESPONSE, ResponseProvenance.SYNTHETIC_FIXTURE)
    session = new_session(response, title="Lake verification", session_id=make_id(SESSION, "v"))
    session, _ = retrieve_for_session(session, searcher)
    store = JsonSessionStore(settings.sessions_dir)
    store.save(session)
    bridge, railway, population = (c.claim_id for c in session.claims)
    return {
        "store": store,
        "catalog": catalog,
        "session_id": session.session_id,
        "bridge": bridge,
        "railway": railway,
        "population": population,
    }


def evidence_containing(session, claim_id: str, text: str) -> str:
    for ev in session.evidence_for(claim_id):
        if text in ev.quoted_passage:
            return ev.evidence_id
    raise AssertionError(f"no evidence for {claim_id} containing {text!r}")


# ---------------------------------------------------------------- benchmark fixtures


class Bench:
    """Helpers around a temporary benchmark store built from the synthetic collection."""

    def __init__(self, store, catalog):
        self.store, self.catalog = store, catalog

    def item(
        self,
        text,
        *,
        passages=(("completed in 1874", "CONTRADICTING"),),
        ready=True,
        domain="history",
        proposed=None,
        context=None,
    ):
        from hallucination_lab.benchmark.capture import extract_sentence
        from hallucination_lab.benchmark.ids import evidence_id_for
        from hallucination_lab.benchmark.models import (
            AuthorType,
            CandidateProvenance,
            CaptureStatus,
            ClaimType,
            Difficulty,
            Domain,
            EvidenceReference,
            ProposedLabel,
            ReviewStatus,
        )
        from hallucination_lab.benchmark.workflow import add_candidate, set_status
        from hallucination_lab.models import EvidenceLabel, Verdict, utc_now

        evidence = []
        for anchor, relation in passages:
            cap = extract_sentence(self.lake_text, anchor)
            locator = f"lake_verenna.md sentence containing {anchor!r}"
            evidence.append(
                EvidenceReference(
                    evidence_id=evidence_id_for(self.lake_ref.reference_id, cap.sha256, locator),
                    reference_id=self.lake_ref.reference_id,
                    capture_status=CaptureStatus.CAPTURED,
                    locator=locator,
                    anchor_text=anchor,
                    passage_text=cap.text,
                    passage_sha256=cap.sha256,
                    captured_at=utc_now(),
                    captured_by="test",
                    proposed_relation=EvidenceLabel(relation),
                )
            )
        label = None
        if proposed:
            label = ProposedLabel(
                verdict=Verdict(proposed),
                proposed_by="claude-assistant",
                proposer_type=AuthorType.AI_ASSISTANT,
                rationale="test",
                proposed_at=utc_now(),
            )
        item, _ = add_candidate(
            self.store,
            claim_text=text,
            domain=Domain(domain),
            difficulty=Difficulty.EASY,
            claim_types=[ClaimType.INCORRECT_DATE],
            original_context=context,
            provenance=CandidateProvenance(
                author="claude-assistant",
                author_type=AuthorType.AI_ASSISTANT,
                created_at=utc_now(),
                method="test fixture",
            ),
            evidence=evidence,
            proposed_label=label,
        )
        if ready:
            item = set_status(
                self.store,
                item.item_id,
                ReviewStatus.READY_FOR_REVIEW,
                "claude-assistant",
                "evidence captured",
            )
        return item

    def review(
        self,
        item,
        reviewer="alice",
        verdict="CONTRADICTED",
        relation="CONTRADICTING",
        kind="first_review",
        now=None,
        **kw,
    ):
        from hallucination_lab.benchmark.models import EvidenceJudgement, ReviewKind
        from hallucination_lab.benchmark.workflow import record_annotation
        from hallucination_lab.models import EvidenceLabel, Verdict

        item = self.store.load_item(item.item_id)
        cite = verdict in {"SUPPORTED", "CONTRADICTED"}
        return record_annotation(
            self.store,
            item.item_id,
            reviewer_id=reviewer,
            review_kind=ReviewKind(kind),
            verdict=Verdict(verdict),
            rationale=kw.pop("rationale", "The passage dates the first bridge to 1874."),
            evidence_judgements=[
                EvidenceJudgement(
                    evidence_id=e.evidence_id, relation=EvidenceLabel(relation), reliability="high"
                )
                for e in item.evidence
            ],
            cited_evidence_ids=[e.evidence_id for e in item.evidence] if cite else [],
            confirmed_own_judgement=kw.pop("confirmed", True),
            now=now,
            **kw,
        )


@pytest.fixture
def bench(tmp_path, catalog, test_collection, monkeypatch):
    from datetime import date

    from hallucination_lab.benchmark.ids import reference_id_for
    from hallucination_lab.benchmark.models import (
        BenchmarkManifest,
        Redistribution,
        ReferenceSource,
        SourceType,
    )
    from hallucination_lab.benchmark.store import BenchmarkStore
    from hallucination_lab.models import utc_now
    from hallucination_lab.retrieval.ingest import load_document

    root = tmp_path / "benchmark"
    monkeypatch.setenv("HALLAB_BENCHMARK_DIR", str(root))
    store = BenchmarkStore(root)
    store.save_manifest(
        BenchmarkManifest(
            name="test-benchmark",
            description="test",
            status_notice="test only",
            created_at=utc_now(),
            partition_rule="hash",
        )
    )
    manifest = catalog.load_manifest(test_collection)
    entry = next(d for d in manifest.documents if d.local_filename == "lake_verenna.md")
    ref = ReferenceSource(
        reference_id=reference_id_for("testrefs/lake_verenna.md", entry.content_sha256),
        title="Lake Verenna (synthetic test fixture)",
        source_type=SourceType.OTHER,
        redistribution=Redistribution.OPEN_LICENSE_TEXT_STORED,
        capture_method="test",
        accessed_on=date(2026, 1, 1),
        local_collection=test_collection,
        local_filename="lake_verenna.md",
        local_content_sha256=entry.content_sha256,
    )
    store.save_reference(ref)
    helper = Bench(store, catalog)
    helper.lake_ref = ref
    helper.lake_text = load_document(
        catalog.documents_dir(test_collection) / "lake_verenna.md"
    ).text
    return helper
