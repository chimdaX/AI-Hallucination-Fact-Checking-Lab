"""Pydantic data models for claims, sources, evidence, verifications and sessions.

Design principles encoded as validation rules (not just documentation):

* A verdict of SUPPORTED or CONTRADICTED must cite evidence. The system never
  asserts truth on its own authority.
* Human judgments require a reviewer ID; automated outputs must not carry one,
  so the two can never be confused.
* Extraction confidence may only be stored together with a description of how
  it was calibrated. Uncalibrated scores are not recorded at all.
* An AI response's ID is a hash of its exact text, so any modification of the
  stored original is detected on load.
* A session is referentially consistent: evidence and verifications only point
  at claims and evidence that exist in the same session.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime, timezone
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from hallucination_lab import __version__
from hallucination_lab.ids import ID_PATTERN, content_sha256, response_id_for
from hallucination_lab.verification.rationale import rationale_errors

# 1.1.0 (Milestone 2): optional retrieval provenance fields on Evidence and
# PassageLocation, plus VerificationSession.retrieval_runs. 1.0.x files load
# unchanged (see migrations.py).
# 1.2.0 (Milestone 3): optional human-verification fields on Verification
# (session_id, evidence_assessments, revision chain, noted_limitations) plus
# VerificationSession.verification_drafts.
SCHEMA_VERSION = "1.2.0"
SUPPORTED_SCHEMA_MAJOR = 1

Identifier = Annotated[str, StringConstraints(pattern=ID_PATTERN)]
NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ScalarValue = str | int | float | bool | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_tz(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware (use UTC)")
    return value


AwareDatetime = Annotated[datetime, AfterValidator(_require_tz)]

_URL_RE = re.compile(r"^https?://\S+$")


def _check_url(value: str | None) -> str | None:
    if value is not None and not _URL_RE.match(value):
        raise ValueError("URL must start with http:// or https:// and contain no spaces")
    return value


OptionalUrl = Annotated[str | None, AfterValidator(_check_url)]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    """Base model: unknown fields are rejected so typos never pass silently."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


# --------------------------------------------------------------------------- enums


class ResponseProvenance(str, Enum):
    """Where an AI response text came from. Fixtures must never pose as real outputs."""

    SYNTHETIC_FIXTURE = "synthetic_fixture"  # hand-written illustrative text
    USER_SUBMITTED = "user_submitted"  # pasted by a user; origin not independently known
    AUTHENTIC_MODEL_OUTPUT = "authentic_model_output"  # captured from a named model


class ExtractionMethod(str, Enum):
    RULE_BASED = "rule_based"
    MANUAL = "manual"
    LLM_ASSISTED = "llm_assisted"  # reserved for the optional future LLM module


class NonCheckableReason(str, Enum):
    """Why the rule-based extractor thinks a segment is not a checkable fact."""

    QUESTION = "question"
    INSTRUCTION = "instruction"
    OPINION = "opinion"
    HYPOTHETICAL = "hypothetical"
    NO_ASSERTION = "no_assertion"  # fragments, headings, greetings


class EditType(str, Enum):
    EDITED = "edited"
    SPLIT_FROM = "split_from"
    SPLIT_INTO = "split_into"
    MERGED_FROM = "merged_from"
    MERGED_INTO = "merged_into"
    ADDED = "added"
    REMOVED = "removed"
    RESTORED = "restored"
    VERIFICATION_FLAG_CHANGED = "verification_flag_changed"


class SourceOrigin(str, Enum):
    """Who put a reference document into a collection."""

    PACKAGED_DEMO = "packaged_demo"  # shipped with this repository
    USER_ADDED = "user_added"  # added by a user with `hallab sources add`
    SYNTHETIC_TEST_FIXTURE = "synthetic_test_fixture"  # invented text for software tests


class RetrievalOutcome(str, Enum):
    """What happened when a claim was searched. Distinguishes "searched, nothing found"
    from "never searched", which matters for INSUFFICIENT_EVIDENCE decisions later."""

    CANDIDATES_FOUND = "candidates_found"
    NO_CANDIDATES = "no_candidates"
    NO_SEARCHABLE_TERMS = "no_searchable_terms"


class RetrievalMethod(str, Enum):
    BM25 = "bm25"
    TFIDF = "tfidf"
    MANUAL = "manual"  # evidence attached by a reviewer, not by a ranking algorithm


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_CHECKABLE = "NOT_CHECKABLE"


class EvidenceLabel(str, Enum):
    """A human reviewer's judgement of what one retrieved passage shows about a claim."""

    SUPPORTING = "SUPPORTING"
    CONTRADICTING = "CONTRADICTING"
    IRRELEVANT = "IRRELEVANT"
    UNCERTAIN = "UNCERTAIN"


class VerificationMethod(str, Enum):
    HUMAN = "human"
    AUTOMATED_RULE = "automated_rule"  # deterministic baseline (e.g. exact numeric match)
    MODEL_SUGGESTION = "model_suggestion"  # never treated as a verified judgment


# --------------------------------------------------------------------- AI response


class AIResponse(StrictModel):
    """The AI-generated text under investigation, stored exactly as submitted."""

    response_id: Identifier
    text: str = Field(min_length=1)
    provenance: ResponseProvenance
    model_name: str | None = None
    prompt: str | None = None
    collected_at: AwareDatetime | None = None
    notes: str | None = None

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        # Intentionally *not* stripped: the original must be preserved byte for byte.
        if not value.strip():
            raise ValueError("response text must not be blank")
        return value

    @model_validator(mode="after")
    def _check_integrity(self) -> AIResponse:
        expected = response_id_for(self.text)
        if self.response_id != expected:
            raise ValueError(
                f"response_id {self.response_id} does not match the text hash ({expected}); "
                "the stored response text may have been modified"
            )
        if self.provenance is ResponseProvenance.AUTHENTIC_MODEL_OUTPUT and not self.model_name:
            raise ValueError("authentic model outputs must name the model that produced them")
        return self

    @classmethod
    def from_text(
        cls,
        text: str,
        provenance: ResponseProvenance = ResponseProvenance.USER_SUBMITTED,
        **kw: Any,
    ) -> AIResponse:
        return cls(response_id=response_id_for(text), text=text, provenance=provenance, **kw)


# --------------------------------------------------------------------------- claims


class TextSpan(StrictModel):
    """Character offsets into ``AIResponse.text`` (half-open: ``text[start:end]``)."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> TextSpan:
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        return self


class ClaimFeatures(StrictModel):
    """Surface features detected by transparent regex rules (see extraction.features)."""

    numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    measurements: list[str] = Field(default_factory=list)
    named_entities: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    quotations: list[str] = Field(default_factory=list)
    temporal_markers: list[str] = Field(default_factory=list)
    starts_with_pronoun: bool = False


class ReviewerEdit(StrictModel):
    """One entry in a claim's audit trail."""

    edit_type: EditType
    reviewer_id: NonBlankStr
    timestamp: AwareDatetime = Field(default_factory=utc_now)
    previous_text: str | None = None
    new_text: str | None = None
    related_claim_ids: list[Identifier] = Field(default_factory=list)
    note: str | None = None


class Claim(StrictModel):
    claim_id: Identifier
    original_text: NonBlankStr
    normalized_claim: NonBlankStr
    source_response_id: Identifier
    span: TextSpan | None = None
    extraction_method: ExtractionMethod
    extractor_version: str | None = None
    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_calibration: str | None = None
    requires_verification: bool
    non_checkable_reason: NonCheckableReason | None = None
    temporal_context: str | None = None
    features: ClaimFeatures = Field(default_factory=ClaimFeatures)
    # Which extraction rules fired and why, so the extractor's decisions are inspectable.
    extraction_notes: list[str] = Field(default_factory=list)
    reviewer_edits: list[ReviewerEdit] = Field(default_factory=list)
    is_active: bool = True

    @model_validator(mode="after")
    def _consistency(self) -> Claim:
        if self.extraction_confidence is not None and not self.confidence_calibration:
            raise ValueError(
                "extraction_confidence requires confidence_calibration describing how the "
                "value was calibrated; leave it empty when no calibrated value exists"
            )
        if self.requires_verification and self.non_checkable_reason is not None:
            raise ValueError(
                "a claim that requires verification cannot have a non-checkable reason"
            )
        return self

    @property
    def is_human_modified(self) -> bool:
        return bool(self.reviewer_edits)


# ----------------------------------------------------------------- sources/evidence


class Source(StrictModel):
    """A reference document in a collection."""

    source_id: Identifier
    title: NonBlankStr
    url: OptionalUrl = None
    document_reference: str | None = None
    publication_date: date | None = None
    document_version: str | None = None
    license: str | None = None
    content_sha256: Sha256Hex | None = None
    ingested_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _locatable(self) -> Source:
        if not self.url and not self.document_reference:
            raise ValueError("a source needs a url or a document_reference so it can be cited")
        return self


class PassageLocation(StrictModel):
    """Where a quoted passage sits inside its source document."""

    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    paragraph_index: int | None = Field(default=None, ge=0)
    # 1-based line numbers in the stored document (schema 1.1+).
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    section: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> PassageLocation:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        if self.line_start and self.line_end and self.line_end < self.line_start:
            raise ValueError("line_end must not be before line_start")
        return self


class Evidence(StrictModel):
    """A passage *retrieved* for a claim.

    Retrieval only means the passage looked relevant to a ranking algorithm.
    Evidence counts as verified only once a human verification cites it
    (see ``VerificationSession.human_cited_evidence_ids``).
    """

    evidence_id: Identifier
    claim_id: Identifier
    source_id: Identifier
    source_title: NonBlankStr
    source_url: OptionalUrl = None
    document_reference: str | None = None
    retrieved_at: AwareDatetime
    quoted_passage: NonBlankStr
    passage_location: PassageLocation | None = None
    document_version: str | None = None
    publication_date: date | None = None
    retrieval_method: RetrievalMethod
    retrieval_rank: int | None = Field(default=None, ge=1)
    # Raw ranking score (e.g. BM25). Not a probability and not a truth estimate.
    retrieval_score: float | None = None
    retrieval_parameters: dict[str, ScalarValue] = Field(default_factory=dict)
    # --- schema 1.1+: retrieval provenance (all optional for backward compatibility)
    passage_id: Identifier | None = None
    # SHA-256 of quoted_passage at retrieval time; any later edit fails validation.
    passage_sha256: Sha256Hex | None = None
    collection_name: str | None = None
    source_content_sha256: Sha256Hex | None = None
    source_origin: SourceOrigin | None = None
    retrieval_run_id: Identifier | None = None
    # Query terms that matched this passage (explains the lexical score).
    matched_terms: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _locatable(self) -> Evidence:
        if not self.source_url and not self.document_reference:
            raise ValueError("evidence needs a source_url or document_reference for attribution")
        if self.passage_sha256 and content_sha256(self.quoted_passage) != self.passage_sha256:
            raise ValueError(
                f"evidence {self.evidence_id}: quoted_passage no longer matches passage_sha256; "
                "retrieved passages must not be edited"
            )
        return self


class ClaimRetrievalResult(StrictModel):
    """Outcome of searching one claim within a retrieval run."""

    claim_id: Identifier
    query_text: NonBlankStr
    query_terms: list[str] = Field(default_factory=list)
    outcome: RetrievalOutcome
    # Ranked best-first; may reference evidence first stored by an earlier run.
    evidence_ids: list[Identifier] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> ClaimRetrievalResult:
        if len(self.scores) != len(self.evidence_ids):
            raise ValueError("scores and evidence_ids must have the same length")
        if (self.outcome is RetrievalOutcome.CANDIDATES_FOUND) != bool(self.evidence_ids):
            raise ValueError("outcome candidates_found must coincide with non-empty evidence_ids")
        return self


class RetrievalRun(StrictModel):
    """One execution of retrieval over a session's claims, recorded for reproducibility."""

    run_id: Identifier
    collection_name: NonBlankStr
    collection_fingerprint: Sha256Hex
    retrieval_method: RetrievalMethod
    parameters: dict[str, ScalarValue] = Field(default_factory=dict)
    started_at: AwareDatetime
    tool_version: str = __version__
    results: list[ClaimRetrievalResult] = Field(default_factory=list)


# --------------------------------------------------------------------- verification


class Provenance(StrictModel):
    """How a verification record was produced, for reproducibility."""

    tool: str = "hallucination_lab"
    tool_version: str = __version__
    method_detail: NonBlankStr
    model_name: str | None = None
    parameters: dict[str, ScalarValue] = Field(default_factory=dict)


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


OptionalText = Annotated[str | None, AfterValidator(_optional_text)]


class EvidenceAssessment(StrictModel):
    """A reviewer's label for one retrieved passage.

    There is deliberately no retrieval score here: lexical relevance and a
    human's evidential judgement are different things and are stored apart.
    """

    evidence_id: Identifier
    label: EvidenceLabel
    explanation: OptionalText = None
    # Hash of the passage the reviewer actually read (equals Evidence.passage_sha256).
    passage_sha256: Sha256Hex | None = None


def _check_assessments(
    assessments: list[EvidenceAssessment],
    supporting: list[str],
    contradicting: list[str],
    require: bool,
) -> None:
    by_id: dict[str, EvidenceAssessment] = {}
    for a in assessments:
        if a.evidence_id in by_id:
            raise ValueError(f"evidence {a.evidence_id} is assessed more than once")
        by_id[a.evidence_id] = a
    for ids, label in (
        (supporting, EvidenceLabel.SUPPORTING),
        (contradicting, EvidenceLabel.CONTRADICTING),
    ):
        for eid in ids:
            assessment = by_id.get(eid)
            if assessment is None:
                if require:
                    raise ValueError(
                        f"cited evidence {eid} has no evidence assessment; a retrieved passage "
                        f"is never treated as {label.value.lower()} automatically"
                    )
                continue
            if assessment.label is not label:
                raise ValueError(
                    f"evidence {eid} is cited as {label.value.lower()} but was assessed as "
                    f"{assessment.label.value}"
                )


class Verification(StrictModel):
    verification_id: Identifier
    claim_id: Identifier
    verdict: Verdict
    reasoning: NonBlankStr
    supporting_evidence_ids: list[Identifier] = Field(default_factory=list)
    contradicting_evidence_ids: list[Identifier] = Field(default_factory=list)
    verification_method: VerificationMethod
    reviewer_id: str | None = None
    verification_timestamp: AwareDatetime = Field(default_factory=utc_now)
    provenance: Provenance
    # --- schema 1.2+: human verification workflow (optional for older records)
    # Set on every record created by the 1.2 workflow; such records must assess
    # every passage they cite.
    session_id: Identifier | None = None
    evidence_assessments: list[EvidenceAssessment] = Field(default_factory=list)
    # Revision chain: a correction is a new record pointing at the one it replaces.
    revision_of: Identifier | None = None
    revision_reason: OptionalText = None
    # Required (checked at session level) when revising another reviewer's record.
    revision_authorization: OptionalText = None
    # Caveats the reviewer was shown, e.g. unconfirmed source credibility.
    noted_limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rules(self) -> Verification:
        problems = rationale_errors(self.reasoning)
        if problems:
            raise ValueError(f"reasoning rejected: {problems[0]}")
        if self.revision_of is not None:
            if self.revision_of == self.verification_id:
                raise ValueError("a verification cannot revise itself")
            reason_problems = rationale_errors(self.revision_reason)
            if reason_problems:
                raise ValueError(f"a revision needs a correction reason: {reason_problems[0]}")
        elif self.revision_reason or self.revision_authorization:
            raise ValueError("revision_reason/revision_authorization require revision_of")
        human = self.verification_method is VerificationMethod.HUMAN
        if human and not (self.reviewer_id and self.reviewer_id.strip()):
            raise ValueError("human verifications must record a reviewer_id")
        if not human and self.reviewer_id is not None:
            raise ValueError("automated or model verifications must not carry a reviewer_id")
        if self.verdict is Verdict.SUPPORTED and not self.supporting_evidence_ids:
            raise ValueError("a SUPPORTED verdict must cite at least one supporting evidence ID")
        if self.verdict is Verdict.CONTRADICTED and not self.contradicting_evidence_ids:
            raise ValueError(
                "a CONTRADICTED verdict must cite at least one contradicting evidence ID"
            )
        overlap = set(self.supporting_evidence_ids) & set(self.contradicting_evidence_ids)
        if overlap:
            raise ValueError(f"evidence cannot both support and contradict: {sorted(overlap)}")
        if self.verification_method is VerificationMethod.MODEL_SUGGESTION and not (
            self.provenance.model_name
        ):
            raise ValueError("model suggestions must record provenance.model_name")
        if self.revision_of and not human:
            raise ValueError("only human verifications can be revised in this version")
        _check_assessments(
            self.evidence_assessments,
            self.supporting_evidence_ids,
            self.contradicting_evidence_ids,
            require=self.session_id is not None,
        )
        return self

    @property
    def is_human_judgment(self) -> bool:
        return self.verification_method is VerificationMethod.HUMAN

    @property
    def cited_evidence_ids(self) -> list[str]:
        return [*self.supporting_evidence_ids, *self.contradicting_evidence_ids]


class VerificationDraft(StrictModel):
    """Unfinished work on one claim by one reviewer. Never counts as a verification."""

    claim_id: Identifier
    reviewer_id: NonBlankStr
    started_at: AwareDatetime
    updated_at: AwareDatetime
    revision_of: Identifier | None = None
    revision_reason: str | None = None  # unvalidated work-in-progress text
    evidence_assessments: list[EvidenceAssessment] = Field(default_factory=list)
    verdict: Verdict | None = None
    rationale: str | None = None  # unvalidated work-in-progress text
    supporting_evidence_ids: list[Identifier] = Field(default_factory=list)
    contradicting_evidence_ids: list[Identifier] = Field(default_factory=list)


# -------------------------------------------------------------------------- session


def _check_schema_version(value: str) -> str:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise ValueError(f"schema_version must look like MAJOR.MINOR.PATCH, got {value!r}")
    if int(match.group(1)) != SUPPORTED_SCHEMA_MAJOR:
        raise ValueError(
            f"unsupported schema major version {match.group(1)} "
            f"(this build reads {SUPPORTED_SCHEMA_MAJOR}.x)"
        )
    current_minor = int(SCHEMA_VERSION.split(".")[1])
    if int(match.group(2)) > current_minor:
        raise ValueError(
            f"schema_version {value} was written by a newer version of this tool "
            f"(this build reads up to {SCHEMA_VERSION}); please upgrade"
        )
    return value


class VerificationSession(StrictModel):
    """Everything about one investigated response: claims, evidence, verdicts."""

    schema_version: Annotated[str, AfterValidator(_check_schema_version)] = SCHEMA_VERSION
    session_id: Identifier
    title: NonBlankStr
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)
    response: AIResponse
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    verifications: list[Verification] = Field(default_factory=list)
    retrieval_runs: list[RetrievalRun] = Field(default_factory=list)
    verification_drafts: list[VerificationDraft] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _referential_integrity(self) -> VerificationSession:
        _require_unique("claim_id", [c.claim_id for c in self.claims])
        _require_unique("evidence_id", [e.evidence_id for e in self.evidence])
        _require_unique("verification_id", [v.verification_id for v in self.verifications])

        for claim in self.claims:
            if claim.source_response_id != self.response.response_id:
                raise ValueError(f"claim {claim.claim_id} belongs to a different response")
            if claim.span and claim.span.end > len(self.response.text):
                raise ValueError(f"claim {claim.claim_id} span exceeds the response length")

        claim_ids = {c.claim_id for c in self.claims}
        evidence_claim = {e.evidence_id: e.claim_id for e in self.evidence}
        for ev in self.evidence:
            if ev.claim_id not in claim_ids:
                raise ValueError(
                    f"evidence {ev.evidence_id} references unknown claim {ev.claim_id}"
                )
        for ver in self.verifications:
            if ver.claim_id not in claim_ids:
                raise ValueError(
                    f"verification {ver.verification_id} references unknown claim {ver.claim_id}"
                )
            if ver.session_id is not None and ver.session_id != self.session_id:
                raise ValueError(
                    f"verification {ver.verification_id} belongs to session {ver.session_id}"
                )
            assessed = [a.evidence_id for a in ver.evidence_assessments]
            for eid in [*ver.supporting_evidence_ids, *ver.contradicting_evidence_ids, *assessed]:
                if eid not in evidence_claim:
                    raise ValueError(
                        f"verification {ver.verification_id} cites unknown evidence {eid}"
                    )
                if evidence_claim[eid] != ver.claim_id:
                    raise ValueError(
                        f"verification {ver.verification_id} cites evidence {eid} "
                        "that was retrieved for a different claim"
                    )
        self._check_retrieval_runs(claim_ids, evidence_claim)
        self._check_revisions()
        self._check_drafts(claim_ids, evidence_claim)
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        return self

    def _check_retrieval_runs(self, claim_ids: set[str], evidence_claim: dict[str, str]) -> None:
        _require_unique("run_id", [r.run_id for r in self.retrieval_runs])
        run_ids = {r.run_id for r in self.retrieval_runs}
        for ev in self.evidence:
            if ev.retrieval_run_id and ev.retrieval_run_id not in run_ids:
                raise ValueError(
                    f"evidence {ev.evidence_id} references unknown retrieval run "
                    f"{ev.retrieval_run_id}"
                )
        for run in self.retrieval_runs:
            for result in run.results:
                if result.claim_id not in claim_ids:
                    raise ValueError(f"run {run.run_id} references unknown claim {result.claim_id}")
                for eid in result.evidence_ids:
                    if evidence_claim.get(eid) != result.claim_id:
                        raise ValueError(
                            f"run {run.run_id} lists evidence {eid} that is missing or "
                            f"belongs to a different claim than {result.claim_id}"
                        )

    def _check_revisions(self) -> None:
        by_id = {v.verification_id: v for v in self.verifications}
        revised = Counter(v.revision_of for v in self.verifications if v.revision_of)
        branched = [vid for vid, n in revised.items() if n > 1]
        if branched:
            raise ValueError(
                f"revision history branches: {sorted(branched)} revised more than once"
            )
        for v in self.verifications:
            if not v.revision_of:
                continue
            prev = by_id.get(v.revision_of)
            if prev is None:
                raise ValueError(
                    f"{v.verification_id} revises unknown verification {v.revision_of}"
                )
            if prev.claim_id != v.claim_id:
                raise ValueError(f"{v.verification_id} revises a verification of another claim")
            if not prev.is_human_judgment:
                raise ValueError(f"{v.verification_id} revises a non-human verification")
            if prev.verification_timestamp >= v.verification_timestamp:
                raise ValueError(f"{v.verification_id} must be later than the record it revises")
            if prev.reviewer_id != v.reviewer_id and not v.revision_authorization:
                raise ValueError(
                    f"{v.reviewer_id} revised {prev.reviewer_id}'s verification "
                    f"{prev.verification_id} without a recorded revision_authorization"
                )
            seen = {v.verification_id}
            cursor = prev
            while cursor is not None:
                if cursor.verification_id in seen:
                    raise ValueError(f"circular revision chain at {cursor.verification_id}")
                seen.add(cursor.verification_id)
                cursor = by_id.get(cursor.revision_of) if cursor.revision_of else None
        # At most one active human verification per claim (legacy 1.0/1.1 records
        # without session_id are tolerated so older sessions still load).
        superseded = set(revised)
        active: dict[str, list[Verification]] = {}
        for v in self.verifications:
            if v.is_human_judgment and v.verification_id not in superseded:
                active.setdefault(v.claim_id, []).append(v)
        for claim_id, records in active.items():
            if len(records) > 1 and any(r.session_id for r in records):
                raise ValueError(
                    f"claim {claim_id} has {len(records)} active human verifications; "
                    "corrections must be recorded as revisions"
                )

    def _check_drafts(self, claim_ids: set[str], evidence_claim: dict[str, str]) -> None:
        keys = [(d.claim_id, d.reviewer_id) for d in self.verification_drafts]
        if len(keys) != len(set(keys)):
            raise ValueError("more than one draft for the same claim and reviewer")
        verification_ids = {v.verification_id for v in self.verifications}
        for d in self.verification_drafts:
            if d.claim_id not in claim_ids:
                raise ValueError(f"draft references unknown claim {d.claim_id}")
            if d.revision_of and d.revision_of not in verification_ids:
                raise ValueError(f"draft revises unknown verification {d.revision_of}")
            ids = [a.evidence_id for a in d.evidence_assessments]
            for eid in [*ids, *d.supporting_evidence_ids, *d.contradicting_evidence_ids]:
                if evidence_claim.get(eid) != d.claim_id:
                    raise ValueError(f"draft for {d.claim_id} references foreign evidence {eid}")

    # --- read helpers -------------------------------------------------------------

    @property
    def active_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.is_active]

    def get_claim(self, claim_id: str) -> Claim:
        for claim in self.claims:
            if claim.claim_id == claim_id:
                return claim
        raise KeyError(claim_id)

    def evidence_for(self, claim_id: str) -> list[Evidence]:
        return [e for e in self.evidence if e.claim_id == claim_id]

    def verifications_for(self, claim_id: str) -> list[Verification]:
        return sorted(
            (v for v in self.verifications if v.claim_id == claim_id),
            key=lambda v: v.verification_timestamp,
        )

    def superseded_ids(self) -> set[str]:
        return {v.revision_of for v in self.verifications if v.revision_of}

    def human_verdict(self, claim_id: str) -> Verification | None:
        """The active human judgment for a claim: the latest record that has not been
        superseded by a revision. Automated suggestions are ignored."""
        superseded = self.superseded_ids()
        human = [
            v
            for v in self.verifications_for(claim_id)
            if v.is_human_judgment and v.verification_id not in superseded
        ]
        return human[-1] if human else None

    def get_verification(self, verification_id: str) -> Verification:
        for v in self.verifications:
            if v.verification_id == verification_id:
                return v
        raise KeyError(verification_id)

    def revision_chain(self, verification_id: str) -> list[Verification]:
        """Oldest-first chain ending at ``verification_id``."""
        chain = [self.get_verification(verification_id)]
        while chain[0].revision_of:
            chain.insert(0, self.get_verification(chain[0].revision_of))
        return chain

    def get_draft(self, claim_id: str, reviewer_id: str) -> VerificationDraft | None:
        for d in self.verification_drafts:
            if d.claim_id == claim_id and d.reviewer_id == reviewer_id:
                return d
        return None

    def pending_review(self) -> list[Claim]:
        """Active claims flagged for verification that have no human verdict yet."""
        return [
            c
            for c in self.active_claims
            if c.requires_verification and self.human_verdict(c.claim_id) is None
        ]

    def human_cited_evidence_ids(self) -> set[str]:
        """Evidence a human reviewer actually relied on (vs. merely retrieved)."""
        cited: set[str] = set()
        superseded = self.superseded_ids()
        for v in self.verifications:
            if v.is_human_judgment and v.verification_id not in superseded:
                cited.update(v.supporting_evidence_ids)
                cited.update(v.contradicting_evidence_ids)
        return cited

    def latest_retrieval(self, claim_id: str) -> ClaimRetrievalResult | None:
        """Most recent retrieval result for a claim, or None if it was never searched."""
        latest: tuple[datetime, ClaimRetrievalResult] | None = None
        for run in self.retrieval_runs:
            for result in run.results:
                if result.claim_id == claim_id and (latest is None or run.started_at >= latest[0]):
                    latest = (run.started_at, result)
        return latest[1] if latest else None

    def touch(self) -> VerificationSession:
        """Return a copy with ``updated_at`` set to now."""
        return self.model_copy(update={"updated_at": utc_now()})


def _require_unique(name: str, values: list[str]) -> None:
    dupes = [value for value, count in Counter(values).items() if count > 1]
    if dupes:
        raise ValueError(f"duplicate {name}: {sorted(dupes)}")
