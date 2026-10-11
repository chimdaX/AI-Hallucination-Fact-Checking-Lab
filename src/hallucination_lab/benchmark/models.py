"""Benchmark data models.

Design rules, enforced by validation rather than convention:

* **Proposed labels are never ground truth.** A candidate may carry a
  ``ProposedLabel`` (written by an AI assistant, a human or a system), stored
  apart from annotations. Only human ``Annotation`` / ``Adjudication`` records
  and an explicit human ``Approval`` produce an official label.
* **Humans only.** Annotations, adjudications and approvals must come from a
  human reviewer ID that has explicitly confirmed the judgement is their own.
  Reserved AI identifiers are rejected.
* **Evidence is traceable.** Each evidence reference names a source, a locator
  and the SHA-256 of the passage as captured. The passage text is stored only
  when the source licence allows redistribution.
* **History is append-only.** Status changes, claim edits and approvals are
  events on the item. Annotations, adjudications and exports are write-once
  files (see ``store.py``).
"""

from __future__ import annotations

import json
import re
from datetime import date
from enum import Enum
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, field_validator, model_validator

from hallucination_lab.ids import content_sha256
from hallucination_lab.models import (
    AwareDatetime,
    EvidenceLabel,
    Identifier,
    NonBlankStr,
    OptionalUrl,
    Sha256Hex,
    StrictModel,
    Verdict,
)
from hallucination_lab.verification.rationale import rationale_errors

BENCHMARK_SCHEMA_VERSION = "1.0.0"
EXPORT_FORMAT = "hallab-benchmark-approved"
PROVISIONAL_FORMAT = "hallab-benchmark-provisional-candidates"

# Identifiers that denote automated authors; they may propose, never review.
RESERVED_AI_IDS = frozenset(
    {
        "claude",
        "claude-assistant",
        "assistant",
        "ai",
        "ai-assistant",
        "model",
        "llm",
        "gpt",
        "system",
        "automated",
        "bot",
    }
)
REVIEWER_RE = re.compile(r"^[\w.@-]{1,64}$")


def _human_reviewer(value: str) -> str:
    value = value.strip()
    if not REVIEWER_RE.fullmatch(value):
        raise ValueError("reviewer IDs use letters, digits, . _ - @ (max 64 characters)")
    if value.lower() in RESERVED_AI_IDS or value.lower().startswith(("claude", "ai-", "bot-")):
        raise ValueError(
            f"{value!r} is reserved for automated authors; reviews and approvals must come "
            "from a human reviewer ID"
        )
    return value


HumanReviewerId = Annotated[str, AfterValidator(_human_reviewer)]


def _rationale(value: str) -> str:
    problems = rationale_errors(value)
    if problems:
        raise ValueError(f"rationale rejected: {problems[0]}")
    return value.strip()


Rationale = Annotated[str, AfterValidator(_rationale)]


# ------------------------------------------------------------------------- enums


class ReviewStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    RESEARCH_NEEDED = "RESEARCH_NEEDED"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    REVIEWED = "REVIEWED"
    DISPUTED = "DISPUTED"
    APPROVED = "APPROVED"
    EXCLUDED = "EXCLUDED"


class Domain(str, Enum):
    HISTORY = "history"
    GEOGRAPHY = "geography"
    MATHEMATICS = "mathematics"
    PHYSICAL_SCIENCES = "physical_sciences"
    TECHNOLOGY = "technology"
    GENERAL_KNOWLEDGE = "general_knowledge"
    ENVIRONMENTAL_SCIENCE = "environmental_science"
    PUBLIC_HEALTH = "public_health"
    ECONOMICS = "economics"
    QUOTATIONS = "quotations"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ClaimType(str, Enum):
    CORRECT_ASSERTION = "correct_assertion"
    INCORRECT_NUMBER = "incorrect_number"
    INCORRECT_DATE = "incorrect_date"
    FABRICATED_REFERENCE = "fabricated_reference"
    UNSUPPORTED_STATISTIC = "unsupported_statistic"
    MISATTRIBUTED_QUOTATION = "misattributed_quotation"
    AMBIGUOUS = "ambiguous"
    TIME_SENSITIVE = "time_sensitive"
    COMPOUND = "compound"
    MISLEADING_CONTEXT = "misleading_context"
    COMMON_MYTH = "common_myth"
    OPINION = "opinion"
    PREDICTION = "prediction"
    NORMATIVE = "normative"


class Partition(str, Enum):
    DEV = "dev"  # may be used freely while developing a system
    TEST = "test"  # held out: final evaluations only, every use is logged


class AuthorType(str, Enum):
    HUMAN = "human"
    AI_ASSISTANT = "ai_assistant"
    AUTOMATED_SYSTEM = "automated_system"


class SourceType(str, Enum):
    OFFICIAL_INSTITUTION = "official_institution"
    PRIMARY_DOCUMENT = "primary_document"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    ACADEMIC = "academic"
    SECONDARY_ENCYCLOPEDIA = "secondary_encyclopedia"
    SEARCH_RECORD = "search_record"
    OTHER = "other"


class Redistribution(str, Enum):
    PUBLIC_DOMAIN_TEXT_STORED = "public_domain_text_stored"
    OPEN_LICENSE_TEXT_STORED = "open_license_text_stored"
    LOCATOR_ONLY = "locator_only"  # licence unclear or restrictive: hash + locator only


class CaptureStatus(str, Enum):
    CAPTURED = "captured"  # passage obtained from the source and hashed at access time
    UNVERIFIED_LEAD = "unverified_lead"  # a pointer someone suggested; not yet checked


class ReviewKind(str, Enum):
    FIRST_REVIEW = "first_review"
    INDEPENDENT_SECOND_REVIEW = "independent_second_review"  # different person
    REPEAT_REVIEW_SAME_REVIEWER = "repeat_review_same_reviewer"  # delayed self-review


class Reliability(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ReviewLevel(str, Enum):
    SINGLE_REVIEW = "single_review"
    REPEAT_REVIEW_SAME_REVIEWER = "repeat_review_same_reviewer"
    INDEPENDENT_DOUBLE_REVIEW = "independent_double_review"
    ADJUDICATED = "adjudicated"


# ---------------------------------------------------------------------- sources


class ReferenceSource(StrictModel):
    """A source document (web page, document, search service) evidence comes from."""

    reference_id: Identifier
    title: NonBlankStr
    author_or_organization: str | None = None
    publisher: str | None = None
    url: OptionalUrl = None
    publication_date: date | None = None
    accessed_on: date | None = None
    document_version: str | None = None
    source_type: SourceType
    license: str | None = None
    redistribution: Redistribution
    reliability_notes: str | None = None
    capture_method: NonBlankStr  # how the content was obtained (reproducibility)
    # Optional link to a document in a local reference collection (for retrieval analysis).
    local_collection: str | None = None
    local_filename: str | None = None
    local_content_sha256: Sha256Hex | None = None

    @model_validator(mode="after")
    def _locatable(self) -> ReferenceSource:
        if not self.url and not self.local_filename:
            raise ValueError("a reference needs a URL or a local collection document")
        if bool(self.local_collection) != bool(self.local_filename):
            raise ValueError("local_collection and local_filename go together")
        return self


class EvidenceReference(StrictModel):
    """A specific passage in a reference source, captured for one benchmark item."""

    evidence_id: Identifier  # see ids.evidence_id_for: source + passage hash (or locator)
    reference_id: Identifier
    capture_status: CaptureStatus
    locator: NonBlankStr  # where in the source (section, sentence anchor, table row, offsets)
    anchor_text: str | None = None  # short phrase used to find the passage again
    end_anchor_text: str | None = None  # for span captures: where the passage ends
    passage_text: str | None = None  # only when the licence allows storing it
    passage_sha256: Sha256Hex | None = None
    captured_at: AwareDatetime | None = None
    captured_by: NonBlankStr
    # The author's expectation of what the passage shows. NOT a human judgement.
    proposed_relation: EvidenceLabel | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _integrity(self) -> EvidenceReference:
        if self.capture_status is CaptureStatus.CAPTURED:
            if not self.passage_sha256 or not self.captured_at:
                raise ValueError("captured evidence needs passage_sha256 and captured_at")
            if self.passage_text is not None and content_sha256(self.passage_text) != (
                self.passage_sha256
            ):
                raise ValueError("passage_text does not match passage_sha256 (altered?)")
        return self


class SearchRecord(StrictModel):
    """A documented search, e.g. for a cited study that may not exist."""

    query: NonBlankStr
    service: NonBlankStr
    request_url: OptionalUrl = None
    performed_at: AwareDatetime
    performed_by: NonBlankStr
    result_summary: NonBlankStr


class ResearchTask(StrictModel):
    task: NonBlankStr
    reason: NonBlankStr
    created_at: AwareDatetime
    created_by: NonBlankStr
    resolved: bool = False


# ------------------------------------------------------------------------ items


class CandidateProvenance(StrictModel):
    author: NonBlankStr
    author_type: AuthorType
    created_at: AwareDatetime
    method: NonBlankStr
    notes: str | None = None


class ProposedLabel(StrictModel):
    """A suggested verdict. Kept apart from human labels; never enters the evaluation set."""

    verdict: Verdict
    proposed_by: NonBlankStr
    proposer_type: AuthorType
    rationale: NonBlankStr
    proposed_at: AwareDatetime
    notice: Literal["proposed label: not human-reviewed ground truth"] = (
        "proposed label: not human-reviewed ground truth"
    )


class SplitRecommendation(StrictModel):
    should_split: bool
    proposed_parts: list[NonBlankStr] = Field(default_factory=list)
    note: str | None = None

    @model_validator(mode="after")
    def _parts(self) -> SplitRecommendation:
        if self.should_split and len(self.proposed_parts) < 2:
            raise ValueError("a recommended split needs at least two proposed parts")
        return self


class StatusEvent(StrictModel):
    from_status: ReviewStatus | None
    to_status: ReviewStatus
    actor: NonBlankStr
    reason: NonBlankStr
    at: AwareDatetime


class ClaimEdit(StrictModel):
    previous_text: NonBlankStr
    new_text: NonBlankStr
    editor: NonBlankStr
    reason: NonBlankStr
    at: AwareDatetime


class Approval(StrictModel):
    """A human approval fixing the official label for one item content version."""

    approval_id: Identifier
    approved_by: HumanReviewerId
    approved_at: AwareDatetime
    final_verdict: Verdict
    review_level: ReviewLevel
    basis_annotation_ids: list[Identifier] = Field(min_length=1)
    adjudication_id: Identifier | None = None
    item_content_sha256: Sha256Hex
    note: str | None = None
    confirmed_own_judgement: Literal[True]


class BenchmarkItem(StrictModel):
    schema_version: Literal["1.0.0"] = BENCHMARK_SCHEMA_VERSION
    item_id: Identifier
    original_claim_text: NonBlankStr  # never changes; item_id derives from it
    claim_text: NonBlankStr  # current wording; edits recorded in claim_edits
    domain: Domain
    difficulty: Difficulty
    claim_types: list[ClaimType] = Field(min_length=1)
    temporal_scope: str | None = None
    original_context: str | None = None
    split_recommendation: SplitRecommendation | None = None
    candidate_provenance: CandidateProvenance
    evidence: list[EvidenceReference] = Field(default_factory=list)
    research_log: list[SearchRecord] = Field(default_factory=list)
    research_tasks: list[ResearchTask] = Field(default_factory=list)
    proposed_label: ProposedLabel | None = None
    partition: Partition
    status_events: list[StatusEvent] = Field(min_length=1)
    claim_edits: list[ClaimEdit] = Field(default_factory=list)
    approvals: list[Approval] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("claim_text", "original_claim_text")
    @classmethod
    def _single_line(cls, value: str) -> str:
        return " ".join(value.split())

    @model_validator(mode="after")
    def _consistency(self) -> BenchmarkItem:
        from hallucination_lab.benchmark.ids import item_id_for

        if self.item_id != item_id_for(self.original_claim_text):
            raise ValueError("item_id must be derived from original_claim_text")
        expected = self.original_claim_text
        for edit in self.claim_edits:
            if edit.previous_text != expected:
                raise ValueError("claim_edits do not form a continuous history")
            expected = edit.new_text
        if expected != self.claim_text:
            raise ValueError("claim_text must equal the last edit (or the original text)")
        for prev, nxt in zip(self.status_events, self.status_events[1:], strict=False):
            if nxt.from_status is not prev.to_status:
                raise ValueError("status_events do not form a continuous history")
        if self.status is ReviewStatus.APPROVED and not self.approvals:
            raise ValueError("an APPROVED item needs an approval record")
        ids = [e.evidence_id for e in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate evidence passage on this item")
        return self

    @property
    def status(self) -> ReviewStatus:
        return self.status_events[-1].to_status

    @property
    def content_sha256(self) -> str:
        """Fingerprint of everything a reviewer judges: claim, context, scope, evidence."""
        payload = {
            "claim_text": self.claim_text,
            "original_context": self.original_context,
            "temporal_scope": self.temporal_scope,
            "evidence": [
                [e.evidence_id, e.reference_id, e.locator, e.passage_sha256] for e in self.evidence
            ],
        }
        return content_sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    @property
    def current_approval(self) -> Approval | None:
        if self.status is not ReviewStatus.APPROVED:
            return None
        return self.approvals[-1]


# ------------------------------------------------------------------ annotations


class EvidenceJudgement(StrictModel):
    evidence_id: Identifier
    relation: EvidenceLabel  # SUPPORTING / CONTRADICTING / IRRELEVANT / UNCERTAIN
    reliability: Reliability
    note: str | None = None


class Annotation(StrictModel):
    """One human review of one item version. Write-once; corrections are new records."""

    annotation_id: Identifier
    item_id: Identifier
    item_content_sha256: Sha256Hex
    reviewer_id: HumanReviewerId
    review_kind: ReviewKind
    verdict: Verdict
    rationale: Rationale
    evidence_judgements: list[EvidenceJudgement] = Field(default_factory=list)
    cited_evidence_ids: list[Identifier] = Field(default_factory=list)
    uncertainty_note: str | None = None
    research_needed: bool = False
    created_at: AwareDatetime
    revision_of: Identifier | None = None
    revision_reason: str | None = None
    proposed_label_seen_before_review: bool = False
    confirmed_own_judgement: Literal[True]

    @model_validator(mode="after")
    def _rules(self) -> Annotation:
        judged = {j.evidence_id: j for j in self.evidence_judgements}
        if len(judged) != len(self.evidence_judgements):
            raise ValueError("each evidence passage may be judged once")
        wanted = {
            Verdict.SUPPORTED: EvidenceLabel.SUPPORTING,
            Verdict.CONTRADICTED: EvidenceLabel.CONTRADICTING,
        }.get(self.verdict)
        if wanted is not None:
            if not self.cited_evidence_ids:
                raise ValueError(f"a {self.verdict.value} review must cite at least one passage")
            for eid in self.cited_evidence_ids:
                if eid not in judged or judged[eid].relation is not wanted:
                    raise ValueError(
                        f"cited passage {eid} must be judged {wanted.value} for a "
                        f"{self.verdict.value} verdict"
                    )
        elif self.cited_evidence_ids:
            raise ValueError("citations are only used for SUPPORTED or CONTRADICTED verdicts")
        if self.revision_of and rationale_errors(self.revision_reason):
            raise ValueError("a revised annotation needs a revision_reason")
        if not self.revision_of and self.revision_reason:
            raise ValueError("revision_reason requires revision_of")
        return self


class Adjudication(StrictModel):
    """Resolution of disagreeing annotations by a human adjudicator."""

    adjudication_id: Identifier
    item_id: Identifier
    item_content_sha256: Sha256Hex
    adjudicator_id: HumanReviewerId
    considered_annotation_ids: list[Identifier] = Field(min_length=2)
    final_verdict: Verdict
    rationale: Rationale
    created_at: AwareDatetime
    confirmed_own_judgement: Literal[True]


# ------------------------------------------------------------------- manifest


class ApprovalPolicy(StrictModel):
    min_reviews: int = Field(default=1, ge=1, le=3)
    require_independent_second_review: bool = False
    min_repeat_review_delay_hours: int = Field(default=24, ge=0)


class BenchmarkManifest(StrictModel):
    schema_version: Literal["1.0.0"] = BENCHMARK_SCHEMA_VERSION
    name: NonBlankStr
    description: NonBlankStr
    status_notice: NonBlankStr
    created_at: AwareDatetime
    policy: ApprovalPolicy = Field(default_factory=ApprovalPolicy)
    partition_rule: NonBlankStr
    composition_targets: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------- exports


class ExportedEvidence(StrictModel):
    evidence_id: Identifier
    reference_id: Identifier
    title: NonBlankStr
    url: str | None = None
    locator: NonBlankStr
    passage_sha256: Sha256Hex | None = None
    passage_text: str | None = None
    redistribution: Redistribution


class ExportedItem(StrictModel):
    item_id: Identifier
    claim_text: NonBlankStr
    domain: Domain
    difficulty: Difficulty
    claim_types: list[ClaimType]
    temporal_scope: str | None = None
    partition: Partition
    label: Verdict
    review_level: ReviewLevel
    approved_by: NonBlankStr
    approved_at: AwareDatetime
    approval_id: Identifier
    item_content_sha256: Sha256Hex
    evidence: list[ExportedEvidence] = Field(default_factory=list)


class ApprovedExport(StrictModel):
    format: Literal["hallab-benchmark-approved"] = EXPORT_FORMAT
    schema_version: Literal["1.0.0"] = BENCHMARK_SCHEMA_VERSION
    benchmark_name: NonBlankStr
    version: Annotated[str, AfterValidator(lambda v: _semver(v))]
    created_at: AwareDatetime
    created_by: NonBlankStr
    policy: ApprovalPolicy
    dataset_sha256: Sha256Hex
    items: list[ExportedItem] = Field(min_length=1)
    notice: NonBlankStr

    @model_validator(mode="after")
    def _hash(self) -> ApprovedExport:
        from hallucination_lab.benchmark.ids import dataset_hash

        if dataset_hash(self.items) != self.dataset_sha256:
            raise ValueError("dataset_sha256 does not match the exported items (modified?)")
        return self


def _semver(value: str) -> str:
    if not re.fullmatch(r"\d+\.\d+\.\d+", value):
        raise ValueError("version must look like MAJOR.MINOR.PATCH")
    return value
