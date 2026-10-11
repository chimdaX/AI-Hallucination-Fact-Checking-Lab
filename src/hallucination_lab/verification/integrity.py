"""Re-check evidence against its source before a human relies on it.

For each retrieved evidence record this module confirms that:

1. the collection and source document are still registered,
2. the document file still has the hash recorded at retrieval time, and
3. the text at the recorded offsets is still exactly the quoted passage.

If the source changed or disappeared, the evidence is **blocked**. It cannot be
cited as supporting or contradicting until retrieval is re-run against the
current documents (revalidation).

The module also reports what is known about each source's credibility. The
tool cannot establish whether a source is reliable. At most, a person can
confirm its metadata. That limitation is shown to reviewers and stored with
their verdicts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

from hallucination_lab.errors import ReferenceCollectionError
from hallucination_lab.ids import content_sha256
from hallucination_lab.models import Evidence, VerificationSession
from hallucination_lab.retrieval.catalog import CollectionCatalog
from hallucination_lab.retrieval.ingest import load_document
from hallucination_lab.retrieval.models import (
    CollectionManifest,
    DocumentEntry,
    DocumentType,
    MetadataStatus,
)


class SourceStatus(str, Enum):
    VERIFIED = "verified"  # document and passage unchanged since retrieval
    MODIFIED = "modified"  # document or passage text differs: blocked
    MISSING = "missing"  # document no longer registered or file gone: blocked
    COLLECTION_UNAVAILABLE = "collection_unavailable"  # cannot re-check (not blocked)
    NOT_TRACEABLE = "not_traceable"  # legacy/manual evidence without hashes (not blocked)


BLOCKING = frozenset({SourceStatus.MODIFIED, SourceStatus.MISSING})


@dataclass(frozen=True)
class EvidenceCheck:
    evidence_id: str
    status: SourceStatus
    detail: str
    credibility_note: str
    metadata_status: str | None = None
    license: str | None = None
    last_updated: str | None = None

    @property
    def blocks_citation(self) -> bool:
        return self.status in BLOCKING


def credibility_note(entry: DocumentEntry | None) -> str:
    if entry is None:
        return "source metadata unavailable; credibility not established"
    meta = entry.metadata
    parts = []
    if meta.status is MetadataStatus.MANUALLY_CONFIRMED:
        parts.append(
            "metadata manually confirmed; reliability of the content is still a reviewer judgement"
        )
    else:
        parts.append(f"credibility not independently established (metadata {meta.status.value})")
    if meta.document_type is DocumentType.ENCYCLOPEDIA_ARTICLE:
        parts.append("tertiary source")
    elif meta.document_type is DocumentType.SYNTHETIC_FIXTURE:
        parts.append("synthetic test fixture, not a real source")
    return "; ".join(parts)


class EvidenceChecker:
    """Caches manifests and document texts while checking many evidence records."""

    def __init__(self, catalog: CollectionCatalog | None) -> None:
        self.catalog = catalog
        self._manifests: dict[str, CollectionManifest | None] = {}
        self._texts: dict[tuple[str, str], str | None] = {}

    def _manifest(self, name: str) -> CollectionManifest | None:
        if name not in self._manifests:
            try:
                self._manifests[name] = self.catalog.load_manifest(name) if self.catalog else None
            except ReferenceCollectionError:
                self._manifests[name] = None
        return self._manifests[name]

    def check(self, ev: Evidence) -> EvidenceCheck:
        if not (
            ev.collection_name
            and ev.source_content_sha256
            and ev.passage_sha256
            and ev.passage_location
        ):
            return EvidenceCheck(
                ev.evidence_id,
                SourceStatus.NOT_TRACEABLE,
                "evidence has no source hashes (manual or pre-1.1 record); it cannot be "
                "re-checked against its source",
                credibility_note(None),
            )
        manifest = self._manifest(ev.collection_name)
        if manifest is None:
            return EvidenceCheck(
                ev.evidence_id,
                SourceStatus.COLLECTION_UNAVAILABLE,
                f"collection {ev.collection_name!r} is not available here; the stored passage "
                "(hash-protected) is shown but the source cannot be re-checked",
                credibility_note(None),
            )
        entry = next((d for d in manifest.documents if d.source_id == ev.source_id), None)

        def result(status: SourceStatus, detail: str) -> EvidenceCheck:
            return EvidenceCheck(
                ev.evidence_id,
                status,
                detail,
                credibility_note(entry),
                metadata_status=entry.metadata.status.value if entry else None,
                license=entry.metadata.license if entry else None,
                last_updated=(
                    entry.metadata.last_updated.isoformat()
                    if entry and entry.metadata.last_updated
                    else None
                ),
            )

        if entry is None:
            return result(
                SourceStatus.MISSING,
                f"source {ev.source_id} is no longer registered in {ev.collection_name!r} "
                "(removed or replaced with different content)",
            )
        path = self.catalog.documents_dir(ev.collection_name) / entry.local_filename
        if not path.is_file():
            return result(SourceStatus.MISSING, f"document file {entry.local_filename} is missing")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != ev.source_content_sha256:
            return result(
                SourceStatus.MODIFIED,
                f"{entry.local_filename} changed after this passage was retrieved",
            )
        key = (ev.collection_name, entry.local_filename)
        if key not in self._texts:
            self._texts[key] = load_document(path, entry.encoding).text
        text = self._texts[key] or ""
        loc = ev.passage_location
        if content_sha256(text[loc.char_start : loc.char_end]) != ev.passage_sha256:
            return result(
                SourceStatus.MODIFIED,
                "the text at the recorded offsets no longer matches the quoted passage",
            )
        return result(
            SourceStatus.VERIFIED, "source document and passage unchanged since retrieval"
        )


def check_session_evidence(
    session: VerificationSession,
    catalog: CollectionCatalog | None,
    claim_id: str | None = None,
) -> dict[str, EvidenceCheck]:
    checker = EvidenceChecker(catalog)
    evidence = session.evidence_for(claim_id) if claim_id else session.evidence
    return {ev.evidence_id: checker.check(ev) for ev in evidence}
