"""Application service used by the web interface (and usable from scripts).

Every operation works on a :class:`LoadedSession`: a validated session plus the
file version it was read from. Each save passes that version to the store, so
if the file changed in the meantime (another browser tab, the CLI) the save
fails with :class:`SessionConflictError` instead of overwriting anything.

No business rules are re-implemented here. Extraction, editing, retrieval and
verification all call the existing engines, and storage keeps its atomic
writes, schema validation and append-only audit checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from hallucination_lab.config import Settings
from hallucination_lab.errors import ClaimEditError, ExtractionError
from hallucination_lab.extraction import editing
from hallucination_lab.models import (
    AIResponse,
    Claim,
    ResponseProvenance,
    RetrievalRun,
    Verification,
    VerificationDraft,
    VerificationSession,
    utc_now,
)
from hallucination_lab.retrieval.catalog import CollectionCatalog
from hallucination_lab.retrieval.search import RetrievalConfig
from hallucination_lab.retrieval.session_retrieval import retrieve_for_session
from hallucination_lab.sessions import new_session
from hallucination_lab.storage import JsonSessionStore, SessionSummary
from hallucination_lab.verification.integrity import EvidenceCheck, check_session_evidence
from hallucination_lab.verification.workflow import (
    VerificationInput,
    build_verification,
    discard_draft,
    record_verification,
    save_draft,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedSession:
    session: VerificationSession
    version: str  # SHA-256 of the file contents this session was read from

    @property
    def session_id(self) -> str:
        return self.session.session_id


class Workspace:
    def __init__(self, settings: Settings, packaged_dir: Path | None = None) -> None:
        self.settings = settings
        self.store = JsonSessionStore(settings.sessions_dir)
        self.catalog = (
            CollectionCatalog(settings, packaged_dir)
            if packaged_dir
            else CollectionCatalog(settings)
        )

    @classmethod
    def from_env(cls) -> Workspace:
        return cls(Settings.from_env())

    # ------------------------------------------------------------------ sessions

    def list_sessions(self) -> tuple[list[SessionSummary], list[str]]:
        return self.store.list_sessions()

    def load(self, session_id: str) -> LoadedSession:
        session, version = self.store.load_versioned(session_id.strip())
        return LoadedSession(session, version)

    def _save(self, loaded: LoadedSession, session: VerificationSession) -> LoadedSession:
        version = self.store.save_versioned(session, expected_version=loaded.version)
        return LoadedSession(self.store.load(session.session_id), version)

    def create_session(
        self, text: str, title: str | None = None, synthetic: bool = False
    ) -> LoadedSession:
        """Extract claims from ``text`` (stored verbatim) and save a new session."""
        if not text or not text.strip():
            raise ExtractionError("paste a response first: the text is empty")
        provenance = (
            ResponseProvenance.SYNTHETIC_FIXTURE if synthetic else ResponseProvenance.USER_SUBMITTED
        )
        session = new_session(AIResponse.from_text(text, provenance), title=title or None)
        version = self.store.save_versioned(session)
        logger.info("created session %s with %d claims", session.session_id, len(session.claims))
        return LoadedSession(session, version)

    # -------------------------------------------------------------------- claims

    @staticmethod
    def _guard_editable(session: VerificationSession, claim_id: str) -> None:
        verdict = session.human_verdict(claim_id)
        if verdict is not None:
            raise ClaimEditError(
                f"this claim already has a human verdict ({verdict.verification_id}); its text "
                "cannot change under a recorded judgement. Correct the verdict instead."
            )
        drafts = [d.reviewer_id for d in session.verification_drafts if d.claim_id == claim_id]
        if drafts:
            raise ClaimEditError(
                f"a verification draft by {', '.join(drafts)} exists for this claim; finish or "
                "discard it before editing the claim"
            )

    def _edit(self, loaded: LoadedSession, claims: list[Claim]) -> LoadedSession:
        return self._save(loaded, loaded.session.model_copy(update={"claims": claims}).touch())

    def edit_claim(
        self,
        loaded: LoadedSession,
        claim_id: str,
        text: str,
        reviewer_id: str,
        note: str | None = None,
    ) -> LoadedSession:
        self._guard_editable(loaded.session, claim_id)
        return self._edit(
            loaded, editing.edit_claim(loaded.session.claims, claim_id, text, reviewer_id, note)
        )

    def set_flag(
        self, loaded: LoadedSession, claim_id: str, value: bool, reviewer_id: str
    ) -> LoadedSession:
        if not value:
            self._guard_editable(loaded.session, claim_id)
        return self._edit(
            loaded,
            editing.set_requires_verification(loaded.session.claims, claim_id, value, reviewer_id),
        )

    def remove_claim(
        self, loaded: LoadedSession, claim_id: str, reviewer_id: str, note: str | None = None
    ) -> LoadedSession:
        self._guard_editable(loaded.session, claim_id)
        return self._edit(
            loaded, editing.remove_claim(loaded.session.claims, claim_id, reviewer_id, note)
        )

    def restore_claim(
        self, loaded: LoadedSession, claim_id: str, reviewer_id: str
    ) -> LoadedSession:
        return self._edit(
            loaded, editing.restore_claim(loaded.session.claims, claim_id, reviewer_id)
        )

    def split_claim(
        self, loaded: LoadedSession, claim_id: str, parts: list[str], reviewer_id: str
    ) -> LoadedSession:
        self._guard_editable(loaded.session, claim_id)
        return self._edit(
            loaded, editing.split_claim(loaded.session.claims, claim_id, parts, reviewer_id)
        )

    def add_claim(self, loaded: LoadedSession, text: str, reviewer_id: str) -> LoadedSession:
        s = loaded.session
        return self._edit(loaded, editing.add_manual_claim(s.claims, s.response, text, reviewer_id))

    # ----------------------------------------------------------------- retrieval

    def retrieve(
        self,
        loaded: LoadedSession,
        collection: str,
        config: RetrievalConfig | None = None,
        claim_ids: list[str] | None = None,
        include_non_checkable: bool = False,
    ) -> tuple[LoadedSession, RetrievalRun, bool]:
        """Returns (updated session, run, whether the index had to be rebuilt)."""
        searcher, rebuilt = self.catalog.searcher(collection)
        session, run = retrieve_for_session(
            loaded.session,
            searcher,
            config,
            claim_ids=claim_ids,
            include_non_checkable=include_non_checkable,
        )
        return self._save(loaded, session), run, rebuilt

    # -------------------------------------------------------------- verification

    def evidence_checks(
        self, session: VerificationSession, claim_id: str | None = None
    ) -> dict[str, EvidenceCheck]:
        return check_session_evidence(session, self.catalog, claim_id)

    def save_draft(self, loaded: LoadedSession, draft: VerificationDraft) -> LoadedSession:
        draft = draft.model_copy(update={"updated_at": max(utc_now(), draft.started_at)})
        return self._save(loaded, save_draft(loaded.session, draft))

    def discard_draft(
        self, loaded: LoadedSession, claim_id: str, reviewer_id: str
    ) -> LoadedSession:
        return self._save(loaded, discard_draft(loaded.session, claim_id, reviewer_id))

    def prepare_verification(
        self, loaded: LoadedSession, reviewer_id: str, inp: VerificationInput
    ) -> tuple[Verification, list[str]]:
        """Validate a decision without saving it (for the confirmation step)."""
        checks = self.evidence_checks(loaded.session, inp.claim_id)
        return build_verification(loaded.session, reviewer_id, inp, checks=checks)

    def commit_verification(
        self, loaded: LoadedSession, verification: Verification
    ) -> LoadedSession:
        """Save a prepared verification. Repeating the same commit is a no-op."""
        if any(
            v.verification_id == verification.verification_id for v in loaded.session.verifications
        ):
            return loaded
        return self._save(loaded, record_verification(loaded.session, verification))
