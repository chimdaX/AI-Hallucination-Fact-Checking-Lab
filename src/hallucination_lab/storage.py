"""Validated JSON persistence for verification sessions.

* Every save re-validates the full session, so an inconsistent session can
  never be written to disk.
* Writes are atomic: data goes to a temporary file in the same directory and
  is then moved over the target with ``os.replace``. A crash mid-write leaves
  the previous version intact instead of a half-written file.
* Every load validates against the schema; corrupt or incompatible files raise
  :class:`SessionLoadError` with a readable explanation.
* Session IDs are checked against a strict pattern before being used as file
  names, which rules out path traversal (``../../etc``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from hallucination_lab.errors import (
    AuditTrailError,
    InvalidIdentifierError,
    SessionConflictError,
    SessionLoadError,
    SessionNotFoundError,
    StorageError,
)
from hallucination_lab.ids import is_valid_id
from hallucination_lab.migrations import migrate_session_data
from hallucination_lab.models import VerificationSession

logger = logging.getLogger(__name__)


def write_json_atomic(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.model_dump_json(indent=2)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def summarize_validation_error(exc: ValidationError, limit: int = 5) -> str:
    parts = []
    for err in exc.errors()[:limit]:
        loc = ".".join(str(p) for p in err["loc"])
        msg = err["msg"].removeprefix("Value error, ")  # Pydantic's prefix for our rules
        parts.append(f"{loc}: {msg}" if loc else msg)
    extra = exc.error_count() - limit
    if extra > 0:
        parts.append(f"... and {extra} more")
    return "; ".join(parts)


def load_session_file(path: Path) -> VerificationSession:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SessionNotFoundError(f"no session file at {path}") from exc
    except OSError as exc:
        raise SessionLoadError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SessionLoadError(f"{path.name} is not valid JSON: {exc}") from exc
    try:
        return VerificationSession.model_validate(migrate_session_data(data))
    except ValidationError as exc:
        raise SessionLoadError(
            f"{path.name} failed schema validation: {summarize_validation_error(exc)}"
        ) from exc


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    title: str
    updated_at: datetime
    claim_count: int
    pending_reviews: int


class JsonSessionStore:
    """One JSON file per session inside ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, session_id: str) -> Path:
        if not is_valid_id(session_id):
            raise InvalidIdentifierError(f"invalid session id: {session_id!r}")
        return self.root / f"{session_id}.json"

    def file_version(self, session_id: str) -> str | None:
        """SHA-256 of the stored file, or None if it does not exist."""
        path = self._path(session_id)
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    def save(self, session: VerificationSession, *, expected_version: str | None = None) -> Path:
        """Validate and atomically write a session.

        * ``expected_version`` (from :meth:`load_versioned`) makes the save fail with
          :class:`SessionConflictError` if the file changed since it was read. This
          prevents one process from silently overwriting another's work. The check
          is best-effort: there is no OS-level file lock.
        * Existing verification and evidence records are append-only: a save that
          would modify or drop one raises :class:`AuditTrailError`.
        """
        # Round-trip through validation so mutated-in-place models are re-checked.
        try:
            validated = VerificationSession.model_validate(session.model_dump())
        except ValidationError as exc:
            raise StorageError(
                f"refusing to save invalid session: {summarize_validation_error(exc)}"
            ) from exc
        path = self._path(validated.session_id)
        if expected_version is not None and self.file_version(validated.session_id) != (
            expected_version
        ):
            raise SessionConflictError(
                f"session {validated.session_id} was changed on disk after it was loaded; "
                "reload it and repeat your last step"
            )
        if path.exists():
            self._check_append_only(path, validated)
        write_json_atomic(path, validated)
        logger.info("saved session %s to %s", validated.session_id, path)
        return path

    def save_versioned(
        self, session: VerificationSession, *, expected_version: str | None = None
    ) -> str:
        """Like :meth:`save` but return the new file version for the next save."""
        self.save(session, expected_version=expected_version)
        version = self.file_version(session.session_id)
        assert version is not None
        return version

    def load_versioned(self, session_id: str) -> tuple[VerificationSession, str]:
        session = self.load(session_id)
        version = self.file_version(session_id)
        assert version is not None
        return session, version

    @staticmethod
    def _check_append_only(path: Path, new: VerificationSession) -> None:
        try:
            old = load_session_file(path)
        except StorageError as exc:
            raise AuditTrailError(
                f"cannot confirm the existing file's audit trail ({exc}); move the "
                "unreadable file aside before saving"
            ) from exc
        for label, old_records, new_records, key in (
            ("verification", old.verifications, new.verifications, "verification_id"),
            ("evidence", old.evidence, new.evidence, "evidence_id"),
        ):
            current = {getattr(r, key): r for r in new_records}
            for record in old_records:
                rid = getattr(record, key)
                if rid not in current:
                    raise AuditTrailError(f"refusing to save: {label} {rid} would be removed")
                if current[rid] != record:
                    raise AuditTrailError(f"refusing to save: {label} {rid} would be modified")

    def load(self, session_id: str) -> VerificationSession:
        path = self._path(session_id)
        if not path.exists():
            raise SessionNotFoundError(f"session {session_id} not found in {self.root}")
        session = load_session_file(path)
        if session.session_id != session_id:
            raise SessionLoadError(
                f"{path.name} contains session {session.session_id}, expected {session_id}"
            )
        return session

    def exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        if not path.exists():
            raise SessionNotFoundError(f"session {session_id} not found in {self.root}")
        path.unlink()
        logger.info("deleted session %s", session_id)

    def list_sessions(self) -> tuple[list[SessionSummary], list[str]]:
        """Return (summaries newest first, problems). Corrupt files are reported, not fatal."""
        summaries: list[SessionSummary] = []
        problems: list[str] = []
        if not self.root.exists():
            return summaries, problems
        for path in sorted(self.root.glob("*.json")):
            try:
                session = load_session_file(path)
            except StorageError as exc:
                logger.warning("skipping unreadable session file: %s", exc)
                problems.append(str(exc))
                continue
            summaries.append(
                SessionSummary(
                    session_id=session.session_id,
                    title=session.title,
                    updated_at=session.updated_at,
                    claim_count=len(session.active_claims),
                    pending_reviews=len(session.pending_review()),
                )
            )
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries, problems
