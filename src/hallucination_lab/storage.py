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

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from hallucination_lab.errors import (
    InvalidIdentifierError,
    SessionLoadError,
    SessionNotFoundError,
    StorageError,
)
from hallucination_lab.ids import is_valid_id
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
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
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
        return VerificationSession.model_validate(data)
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

    def save(self, session: VerificationSession) -> Path:
        # Round-trip through validation so mutated-in-place models are re-checked.
        try:
            validated = VerificationSession.model_validate(session.model_dump())
        except ValidationError as exc:
            raise StorageError(
                f"refusing to save invalid session: {summarize_validation_error(exc)}"
            ) from exc
        path = self._path(validated.session_id)
        write_json_atomic(path, validated)
        logger.info("saved session %s to %s", validated.session_id, path)
        return path

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
