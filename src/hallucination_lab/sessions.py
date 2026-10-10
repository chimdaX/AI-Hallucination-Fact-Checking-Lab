"""Creating verification sessions."""

from __future__ import annotations

from hallucination_lab.extraction import RuleBasedClaimExtractor
from hallucination_lab.ids import SESSION, random_id
from hallucination_lab.models import AIResponse, VerificationSession


def default_title(response: AIResponse, limit: int = 60) -> str:
    first_line = " ".join(response.text.split())
    return first_line if len(first_line) <= limit else first_line[: limit - 1].rstrip() + "…"


def new_session(
    response: AIResponse,
    title: str | None = None,
    extractor: RuleBasedClaimExtractor | None = None,
    session_id: str | None = None,
) -> VerificationSession:
    """Start a session for a response and run candidate claim extraction."""
    claims = (extractor or RuleBasedClaimExtractor()).extract(response)
    return VerificationSession(
        session_id=session_id or random_id(SESSION),
        title=title or default_title(response),
        response=response,
        claims=claims,
    )
