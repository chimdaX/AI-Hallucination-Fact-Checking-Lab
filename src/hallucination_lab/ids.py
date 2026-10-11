"""Deterministic identifiers.

Identifiers are derived from content with SHA-256 so that the same input always
produces the same ID. That makes results reproducible across runs and machines
and lets tests assert exact IDs. Every ID has the shape ``<prefix>_<16 hex>``,
which is also safe to use as a file name.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid

ID_PATTERN = r"^[a-z]{3}_[0-9a-f]{16}$"
_ID_RE = re.compile(ID_PATTERN)

# Prefixes in use. Keeping them in one place avoids accidental collisions.
RESPONSE = "rsp"
CLAIM = "clm"
SOURCE = "src"
EVIDENCE = "evd"
VERIFICATION = "ver"
SESSION = "ses"
PASSAGE = "psg"
RETRIEVAL_RUN = "run"
COLLECTION = "col"


def content_sha256(text: str) -> str:
    """Full SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_id(prefix: str, *parts: object) -> str:
    """Build a deterministic ID from a prefix and JSON-serialisable parts.

    Parts are serialised as a JSON array, so ``("ab", "c")`` and ``("a", "bc")``
    hash differently.
    """
    if not re.fullmatch(r"[a-z]{3}", prefix):
        raise ValueError(f"ID prefix must be three lowercase letters, got {prefix!r}")
    payload = json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"), default=str)
    return f"{prefix}_{content_sha256(payload)[:16]}"


def random_id(prefix: str) -> str:
    """Non-deterministic ID for objects with no natural content key (sessions)."""
    return make_id(prefix, uuid.uuid4().hex)


def is_valid_id(value: str) -> bool:
    return bool(_ID_RE.fullmatch(value))


def response_id_for(text: str) -> str:
    """A response's ID is a pure function of its exact text."""
    return make_id(RESPONSE, text)
