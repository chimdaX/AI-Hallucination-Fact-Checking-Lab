"""Explicit, versioned migrations for stored session JSON.

Each migration transforms raw JSON data from one schema version to the next
before Pydantic validation. Migrations only *add* information or relabel the
version; they never drop or rewrite claims, evidence or verdicts.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from hallucination_lab.models import SCHEMA_VERSION

logger = logging.getLogger(__name__)

Migration = Callable[[dict[str, Any]], dict[str, Any]]


def _from_1_0(data: dict[str, Any]) -> dict[str, Any]:
    # 1.1.0 only added optional fields (Evidence retrieval provenance,
    # PassageLocation line numbers/section, VerificationSession.retrieval_runs),
    # so existing content is valid as-is.
    return {**data, "schema_version": "1.1.0"}


# Keyed by (major, minor) of the version being migrated *from*.
MIGRATIONS: dict[tuple[int, int], Migration] = {(1, 0): _from_1_0}


def _major_minor(version: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else None


def migrate_session_data(data: Any) -> Any:
    """Apply migrations until the data reaches the current schema version.

    Unknown or malformed versions are passed through unchanged so that schema
    validation can report a precise error.
    """
    if not isinstance(data, dict):
        return data
    original = data.get("schema_version", "1.0.0")
    current = _major_minor(str(original))
    target = _major_minor(SCHEMA_VERSION)
    while current is not None and current != target and current in MIGRATIONS:
        data = MIGRATIONS[current](data)
        current = _major_minor(data["schema_version"])
    if data.get("schema_version") != original:
        logger.info("migrated session data from schema %s to %s", original, data["schema_version"])
    return data
