"""Stable identifiers and hashes for benchmark records."""

from __future__ import annotations

import json
import unicodedata

from hallucination_lab.ids import content_sha256, make_id

ITEM = "bmk"
REFERENCE = "ref"
ANNOTATION = "ann"
ADJUDICATION = "adj"
APPROVAL = "apr"
EVAL_RUN = "evr"
EVIDENCE = "bev"


def normalize_claim(text: str) -> str:
    """Case-, accent- and whitespace-insensitive form used for IDs and duplicate checks."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = "".join(
        ch for ch in unicodedata.normalize("NFKD", folded) if not unicodedata.combining(ch)
    )
    return " ".join(folded.replace("\u2019", "'").split()).rstrip(".!? ")


def item_id_for(claim_text: str) -> str:
    return make_id(ITEM, normalize_claim(claim_text))


def reference_id_for(url_or_path: str, version: str | None) -> str:
    return make_id(REFERENCE, url_or_path, version)


def evidence_id_for(reference_id: str, passage_sha256: str | None, locator: str) -> str:
    """Stable ID of one evidence passage: its source plus its content (or locator)."""
    return make_id(EVIDENCE, reference_id, passage_sha256 or locator)


def dataset_hash(items) -> str:
    """SHA-256 over the canonical JSON of exported items (order-independent)."""
    canonical = sorted((item.model_dump(mode="json") for item in items), key=lambda d: d["item_id"])
    return content_sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False))
