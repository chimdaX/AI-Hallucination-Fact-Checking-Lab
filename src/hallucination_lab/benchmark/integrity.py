"""Detect changes to the sources behind benchmark evidence.

* **Offline** (always): stored passage text must match its hash (enforced on load).
  Evidence from a local collection must still exist unchanged: the document hash
  must equal the one recorded and the passage must still be in the document.
* **Online** (optional): web pages captured as HTML are re-fetched, and the
  passage is re-extracted with the same anchors and compared by hash. Wikipedia
  evidence points to a fixed revision (``oldid``), which cannot change, so it is
  not re-fetched.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from hallucination_lab.benchmark.capture import recheck_html_evidence
from hallucination_lab.benchmark.store import BenchmarkStore
from hallucination_lab.retrieval.catalog import CollectionCatalog
from hallucination_lab.retrieval.ingest import load_document

USER_AGENT = "ai-hallucination-verification-lab/0.1 (benchmark source re-check)"


@dataclass(frozen=True)
class SourceCheck:
    item_id: str
    evidence_id: str
    reference_title: str
    status: str  # ok | changed | missing | unavailable | anchor_missing | not_checked
    detail: str


def default_fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def check_sources(
    store: BenchmarkStore,
    catalog: CollectionCatalog | None = None,
    *,
    online: bool = False,
    fetch: Callable[[str], str] = default_fetch,
) -> list[SourceCheck]:
    refs = store.references()
    items, _ = store.load_items()
    texts: dict[str, str | None] = {}
    results: list[SourceCheck] = []
    fetched: dict[str, str] = {}

    def cached_fetch(url: str) -> str:
        if url not in fetched:
            fetched[url] = fetch(url)
        return fetched[url]

    for item in items:
        for e in item.evidence:
            ref = refs.get(e.reference_id)
            if ref is None:
                results.append(
                    SourceCheck(
                        item.item_id,
                        e.evidence_id,
                        "?",
                        "missing",
                        "reference metadata file is missing",
                    )
                )
                continue
            title = ref.title
            if ref.local_collection and catalog is not None:
                key = f"{ref.local_collection}/{ref.local_filename}"
                if key not in texts:
                    path = catalog.documents_dir(ref.local_collection) / ref.local_filename
                    if not path.is_file():
                        texts[key] = None
                    elif hashlib.sha256(path.read_bytes()).hexdigest() != (
                        ref.local_content_sha256
                    ):
                        texts[key] = ""
                    else:
                        texts[key] = load_document(path).text
                text = texts[key]
                if text is None:
                    results.append(
                        SourceCheck(
                            item.item_id,
                            e.evidence_id,
                            title,
                            "missing",
                            f"local document {key} not found",
                        )
                    )
                elif text == "":
                    results.append(
                        SourceCheck(
                            item.item_id,
                            e.evidence_id,
                            title,
                            "changed",
                            f"local document {key} changed since capture",
                        )
                    )
                elif e.passage_text and e.passage_text not in text:
                    results.append(
                        SourceCheck(
                            item.item_id,
                            e.evidence_id,
                            title,
                            "changed",
                            "passage no longer present in the document",
                        )
                    )
                else:
                    results.append(
                        SourceCheck(
                            item.item_id,
                            e.evidence_id,
                            title,
                            "ok",
                            "local document and passage unchanged",
                        )
                    )
                continue
            if "oldid=" in (ref.url or ""):
                results.append(
                    SourceCheck(
                        item.item_id,
                        e.evidence_id,
                        title,
                        "ok",
                        "pinned Wikipedia revision (immutable)",
                    )
                )
                continue
            if not online:
                results.append(
                    SourceCheck(
                        item.item_id,
                        e.evidence_id,
                        title,
                        "not_checked",
                        "web page: run with --online to re-fetch",
                    )
                )
                continue
            check = recheck_html_evidence(
                ref.url,
                e.anchor_text or "",
                e.end_anchor_text,
                e.passage_sha256 or "",
                cached_fetch,
            )
            status = "ok" if check.status == "unchanged" else check.status
            results.append(SourceCheck(item.item_id, e.evidence_id, title, status, check.detail))
    return results
