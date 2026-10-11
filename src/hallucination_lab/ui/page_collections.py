"""Reference collections: browse sources, metadata, licences, hashes and integrity;
add documents to user collections through the existing validated ingestion."""

from __future__ import annotations

import re
import tempfile
from datetime import date
from pathlib import Path, PureWindowsPath

import pandas as pd
import streamlit as st

from hallucination_lab.errors import HallucinationLabError, IndexNotFoundError, StaleIndexError
from hallucination_lab.retrieval.ingest import SUPPORTED_EXTENSIONS
from hallucination_lab.retrieval.models import DocumentType, SourceMetadata
from hallucination_lab.ui import common


def safe_filename(name: str) -> str:
    """Reduce an uploaded file name to a plain, portable file name.

    Strips any directory part (POSIX or Windows), replaces characters outside
    letters, digits, ``.``, ``_`` and ``-``, and keeps the extension.
    """
    base = PureWindowsPath(name).name  # handles both "\\" and "/" separators
    base = Path(base).name
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.ASCII).strip("._") or "document"
    ext = re.sub(r"[^\w]+", "", ext, flags=re.ASCII).lower()
    return f"{stem}.{ext}" if ext else stem


def _parse_date(value: str, label: str) -> date | None:
    value = value.strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HallucinationLabError(f"{label} must be YYYY-MM-DD or empty") from exc


def _index_status(ws, name: str) -> str:
    try:
        ws.catalog.load_index(name)
        return "indexed and up to date"
    except IndexNotFoundError:
        return "not indexed yet (built automatically on first search)"
    except StaleIndexError:
        return "index out of date (rebuilt automatically on next search)"
    except HallucinationLabError as exc:
        return f"cannot be indexed: {exc}"


def _browse(ws) -> None:
    collections = ws.catalog.list_collections()
    if not collections:
        st.info("No collections found.")
        return
    names = [c.name for c in collections]
    name = st.selectbox("Collection", names, index=names.index("demo") if "demo" in names else 0)
    info = next(c for c in collections if c.name == name)
    manifest = ws.catalog.load_manifest(name)
    owner = (
        "Packaged with the repository (read-only)"
        if ws.catalog.is_packaged(name)
        else "User-supplied (stored in your data folder)"
    )
    st.markdown(f"**{name}**: {owner}")
    if manifest.description:
        st.caption(manifest.description)
    st.caption(
        f"Location: `{info.path}` · {len(manifest.documents)} document(s) · "
        f"{_index_status(ws, name)}"
    )

    problems = ws.catalog.verify(name)
    if problems:
        for filename, issue in sorted(problems.items()):
            st.error(
                f"{filename}: {issue}. Searches are blocked until it is restored or "
                "re-registered with `hallab sources add --replace`."
            )
    elif manifest.documents:
        st.success(
            "Integrity check: every document matches the hash recorded when it was registered."
        )

    rows = [
        {
            "file": d.local_filename,
            "title": d.metadata.title,
            "licence": d.metadata.license or "unknown",
            "metadata status": d.metadata.status.value,
            "accessed": str(d.metadata.accessed_on or ""),
            "last updated": str(d.metadata.last_updated or ""),
            "registered (UTC)": d.added_at.strftime("%Y-%m-%d %H:%M"),
            "bytes": d.byte_size,
            "sha256": d.content_sha256[:16] + "…",
            "integrity": problems.get(d.local_filename, "ok"),
        }
        for d in manifest.documents
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "Metadata status says where title, URL and dates came from. It is not a "
            "judgement of the source's reliability."
        )
        files = [d.local_filename for d in manifest.documents]
        chosen = st.selectbox(
            "Inspect a document", files, index=None, placeholder="Choose a document"
        )
        if chosen:
            entry = next(d for d in manifest.documents if d.local_filename == chosen)
            st.json(entry.model_dump(mode="json"))
            path = ws.catalog.documents_dir(name) / chosen
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                st.caption(f"First 3,000 of {len(text):,} characters:")
                st.text(text[:3000])
            else:
                st.error("The document file is missing.")


def _upload(ws) -> None:
    st.markdown(
        "Add a `.txt` or `.md` document to **your own** collection. The packaged "
        "`demo` collection is read-only. The file is validated (encoding, size, "
        "duplicates) by the same ingestion code as `hallab sources add`."
    )
    st.caption(
        "Only add documents you are allowed to store and use. Metadata you type is "
        "recorded as *user-supplied* and is not verified by the tool."
    )
    user_collections = [
        c.name for c in ws.catalog.list_collections() if not ws.catalog.is_packaged(c.name)
    ]
    with st.form("upload", clear_on_submit=False):
        collection = st.text_input(
            "Collection name (existing or new; lowercase letters, digits, - and _)",
            value=user_collections[0] if user_collections else "my-references",
        )
        upload = st.file_uploader("Document", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS])
        title = st.text_input("Title (required)")
        c1, c2 = st.columns(2)
        author = c1.text_input("Author or organization")
        publisher = c2.text_input("Publisher")
        url = c1.text_input("Source URL (http/https)")
        license_ = c2.text_input("Licence")
        published = c1.text_input("Publication date (YYYY-MM-DD)")
        accessed = c2.text_input("Date accessed (YYYY-MM-DD)")
        version = c1.text_input("Source version")
        doc_type = c2.selectbox(
            "Document type",
            [t.value for t in DocumentType],
            index=[t.value for t in DocumentType].index("other"),
        )
        submitted = st.form_submit_button("Validate and add document", type="primary")
    if not submitted:
        return
    if upload is None or not title.strip():
        st.error("Choose a file and enter a title.")
        return

    def add() -> None:
        name = collection.strip()
        if ws.catalog.is_packaged(name):
            raise HallucinationLabError(f"'{name}' is a packaged, read-only collection.")
        metadata = SourceMetadata(
            title=title,
            author_or_organization=author or None,
            publisher=publisher or None,
            url=url.strip() or None,
            license=license_ or None,
            publication_date=_parse_date(published, "Publication date"),
            accessed_on=_parse_date(accessed, "Date accessed"),
            source_version=version or None,
            document_type=DocumentType(doc_type),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / safe_filename(upload.name)
            path.write_bytes(upload.getvalue())
            entry, warnings = ws.catalog.add_document(name, path, metadata)
        for w in warnings:
            st.warning(f"{w.code}: {w.message}")
        st.success(
            f"Added {entry.local_filename} to '{name}' as {entry.source_id}. It becomes "
            "searchable when you next retrieve from this collection."
        )

    common.run_action(add)


@common.safe_page
def render() -> None:
    st.title("Reference collections")
    st.caption(
        "Trusted documents that retrieval searches. Sources are identified by the "
        "SHA-256 of their bytes, so any change to a document is detected."
    )
    ws = common.workspace()
    browse, add = st.tabs(["Browse", "Add a document"])
    with browse:
        _browse(ws)
    with add:
        _upload(ws)
