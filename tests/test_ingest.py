"""Document loading, validation, canonical text and passage segmentation."""

from itertools import pairwise

import pytest

from hallucination_lab.errors import DocumentIngestionError, DocumentModifiedError
from hallucination_lab.models import SourceOrigin
from hallucination_lab.retrieval.ingest import (
    canonicalize,
    collection_fingerprint,
    ingest_collection,
    load_document,
    new_document_entry,
    segment_passages,
)
from hallucination_lab.retrieval.models import ChunkingConfig, CollectionManifest
from tests.conftest import FIXED_TIME, LAKE_DOC, synthetic_metadata

SID = "src_0000000000000001"


def _write(tmp_path, name, data):
    path = tmp_path / name
    (path.write_bytes if isinstance(data, bytes) else path.write_text)(data)
    return path


# ------------------------------------------------------------------- validation


@pytest.mark.parametrize("name", ["doc.pdf", "doc.docx", "doc.html", "noext"])
def test_unsupported_formats_rejected(tmp_path, name):
    with pytest.raises(DocumentIngestionError, match="unsupported file type"):
        load_document(_write(tmp_path, name, "Some text here."))


@pytest.mark.parametrize("content", [b"", b"   \n\t  "])
def test_empty_documents_rejected(tmp_path, content):
    with pytest.raises(DocumentIngestionError, match="empty"):
        load_document(_write(tmp_path, "e.txt", content))


def test_too_large_rejected(tmp_path):
    path = _write(tmp_path, "big.txt", "word " * 300)
    with pytest.raises(DocumentIngestionError, match="exceeds"):
        load_document(path, max_bytes=1000)


def test_binary_rejected(tmp_path):
    with pytest.raises(DocumentIngestionError, match="binary"):
        load_document(_write(tmp_path, "b.txt", b"text\x00more text"))


def test_invalid_utf8_gives_actionable_error(tmp_path):
    path = _write(tmp_path, "l.txt", "Café au lait.".encode("latin-1"))
    with pytest.raises(DocumentIngestionError, match="not valid utf-8"):
        load_document(path)
    assert load_document(path, encoding="latin-1").text == "Café au lait."


def test_unsupported_encoding_name(tmp_path):
    with pytest.raises(DocumentIngestionError, match="unsupported encoding"):
        load_document(_write(tmp_path, "x.txt", "abc def ghi"), encoding="ebcdic")


@pytest.mark.parametrize(
    "data,encoding",
    [
        ("﻿BOM text here.".encode(), "utf-8"),
        ("UTF-16 text here.".encode("utf-16"), "utf-16"),
    ],
)
def test_byte_order_marks_detected(tmp_path, data, encoding):
    doc = load_document(_write(tmp_path, "bom.txt", data))
    assert doc.system.encoding == encoding
    assert not doc.text.startswith("﻿")


def test_canonical_text_only_normalises_newlines():
    assert canonicalize("A  b\r\nC\rDé") == "A  b\nC\nDé"


def test_system_facts_computed_from_bytes(tmp_path):
    path = _write(tmp_path, "d.txt", b"Line one.\r\nLine two.\r\n")
    doc = load_document(path)
    import hashlib

    assert doc.system.content_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert doc.system.byte_size == 22 and doc.text == "Line one.\nLine two.\n"
    assert doc.system.line_count == 3


def test_short_document_warning(tmp_path):
    doc = load_document(_write(tmp_path, "s.txt", "Tiny but valid text."))
    assert [w.code for w in doc.warnings] == ["short_document"]


# ------------------------------------------------------------------ segmentation


def _passages(text, **cfg):
    return segment_passages(text, SID, ChunkingConfig(**cfg))[0]


def test_passages_are_exact_slices_with_line_numbers():
    passages = _passages(LAKE_DOC)
    lines = LAKE_DOC.split("\n")
    for p in passages:
        assert LAKE_DOC[p.char_start : p.char_end] == p.text
        assert lines[p.line_start - 1] and p.text.startswith(lines[p.line_start - 1].strip()[:10])


def test_markdown_and_wiki_sections_recorded():
    sections = {p.section for p in _passages(LAKE_DOC)}
    assert {
        "Lake Verenna (synthetic test fixture) > Geography",
        "Lake Verenna (synthetic test fixture) > History",
    } <= sections
    wiki = "Intro text here.\n== History ==\nBuilt in 1900.\n=== Later ===\nRebuilt in 1950.\n"
    assert [p.section for p in _passages(wiki)] == [None, "History", "History > Later"]


def test_headings_are_not_passages():
    assert all(not p.text.startswith(("#", "==")) for p in _passages(LAKE_DOC))


def test_passage_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    passages = _passages(text, max_chars=120, overlap_sentences=1)
    assert len(passages) > 3
    assert all(len(p.text) <= 120 for p in passages)
    for a, b in pairwise(passages):
        assert b.char_start < a.char_end  # one shared sentence
    no_overlap = _passages(text, max_chars=120, overlap_sentences=0)
    for a, b in pairwise(no_overlap):
        assert b.char_start >= a.char_end


def test_long_unpunctuated_text_is_cut_with_warning():
    text = "\n".join(f"Row {i} | value {i * 3} | unit" for i in range(80))
    passages, warnings = segment_passages(text, SID, ChunkingConfig(max_chars=200))
    assert all(len(p.text) <= 200 for p in passages)
    assert [w.code for w in warnings] == ["long_unpunctuated_text"]
    assert "".join(p.text for p in passages).replace("\n", "") == text.replace("\n", "")


def test_passage_ids_stable_and_config_dependent():
    a = [p.passage_id for p in _passages(LAKE_DOC)]
    assert a == [p.passage_id for p in _passages(LAKE_DOC)]
    assert a != [p.passage_id for p in _passages(LAKE_DOC, max_chars=100)]


# -------------------------------------------------------------------- collection


def _manifest(tmp_path, files):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(exist_ok=True)
    entries = []
    for name, text in files.items():
        path = _write(docs_dir, name, text)
        entries.append(
            new_document_entry(
                load_document(path),
                name,
                synthetic_metadata(name),
                SourceOrigin.SYNTHETIC_TEST_FIXTURE,
                now=FIXED_TIME,
            )
        )
    manifest = CollectionManifest(
        name="t",
        origin=SourceOrigin.SYNTHETIC_TEST_FIXTURE,
        created_at=FIXED_TIME,
        documents=entries,
    )
    return manifest, docs_dir


def test_ingest_collection(tmp_path):
    manifest, docs = _manifest(tmp_path, {"lake.md": LAKE_DOC})
    processed = ingest_collection(manifest, docs, now=FIXED_TIME)
    assert processed.documents[0].passage_count == len(processed.passages) > 0
    assert processed.fingerprint == collection_fingerprint(manifest, ChunkingConfig())


def test_modified_document_detected(tmp_path):
    manifest, docs = _manifest(tmp_path, {"lake.md": LAKE_DOC})
    (docs / "lake.md").write_text(LAKE_DOC.replace("1874", "1875"), encoding="utf-8")
    with pytest.raises(DocumentModifiedError, match="modified since it was registered"):
        ingest_collection(manifest, docs)


def test_missing_document_detected(tmp_path):
    manifest, docs = _manifest(tmp_path, {"lake.md": LAKE_DOC})
    (docs / "lake.md").unlink()
    with pytest.raises(DocumentModifiedError, match="missing"):
        ingest_collection(manifest, docs)


def test_empty_collection_cannot_be_indexed(tmp_path):
    manifest = CollectionManifest(name="t", origin=SourceOrigin.USER_ADDED, created_at=FIXED_TIME)
    with pytest.raises(DocumentIngestionError, match="no documents"):
        ingest_collection(manifest, tmp_path)


def test_fingerprint_changes_with_settings_and_metadata(tmp_path):
    manifest, _ = _manifest(tmp_path, {"lake.md": LAKE_DOC})
    base = collection_fingerprint(manifest, ChunkingConfig())
    assert base != collection_fingerprint(manifest, ChunkingConfig(max_chars=500))
    doc = manifest.documents[0]
    retitled = manifest.model_copy(
        update={
            "documents": [
                doc.model_copy(
                    update={"metadata": doc.metadata.model_copy(update={"title": "New"})}
                )
            ]
        }
    )
    assert base != collection_fingerprint(retitled, ChunkingConfig())
