import json

import pytest

from hallucination_lab.cli import main


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HALLAB_DATA_DIR", str(tmp_path / "lab"))
    return tmp_path / "lab"


def test_sources_add_list_show_verify(capsys, data_dir, synthetic_files, tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {"title": "Lake Verenna", "document_type": "synthetic_fixture", "license": "CC0 (test)"}
        )
    )
    assert (
        main(
            [
                "sources",
                "add",
                "--collection",
                "mine",
                "--file",
                str(synthetic_files["lake"]),
                "--metadata",
                str(meta),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "source_id:" in out and "not verified by the system" in out
    source_id = next(line.split()[-1] for line in out.splitlines() if "source_id:" in line)

    assert (
        main(
            [
                "sources",
                "add",
                "--collection",
                "mine",
                "--file",
                str(synthetic_files["railway"]),
                "--title",
                "Railway",
            ]
        )
        == 0
    )
    assert main(["sources", "list", "--collection", "mine"]) == 0
    assert "Lake Verenna | CC0 (test) | metadata user_supplied" in capsys.readouterr().out
    assert main(["sources", "show", "--collection", "mine", source_id]) == 0
    assert json.loads(capsys.readouterr().out)["source_id"] == source_id
    assert main(["sources", "verify", "--collection", "mine"]) == 0


def test_sources_add_errors(capsys, data_dir, synthetic_files, tmp_path):
    assert (
        main(
            [
                "sources",
                "add",
                "--collection",
                "demo",
                "--file",
                str(synthetic_files["lake"]),
                "--title",
                "x",
            ]
        )
        == 2
    )
    assert "read-only" in capsys.readouterr().err
    assert (
        main(["sources", "add", "--collection", "mine", "--file", str(synthetic_files["lake"])])
        == 2
    )
    assert "--title" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text('{"title": "x", "url": "not a url"}')
    assert (
        main(
            [
                "sources",
                "add",
                "--collection",
                "mine",
                "--file",
                str(synthetic_files["lake"]),
                "--metadata",
                str(bad),
            ]
        )
        == 2
    )
    assert "invalid" in capsys.readouterr().err


def test_verify_reports_modification(capsys, data_dir, synthetic_files):
    main(
        [
            "sources",
            "add",
            "--collection",
            "mine",
            "--file",
            str(synthetic_files["lake"]),
            "--title",
            "Lake",
        ]
    )
    doc = data_dir / "collections" / "mine" / "documents" / "lake_verenna.md"
    doc.write_text("tampered")
    assert main(["sources", "verify", "--collection", "mine"]) == 1
    assert "modified" in capsys.readouterr().out


def test_index_and_search_label_candidates(capsys, data_dir):
    assert main(["index", "build", "--collection", "demo", "--max-chars", "600"]) == 0
    assert "passages" in capsys.readouterr().out
    assert (
        main(
            [
                "search",
                "--collection",
                "demo",
                "--top-k",
                "2",
                "--claim",
                "The Eiffel Tower was completed in 1889.",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "RETRIEVAL CANDIDATES" in out and "NOT verified" in out
    assert "BM25 score" in out and "Passage:  psg_" in out and "URL:" in out


def test_search_json_and_errors(capsys, data_dir):
    assert main(["search", "--collection", "demo", "--json", "--claim", "Pi is 3.14159"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["label"] == "retrieval_candidates_not_verified_evidence"
    assert payload["hits"][0]["bm25_score"] > 0 and payload["hits"][0]["source_url"]
    assert main(["search", "--collection", "demo", "--claim", "   "]) == 2
    assert "empty" in capsys.readouterr().err
    assert main(["search", "--collection", "nope", "--claim", "x y"]) == 2
    assert "no reference collection" in capsys.readouterr().err


def test_retrieve_into_saved_session_and_show(capsys, data_dir):
    main(
        [
            "extract",
            "--save",
            "--text",
            "The Eiffel Tower was completed in 1899. Do you like Paris?",
        ]
    )
    session_id = capsys.readouterr().err.split("Saved session ")[1].split()[0]
    assert main(["retrieve", "--session", session_id, "--collection", "demo", "--top-k", "2"]) == 0
    out = capsys.readouterr().out
    assert "claims searched: 1" in out and "RETRIEVAL CANDIDATES" in out
    assert main(["sessions", "show", session_id]) == 0
    assert "Eiffel Tower (Wikipedia)" in capsys.readouterr().out


def test_eval_retrieval(capsys, data_dir):
    assert main(["eval", "retrieval"]) == 0
    out = capsys.readouterr().out
    assert "recall@5" in out and "not hallucination detection" in out
