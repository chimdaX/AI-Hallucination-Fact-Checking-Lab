"""Capture, source re-checks, retrieval analysis, CLI and the Streamlit page."""

import json
from pathlib import Path

import pytest

from hallucination_lab.benchmark import workflow as wf
from hallucination_lab.benchmark.capture import (
    CaptureError,
    extract_sentence,
    extract_span,
    html_to_text,
    recheck_html_evidence,
)
from hallucination_lab.benchmark.integrity import check_sources
from hallucination_lab.benchmark.retrieval_analysis import analyse_retrieval
from hallucination_lab.benchmark.store import BenchmarkStore
from hallucination_lab.cli import main
from hallucination_lab.ids import content_sha256
from tests.conftest import ScriptedPrompter

CLAIM = "The first bridge across Lake Verenna was completed in 1880."
PAGE = (
    "<html><head><style>x{}</style><script>var a=1;</script></head><body><nav>Menu</nav>"
    "<p>The tower opened in 1889. It is 330&nbsp;m tall.</p><p>Other text.</p></body></html>"
)


# ------------------------------------------------------------------- capture


def test_html_to_text_and_extraction():
    text = html_to_text(PAGE)
    assert "var a" not in text and "330 m" in text
    cap = extract_sentence(text, "330 m")
    assert cap.text == "It is 330 m tall." and cap.sha256 == content_sha256(cap.text)
    span = extract_span(text, "The tower", "1889.")
    assert span.text == "The tower opened in 1889."
    with pytest.raises(CaptureError):
        extract_sentence(text, "not there")


def test_online_recheck_statuses():
    sha = extract_sentence(html_to_text(PAGE), "330 m").sha256
    same = recheck_html_evidence("https://example.org", "330 m", None, sha, lambda u: PAGE)
    assert same.status == "unchanged"
    changed = recheck_html_evidence(
        "https://example.org",
        "330 m",
        None,
        sha,
        lambda u: PAGE.replace("330&nbsp;m tall", "330 m high"),
    )
    assert changed.status == "changed"
    gone = recheck_html_evidence(
        "https://example.org", "330 m", None, sha, lambda u: "<p>nothing</p>"
    )
    assert gone.status == "anchor_missing"

    def boom(url):
        raise OSError("offline")

    assert (
        recheck_html_evidence("https://example.org", "330 m", None, sha, boom).status
        == "unavailable"
    )


def test_check_sources_detects_modified_local_document(bench):
    bench.item(CLAIM)
    assert {r.status for r in check_sources(bench.store, bench.catalog)} == {"ok"}
    doc = bench.catalog.documents_dir("testrefs") / "lake_verenna.md"
    doc.write_text(doc.read_text().replace("1874", "1875"))
    assert {r.status for r in check_sources(bench.store, bench.catalog)} == {"changed"}
    doc.unlink()
    assert {r.status for r in check_sources(bench.store, bench.catalog)} == {"missing"}


# ---------------------------------------------------------- retrieval analysis


def test_retrieval_analysis_counts_contradicting_evidence_as_relevant(bench):
    bench.item(CLAIM)  # its evidence CONTRADICTS the claim
    analysis = analyse_retrieval(bench.store, bench.catalog.searcher("testrefs")[0])
    assert analysis.items_with_local_evidence == 1 and analysis.evidence_coverage == 1.0
    assert analysis.report.recall_at(1) == 1.0  # found, although it contradicts the claim
    assert analysis.by_relation == {"CONTRADICTING": {"items": 1, "found_in_top_5": 1}}
    assert "independent of claim truth" in analysis.notice
    approved_only = analyse_retrieval(
        bench.store, bench.catalog.searcher("testrefs")[0], approved_only=True
    )
    assert approved_only.report is None and approved_only.items_considered == 0


# ------------------------------------------------------------------------- CLI


@pytest.fixture
def cli(bench, settings, monkeypatch):
    monkeypatch.setenv("HALLAB_DATA_DIR", str(settings.data_dir))
    return bench


def test_cli_status_list_show_validate(cli, capsys):
    item = cli.item(CLAIM, proposed="SUPPORTED")
    assert main(["benchmark", "status"]) == 0
    out = capsys.readouterr().out
    assert "Human-APPROVED labels: 0" in out and "NOT ground truth" in out
    assert main(["benchmark", "list", "--status", "ready_for_review"]) == 0
    assert item.item_id in capsys.readouterr().out
    assert main(["benchmark", "show", item.item_id]) == 0
    out = capsys.readouterr().out
    assert "hidden to avoid biasing" in out and "AI-PROPOSED" not in out
    assert main(["benchmark", "show", item.item_id, "--reveal-proposed"]) == 0
    assert "AI-PROPOSED label (not ground truth): SUPPORTED" in capsys.readouterr().out
    assert main(["benchmark", "validate"]) == 0


def test_cli_review_approve_export_evaluate(cli, capsys, monkeypatch, tmp_path):
    from hallucination_lab import cli_benchmark

    item = cli.item(CLAIM)
    answers = [
        "n",
        "c",
        "h",
        "",
        "c",
        "Looks fine.",
        "n",
        "The passage dates the first bridge to 1874, not 1880.",
        "",
        "n",
        "y",
    ]
    monkeypatch.setattr(cli_benchmark, "PROMPTER_FACTORY", lambda: ScriptedPrompter(answers))
    assert main(["benchmark", "review", item.item_id, "--reviewer-id", "alice"]) == 0
    annotation = cli.store.annotations(item.item_id)[0]
    assert annotation.verdict.value == "CONTRADICTED" and not annotation.research_needed
    assert main(["benchmark", "approve", item.item_id, "--approver-id", "alice"]) == 2
    assert "--confirm-own-judgement" in capsys.readouterr().err
    assert (
        main(
            [
                "benchmark",
                "approve",
                item.item_id,
                "--approver-id",
                "claude",
                "--confirm-own-judgement",
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "benchmark",
                "approve",
                item.item_id,
                "--approver-id",
                "alice",
                "--confirm-own-judgement",
            ]
        )
        == 0
    )
    assert main(["benchmark", "export", "--version", "1.0.0", "--created-by", "alice"]) == 0
    export = cli.store.load_export(cli.store.exports()[0])
    preds = tmp_path / "preds.json"
    preds.write_text(
        json.dumps(
            {
                "format": "hallab-benchmark-predictions",
                "benchmark_version": "1.0.0",
                "dataset_sha256": export.dataset_sha256,
                "system": {"name": "s", "version": "1"},
                "created_at": "2026-02-01T00:00:00Z",
                "predictions": [{"item_id": item.item_id, "verdict": "CONTRADICTED"}],
            }
        )
    )
    capsys.readouterr()
    part = item.partition.value
    assert main(["benchmark", "evaluate", "--predictions", str(preds), "--partition", part]) == 0
    out = capsys.readouterr().out
    assert "strict accuracy 1.000" in out and "Coverage 1.000" in out


def test_cli_review_quit_saves_nothing(cli, monkeypatch, capsys):
    from hallucination_lab import cli_benchmark

    item = cli.item(CLAIM)
    monkeypatch.setattr(cli_benchmark, "PROMPTER_FACTORY", lambda: ScriptedPrompter(["n", "q"]))
    assert main(["benchmark", "review", item.item_id, "--reviewer-id", "alice"]) == 0
    assert cli.store.annotations(item.item_id) == []


def test_cli_evaluate_without_export_refuses(cli, capsys, tmp_path):
    assert main(["benchmark", "evaluate", "--predictions", str(tmp_path / "x.json")]) == 2
    assert "no approved export" in capsys.readouterr().err


def test_cli_retrieval_and_sources(cli, capsys):
    cli.item(CLAIM)
    assert main(["benchmark", "retrieval", "--collection", "testrefs"]) == 0
    out = capsys.readouterr().out
    assert "not verification accuracy" in out and "recall@1" in out
    assert main(["benchmark", "check-sources"]) == 0


# ----------------------------------------------------------- packaged dataset


def test_packaged_candidate_set_is_valid_and_provisional():
    store = BenchmarkStore(Path(__file__).resolve().parents[1] / "data" / "benchmark")
    items, problems = store.load_items()
    assert not problems and len(items) == 45
    errors, _ = wf.validate_benchmark(store)
    assert errors == []
    refs = store.references()
    proposed = {}
    for item in items:
        assert item.candidate_provenance.author_type.value == "ai_assistant"
        assert item.status.value in {"READY_FOR_REVIEW", "RESEARCH_NEEDED"}  # none approved
        assert not item.approvals
        proposed[item.proposed_label.verdict.value] = (
            proposed.get(item.proposed_label.verdict.value, 0) + 1
        )
        for e in item.evidence:
            ref = refs[e.reference_id]
            if ref.redistribution.value == "locator_only":
                assert e.passage_text is None  # licence-restricted text never stored
            assert e.passage_sha256 and e.captured_at
    assert proposed == {
        "SUPPORTED": 15,
        "CONTRADICTED": 15,
        "INSUFFICIENT_EVIDENCE": 10,
        "NOT_CHECKABLE": 5,
    }
    assert store.exports() == []  # nothing has been approved or exported


# --------------------------------------------------------------- Streamlit page


def test_benchmark_page_renders_and_records_blind_review(cli):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    item = cli.item(CLAIM, proposed="SUPPORTED")
    script = "from hallucination_lab.ui import page_benchmark\npage_benchmark.render()\n"
    at = AppTest.from_string(script, default_timeout=60)
    at.session_state["reviewer_id"] = "alice"
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("Under development" in w.value for w in at.warning)
    assert not any("AI-proposed label (NOT" in i.value for i in at.info)  # hidden by default
    at.radio(key=f"bm_rel_{item.item_id}_1").set_value("CONTRADICTING")
    at.selectbox(key=f"bm_verdict_{item.item_id}").set_value("CONTRADICTED")
    at.multiselect(key=f"bm_cited_{item.item_id}").set_value([item.evidence[0].evidence_id])
    at.text_area(key=f"bm_rat_{item.item_id}").input(
        "The passage dates the first bridge to 1874, not 1880."
    )
    next(b for b in at.button if b.label == "Save review").click().run()
    assert any("Judge" in e.value or "confirm" in e.value.lower() for e in at.error)
    assert cli.store.annotations(item.item_id) == []  # not confirmed -> nothing saved
    at.checkbox(key=f"bm_confirm_{item.item_id}").check()
    next(b for b in at.button if b.label == "Save review").click().run()
    assert not at.exception
    (annotation,) = cli.store.annotations(item.item_id)
    assert annotation.verdict.value == "CONTRADICTED"
    assert annotation.proposed_label_seen_before_review is False
