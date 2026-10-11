"""Report generation: content, disclosure, determinism and the three export formats."""

import csv
import io
import json
from datetime import datetime, timezone

import pytest

from hallucination_lab.cli import main
from hallucination_lab.models import EvidenceAssessment, EvidenceLabel, Verdict
from hallucination_lab.services.reports import (
    CSV_COLUMNS,
    build_report,
    export,
    report_filename,
    to_csv,
    to_markdown,
)
from hallucination_lab.services.workspace import Workspace
from hallucination_lab.verification.workflow import VerificationInput
from tests.conftest import evidence_containing

T1 = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def ws(settings, test_collection):
    return Workspace(settings)


@pytest.fixture
def reviewed(ws, verification_env):
    """Bridge claim contradicted then revised; railway pending; population pending."""
    env = verification_env
    loaded = ws.load(env["session_id"])
    eid = evidence_containing(loaded.session, env["bridge"], "completed in 1874")
    other = [e for e in loaded.session.latest_retrieval(env["bridge"]).evidence_ids if e != eid]

    def inp(**kw):
        return VerificationInput(
            claim_id=env["bridge"],
            verdict=Verdict.CONTRADICTED,
            rationale="The passage dates the first bridge to 1874, not 1880.",
            assessments=[
                EvidenceAssessment(evidence_id=eid, label=EvidenceLabel.CONTRADICTING),
                *[EvidenceAssessment(evidence_id=o, label=EvidenceLabel.IRRELEVANT) for o in other],
            ],
            contradicting_evidence_ids=[eid],
            **kw,
        )

    v1, _ = ws.prepare_verification(loaded, "r1", inp())
    loaded = ws.commit_verification(loaded, v1)
    v2, _ = ws.prepare_verification(
        loaded,
        "r1",
        inp(revision_of=v1.verification_id, revision_reason="Added the reviewer note."),
    )
    loaded = ws.commit_verification(loaded, v2)
    return env, loaded, v1, v2, eid


def _report(ws, loaded, when=T1):
    return build_report(loaded.session, ws.evidence_checks(loaded.session), generated_at=when)


def test_report_contents(ws, reviewed):
    env, loaded, _, v2, eid = reviewed
    r = _report(ws, loaded)
    assert r["session"]["session_id"] == loaded.session_id
    assert r["response"]["text"] == loaded.session.response.text
    assert r["response"]["provenance"] == "synthetic_fixture"
    bridge = next(c for c in r["claims"] if c["claim_id"] == env["bridge"])
    assert bridge["review_status"] == "verified"
    verdict = bridge["human_verdict"]
    assert verdict["verification_id"] == v2.verification_id and verdict["revision_number"] == 2
    assert verdict["reviewer_id"] == "r1" and verdict["rationale"]
    cited = verdict["cited_evidence"][0]
    assert cited["evidence_id"] == eid and "1874" in cited["quoted_passage"]
    assert cited["document_reference"] and cited["source_id"] and cited["passage_sha256"]
    assert cited["source_check"] == "verified"
    # history preserved, active not double-counted
    assert [h["status"] for h in bridge["revision_history"]] == ["superseded", "active"]
    assert r["summary"]["verified_by_human"] == 1
    assert r["summary"]["active_verdicts"]["CONTRADICTED"] == 1
    assert r["summary"]["superseded_records_preserved"] == 1


def test_candidates_distinguished_from_assessed_evidence(ws, reviewed):
    env, loaded, *_ = reviewed
    r = _report(ws, loaded)
    bridge = next(c for c in r["claims"] if c["claim_id"] == env["bridge"])
    labels = {c["reviewer_assessment"] for c in bridge["retrieval"]["candidates"]}
    assert labels <= {"CONTRADICTING", "IRRELEVANT"}
    railway = next(c for c in r["claims"] if c["claim_id"] == env["railway"])
    assert railway["human_verdict"] is None
    assert {c["reviewer_assessment"] for c in railway["retrieval"]["candidates"]} == {
        "not assessed"
    }
    assert "not verified evidence" in railway["retrieval"]["note"]


def test_unresolved_and_limitations_disclosed(ws, reviewed):
    env, loaded, *_ = reviewed
    r = _report(ws, loaded)
    assert {u["claim_id"] for u in r["unresolved_claims"]} == {env["railway"], env["population"]}
    assert r["summary"]["complete"] is False
    assert any("synthetic fixture" in item for item in r["limitations"])
    assert any("not a general LLM hallucination rate" in item for item in r["limitations"])


def test_integrity_warning_for_modified_source(ws, reviewed):
    _, loaded, *_ = reviewed
    doc = ws.catalog.documents_dir("testrefs") / "lake_verenna.md"
    doc.write_text(doc.read_text().replace("1874", "1875"))
    r = _report(ws, loaded)
    flagged = [w for w in r["integrity_warnings"] if w["status"] == "modified"]
    assert flagged and any(w["cited_by_active_verdict"] for w in flagged)


def test_reports_are_deterministic(ws, reviewed):
    _, loaded, *_ = reviewed
    a, b = _report(ws, loaded, T1), _report(ws, loaded, T2)
    assert a["content_sha256"] == b["content_sha256"]
    assert a["generated_at"] != b["generated_at"]
    for fmt in ("json", "csv", "md"):
        assert export(a, fmt) == export(_report(ws, loaded, T1), fmt)


def test_json_export_round_trips(ws, reviewed):
    _, loaded, *_ = reviewed
    r = _report(ws, loaded)
    assert json.loads(export(r, "json")) == r
    assert report_filename(r, "json") == f"hallab-report-{loaded.session_id}.json"


def test_csv_export(ws, reviewed):
    env, loaded, _, v2, eid = reviewed
    rows = list(csv.DictReader(io.StringIO(to_csv(_report(ws, loaded)))))
    assert list(rows[0]) == CSV_COLUMNS
    assert len(rows) == len(loaded.session.claims)
    bridge = next(r for r in rows if r["claim_id"] == env["bridge"])
    assert bridge["verdict"] == "CONTRADICTED" and bridge["verification_id"] == v2.verification_id
    assert bridge["superseded_records"] == "1" and eid in bridge["cited_evidence_ids"]
    pending = next(r for r in rows if r["claim_id"] == env["railway"])
    assert pending["verdict"] == "" and pending["review_status"] == "awaiting_review"


def test_markdown_export(ws, reviewed):
    _, loaded, v1, _, _ = reviewed
    md = to_markdown(_report(ws, loaded))
    for heading in (
        "# Verification report",
        "## Summary",
        "## Original AI response",
        "## Claims",
        "## Unresolved claims",
        "## Document integrity warnings",
        "## Methodological limitations",
    ):
        assert heading in md
    assert "INCOMPLETE" in md and v1.verification_id in md and "superseded" in md
    assert "BM25 (relevance only)" in md


def test_empty_session_report(ws):
    loaded = ws.create_session("???")
    r = build_report(loaded.session)
    assert r["summary"]["verified_by_human"] == 0 and r["unresolved_claims"] == []
    assert any("not re-checked" in item for item in r["limitations"])
    for fmt in ("json", "csv", "md"):
        assert export(r, fmt)


def test_unknown_format():
    with pytest.raises(ValueError, match="unknown report format"):
        export({}, "pdf")


def test_cli_report_command(ws, reviewed, settings, monkeypatch, tmp_path, capsys):
    _, loaded, *_ = reviewed
    monkeypatch.setenv("HALLAB_DATA_DIR", str(settings.data_dir))
    out = tmp_path / "out dir" / "report.csv"  # space in path on purpose
    assert (
        main(["report", "--session", loaded.session_id, "--format", "csv", "--output", str(out)])
        == 0
    )
    assert out.read_text(encoding="utf-8").startswith("session_id,claim_id")
    assert main(["report", "--session", loaded.session_id, "--format", "md"]) == 0
    assert "# Verification report" in capsys.readouterr().out


def test_report_refuses_corrupted_session(settings, monkeypatch, capsys):
    monkeypatch.setenv("HALLAB_DATA_DIR", str(settings.data_dir))
    settings.sessions_dir.mkdir(parents=True)
    (settings.sessions_dir / "ses_ffffffffffffffff.json").write_text("{broken")
    assert main(["report", "--session", "ses_ffffffffffffffff"]) == 2
    assert "not valid JSON" in capsys.readouterr().err
