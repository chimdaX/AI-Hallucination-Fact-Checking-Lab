from hallucination_lab.models import EvidenceAssessment, EvidenceLabel, Verdict
from hallucination_lab.services.dashboard import compute_dashboard
from hallucination_lab.services.workspace import Workspace
from hallucination_lab.storage import JsonSessionStore
from hallucination_lab.verification.workflow import VerificationInput
from tests.conftest import evidence_containing


def test_empty_store(tmp_path):
    stats = compute_dashboard(JsonSessionStore(tmp_path / "none"))
    assert stats.empty and stats.totals.verified == 0 and stats.sessions == []


def test_counts_come_from_sessions_and_exclude_superseded(
    settings, test_collection, verification_env
):
    ws = Workspace(settings)
    env = verification_env
    loaded = ws.load(env["session_id"])
    eid = evidence_containing(loaded.session, env["bridge"], "completed in 1874")
    base = dict(
        claim_id=env["bridge"],
        verdict=Verdict.CONTRADICTED,
        rationale="The passage dates the first bridge to 1874, not 1880.",
        assessments=[EvidenceAssessment(evidence_id=eid, label=EvidenceLabel.CONTRADICTING)],
        contradicting_evidence_ids=[eid],
    )
    v1, _ = ws.prepare_verification(loaded, "r1", VerificationInput(**base))
    loaded = ws.commit_verification(loaded, v1)
    v2, _ = ws.prepare_verification(
        loaded,
        "r1",
        VerificationInput(
            **base, revision_of=v1.verification_id, revision_reason="Re-read the passage."
        ),
    )
    ws.commit_verification(loaded, v2)
    ws.create_session("Brindle is a town on Lake Verenna.")  # user-submitted, unreviewed

    stats = compute_dashboard(ws.store)
    t = stats.totals
    assert t.sessions == 2 and t.verified == 1 and t.revisions == 1 and t.superseded == 1
    assert t.verdicts["CONTRADICTED"] == 1 and sum(t.verdicts.values()) == 1
    assert t.pending == 2 + 1
    assert t.evidence_cited == 1 and t.evidence_retrieved >= t.evidence_assessed >= 1
    assert set(stats.by_provenance) == {"synthetic_fixture", "user_submitted"}
    assert stats.by_provenance["synthetic_fixture"].verified == 1
    assert stats.by_provenance["user_submitted"].verified == 0
    assert {r["data"] for r in stats.sessions} == {"Illustrative fixture", "User-submitted"}


def test_unreadable_files_reported(settings, verification_env):
    (settings.sessions_dir / "ses_ffffffffffffffff.json").write_text("not json")
    stats = compute_dashboard(JsonSessionStore(settings.sessions_dir))
    assert stats.totals.sessions == 1 and len(stats.unreadable) == 1
