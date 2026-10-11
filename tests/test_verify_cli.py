"""Interactive verification flow driven by scripted reviewer input."""

import json

import pytest

from hallucination_lab import cli_verify
from hallucination_lab.cli import main
from hallucination_lab.errors import VerificationError
from hallucination_lab.models import Verdict
from hallucination_lab.verification.interactive import run_verification
from tests.conftest import ScriptedPrompter


def _run(env, answers, reviewer="r1", **kw):
    io = ScriptedPrompter(answers)
    saved = run_verification(
        env["store"], env["session_id"], reviewer, catalog=env["catalog"], prompter=io, **kw
    )
    return saved, io


def _labels_for(env, claim_key, wanted):
    """Answers for the passage loop: label the passage containing `wanted`, others irrelevant."""
    session = env["store"].load(env["session_id"])
    latest = session.latest_retrieval(env[claim_key])
    evidence = {e.evidence_id: e for e in session.evidence}
    answers = []
    for eid in latest.evidence_ids:
        hit = any(w in evidence[eid].quoted_passage for w in wanted)
        answers += [wanted[next(iter(wanted))] if hit else "i", ""]
    return answers


def bridge_answers(env):
    return [
        *_labels_for(env, "bridge", {"completed in 1874": "c"}),
        "c",
        "",
        "The passage dates the first bridge to 1874, not 1880.",
        "y",
    ]


def test_full_review_saves_after_confirmation(verification_env):
    env = verification_env
    saved, io = _run(env, [*bridge_answers(env), "n"], claim_ids=[env["bridge"]])
    (v,) = saved
    assert v.verdict is Verdict.CONTRADICTED and v.reviewer_id == "r1"
    stored = env["store"].load(env["session_id"])
    assert stored.human_verdict(env["bridge"]) == v
    assert stored.verification_drafts == []
    assert "REVIEW SUMMARY" in io.text and "NOT support" in io.text
    assert "Response context" in io.text and ">>>" in io.text


def test_declining_to_save_keeps_draft_only(verification_env):
    env = verification_env
    answers = [*bridge_answers(env)[:-1], "n"]
    saved, _ = _run(env, answers, claim_ids=[env["bridge"]])
    assert saved == []
    stored = env["store"].load(env["session_id"])
    assert stored.verifications == []
    draft = stored.get_draft(env["bridge"], "r1")
    assert draft.verdict is Verdict.CONTRADICTED and draft.rationale


def test_interrupt_and_resume_from_draft(verification_env):
    env = verification_env
    first_passage = bridge_answers(env)[:2]
    saved, io = _run(env, [*first_passage, "q"], claim_ids=[env["bridge"]])  # quit mid-way
    assert saved == [] and "Paused" in io.text
    draft = env["store"].load(env["session_id"]).get_draft(env["bridge"], "r1")
    assert len(draft.evidence_assessments) == 1

    remaining = bridge_answers(env)[2:]
    saved, io = _run(env, ["y", "y", *remaining, "n"], claim_ids=[env["bridge"]])
    assert "Resume your draft" in io.text
    assert len(saved) == 1 and len(saved[0].evidence_assessments) == len(
        env["store"].load(env["session_id"]).evidence_for(env["bridge"])
    )


def test_eof_is_a_safe_pause(verification_env):
    env = verification_env
    saved, io = _run(env, [], claim_ids=[env["bridge"]])
    assert saved == [] and "Paused" in io.text
    assert env["store"].load(env["session_id"]).verifications == []


def test_invalid_input_and_rationale_are_reprompted(verification_env):
    env = verification_env
    labels = _labels_for(env, "bridge", {"completed in 1874": "c"})
    answers = [
        "zzz",
        *labels,
        "s",
        "c",
        "",
        "n/a",
        "Looks fine.",
        "n",
        "The passage dates the first bridge to 1874, not 1880.",
        "y",
        "n",
    ]
    saved, io = _run(env, answers, claim_ids=[env["bridge"]])
    assert len(saved) == 1
    assert "Please type one of" in io.text
    assert "SUPPORTED needs at least one passage" in io.text
    assert "placeholder" in io.text and "generic" in io.text


def test_high_score_never_auto_selected(verification_env):
    """Marking every passage irrelevant makes SUPPORTED impossible, whatever the scores."""
    env = verification_env
    session = env["store"].load(env["session_id"])
    n = len(session.latest_retrieval(env["railway"]).evidence_ids)
    answers = ["i", ""] * n + [
        "s",
        "i",
        "No passage was judged relevant by the reviewer here.",
        "y",
        "n",
    ]
    saved, _ = _run(env, answers, claim_ids=[env["railway"]])
    assert saved[0].verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert saved[0].supporting_evidence_ids == []


def test_revision_flow_preserves_original(verification_env):
    env = verification_env
    (v1,), _ = _run(env, [*bridge_answers(env), "n"], claim_ids=[env["bridge"]])
    with pytest.raises(VerificationError, match="--revise"):
        _run(env, [], claim_ids=[env["bridge"]])
    answers = [
        "Re-checked the passage; adding the second bridge date.",
        *["y"] * 3,
        "c",
        "",
        "The passage dates the first bridge to 1874 and a second to 1921; neither 1880.",
        "y",
    ]
    (v2,), _ = _run(env, answers, revise=v1.verification_id)
    stored = env["store"].load(env["session_id"])
    assert stored.get_verification(v1.verification_id) == v1
    assert v2.revision_of == v1.verification_id and stored.human_verdict(env["bridge"]) == v2


def test_cross_reviewer_revision_requires_authorization(verification_env):
    env = verification_env
    (v1,), _ = _run(env, [*bridge_answers(env), "n"], claim_ids=[env["bridge"]])
    with pytest.raises(VerificationError, match="--authorization"):
        _run(env, [], reviewer="r2", revise=v1.verification_id)


def test_claim_without_retrieval_can_be_skipped(verification_env):
    from hallucination_lab.extraction.editing import add_manual_claim

    env = verification_env
    store = env["store"]
    session, version = store.load_versioned(env["session_id"])
    claims = add_manual_claim(
        session.claims, session.response, "Brindle hosts an annual regatta.", "r1"
    )
    store.save(session.model_copy(update={"claims": claims}), expected_version=version)
    new_id = claims[-1].claim_id
    saved, io = _run(env, ["n"], claim_ids=[new_id])
    assert saved == [] and "NOT SEARCHED" in io.text


# ------------------------------------------------------------------ CLI wrapper


@pytest.fixture
def cli_env(verification_env, settings, monkeypatch):
    monkeypatch.setenv("HALLAB_DATA_DIR", str(settings.data_dir))
    return verification_env


def test_cli_verify_status_and_summary(cli_env, capsys, monkeypatch):
    env = cli_env
    answers = [*bridge_answers(env), "n"]
    monkeypatch.setattr(cli_verify, "PROMPTER_FACTORY", lambda: ScriptedPrompter(answers))
    assert (
        main(
            [
                "verify",
                "--session",
                env["session_id"],
                "--reviewer-id",
                "r1",
                "--claim-id",
                env["bridge"],
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["verify", "status", "--session", env["session_id"]]) == 0
    out = capsys.readouterr().out
    assert "INCOMPLETE (2 claim(s) still need human review)" in out
    assert "CONTRADICTED                               1" in out
    assert main(["verify", "summary", "--session", env["session_id"]]) == 0
    out = capsys.readouterr().out
    assert "INCOMPLETE" in out and "Rationale:" in out and "Reviewer r1" in out
    assert main(["verify", "summary", "--session", env["session_id"], "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["label"] == "human_verification_records" and len(data["records"]) == 1


def test_cli_argument_errors(cli_env, capsys):
    env = cli_env
    assert main(["verify", "--session", env["session_id"]]) == 2
    assert "--reviewer-id is required" in capsys.readouterr().err
    assert (
        main(
            [
                "verify",
                "--session",
                env["session_id"],
                "--reviewer-id",
                "r1",
                "--authorization",
                "x",
            ]
        )
        == 2
    )
    assert "only used together with --revise" in capsys.readouterr().err
    assert (
        main(
            [
                "verify",
                "--session",
                env["session_id"],
                "--reviewer-id",
                "r1",
                "--revise",
                "ver_ffffffffffffffff",
            ]
        )
        == 2
    )
    assert "no verification" in capsys.readouterr().err
