"""Interactive terminal workflow for human claim verification (``hallab verify``).

Per claim, the reviewer:

1. reads the claim, its original wording and the surrounding response text,
2. sees how it was extracted and how (or whether) evidence was retrieved,
3. inspects each retrieved passage with source, URL, date, location, source
   integrity status and credibility note, and labels it SUPPORTING,
   CONTRADICTING, IRRELEVANT or UNCERTAIN (or skips it),
4. chooses a verdict and the passages that justify it,
5. writes a rationale (checked for obvious problems),
6. reviews a summary and saves it, only after typing "y".

Progress is saved as a *draft* after every step. Typing "q", pressing Ctrl-D or
Ctrl-C stops safely, and the same command resumes later. Drafts are never
counted as verifications. Nothing in this flow assigns a verdict by itself.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from hallucination_lab.errors import VerificationError
from hallucination_lab.models import (
    Claim,
    EvidenceAssessment,
    EvidenceLabel,
    Verdict,
    Verification,
    VerificationDraft,
    VerificationSession,
    utc_now,
)
from hallucination_lab.retrieval.catalog import CollectionCatalog
from hallucination_lab.storage import JsonSessionStore
from hallucination_lab.verification.integrity import EvidenceCheck, check_session_evidence
from hallucination_lab.verification.rationale import check_rationale, rationale_errors
from hallucination_lab.verification.workflow import (
    VerificationInput,
    build_verification,
    check_revision_allowed,
    claims_awaiting_review,
    discard_draft,
    ordered_evidence_ids,
    record_verification,
    save_draft,
)

WIDTH = 92
LABEL_KEYS = {
    "s": EvidenceLabel.SUPPORTING,
    "c": EvidenceLabel.CONTRADICTING,
    "i": EvidenceLabel.IRRELEVANT,
    "u": EvidenceLabel.UNCERTAIN,
}
VERDICT_KEYS = {
    "s": Verdict.SUPPORTED,
    "c": Verdict.CONTRADICTED,
    "i": Verdict.INSUFFICIENT_EVIDENCE,
    "n": Verdict.NOT_CHECKABLE,
}
ASSESS_HELP = """\
  s = SUPPORTING     the passage establishes the claim (or the part you are checking)
  c = CONTRADICTING  the passage establishes that the claim is false or materially wrong
  i = IRRELEVANT     the passage does not bear on the claim (shared words only)
  u = UNCERTAIN      it might bear on the claim, but is ambiguous, partial or unclear
  k = skip for now   leave this passage unassessed
  f = show the full passage again
  q = save a draft and quit (resume later with the same command)
The BM25 score only says how many rare words the passage shares with the claim.
It is not evidence that the passage supports anything."""
VERDICT_HELP = """\
  s = SUPPORTED              reliable evidence you marked SUPPORTING establishes the claim
  c = CONTRADICTED           reliable evidence you marked CONTRADICTING shows it is false
  i = INSUFFICIENT_EVIDENCE  not enough reliable evidence: none found, weak, ambiguous,
                             partial, or conflicting sources you cannot resolve
  n = NOT_CHECKABLE          opinion, subjective, or too vague to be an objective fact
                             (not for claims that are merely hard to research)
  q = save a draft and quit
See docs/HUMAN_VERIFICATION_GUIDE.md for full decision criteria."""


class QuitRequested(Exception):
    """The reviewer asked to stop; the draft has been saved."""


class Prompter:
    """Terminal I/O. Tests substitute a scripted implementation."""

    def ask(self, prompt: str) -> str:
        return input(prompt)

    def say(self, text: str = "") -> None:
        print(text)


@dataclass
class _State:
    store: JsonSessionStore
    session: VerificationSession
    version: str
    reviewer_id: str
    catalog: CollectionCatalog | None
    io: Prompter
    now: Callable[[], datetime]
    saved: list[Verification] = field(default_factory=list)


def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(
        " ".join(text.split()), WIDTH, initial_indent=indent, subsequent_indent=indent
    )


def _ask(state: _State, prompt: str) -> str:
    try:
        return state.io.ask(prompt).strip()
    except EOFError as exc:
        raise QuitRequested from exc


def _choose(state: _State, prompt: str, keys: dict, help_text: str, extra: str = "") -> str:
    allowed = [*keys, *extra]
    while True:
        answer = _ask(state, prompt).lower()
        if answer == "?":
            state.io.say(help_text)
        elif answer == "q":
            raise QuitRequested
        elif answer in allowed:
            return answer
        else:
            state.io.say(f"  Please type one of: {', '.join(allowed)}, ? (help) or q (quit).")


def _yes(state: _State, prompt: str, default: bool) -> bool:
    while True:
        answer = _ask(state, prompt).lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        state.io.say("  Please answer y or n.")


def _save(state: _State) -> None:
    state.version = state.store.save_versioned(state.session, expected_version=state.version)


def _save_draft(state: _State, draft: VerificationDraft) -> VerificationDraft:
    draft = draft.model_copy(update={"updated_at": max(state.now(), draft.started_at)})
    state.session = save_draft(state.session, draft)
    _save(state)
    return draft


# ------------------------------------------------------------------ display


def _context(session: VerificationSession, claim: Claim, radius: int = 220) -> str:
    text = session.response.text
    if claim.span is None:
        return "(no position in the response: the claim was written or merged by a reviewer)"
    start, end = claim.span.start, claim.span.end
    before = text[max(0, start - radius) : start]
    after = text[end : end + radius]
    prefix = "..." if start > radius else ""
    suffix = "..." if end + radius < len(text) else ""
    return f"{prefix}{before}>>> {text[start:end]} <<<{after}{suffix}"


def _show_claim(state: _State, claim: Claim, position: str) -> None:
    s, io = state.session, state.io
    io.say("=" * WIDTH)
    io.say(f"CLAIM {position}   [{claim.claim_id}]")
    io.say(_wrap(claim.normalized_claim, "  "))
    if claim.original_text != claim.normalized_claim:
        io.say("  Original wording:")
        io.say(_wrap(claim.original_text))
    io.say("  Response context (claim marked >>> <<<):")
    io.say(_wrap(_context(s, claim)))
    io.say(
        f"  Extraction: {claim.extraction_method.value}"
        + (f"; reviewer edits: {len(claim.reviewer_edits)}" if claim.reviewer_edits else "")
    )
    if claim.temporal_context:
        io.say(f"  Time context: {claim.temporal_context} (check the evidence date)")
    latest = s.latest_retrieval(claim.claim_id)
    if latest is None:
        io.say("  Retrieval: NOT SEARCHED. Run `hallab retrieve` to find candidate passages.")
        return
    run = next(r for r in s.retrieval_runs if latest in r.results)
    p = run.parameters
    io.say(
        f"  Retrieval: {latest.outcome.value} | run {run.run_id} on '{run.collection_name}' "
        f"at {run.started_at:%Y-%m-%d %H:%M} UTC"
    )
    io.say(
        f"    method {p.get('retrieval_method')} k1={p.get('k1')} b={p.get('b')} "
        f"top_k={p.get('top_k')} tokenizer={p.get('tokenizer_version')}; "
        f"query terms: {', '.join(latest.query_terms) or '-'}"
    )


def _show_evidence(
    state: _State, n: int, total: int, eid: str, check: EvidenceCheck | None, full: bool = True
) -> None:
    ev = next(e for e in state.session.evidence if e.evidence_id == eid)
    io = state.io
    io.say("-" * WIDTH)
    rank = f"rank {ev.retrieval_rank}, " if ev.retrieval_rank else ""
    score = f"BM25 {ev.retrieval_score:.3f}" if ev.retrieval_score is not None else "no score"
    io.say(f"  PASSAGE {n}/{total}  [{eid}]  ({rank}{score}: lexical overlap, NOT support)")
    io.say(
        f"  Source:   {ev.source_title}"
        + (f"  ({ev.source_origin.value})" if ev.source_origin else "")
    )
    if ev.source_url:
        io.say(f"  URL:      {ev.source_url}")
    dates = []
    if ev.publication_date:
        dates.append(f"published {ev.publication_date}")
    if check and check.last_updated:
        dates.append(f"last updated {check.last_updated}")
    if ev.document_version:
        dates.append(ev.document_version)
    if dates:
        io.say(f"  Version:  {'; '.join(dates)}")
    loc = ev.passage_location
    where = ev.document_reference or ""
    if loc and loc.section:
        where += f"  (section: {loc.section})"
    io.say(f"  Location: {where}")
    io.say(
        f"  Retrieved {ev.retrieved_at:%Y-%m-%d %H:%M} UTC; matched terms: "
        f"{', '.join(ev.matched_terms) or '-'}"
    )
    if check:
        flag = "BLOCKED: " if check.blocks_citation else ""
        io.say(f"  Source check: {flag}{check.status.value} ({check.detail})")
        io.say(f"  Credibility: {check.credibility_note}")
        if check.license:
            io.say(f"  Licence: {check.license}")
    text = ev.quoted_passage if full else ev.quoted_passage[:400]
    io.say("  Passage:")
    for para in text.split("\n"):
        if para.strip():
            io.say(_wrap(para, "    | "))


# ------------------------------------------------------------- claim review


def _assess_passages(
    state: _State, claim: Claim, draft: VerificationDraft, checks: dict[str, EvidenceCheck]
) -> VerificationDraft:
    ids = ordered_evidence_ids(state.session, claim.claim_id)
    if not ids:
        state.io.say("  No retrieved passages for this claim.")
        return draft
    assessed = {a.evidence_id: a for a in draft.evidence_assessments}
    state.io.say(f"\n  Assess each passage (? for help). {len(assessed)} already assessed.")
    for n, eid in enumerate(ids, 1):
        check = checks.get(eid)
        _show_evidence(state, n, len(ids), eid, check)
        if eid in assessed:
            prior = assessed[eid]
            state.io.say(
                f"  Current assessment: {prior.label.value}"
                + (f" - {prior.explanation}" if prior.explanation else "")
            )
            if _yes(state, "  Keep it? [Y/n] ", default=True):
                continue
        blocked = bool(check and check.blocks_citation)
        if blocked:
            state.io.say(
                "  This passage's source changed after retrieval: you may only mark it "
                "IRRELEVANT or UNCERTAIN until retrieval is re-run."
            )
        while True:
            key = _choose(
                state,
                "  Assessment [s/c/i/u, k=skip, f=full, ?=help, q=quit]: ",
                LABEL_KEYS,
                ASSESS_HELP,
                extra="kf",
            )
            if key == "f":
                _show_evidence(state, n, len(ids), eid, check)
                continue
            if blocked and key in {"s", "c"}:
                state.io.say("  Not allowed for a blocked passage.")
                continue
            break
        if key == "k":
            continue
        explanation = _ask(state, "  Optional note on this passage (Enter to skip): ")
        assessed[eid] = EvidenceAssessment(
            evidence_id=eid,
            label=LABEL_KEYS[key],
            explanation=explanation or None,
        )
        draft = _save_draft(
            state,
            draft.model_copy(
                update={"evidence_assessments": [assessed[i] for i in ids if i in assessed]}
            ),
        )
        state.io.say("  (draft saved)")
    return draft


def _citable(
    draft: VerificationDraft, label: EvidenceLabel, checks: dict[str, EvidenceCheck]
) -> list[str]:
    return [
        a.evidence_id
        for a in draft.evidence_assessments
        if a.label is label
        and not (a.evidence_id in checks and checks[a.evidence_id].blocks_citation)
    ]


def _pick_evidence(state: _State, candidates: list[str], role: str) -> list[str]:
    state.io.say(f"  Passages you marked {role}:")
    for n, eid in enumerate(candidates, 1):
        ev = next(e for e in state.session.evidence if e.evidence_id == eid)
        state.io.say(f"    {n}. {ev.source_title} {ev.document_reference or ''} [{eid}]")
    while True:
        answer = _ask(state, f"  Cite which as {role}? (numbers, e.g. 1,2; Enter = all): ")
        if answer.lower() == "q":
            raise QuitRequested
        if not answer:
            return candidates
        try:
            picks = sorted({int(x) for x in answer.replace(" ", "").split(",") if x})
        except ValueError:
            state.io.say("  Please enter numbers separated by commas.")
            continue
        if picks and all(1 <= p <= len(candidates) for p in picks):
            return [candidates[p - 1] for p in picks]
        state.io.say(f"  Choose numbers between 1 and {len(candidates)}.")


def _choose_verdict(
    state: _State, draft: VerificationDraft, checks: dict[str, EvidenceCheck]
) -> VerificationDraft:
    labels = {a.label for a in draft.evidence_assessments}
    while True:
        key = _choose(
            state,
            "\n  Verdict [s=SUPPORTED c=CONTRADICTED i=INSUFFICIENT_EVIDENCE "
            "n=NOT_CHECKABLE, ?=help, q=quit]: ",
            VERDICT_KEYS,
            VERDICT_HELP,
        )
        verdict = VERDICT_KEYS[key]
        supporting, contradicting = [], []
        if verdict is Verdict.SUPPORTED:
            candidates = _citable(draft, EvidenceLabel.SUPPORTING, checks)
            if not candidates:
                state.io.say(
                    "  SUPPORTED needs at least one passage you assessed as SUPPORTING "
                    "(and whose source is not blocked). Choose another verdict."
                )
                continue
            if EvidenceLabel.CONTRADICTING in labels:
                state.io.say(
                    "  Note: you also marked contradicting passages. Explain in your "
                    "rationale why they do not outweigh the support, or consider "
                    "INSUFFICIENT_EVIDENCE."
                )
            supporting = _pick_evidence(state, candidates, "supporting")
        elif verdict is Verdict.CONTRADICTED:
            candidates = _citable(draft, EvidenceLabel.CONTRADICTING, checks)
            if not candidates:
                state.io.say(
                    "  CONTRADICTED needs at least one passage you assessed as "
                    "CONTRADICTING. Missing evidence is not a contradiction: consider "
                    "INSUFFICIENT_EVIDENCE."
                )
                continue
            if EvidenceLabel.SUPPORTING in labels:
                state.io.say(
                    "  Note: you also marked supporting passages. Explain the conflict "
                    "in your rationale, or consider INSUFFICIENT_EVIDENCE."
                )
            contradicting = _pick_evidence(state, candidates, "contradicting")
        return _save_draft(
            state,
            draft.model_copy(
                update={
                    "verdict": verdict,
                    "supporting_evidence_ids": supporting,
                    "contradicting_evidence_ids": contradicting,
                }
            ),
        )


def _write_rationale(state: _State, claim: Claim, draft: VerificationDraft) -> VerificationDraft:
    prompts = {
        Verdict.SUPPORTED: "which passage establishes the claim, and how",
        Verdict.CONTRADICTED: "what the evidence says and how it conflicts with the claim",
        Verdict.INSUFFICIENT_EVIDENCE: "what is missing, weak, ambiguous or conflicting",
        Verdict.NOT_CHECKABLE: "why this is not an objectively checkable statement",
    }
    assert draft.verdict is not None
    cited = [*draft.supporting_evidence_ids, *draft.contradicting_evidence_ids]
    passages = [e.quoted_passage for e in state.session.evidence if e.evidence_id in cited]
    if draft.rationale:
        state.io.say(f"  Draft rationale: {draft.rationale}")
    while True:
        text = _ask(state, f"  Rationale ({prompts[draft.verdict]}; Enter keeps draft):\n  > ")
        if text.lower() == "q":
            raise QuitRequested
        text = text or (draft.rationale or "")
        result = check_rationale(text, draft.verdict.value, claim.normalized_claim, passages)
        if result.errors:
            state.io.say(f"  Not accepted: {result.errors[0]}. Please explain your reasoning.")
            continue
        if result.warnings:
            for w in result.warnings:
                state.io.say(f"  Warning: {w}")
            if not _yes(state, "  Keep this rationale anyway? [y/N] ", default=False):
                continue
        return _save_draft(state, draft.model_copy(update={"rationale": text}))


def _ask_revision_reason(state: _State, draft: VerificationDraft) -> VerificationDraft:
    if draft.revision_reason:
        state.io.say(f"  Draft correction reason: {draft.revision_reason}")
        if _yes(state, "  Keep it? [Y/n] ", default=True):
            return draft
    while True:
        text = _ask(state, "  Reason for this correction (required):\n  > ")
        if text.lower() == "q":
            raise QuitRequested
        problems = rationale_errors(text)
        if problems:
            state.io.say(f"  Not accepted: {problems[0]}.")
            continue
        return _save_draft(state, draft.model_copy(update={"revision_reason": text}))


def _show_previous(state: _State, previous: Verification) -> None:
    io = state.io
    chain = state.session.revision_chain(previous.verification_id)
    io.say(
        f"  REVISING {previous.verification_id} (revision {len(chain)} of this claim's "
        f"history) by {previous.reviewer_id} at {previous.verification_timestamp:%Y-%m-%d %H:%M}"
    )
    io.say(f"    verdict: {previous.verdict.value}")
    io.say(_wrap(f"rationale: {previous.reasoning}"))
    for a in previous.evidence_assessments:
        io.say(
            f"    {a.evidence_id}: {a.label.value}"
            + (f" - {a.explanation}" if a.explanation else "")
        )
    io.say("  The original record is kept unchanged; your correction becomes a new record.")


def _show_summary(state: _State, claim: Claim, v: Verification, warnings: list[str]) -> None:
    io = state.io
    io.say("\n" + "-" * WIDTH)
    io.say("  REVIEW SUMMARY (not yet saved)")
    io.say(_wrap(f"Claim: {claim.normalized_claim}", "  "))
    io.say(f"  Verdict: {v.verdict.value}")
    evidence = {e.evidence_id: e for e in state.session.evidence}
    for role, ids in (
        ("supporting", v.supporting_evidence_ids),
        ("contradicting", v.contradicting_evidence_ids),
    ):
        for eid in ids:
            ev = evidence[eid]
            io.say(f"  Cited ({role}): {ev.source_title} {ev.document_reference or ''}")
    counts = {
        lbl.value: sum(a.label is lbl for a in v.evidence_assessments) for lbl in EvidenceLabel
    }
    io.say(f"  Passages assessed: {counts}")
    io.say(_wrap(f"Rationale: {v.reasoning}", "  "))
    if v.revision_of:
        io.say(f"  Revises: {v.revision_of}; reason: {v.revision_reason}")
        if v.revision_authorization:
            io.say(f"  Authorization: {v.revision_authorization}")
    for note in v.noted_limitations:
        io.say(f"  Limitation: {note}")
    for w in warnings:
        io.say(f"  Rationale warning (accepted by you): {w}")
    io.say(f"  Reviewer: {v.reviewer_id}   time: {v.verification_timestamp:%Y-%m-%d %H:%M:%S} UTC")


def _review_claim(
    state: _State,
    claim: Claim,
    position: str,
    previous: Verification | None,
    authorization: str | None,
) -> bool:
    """Review one claim. Returns True if a verification was saved."""
    _show_claim(state, claim, position)
    if previous:
        _show_previous(state, previous)
    checks = check_session_evidence(state.session, state.catalog, claim.claim_id)
    blocked = [c for c in checks.values() if c.blocks_citation]
    if blocked:
        state.io.say(
            f"  WARNING: {len(blocked)} passage(s) have sources that changed or "
            "disappeared since retrieval. They cannot be cited until you re-run "
            "`hallab retrieve` for this session."
        )
    if state.session.latest_retrieval(claim.claim_id) is None and not _yes(
        state,
        "  Review without evidence? Only INSUFFICIENT_EVIDENCE or NOT_CHECKABLE will "
        "be possible. [y/N] ",
        default=False,
    ):
        return False

    now = state.now()
    draft = state.session.get_draft(claim.claim_id, state.reviewer_id)
    wanted_revision = previous.verification_id if previous else None
    if draft is not None:
        if draft.revision_of != wanted_revision:
            state.io.say(
                "  You have a draft for this claim from a different task "
                f"(revision_of={draft.revision_of})."
            )
            if not _yes(state, "  Discard that draft and start fresh? [y/N] ", default=False):
                return False
            draft = None
        elif _yes(
            state,
            f"  Resume your draft from {draft.updated_at:%Y-%m-%d %H:%M}? [Y/n] ",
            default=True,
        ):
            state.io.say("  Resuming draft.")
        else:
            draft = None
        if draft is None:
            state.session = discard_draft(state.session, claim.claim_id, state.reviewer_id)
            _save(state)
    if draft is None:
        # A revision starts from the previous record's passage assessments; the
        # reviewer confirms or changes each one. The previous record is untouched.
        draft = VerificationDraft(
            claim_id=claim.claim_id,
            reviewer_id=state.reviewer_id,
            started_at=now,
            updated_at=now,
            revision_of=wanted_revision,
            evidence_assessments=list(previous.evidence_assessments) if previous else [],
        )
        draft = _save_draft(state, draft)

    if previous:
        draft = _ask_revision_reason(state, draft)
    draft = _assess_passages(state, claim, draft, checks)
    while True:
        draft = _choose_verdict(state, draft, checks)
        draft = _write_rationale(state, claim, draft)
        try:
            verification, warnings = build_verification(
                state.session,
                state.reviewer_id,
                VerificationInput(
                    claim_id=claim.claim_id,
                    verdict=draft.verdict,
                    rationale=draft.rationale,
                    assessments=draft.evidence_assessments,
                    supporting_evidence_ids=draft.supporting_evidence_ids,
                    contradicting_evidence_ids=draft.contradicting_evidence_ids,
                    revision_of=draft.revision_of,
                    revision_reason=draft.revision_reason,
                    revision_authorization=authorization,
                ),
                checks=checks,
                now=state.now(),
            )
        except VerificationError as exc:
            state.io.say(f"  Cannot record this verification: {exc}")
            state.io.say("  Choose the verdict again (your draft is kept).")
            continue
        _show_summary(state, claim, verification, warnings)
        choice = _ask(
            state,
            "  Save this verification? [y = save, e = edit verdict/rationale, "
            "n = keep as draft and move on]: ",
        ).lower()
        if choice in {"y", "yes"}:
            state.session = record_verification(state.session, verification)
            _save(state)
            state.saved.append(verification)
            state.io.say(f"  Saved {verification.verification_id}.")
            return True
        if choice == "q":
            raise QuitRequested
        if choice in {"e", "edit"}:
            continue
        state.io.say("  Not saved. Your answers are kept as a draft.")
        return False


def run_verification(
    store: JsonSessionStore,
    session_id: str,
    reviewer_id: str,
    *,
    catalog: CollectionCatalog | None = None,
    prompter: Prompter | None = None,
    claim_ids: list[str] | None = None,
    revise: str | None = None,
    authorization: str | None = None,
    now: Callable[[], datetime] = utc_now,
) -> list[Verification]:
    """Run the interactive review. Returns the verifications saved in this run."""
    if not reviewer_id or not reviewer_id.strip():
        raise VerificationError("--reviewer-id is required")
    session, version = store.load_versioned(session_id)
    io = prompter or Prompter()
    state = _State(store, session, version, reviewer_id.strip(), catalog, io, now)

    previous = None
    if revise:
        previous = check_revision_allowed(session, revise, state.reviewer_id, authorization)
        claim = session.get_claim(previous.claim_id)
        if not claim.is_active:
            raise VerificationError("the claim of this verification is no longer active")
        queue = [claim]
    elif claim_ids:
        queue = []
        for cid in claim_ids:
            try:
                claim = session.get_claim(cid)
            except KeyError as exc:
                raise VerificationError(f"claim {cid} is not in this session") from exc
            if session.human_verdict(cid) is not None:
                active = session.human_verdict(cid)
                raise VerificationError(
                    f"claim {cid} already has active verification {active.verification_id}; "
                    f"use --revise {active.verification_id}"
                )
            queue.append(claim)
    else:
        queue = claims_awaiting_review(session)
        drafted = {
            d.claim_id
            for d in session.verification_drafts
            if d.reviewer_id == state.reviewer_id and d.revision_of is None
        }
        queue.sort(key=lambda c: c.claim_id not in drafted)  # resume drafts first

    io.say("HUMAN VERIFICATION. You decide every verdict; retrieval only suggested passages.")
    io.say(f"Session {session.session_id}: {session.title}")
    if not queue:
        io.say(
            "Nothing to review: every claim flagged for verification has an active human "
            "verdict. Use `hallab verify status` or `--revise VERIFICATION_ID`."
        )
        return []
    io.say(f"{len(queue)} claim(s) to review. Type ? at any prompt for help, q to pause.")
    try:
        for n, claim in enumerate(queue, 1):
            _review_claim(state, claim, f"{n} of {len(queue)}", previous, authorization)
            if n < len(queue) and not _yes(
                state, "\nContinue to the next claim? [Y/n] ", default=True
            ):
                break
    except (QuitRequested, KeyboardInterrupt):
        io.say("\nPaused. Your progress is saved as a draft; run the same command to resume.")
    io.say(f"Saved {len(state.saved)} verification(s) in this run.")
    return state.saved
