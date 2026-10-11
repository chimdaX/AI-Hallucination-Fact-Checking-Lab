"""Command-line interface (``hallab``).

Commands:
  hallab extract --text "..." | --file response.txt [--json]
  hallab demo [--save]
  hallab sessions list
  hallab sessions show SESSION_ID

Retrieval commands (sources, index, search, retrieve, eval) live in cli_retrieval.py;
human verification commands (verify, verify status, verify summary) in cli_verify.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from hallucination_lab import __version__, cli_retrieval, cli_verify
from hallucination_lab.config import Settings
from hallucination_lab.errors import HallucinationLabError
from hallucination_lab.examples import load_examples
from hallucination_lab.logging_utils import configure_logging
from hallucination_lab.models import AIResponse, Claim, ResponseProvenance
from hallucination_lab.sessions import new_session
from hallucination_lab.storage import JsonSessionStore, summarize_validation_error


def _format_claim(n: int, claim: Claim) -> str:
    if claim.requires_verification:
        status = "CHECK"
    else:
        reason = claim.non_checkable_reason.value if claim.non_checkable_reason else "reviewer"
        status = f"SKIP ({reason})"
    lines = [f"{n:>3}. [{status}] {claim.normalized_claim}"]
    f = claim.features
    tags = [
        f"{label}: {', '.join(values)}"
        for label, values in [
            ("numbers", f.numbers),
            ("dates", f.dates),
            ("measurements", f.measurements),
            ("entities", f.named_entities),
            ("citations", f.citations),
            ("quotes", f.quotations),
            ("temporal", f.temporal_markers),
        ]
        if values
    ]
    if tags:
        lines.append("       " + " | ".join(tags))
    return "\n".join(lines)


def _print_claims(claims: list[Claim], as_json: bool) -> None:
    if as_json:
        print(json.dumps([c.model_dump(mode="json") for c in claims], indent=2))
        return
    for n, claim in enumerate(claims, 1):
        print(_format_claim(n, claim))


def cmd_extract(args: argparse.Namespace) -> int:
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    provenance = (
        ResponseProvenance.SYNTHETIC_FIXTURE
        if args.synthetic
        else ResponseProvenance.USER_SUBMITTED
    )
    response = AIResponse.from_text(text, provenance)
    session = new_session(response)
    _print_claims(session.claims, args.json)
    if args.save:
        path = JsonSessionStore(Settings.from_env().sessions_dir).save(session)
        print(f"\nSaved session {session.session_id} -> {path}", file=sys.stderr)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    dataset = load_examples()
    store = JsonSessionStore(Settings.from_env().sessions_dir)
    print(f"{dataset.dataset}: {len(dataset.items)} synthetic fixtures (not real model outputs)\n")
    for item in dataset.items:
        session = new_session(item.response, title=f"[fixture] {item.topic}")
        checkable = sum(c.requires_verification for c in session.claims)
        print(f"== {item.topic} ({len(session.claims)} candidates, {checkable} to check)")
        _print_claims(session.claims, as_json=False)
        if args.save:
            print(f"   saved -> {store.save(session)}")
        print()
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    store = JsonSessionStore(Settings.from_env().sessions_dir)
    if args.action == "list":
        summaries, problems = store.list_sessions()
        if not summaries:
            print(f"No sessions in {store.root}")
        for s in summaries:
            print(
                f"{s.session_id}  {s.updated_at:%Y-%m-%d %H:%M}  claims={s.claim_count:<3} "
                f"pending={s.pending_reviews:<3} {s.title}"
            )
        for problem in problems:
            print(f"WARNING: {problem}", file=sys.stderr)
        return 0
    session = store.load(args.session_id)
    print(
        f"{session.title}\nresponse: {session.response.response_id} "
        f"({session.response.provenance.value})\n"
    )
    _print_claims(session.active_claims, as_json=False)
    if session.retrieval_runs:
        print(f"\n{cli_retrieval.RETRIEVAL_BANNER}\n")
        cli_retrieval._print_session_evidence(session)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hallab", description="AI Hallucination Detection & Fact-Checking Lab"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING (default)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="extract candidate claims from text")
    src = p_extract.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="response text")
    src.add_argument("--file", help="path to a UTF-8 text file")
    p_extract.add_argument("--json", action="store_true", help="print claims as JSON")
    p_extract.add_argument("--save", action="store_true", help="save as a new session")
    p_extract.add_argument(
        "--synthetic",
        action="store_true",
        help="label the text as an illustrative synthetic fixture (not a real model output)",
    )
    p_extract.set_defaults(func=cmd_extract)

    p_demo = sub.add_parser("demo", help="run extraction on the bundled synthetic fixtures")
    p_demo.add_argument("--save", action="store_true", help="save each fixture as a session")
    p_demo.set_defaults(func=cmd_demo)

    p_sessions = sub.add_parser("sessions", help="inspect saved sessions")
    s_sub = p_sessions.add_subparsers(dest="action", required=True)
    s_sub.add_parser("list")
    p_show = s_sub.add_parser("show")
    p_show.add_argument("session_id")
    p_sessions.set_defaults(func=cmd_sessions)

    cli_retrieval.register(sub)
    cli_verify.register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        return args.func(args)
    except HallucinationLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(f"error: invalid input: {summarize_validation_error(exc)}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
