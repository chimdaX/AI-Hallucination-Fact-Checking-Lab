"""CLI commands for reports and the web interface.

hallab report --session ID [--format json|csv|md] [--output PATH]
hallab ui [--port 8501]
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from hallucination_lab.errors import HallucinationLabError
from hallucination_lab.services.reports import build_report, export
from hallucination_lab.services.workspace import Workspace

APP_PATH = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"


def cmd_report(args: argparse.Namespace) -> int:
    workspace = Workspace.from_env()
    session = workspace.load(args.session).session
    report = build_report(session, workspace.evidence_checks(session))
    text = export(report, args.format)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        print(f"Wrote {args.format.upper()} report to {out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("streamlit") is None:
        raise HallucinationLabError(
            'the web interface needs Streamlit: pip install -e ".[ui]"  (CLI features work '
            "without it)"
        )
    if not APP_PATH.is_file():
        raise HallucinationLabError(f"app not found at {APP_PATH}; run from a source checkout")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(args.port),
    ]
    if args.headless:
        command += ["--server.headless", "true"]
    return subprocess.call(command)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("report", help="export a verification report (json, csv, md)")
    p.add_argument("--session", required=True)
    p.add_argument("--format", choices=["json", "csv", "md"], default="md")
    p.add_argument("--output", help="file to write (default: print to stdout)")
    p.set_defaults(func=cmd_report)

    p_ui = sub.add_parser("ui", help="start the Streamlit web interface (needs the [ui] extra)")
    p_ui.add_argument("--port", type=int, default=8501)
    p_ui.add_argument("--headless", action="store_true", help="do not open a browser")
    p_ui.set_defaults(func=cmd_ui)
