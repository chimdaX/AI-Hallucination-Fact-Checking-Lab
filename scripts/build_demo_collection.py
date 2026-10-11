"""Capture the packaged ``demo`` reference collection from its original sources.

Usage:
    python scripts/build_demo_collection.py --cache DIR   # reuse previously downloaded files
    python scripts/build_demo_collection.py --download DIR  # download into DIR, then build

What it does, per source:

* Wikipedia / Wikiquote: one MediaWiki API request returns the page's
  plain-text extract (TextExtracts, ``explaintext``) **and** the ID and
  timestamp of the revision it was produced from. The extract is written
  verbatim. Infoboxes, tables, references and some formatting are not part of
  TextExtracts output.
* NASA fact sheet: the HTML page is converted to text by this script. Table
  cells are joined with " | ", superscripts become "^", and the footer contact
  block (staff email addresses) is omitted.

Re-running with ``--download`` captures whatever revisions are current at that
time. The committed files in ``data/references/demo`` are the canonical
snapshot, and each document's metadata pins its exact revision.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

from hallucination_lab.models import SourceOrigin
from hallucination_lab.retrieval.ingest import load_document, new_document_entry
from hallucination_lab.retrieval.models import (
    CollectionManifest,
    DocumentType,
    MetadataStatus,
    SourceMetadata,
)
from hallucination_lab.storage import write_json_atomic

USER_AGENT = "ai-hallucination-verification-lab/0.1 (open-source research tool; build-time capture)"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "references" / "demo"

WIKI_SOURCES = [
    # (api host, page title, cache file, output file, document type)
    ("en.wikipedia.org", "Eiffel_Tower", "wp_Eiffel_Tower.json", "wikipedia_eiffel_tower.txt"),
    ("en.wikipedia.org", "Mount_Everest", "wp_Mount_Everest.json", "wikipedia_mount_everest.txt"),
    (
        "en.wikipedia.org",
        "Properties_of_water",
        "wp_Properties_of_water.json",
        "wikipedia_properties_of_water.txt",
    ),
    ("en.wikipedia.org", "Sunlight", "wp_Sunlight.json", "wikipedia_sunlight.txt"),
    ("en.wikipedia.org", "Leap_year", "wp_Leap_year.json", "wikipedia_leap_year.txt"),
    ("en.wikipedia.org", "Pi", "wp_Pi.json", "wikipedia_pi.txt"),
    (
        "en.wikipedia.org",
        "Python_(programming_language)",
        "wp_Python_(programming_language).json",
        "wikipedia_python_programming_language.txt",
    ),
    ("en.wikipedia.org", "Burj_Khalifa", "wp_Burj_Khalifa.json", "wikipedia_burj_khalifa.txt"),
    (
        "en.wikiquote.org",
        "Narcotics_Anonymous",
        "wq_Narcotics_Anonymous.json",
        "wikiquote_narcotics_anonymous.txt",
    ),
]
NASA_URL = "https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html"
NASA_CACHE = "nasa_mars.html"


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        dest.write_bytes(response.read())


def _wiki_api_url(host: str, title: str) -> str:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts|revisions|info",
        "inprop": "url",
        "explaintext": "1",
        "exsectionformat": "wiki",
        "rvprop": "ids|timestamp",
        "titles": title,
    }
    return f"https://{host}/w/api.php?{urllib.parse.urlencode(params)}"


class _FactSheetText(HTMLParser):
    BLOCK: ClassVar[frozenset[str]] = frozenset(
        {"p", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "div", "li", "table", "hr"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = [""]
        self.skip = 0
        self.cells_in_row = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
        elif tag in self.BLOCK:
            self.lines.append("")
            self.cells_in_row = 0
        elif tag in {"td", "th"}:
            if self.cells_in_row:
                self.lines[-1] += " | "
            self.cells_in_row += 1
        elif tag == "sup":
            self.lines[-1] += "^"

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.lines[-1] += data


def _convert_fact_sheet(html: str) -> str:
    parser = _FactSheetText()
    parser.feed(html)
    out = []
    for line in parser.lines:
        line = re.sub(r"\s+", " ", line.replace("\xa0", " ")).strip(" |")
        line = re.sub(r"\^\s+", "^", line)
        if line.startswith("Author/Curator"):
            break  # footer with staff contact details is intentionally omitted
        if line:
            out.append(line)
    # Blank line between rows so each table row is its own paragraph.
    return "\n\n".join(out) + "\n"


def build(cache: Path, accessed: date) -> None:
    docs_dir = OUT_DIR / "documents"
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    docs_dir.mkdir(parents=True)
    entries = []
    now = datetime.now(timezone.utc)

    for host, _title, cache_name, out_name in WIKI_SOURCES:
        page = json.loads((cache / cache_name).read_text(encoding="utf-8"))["query"]["pages"][0]
        revision = page["revisions"][0]
        rev_ts = datetime.fromisoformat(revision["timestamp"].replace("Z", "+00:00"))
        (docs_dir / out_name).write_text(page["extract"] + "\n", encoding="utf-8", newline="\n")
        project = "Wikiquote" if "wikiquote" in host else "Wikipedia"
        page_path = urllib.parse.quote(page["title"].replace(" ", "_"))
        base = f"https://{host}/w/index.php?title={page_path}"
        doc_type = (
            DocumentType.QUOTATION_COMPILATION
            if project == "Wikiquote"
            else DocumentType.ENCYCLOPEDIA_ARTICLE
        )
        metadata = SourceMetadata(
            title=f"{page['title']} ({project})",
            author_or_organization=f"{project} contributors",
            publisher="Wikimedia Foundation",
            last_updated=rev_ts.date(),
            url=f"{base}&oldid={revision['revid']}",
            accessed_on=accessed,
            document_type=doc_type,
            license="CC BY-SA 4.0",
            license_url="https://creativecommons.org/licenses/by-sa/4.0/",
            source_version=f"revision {revision['revid']} ({revision['timestamp']})",
            status=MetadataStatus.CAPTURED_AT_DOWNLOAD,
            status_note=(
                f"Title, revision ID and revision timestamp recorded from the MediaWiki API "
                f"response on {accessed.isoformat()}; the text is the API's plain-text "
                "extract of that revision."
            ),
            notes=(
                f"Tertiary source (community-edited {project}). Plain-text extract: "
                "infoboxes, tables, citations and some formatting are not included. "
                "Article history: "
                f"{base}&action=history"
            ),
        )
        loaded = load_document(docs_dir / out_name)
        entries.append(
            new_document_entry(loaded, out_name, metadata, SourceOrigin.PACKAGED_DEMO, now=now)
        )

    nasa_text = _convert_fact_sheet((cache / NASA_CACHE).read_text(encoding="utf-8"))
    match = re.search(r"Last Updated: (\d{1,2} \w+ \d{4})", (cache / NASA_CACHE).read_text())
    last_updated = datetime.strptime(match.group(1), "%d %B %Y").date() if match else None
    out_name = "nasa_mars_fact_sheet.txt"
    (docs_dir / out_name).write_text(nasa_text, encoding="utf-8", newline="\n")
    metadata = SourceMetadata(
        title="Mars Fact Sheet",
        author_or_organization="Dr. David R. Williams (Author/Curator), NSSDCA, "
        "NASA Goddard Space Flight Center",
        publisher="NASA Space Science Data Coordinated Archive (NSSDCA)",
        last_updated=last_updated,
        url=NASA_URL,
        accessed_on=accessed,
        document_type=DocumentType.FACT_SHEET,
        license="U.S. Government work (NASA); believed public domain in the U.S. "
        "- requires manual confirmation",
        license_url="https://www.nasa.gov/nasa-brand-center/images-and-media/",
        source_version=f"page last updated {match.group(1)}" if match else None,
        status=MetadataStatus.CAPTURED_AT_DOWNLOAD,
        status_note=(
            f"Author/curator and 'Last Updated' date copied from the page footer on "
            f"{accessed.isoformat()}. Licence status inferred from NASA's media usage "
            "guidelines and not independently confirmed."
        ),
        notes=(
            "Primary/official data source. Converted from HTML by "
            "scripts/build_demo_collection.py: table cells joined with ' | ', superscripts "
            "written as '^', footer contact block omitted."
        ),
    )
    loaded = load_document(docs_dir / out_name)
    entries.append(
        new_document_entry(loaded, out_name, metadata, SourceOrigin.PACKAGED_DEMO, now=now)
    )

    manifest = CollectionManifest(
        name="demo",
        description=(
            "Packaged offline demonstration collection. Real source texts captured from "
            "Wikipedia/Wikiquote (CC BY-SA 4.0, pinned revisions) and a NASA fact sheet. "
            "See CONTENT_LICENSE.md. Retrieved passages are candidates, not verified facts."
        ),
        origin=SourceOrigin.PACKAGED_DEMO,
        created_at=now,
        documents=entries,
    )
    write_json_atomic(OUT_DIR / "collection.json", manifest)
    print(f"wrote {len(entries)} documents to {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cache", type=Path, help="directory with previously downloaded files")
    group.add_argument("--download", type=Path, help="download sources into this directory")
    parser.add_argument(
        "--accessed",
        type=date.fromisoformat,
        default=None,
        help="access date to record (default: today)",
    )
    args = parser.parse_args()
    cache = args.cache or args.download
    if args.download:
        cache.mkdir(parents=True, exist_ok=True)
        for host, title, cache_name, _ in WIKI_SOURCES:
            _download(_wiki_api_url(host, title), cache / cache_name)
        _download(NASA_URL, cache / NASA_CACHE)
    build(cache, args.accessed or date.today())


if __name__ == "__main__":
    main()
