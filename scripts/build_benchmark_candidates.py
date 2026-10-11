"""Create the benchmark's candidate claims and evidence (AI-authored, provisional).

Usage:
    python scripts/build_benchmark_candidates.py --cache DIR      # reuse downloaded sources
    python scripts/build_benchmark_candidates.py --download DIR   # fetch sources into DIR first

What this script does and does not do:

* Every claim, proposed label and evidence choice here was written by an AI coding
  assistant (``claude-assistant``). Each is recorded as such and is NOT ground truth.
* Evidence passages are captured verbatim from real sources: the packaged ``demo``
  collection, Wikipedia revisions via the MediaWiki API, and official web pages. Each
  passage is stored with its SHA-256 and access date. Passage text is stored only where
  the licence allows redistribution; otherwise only a locator and hash are kept.
* Sources that refused automated access (HTTP 403) are recorded as research tasks, not
  as evidence.
* Searches for possibly fabricated studies are documented as search records (Crossref
  queries with their results), not as proof that something does not exist.
* Existing items are never modified. The script only adds missing candidates, so
  re-running it cannot overwrite human annotations.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

from hallucination_lab.benchmark.capture import (
    HTML_TEXT_VERSION,
    extract_sentence,
    extract_span,
    html_to_text,
)
from hallucination_lab.benchmark.ids import evidence_id_for, item_id_for, reference_id_for
from hallucination_lab.benchmark.models import (
    ApprovalPolicy,
    AuthorType,
    BenchmarkManifest,
    CandidateProvenance,
    CaptureStatus,
    ClaimType,
    Difficulty,
    Domain,
    EvidenceReference,
    ProposedLabel,
    Redistribution,
    ReferenceSource,
    ResearchTask,
    ReviewStatus,
    SearchRecord,
    SourceType,
    SplitRecommendation,
)
from hallucination_lab.benchmark.store import BenchmarkStore
from hallucination_lab.benchmark.workflow import add_candidate, set_status
from hallucination_lab.models import EvidenceLabel, Verdict
from hallucination_lab.retrieval.ingest import load_document
from hallucination_lab.retrieval.models import CollectionManifest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "references" / "demo"
AUTHOR = "claude-assistant"
CAPTURED_BY = "claude-assistant via scripts/build_benchmark_candidates.py"
USER_AGENT = "ai-hallucination-verification-lab/0.1 (open-source research tool; benchmark capture)"

SUP, CON, IRR, UNC = (
    EvidenceLabel.SUPPORTING,
    EvidenceLabel.CONTRADICTING,
    EvidenceLabel.IRRELEVANT,
    EvidenceLabel.UNCERTAIN,
)
D, DIFF, T = Domain, Difficulty, ClaimType

# ------------------------------------------------------------------------- sources

WIKI = {  # key -> (cache file, page title)
    "wp_euclid": ("wp_Euclids_theorem.json", "Euclid's_theorem"),
    "wp_sqrt2": ("wp_Square_root_of_.json", "Square_root_of_2"),
    "wp_brain": ("wp_Ten_percent_of_the_brain_myth.json", "Ten_percent_of_the_brain_myth"),
    "wp_cogito": ("wp_Cogito_ergo_sum.json", "Cogito,_ergo_sum"),
    "wp_canberra": ("wp_Canberra.json", "Canberra"),
    "wp_nile": ("wp_Nile.json", "Nile"),
    "wp_amazon": ("wp_Amazon_River.json", "Amazon_River"),
    "wp_cake": ("wp_Let_them_eat_cake.json", "Let_them_eat_cake"),
    "wp_greatwall": ("wp_Great_Wall_of_China.json", "Great_Wall_of_China"),
    "wp_abszero": ("wp_Absolute_zero.json", "Absolute_zero"),
    "wp_antibiotic": ("wp_Antibiotic.json", "Antibiotic"),
    "wp_alexandria": ("wp_Library_of_Alexandria.json", "Library_of_Alexandria"),
}

HTML = {
    "nist_c": dict(
        cache="nist_c.html",
        url="https://physics.nist.gov/cgi-bin/cuu/Value?c",
        title="CODATA value: speed of light in vacuum",
        org="National Institute of Standards and Technology (NIST)",
        publisher="NIST",
        version="2022 CODATA recommended values (as stated on the page)",
        source_type=SourceType.OFFICIAL_INSTITUTION,
        license="U.S. Government work; believed not subject to copyright in the U.S. "
        "(requires manual confirmation)",
        redistribution=Redistribution.PUBLIC_DOMAIN_TEXT_STORED,
        reliability="Primary authoritative source for fundamental physical constants.",
    ),
    "who_smallpox": dict(
        cache="who_smallpox.html",
        url="https://www.who.int/health-topics/smallpox",
        title="Smallpox (health topic)",
        org="World Health Organization",
        publisher="WHO",
        version=None,
        source_type=SourceType.OFFICIAL_INSTITUTION,
        license="WHO website terms; redistribution status not confirmed",
        redistribution=Redistribution.LOCATOR_ONLY,
        reliability="Official source of the organisation that declared eradication.",
    ),
    "fed_faq": dict(
        cache="fed_faq.html",
        url="https://www.federalreserve.gov/faqs/about_12594.htm",
        title="What is the purpose of the Federal Reserve System? (FAQ)",
        org="Board of Governors of the Federal Reserve System",
        publisher="Federal Reserve",
        version=None,
        source_type=SourceType.OFFICIAL_INSTITUTION,
        license="Licence status not confirmed",
        redistribution=Redistribution.LOCATOR_ONLY,
        reliability="Official institutional source about its own history.",
    ),
    "python_license": dict(
        cache="python_license.html",
        url="https://docs.python.org/3/license.html",
        title="History and License (Python documentation)",
        org="Python Software Foundation",
        publisher="Python Software Foundation",
        version="Python 3 documentation page as served on the access date",
        source_type=SourceType.TECHNICAL_DOCUMENTATION,
        license="Python documentation is distributed under the PSF License "
        "(requires manual confirmation of terms for excerpts)",
        redistribution=Redistribution.OPEN_LICENSE_TEXT_STORED,
        reliability="Primary documentation maintained by the language's steward.",
    ),
    "un_population": dict(
        cache="un_population.html",
        url="https://www.un.org/en/global-issues/population",
        title="Global Issues: Population",
        org="United Nations",
        publisher="United Nations",
        version=None,
        source_type=SourceType.OFFICIAL_INSTITUTION,
        license="UN website terms; redistribution status not confirmed",
        redistribution=Redistribution.LOCATOR_ONLY,
        reliability="Official UN overview; figures come from UN DESA population estimates.",
    ),
}

# ---------------------------------------------------------------------- searches

SEARCHES = {  # key -> (cache file, service, query, request URL, what was looked for)
    "cr_journal_coastal": (
        "crossref_journal.json",
        "Crossref API (journals)",
        "Journal of Coastal Futures",
        "https://api.crossref.org/journals?query=Journal+of+Coastal+Futures&rows=5",
        "a journal named 'Journal of Coastal Futures'",
    ),
    "cr_works_hartwell": (
        "crossref_works.json",
        "Crossref API (works)",
        "Hartwell 2019 coastal cities underwater 2050",
        "https://api.crossref.org/works?query.bibliographic=Hartwell+2019+coastal+cities+"
        "underwater+2050&rows=5",
        "a 2019 study by Hartwell et al. on coastal flooding",
    ),
    "cr_journal_gorc": (
        "crossref_gorc_journal.json",
        "Crossref API (journals)",
        "Global Ocean Restoration Council",
        "https://api.crossref.org/journals?query=Global+Ocean+Restoration+Council&rows=5",
        "a publication venue named 'Global Ocean Restoration Council'",
    ),
    "cr_works_coral": (
        "crossref_coral.json",
        "Crossref API (works)",
        "Global Ocean Restoration Council 2021 coral cover increased 40 percent",
        "https://api.crossref.org/works?query.bibliographic=Global+Ocean+Restoration+Council+"
        "2021+coral+cover+increased+40+percent&rows=5",
        "a 2021 report finding a 40% worldwide increase in coral cover",
    ),
    "cr_works_tea": (
        "crossref_tea.json",
        "Crossref API (works)",
        "2017 survey Canadians prefer tea over coffee 64 percent",
        "https://api.crossref.org/works?query.bibliographic=2017+survey+Canadians+prefer+tea+"
        "over+coffee+64+percent&rows=5",
        "a 2017 survey of Canadians' tea/coffee preference",
    ),
    "cr_works_houseplants": (
        "crossref_houseplants.json",
        "Crossref API (works)",
        "2018 study houseplants reduce office stress 37 percent",
        "https://api.crossref.org/works?query.bibliographic=2018+study+houseplants+reduce+"
        "office+stress+37+percent&rows=5",
        "a 2018 study reporting a 37% stress reduction",
    ),
}


def _fixture_text(topic_prefix: str) -> str:
    data = json.loads((ROOT / "data" / "examples" / "synthetic_responses.json").read_text())
    for item in data["items"]:
        if item["topic"].startswith(topic_prefix):
            return item["response"]["text"]
    raise KeyError(topic_prefix)


def ev(source, anchor, relation, end=None, note=None):
    return dict(source=source, anchor=anchor, end=end, relation=relation, note=note)


def task(text, reason):
    return dict(task=text, reason=reason)


BLOCKED = "returned HTTP 403 to automated access on {date}; not bypassed"

# --------------------------------------------------------------------------- claims
# proposed = (verdict, rationale). Proposed labels are AI suggestions for sampling
# balance and review triage only; they are hidden from reviewers by default.

CLAIMS = [
    # ---------------------------------------------------------------- SUPPORTED
    dict(
        text="Pi is approximately 3.14159.",
        domain=D.MATHEMATICS,
        diff=DIFF.EASY,
        types=[T.CORRECT_ASSERTION],
        context="Mathematics and calendar facts",
        proposed=(Verdict.SUPPORTED, "Encyclopedic definition gives pi as approximately 3.14159."),
        evidence=[ev("demo:wikipedia_pi.txt", "approximately equal to 3.14159", SUP)],
    ),
    dict(
        text="The chemical formula of water is H2O, and each water molecule contains two "
        "hydrogen atoms and one oxygen atom.",
        domain=D.PHYSICAL_SCIENCES,
        diff=DIFF.EASY,
        types=[T.CORRECT_ASSERTION, T.COMPOUND],
        context="Science: water and light",
        split=(
            [
                "The chemical formula of water is H2O.",
                "Each water molecule contains two hydrogen atoms and one oxygen atom.",
            ],
            "Two propositions; both supported by the same source, but split if reviewers "
            "need per-proposition labels.",
        ),
        proposed=(Verdict.SUPPORTED, "Both parts are stated in the water article."),
        evidence=[ev("demo:wikipedia_properties_of_water.txt", "chemical formula H2O", SUP)],
    ),
    dict(
        text="The speed of light in vacuum is exactly 299,792,458 metres per second.",
        domain=D.PHYSICAL_SCIENCES,
        diff=DIFF.EASY,
        types=[T.CORRECT_ASSERTION],
        proposed=(Verdict.SUPPORTED, "NIST lists 299 792 458 m s-1 with exact uncertainty."),
        evidence=[ev("nist_c", "speed of light in vacuum\nNumerical value", SUP, end="(exact)")],
    ),
    dict(
        text="Python was created by Guido van Rossum as a successor to the ABC language.",
        domain=D.TECHNOLOGY,
        diff=DIFF.EASY,
        types=[T.CORRECT_ASSERTION],
        proposed=(Verdict.SUPPORTED, "Python's own documentation states both facts."),
        evidence=[
            ev(
                "python_license",
                "Python was created in the early 1990s",
                SUP,
                end="successor of a language called ABC.",
            ),
            ev("demo:wikipedia_python_programming_language.txt", "as a successor to the ABC", SUP),
        ],
    ),
    dict(
        text="A year on Mars lasts about 687 Earth days.",
        domain=D.PHYSICAL_SCIENCES,
        diff=DIFF.MEDIUM,
        types=[T.CORRECT_ASSERTION],
        proposed=(Verdict.SUPPORTED, "NASA fact sheet: sidereal orbit period 686.980 days."),
        evidence=[
            ev(
                "demo:nasa_mars_fact_sheet.txt",
                "Sidereal orbit period (days) | 686.980",
                SUP,
                note="Vocabulary differs from the claim ('orbit period' vs 'year').",
            )
        ],
    ),
    dict(
        text="Sunlight takes about eight minutes to reach Earth.",
        domain=D.PHYSICAL_SCIENCES,
        diff=DIFF.EASY,
        types=[T.CORRECT_ASSERTION],
        context="Science: water and light",
        proposed=(Verdict.SUPPORTED, "Source gives about 8.3 minutes; 'about eight' fits."),
        evidence=[
            ev(
                "demo:wikipedia_sunlight.txt",
                "Sunlight takes about 8.3 minutes to reach Earth",
                SUP,
            )
        ],
    ),
    dict(
        text="Edmund Hillary and Tenzing Norgay made the first documented ascent of Mount "
        "Everest in 1953.",
        domain=D.HISTORY,
        diff=DIFF.EASY,
        types=[T.CORRECT_ASSERTION],
        context="Geography: Mount Everest",
        proposed=(Verdict.SUPPORTED, "Stated directly in the Everest article."),
        evidence=[
            ev(
                "demo:wikipedia_mount_everest.txt",
                "made the first documented ascent of Everest in 1953",
                SUP,
            )
        ],
    ),
    dict(
        text="The Eiffel Tower was completed in 1889.",
        domain=D.HISTORY,
        diff=DIFF.EASY,
        types=[T.CORRECT_ASSERTION],
        proposed=(Verdict.SUPPORTED, "Article: built from 1887 to 1889; completed in 1889."),
        evidence=[
            ev("demo:wikipedia_eiffel_tower.txt", "tallest structure when completed in 1889", SUP),
            ev(
                "demo:wikipedia_eiffel_tower.txt",
                "designed and built the tower from 1887 to 1889",
                SUP,
            ),
        ],
        tasks=[
            task(
                "Capture the official history page (toureiffel.paris) as a primary source",
                "official site " + BLOCKED,
            )
        ],
    ),
    dict(
        text="There are infinitely many prime numbers.",
        domain=D.MATHEMATICS,
        diff=DIFF.EASY,
        types=[T.CORRECT_ASSERTION],
        proposed=(Verdict.SUPPORTED, "Euclid's theorem."),
        evidence=[ev("wp_euclid", "asserts that there are infinitely many prime numbers", SUP)],
    ),
    dict(
        text="Absolute zero is equivalent to -273.15 degrees Celsius.",
        domain=D.PHYSICAL_SCIENCES,
        diff=DIFF.EASY,
        types=[T.CORRECT_ASSERTION],
        proposed=(Verdict.SUPPORTED, "Kelvin scale definition: 0 K = -273.15 °C."),
        evidence=[
            ev(
                "wp_abszero",
                "equivalent to \u2212273.15 °C",
                SUP,
                note="Source uses the Unicode minus sign.",
            )
        ],
        tasks=[
            task(
                "Add a primary metrology source (e.g. the BIPM SI Brochure)",
                "evidence is currently an encyclopedia only",
            )
        ],
    ),
    dict(
        text="The World Health Organization declared smallpox eradicated in 1980.",
        domain=D.PUBLIC_HEALTH,
        diff=DIFF.EASY,
        types=[T.CORRECT_ASSERTION],
        proposed=(Verdict.SUPPORTED, "WHO page: 'In 1980 WHO declared smallpox eradicated'."),
        evidence=[ev("who_smallpox", "In 1980 WHO declared", SUP, end="achieve this distinction.")],
    ),
    dict(
        text="Antibiotics are not effective against viral infections.",
        domain=D.PUBLIC_HEALTH,
        diff=DIFF.EASY,
        types=[T.CORRECT_ASSERTION],
        proposed=(Verdict.SUPPORTED, "Antibiotics act on bacteria, not viruses."),
        evidence=[ev("wp_antibiotic", "antibiotics are not effective against viruses", SUP)],
        tasks=[
            task(
                "Capture an official public-health source (e.g. CDC antibiotic-use page)",
                "cdc.gov " + BLOCKED,
            )
        ],
    ),
    dict(
        text="The Federal Reserve was created in 1913.",
        domain=D.ECONOMICS,
        diff=DIFF.EASY,
        types=[T.CORRECT_ASSERTION],
        proposed=(Verdict.SUPPORTED, "Federal Reserve FAQ: created on December 23, 1913."),
        evidence=[ev("fed_faq", "The Federal Reserve was created on December 23, 1913", SUP)],
    ),
    dict(
        text="Canberra is the capital city of Australia.",
        domain=D.GEOGRAPHY,
        diff=DIFF.EASY,
        types=[T.CORRECT_ASSERTION],
        proposed=(Verdict.SUPPORTED, "Stated in the first sentence of the Canberra article."),
        evidence=[ev("wp_canberra", "is the capital city of Australia", SUP)],
    ),
    dict(
        text="The world's population reached 8 billion people in November 2022.",
        domain=D.GENERAL_KNOWLEDGE,
        diff=DIFF.MEDIUM,
        types=[T.CORRECT_ASSERTION, T.TIME_SENSITIVE],
        scope="Event dated 15 November 2022",
        proposed=(
            Verdict.SUPPORTED,
            "UN: 'On 15 November 2022, the world's population reached 8 billion people'.",
        ),
        evidence=[
            ev("un_population", "On 15 November 2022", SUP, end="milestone in human development.")
        ],
    ),
    # ------------------------------------------------------------- CONTRADICTED
    dict(
        text="The Eiffel Tower was completed in 1899.",
        domain=D.HISTORY,
        diff=DIFF.EASY,
        types=[T.INCORRECT_DATE],
        context="History: Eiffel Tower",
        proposed=(Verdict.CONTRADICTED, "Sources give 1889, not 1899."),
        evidence=[
            ev("demo:wikipedia_eiffel_tower.txt", "tallest structure when completed in 1889", CON),
            ev(
                "demo:wikipedia_eiffel_tower.txt",
                "designed and built the tower from 1887 to 1889",
                CON,
            ),
        ],
    ),
    dict(
        text="Mount Everest is about 9,848 metres high.",
        domain=D.GEOGRAPHY,
        diff=DIFF.EASY,
        types=[T.INCORRECT_NUMBER],
        context="Geography: Mount Everest",
        proposed=(Verdict.CONTRADICTED, "2020 survey height is 8,848.86 m, 1,000 m lower."),
        evidence=[ev("demo:wikipedia_mount_everest.txt", "8,848.86 m", CON)],
        tasks=[
            task(
                "Capture a primary source for the 2020 China-Nepal survey height",
                "evidence is currently an encyclopedia only",
            )
        ],
    ),
    dict(
        text="A leap year has 365 days.",
        domain=D.MATHEMATICS,
        diff=DIFF.EASY,
        types=[T.INCORRECT_NUMBER],
        context="Mathematics and calendar facts",
        proposed=(Verdict.CONTRADICTED, "A leap year has 366 days."),
        evidence=[ev("demo:wikipedia_leap_year.txt", "a leap year of 366 days", CON)],
    ),
    dict(
        text="The Great Wall of China can be seen with the naked eye from the Moon.",
        domain=D.GENERAL_KNOWLEDGE,
        diff=DIFF.MEDIUM,
        types=[T.COMMON_MYTH],
        proposed=(Verdict.CONTRADICTED, "It cannot be seen by the naked eye from the Moon."),
        evidence=[ev("wp_greatwall", "cannot be seen by the naked human eye from the Moon", CON)],
        tasks=[
            task(
                "Add a primary NASA source on visibility from orbit and the Moon",
                "the NASA article URL redirected to a generic page on the access date",
            )
        ],
    ),
    dict(
        text="Humans use only 10 percent of their brains.",
        domain=D.PUBLIC_HEALTH,
        diff=DIFF.MEDIUM,
        types=[T.COMMON_MYTH],
        proposed=(Verdict.CONTRADICTED, "Imaging shows activity in all parts of the brain."),
        evidence=[ev("wp_brain", "all parts of the brain show some level of activity", CON)],
        tasks=[
            task(
                "Add a peer-reviewed or institutional neuroscience source",
                "evidence is currently an encyclopedia only",
            )
        ],
    ),
    dict(
        text="Sydney is the capital city of Australia.",
        domain=D.GEOGRAPHY,
        diff=DIFF.EASY,
        types=[T.MISLEADING_CONTEXT],
        proposed=(Verdict.CONTRADICTED, "Canberra is the capital."),
        evidence=[ev("wp_canberra", "is the capital city of Australia", CON)],
    ),
    dict(
        text="The square root of 2 is a rational number.",
        domain=D.MATHEMATICS,
        diff=DIFF.EASY,
        types=[T.CORRECT_ASSERTION],
        proposed=(Verdict.CONTRADICTED, "The square root of two is irrational."),
        evidence=[ev("wp_sqrt2", "that the square root of two is irrational", CON)],
    ),
    dict(
        text="Python was first released in 2001.",
        domain=D.TECHNOLOGY,
        diff=DIFF.EASY,
        types=[T.INCORRECT_DATE],
        proposed=(Verdict.CONTRADICTED, "First released in 1991 as Python 0.9.0."),
        evidence=[
            ev("demo:wikipedia_python_programming_language.txt", "first released it in 1991", CON)
        ],
    ),
    dict(
        text="Light from the Sun takes about eight hours to reach Earth.",
        domain=D.PHYSICAL_SCIENCES,
        diff=DIFF.EASY,
        types=[T.INCORRECT_NUMBER],
        proposed=(Verdict.CONTRADICTED, "About 8.3 minutes, not hours."),
        evidence=[
            ev(
                "demo:wikipedia_sunlight.txt",
                "Sunlight takes about 8.3 minutes to reach Earth",
                CON,
            )
        ],
    ),
    dict(
        text="Mars has three natural moons.",
        domain=D.PHYSICAL_SCIENCES,
        diff=DIFF.EASY,
        types=[T.INCORRECT_NUMBER],
        proposed=(Verdict.CONTRADICTED, "NASA fact sheet: 2 natural satellites."),
        evidence=[ev("demo:nasa_mars_fact_sheet.txt", "Number of natural satellites | 2", CON)],
    ),
    dict(
        text="The world's population is currently about 6 billion people.",
        domain=D.GENERAL_KNOWLEDGE,
        diff=DIFF.MEDIUM,
        types=[T.TIME_SENSITIVE, T.INCORRECT_NUMBER],
        scope="Present tense; judge against evidence current on the access date",
        proposed=(Verdict.CONTRADICTED, "UN: 8 billion reached in November 2022."),
        evidence=[
            ev("un_population", "On 15 November 2022", CON, end="milestone in human development.")
        ],
    ),
    dict(
        text="The Federal Reserve was established in 1950.",
        domain=D.ECONOMICS,
        diff=DIFF.EASY,
        types=[T.INCORRECT_DATE],
        proposed=(Verdict.CONTRADICTED, "Created in 1913."),
        evidence=[ev("fed_faq", "The Federal Reserve was created on December 23, 1913", CON)],
    ),
    dict(
        text="Smallpox was declared eradicated in 1990.",
        domain=D.PUBLIC_HEALTH,
        diff=DIFF.EASY,
        types=[T.INCORRECT_DATE],
        proposed=(Verdict.CONTRADICTED, "Declared eradicated in 1980."),
        evidence=[ev("who_smallpox", "In 1980 WHO declared", CON, end="achieve this distinction.")],
    ),
    dict(
        text="The speed of light in vacuum is about 186,000 kilometres per second.",
        domain=D.PHYSICAL_SCIENCES,
        diff=DIFF.MEDIUM,
        types=[T.INCORRECT_NUMBER, T.MISLEADING_CONTEXT],
        proposed=(
            Verdict.CONTRADICTED,
            "About 299,792 km/s; 186,000 is the figure in miles per second (unit confusion).",
        ),
        evidence=[ev("nist_c", "speed of light in vacuum\nNumerical value", CON, end="(exact)")],
    ),
    dict(
        text="The phrase 'I think, therefore I am' was first published by Isaac Newton.",
        domain=D.QUOTATIONS,
        diff=DIFF.MEDIUM,
        types=[T.MISATTRIBUTED_QUOTATION],
        proposed=(Verdict.CONTRADICTED, "Published by Descartes in 1637."),
        evidence=[ev("wp_cogito", "He originally published it in French", CON)],
    ),
    # ---------------------------------------------------- INSUFFICIENT_EVIDENCE
    dict(
        text="According to a 2019 study by Hartwell et al. published in the Journal of "
        "Coastal Futures, 73% of coastal cities will be underwater by 2050.",
        domain=D.ENVIRONMENTAL_SCIENCE,
        diff=DIFF.HARD,
        types=[T.FABRICATED_REFERENCE, T.UNSUPPORTED_STATISTIC],
        context="Fabricated reference and unsupported statistic",
        proposed=(
            Verdict.INSUFFICIENT_EVIDENCE,
            "No trace of the journal or study in the "
            "documented searches; absence of evidence is not a contradiction.",
        ),
        searches=["cr_journal_coastal", "cr_works_hartwell"],
        tasks=[
            task(
                "Search Google Scholar and a library database manually",
                "Crossref coverage is incomplete",
            )
        ],
    ),
    dict(
        text="An unknown research group published a revolutionary energy discovery in 2040.",
        domain=D.GENERAL_KNOWLEDGE,
        diff=DIFF.HARD,
        types=[T.AMBIGUOUS, T.TIME_SENSITIVE],
        scope="Refers to 2040, after every available source",
        proposed=(
            Verdict.INSUFFICIENT_EVIDENCE,
            "Future-dated and vague; reviewers may argue NOT_CHECKABLE instead (boundary case).",
        ),
        ready="future-dated, vague claim: no evidence can exist before 2040; reviewer decides "
        "between INSUFFICIENT_EVIDENCE and NOT_CHECKABLE",
    ),
    dict(
        text="Albert Einstein once said: 'The definition of insanity is doing the same thing "
        "over and over and expecting different results.'",
        domain=D.QUOTATIONS,
        diff=DIFF.HARD,
        types=[T.MISATTRIBUTED_QUOTATION],
        context="Quotation, time-sensitive and hypothetical statements",
        proposed=(
            Verdict.INSUFFICIENT_EVIDENCE,
            "Earliest known print appearance is 1981 "
            "(Narcotics Anonymous); no source shows Einstein said it. Reviewers may "
            "judge this strong enough for CONTRADICTED (boundary case).",
        ),
        evidence=[
            ev(
                "demo:wikiquote_narcotics_anonymous.txt",
                "have been attributed to Albert Einstein",
                UNC,
            )
        ],
    ),
    dict(
        text="Marie Antoinette said 'Let them eat cake' when told the peasants had no bread.",
        domain=D.QUOTATIONS,
        diff=DIFF.HARD,
        types=[T.MISATTRIBUTED_QUOTATION],
        proposed=(
            Verdict.INSUFFICIENT_EVIDENCE,
            "No evidence she said it; the phrase "
            "predates her arrival in France. Boundary between INSUFFICIENT and "
            "CONTRADICTED.",
        ),
        evidence=[
            ev("wp_cake", "there is no evidence that she ever uttered it", UNC),
            ev("wp_cake", "The phrase can actually be traced back to Jean-Jacques", UNC),
        ],
    ),
    dict(
        text="The Nile is the longest river in the world.",
        domain=D.GEOGRAPHY,
        diff=DIFF.HARD,
        types=[T.AMBIGUOUS],
        proposed=(
            Verdict.INSUFFICIENT_EVIDENCE,
            "Sources conflict: the title is disputed with the Amazon depending on measurement.",
        ),
        evidence=[
            ev("wp_nile", "it is the longest river in the world", SUP),
            ev("wp_amazon", "a title which is disputed with the Nile", UNC),
        ],
    ),
    dict(
        text="A 2021 report by the Global Ocean Restoration Council found that worldwide "
        "coral cover increased by 40 percent.",
        domain=D.ENVIRONMENTAL_SCIENCE,
        diff=DIFF.HARD,
        types=[T.FABRICATED_REFERENCE, T.UNSUPPORTED_STATISTIC],
        proposed=(
            Verdict.INSUFFICIENT_EVIDENCE,
            "Organisation and report not found in the "
            "documented searches. The organisation name was invented for this item.",
        ),
        searches=["cr_journal_gorc", "cr_works_coral"],
        tasks=[
            task(
                "Search for the organisation name in web search and NGO registries",
                "Crossref covers scholarly publications only",
            )
        ],
    ),
    dict(
        text="In a 2017 survey, 64 percent of Canadians said they prefer tea over coffee.",
        domain=D.GENERAL_KNOWLEDGE,
        diff=DIFF.MEDIUM,
        types=[T.UNSUPPORTED_STATISTIC],
        proposed=(Verdict.INSUFFICIENT_EVIDENCE, "No survey identified."),
        searches=["cr_works_tea"],
        tasks=[
            task(
                "Search polling organisations' archives for such a survey",
                "polls are rarely indexed in Crossref",
            )
        ],
    ),
    dict(
        text="Bananas are the most consumed fruit in the world.",
        domain=D.GENERAL_KNOWLEDGE,
        diff=DIFF.MEDIUM,
        types=[T.AMBIGUOUS, T.UNSUPPORTED_STATISTIC],
        proposed=(
            Verdict.INSUFFICIENT_EVIDENCE,
            "Depends on the metric (tonnage, per capita, trade); no evidence captured.",
        ),
        tasks=[
            task(
                "Capture FAO production and consumption statistics",
                "no evidence captured; the claim's metric is undefined",
            )
        ],
    ),
    dict(
        text="The ancient Library of Alexandria held exactly 400,000 scrolls.",
        domain=D.HISTORY,
        diff=DIFF.HARD,
        types=[T.UNSUPPORTED_STATISTIC, T.AMBIGUOUS],
        proposed=(
            Verdict.INSUFFICIENT_EVIDENCE,
            "The number is unknown; estimates range from 40,000 to 700,000.",
        ),
        evidence=[ev("wp_alexandria", "It is unknown how many scrolls were housed", UNC)],
    ),
    dict(
        text="A 2018 study found that houseplants reduce office stress by 37 percent.",
        domain=D.PUBLIC_HEALTH,
        diff=DIFF.MEDIUM,
        types=[T.UNSUPPORTED_STATISTIC, T.FABRICATED_REFERENCE],
        proposed=(Verdict.INSUFFICIENT_EVIDENCE, "No study identified; figure unsourced."),
        searches=["cr_works_houseplants"],
        tasks=[task("Search PubMed for the claimed study", "no matching study found yet")],
    ),
    # ------------------------------------------------------------ NOT_CHECKABLE
    dict(
        text="Paris is the most beautiful city in the world.",
        domain=D.GENERAL_KNOWLEDGE,
        diff=DIFF.EASY,
        types=[T.OPINION],
        proposed=(Verdict.NOT_CHECKABLE, "Aesthetic opinion."),
        ready="opinion: no factual evidence is required to review it",
    ),
    dict(
        text="Python is the best programming language for beginners.",
        domain=D.TECHNOLOGY,
        diff=DIFF.MEDIUM,
        types=[T.OPINION],
        proposed=(Verdict.NOT_CHECKABLE, "Evaluative judgement without a defined criterion."),
        ready="opinion: no factual evidence is required to review it",
    ),
    dict(
        text="Bitcoin will be worth more than gold by 2040.",
        domain=D.ECONOMICS,
        diff=DIFF.MEDIUM,
        types=[T.PREDICTION],
        scope="Prediction about 2040",
        proposed=(Verdict.NOT_CHECKABLE, "Prediction about the future."),
        ready="prediction: cannot be checked against present evidence",
    ),
    dict(
        text="Everyone should drink coffee every morning.",
        domain=D.PUBLIC_HEALTH,
        diff=DIFF.EASY,
        types=[T.NORMATIVE],
        proposed=(Verdict.NOT_CHECKABLE, "Normative recommendation, not a factual claim."),
        ready="normative statement: no factual evidence is required to review it",
    ),
    dict(
        text="Modern architecture is more inspiring than classical architecture.",
        domain=D.GENERAL_KNOWLEDGE,
        diff=DIFF.EASY,
        types=[T.OPINION],
        proposed=(Verdict.NOT_CHECKABLE, "Subjective comparison."),
        ready="opinion: no factual evidence is required to review it",
    ),
]

TARGETS = {"SUPPORTED": 15, "CONTRADICTED": 15, "INSUFFICIENT_EVIDENCE": 10, "NOT_CHECKABLE": 5}


# ------------------------------------------------------------------------ build


class Sources:
    def __init__(self, cache: Path, accessed: date) -> None:
        self.cache, self.accessed = cache, accessed
        self.texts: dict[str, str] = {}
        self.refs: dict[str, ReferenceSource] = {}
        self.demo = CollectionManifest.model_validate_json(
            (DEMO / "collection.json").read_text(encoding="utf-8")
        )

    def get(self, key: str) -> tuple[ReferenceSource, str]:
        if key not in self.refs:
            self.refs[key], self.texts[key] = self._load(key)
        return self.refs[key], self.texts[key]

    def _load(self, key: str) -> tuple[ReferenceSource, str]:
        if key.startswith("demo:"):
            filename = key.split(":", 1)[1]
            entry = next(d for d in self.demo.documents if d.local_filename == filename)
            m = entry.metadata
            text = load_document(DEMO / "documents" / filename).text
            public_domain = "Government" in (m.license or "")
            return ReferenceSource(
                reference_id=reference_id_for(m.url, m.source_version),
                title=m.title,
                author_or_organization=m.author_or_organization,
                publisher=m.publisher,
                url=m.url,
                accessed_on=m.accessed_on,
                document_version=m.source_version,
                source_type=(
                    SourceType.OFFICIAL_INSTITUTION
                    if public_domain
                    else SourceType.SECONDARY_ENCYCLOPEDIA
                ),
                license=m.license,
                redistribution=(
                    Redistribution.PUBLIC_DOMAIN_TEXT_STORED
                    if public_domain
                    else Redistribution.OPEN_LICENSE_TEXT_STORED
                ),
                reliability_notes=m.notes,
                capture_method="packaged demo collection document "
                "(see data/references/demo/collection.json)",
                local_collection="demo",
                local_filename=filename,
                local_content_sha256=entry.content_sha256,
            ), text
        if key in WIKI:
            cache_file, _ = WIKI[key]
            page = json.loads((self.cache / cache_file).read_text())["query"]["pages"][0]
            rev = page["revisions"][0]
            url = (
                f"https://en.wikipedia.org/w/index.php?title="
                f"{urllib.parse.quote(page['title'].replace(' ', '_'))}&oldid={rev['revid']}"
            )
            version = f"revision {rev['revid']} ({rev['timestamp']})"
            return ReferenceSource(
                reference_id=reference_id_for(url, version),
                title=f"{page['title']} (Wikipedia)",
                author_or_organization="Wikipedia contributors",
                publisher="Wikimedia Foundation",
                url=url,
                accessed_on=self.accessed,
                document_version=version,
                source_type=SourceType.SECONDARY_ENCYCLOPEDIA,
                license="CC BY-SA 4.0",
                redistribution=Redistribution.OPEN_LICENSE_TEXT_STORED,
                reliability_notes="Tertiary, community-edited source; used as supplementary "
                "evidence. Prefer a primary source where one is listed.",
                capture_method="MediaWiki API plain-text extract (TextExtracts) of the revision "
                "current on the access date",
            ), page["extract"]
        spec = HTML[key]
        text = html_to_text((self.cache / spec["cache"]).read_text(encoding="utf-8"))
        return ReferenceSource(
            reference_id=reference_id_for(spec["url"], f"accessed {self.accessed}"),
            title=spec["title"],
            author_or_organization=spec["org"],
            publisher=spec["publisher"],
            url=spec["url"],
            accessed_on=self.accessed,
            document_version=spec["version"],
            source_type=spec["source_type"],
            license=spec["license"],
            redistribution=spec["redistribution"],
            reliability_notes=spec["reliability"],
            capture_method=f"HTTP GET on {self.accessed}; text extracted with {HTML_TEXT_VERSION}",
        ), text


def _evidence(sources: Sources, spec: dict, now: datetime) -> EvidenceReference:
    ref, text = sources.get(spec["source"])
    if spec["end"]:
        cap = extract_span(text, spec["anchor"], spec["end"])
        how = f"span from {spec['anchor']!r} to {spec['end']!r}"
    else:
        cap = extract_sentence(text, spec["anchor"])
        how = f"sentence containing {spec['anchor']!r}"
    where = f"{ref.local_filename} " if ref.local_filename else ""
    store_text = ref.redistribution is not Redistribution.LOCATOR_ONLY
    locator = where + cap.locator(how)
    return EvidenceReference(
        evidence_id=evidence_id_for(ref.reference_id, cap.sha256, locator),
        reference_id=ref.reference_id,
        capture_status=CaptureStatus.CAPTURED,
        locator=locator,
        anchor_text=spec["anchor"],
        end_anchor_text=spec["end"],
        passage_text=cap.text if store_text else None,
        passage_sha256=cap.sha256,
        captured_at=now,
        captured_by=CAPTURED_BY,
        proposed_relation=spec["relation"],
        note=spec["note"],
    )


def _search(cache: Path, key: str, now: datetime) -> SearchRecord:
    cache_file, service, query, url, looked_for = SEARCHES[key]
    message = json.loads((cache / cache_file).read_text())["message"]
    total = message["total-results"]
    titles = [(i.get("title") or [""])[0] for i in message.get("items", [])]
    if "journals" in url:
        summary = (
            f"{total} journal(s) matched the query."
            if total
            else f"No journal matched the query (0 results); searched for {looked_for}."
        )
    else:
        summary = (
            f"{total:,} works returned (relevance-ranked fuzzy match); the top "
            f"{len(titles)} were inspected and none is {looked_for}: "
            + "; ".join(t[:80] for t in titles)
        )
    return SearchRecord(
        query=query,
        service=service,
        request_url=url,
        performed_at=now,
        performed_by=f"{AUTHOR} (automated HTTP query; result assessment by the same AI assistant)",
        result_summary=summary,
    )


def build(store: BenchmarkStore, cache: Path, accessed: date) -> None:
    now = datetime.now(timezone.utc)
    if not store.exists():
        store.save_manifest(
            BenchmarkManifest(
                name="hallab-factual-verification-benchmark",
                description="Claim-level factual verification benchmark with traceable evidence "
                "and human-approved labels.",
                status_notice="UNDER DEVELOPMENT. Candidates and proposed labels are AI-authored "
                "and provisional. Only human-APPROVED items form the benchmark.",
                created_at=now,
                policy=ApprovalPolicy(),
                partition_rule="partition = dev if int(item_id hex) % 3 == 0 else test "
                "(deterministic, about 1/3 dev and 2/3 held-out test)",
                composition_targets=TARGETS,
            )
        )
    sources = Sources(cache, accessed)
    created = skipped = 0
    existing = set(store.item_ids())
    for spec in CLAIMS:
        if item_id_for(spec["text"]) in existing:
            skipped += 1
            continue
        evidence = [_evidence(sources, e, now) for e in spec.get("evidence", [])]
        for ref in sources.refs.values():
            store.save_reference(ref)
        searches = [_search(cache, k, now) for k in spec.get("searches", [])]
        tasks = [
            ResearchTask(
                task=t["task"],
                reason=t["reason"].format(date=accessed),
                created_at=now,
                created_by=AUTHOR,
            )
            for t in spec.get("tasks", [])
        ]
        split = spec.get("split")
        verdict, rationale = spec["proposed"]
        item, warnings = add_candidate(
            store,
            claim_text=spec["text"],
            domain=spec["domain"],
            difficulty=spec["diff"],
            claim_types=spec["types"],
            temporal_scope=spec.get("scope"),
            original_context=_fixture_text(spec["context"]) if spec.get("context") else None,
            split_recommendation=SplitRecommendation(
                should_split=True, proposed_parts=split[0], note=split[1]
            )
            if split
            else None,
            proposed_label=ProposedLabel(
                verdict=verdict,
                proposed_by=AUTHOR,
                proposer_type=AuthorType.AI_ASSISTANT,
                rationale=rationale,
                proposed_at=now,
            ),
            provenance=CandidateProvenance(
                author=AUTHOR,
                author_type=AuthorType.AI_ASSISTANT,
                created_at=now,
                method=(
                    f"adapted from the synthetic fixture '{spec['context']}'"
                    if spec.get("context")
                    else "written by an AI assistant to cover the benchmark composition targets"
                ),
                notes="AI-authored candidate. Not human-reviewed.",
            ),
            evidence=evidence,
            research_log=searches,
            research_tasks=tasks,
            now=now,
        )
        for w in warnings:
            print(f"  warning ({item.item_id}): {w}")
        ready = spec.get("ready")
        if evidence or searches or ready:
            reason = ready or ("evidence captured" if evidence else "search records documented")
            set_status(store, item.item_id, ReviewStatus.READY_FOR_REVIEW, AUTHOR, reason, now)
        else:
            set_status(
                store,
                item.item_id,
                ReviewStatus.RESEARCH_NEEDED,
                AUTHOR,
                "no evidence captured yet",
                now,
            )
        created += 1
    print(
        f"created {created} candidate(s), skipped {skipped} existing; "
        f"{len(sources.refs)} reference source(s) used"
    )


def download(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)

    def get(url: str, dest: str) -> None:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            (cache / dest).write_bytes(r.read())
        time.sleep(1)

    for cache_file, title in WIKI.values():
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
            "prop": "extracts|revisions|info",
            "inprop": "url",
            "explaintext": "1",
            "exsectionformat": "wiki",
            "rvprop": "ids|timestamp",
            "titles": title,
        }
        get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params), cache_file)
    for spec in HTML.values():
        get(spec["url"], spec["cache"])
    for cache_file, _, _, url, _ in SEARCHES.values():
        get(
            url + ("&select=title,author,container-title,issued,DOI" if "works" in url else ""),
            cache_file,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cache", type=Path, help="directory with previously downloaded sources")
    group.add_argument("--download", type=Path, help="download sources into this directory")
    parser.add_argument("--accessed", type=date.fromisoformat, default=None)
    parser.add_argument("--root", type=Path, default=None, help="benchmark directory")
    args = parser.parse_args()
    cache = args.cache or args.download
    if args.download:
        download(cache)
    build(BenchmarkStore(args.root), cache, args.accessed or date.today())


if __name__ == "__main__":
    main()
