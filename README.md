# AI Hallucination Detection & Fact-Checking Lab

**Claim-level factual verification for AI-generated text, with evidence you can trace.**

This project breaks an AI response into individual factual claims, retrieves evidence
for each claim from curated reference documents, and records evidence-backed verdicts
(`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `NOT_CHECKABLE`). The output is
a transparent fact-checking report.

The tool does not decide what is true on its own authority. Every verdict must cite
evidence and record how it was produced. Human judgments and automated suggestions are
kept apart at the schema level.

> **Status: pre-alpha (milestone 4 of 8).** Claim extraction, offline BM25 evidence
> retrieval, **human verification** (passage assessments, evidence-backed verdicts,
> drafts, revision history), a **local Streamlit web interface**, and **JSON / CSV /
> Markdown reports** work and are tested. **All verdicts are human. No automated
> verdicts exist.** The labelled verdict benchmark and an automated baseline have
> **not been built yet**. The web interface runs locally; it is not deployed anywhere.
> See [What works today](#what-works-today) and the [roadmap](docs/ROADMAP.md).

---

## Why this project exists

General response-quality evaluation (helpfulness, tone, rubric scores) asks *"is this a
good answer?"*. This lab asks a narrower question: *"which specific statements in this
answer are factually supported, and by what evidence?"* It works at the level of single
claims instead of whole responses.

## What works today

| Capability | Status | Where |
|---|---|---|
| Pydantic schemas for responses, claims, sources, evidence, verifications, sessions | ✅ implemented, tested | `src/hallucination_lab/models.py` |
| Integrity rules enforced by validation (see below) | ✅ implemented, tested | `models.py` |
| Deterministic, content-hashed identifiers | ✅ implemented, tested | `ids.py` |
| Atomic, validated JSON session storage (save / load / list / delete) | ✅ implemented, tested | `storage.py` |
| Rule-based sentence segmentation with exact character offsets | ✅ prototype, tested | `extraction/segmenter.py` |
| Compound-sentence splitting (conservative) | ✅ prototype, tested | `extraction/segmenter.py` |
| Flags for questions, opinions, instructions, hypotheticals, filler | ✅ prototype, tested | `extraction/rules.py` |
| Detection of numbers, dates, measurements, entities, citations, quotes, time-sensitive wording | ✅ prototype (heuristic), tested | `extraction/features.py` |
| Manual claim edit / split / merge / add / remove / restore with audit trail | ✅ implemented, tested | `extraction/editing.py` |
| Synthetic example responses (clearly labelled, not real model outputs) | ✅ 6 fixtures | `data/examples/` |
| Command-line interface (`hallab`) | ✅ implemented, tested | `cli.py`, `cli_retrieval.py` |
| Reference collections with declared vs computed source metadata | ✅ implemented, tested | `retrieval/models.py`, `retrieval/catalog.py` |
| Ingestion of `.txt`/`.md` (validation, duplicate and modification detection, passages with offsets, lines, and sections) | ✅ implemented, tested | `retrieval/ingest.py` |
| BM25 retrieval (direct implementation, per-term score explanation, de-duplication) | ✅ implemented, tested | `retrieval/bm25.py`, `retrieval/search.py` |
| Retrieval results saved to sessions with full provenance (schema 1.1.0, 1.0.0 still loads) | ✅ implemented, tested | `retrieval/session_retrieval.py`, `migrations.py` |
| Offline demo collection: 10 real source texts (Wikipedia/Wikiquote CC BY-SA, NASA) | ✅ packaged | `data/references/demo/` |
| Retrieval evaluation (Recall@k, Precision@k, MRR) on 24 labelled queries | ✅ implemented | `retrieval/evaluation.py`, `data/evaluation/` |
| Interactive human verification (`hallab verify`): passage assessments, verdicts, cited evidence, rationale checks | ✅ implemented, tested | `verification/interactive.py`, `verification/workflow.py` |
| Drafts with pause and resume; conflict-safe, append-only session saves | ✅ implemented, tested | `storage.py`, `verification/workflow.py` |
| Revision chains (linear, authorized cross-reviewer corrections, history preserved) | ✅ implemented, tested | `models.py`, `verification/workflow.py` |
| Source re-check before citing (blocks evidence whose document changed) | ✅ implemented, tested | `verification/integrity.py` |
| Verification progress (`verify status`) and summary (`verify summary [--json]`) | ✅ implemented, tested | `cli_verify.py` |
| Automated baseline verification (deterministic cases only, labelled as automated) | ⏳ Phase 4 | — |
| Local Streamlit web interface: dashboard, analysis, verification, reports, collections | ✅ implemented, tested (AppTest and headless-browser smoke test) | `app/streamlit_app.py`, `src/hallucination_lab/ui/` |
| Verification reports: JSON / CSV / Markdown, deterministic fingerprint, full audit trail | ✅ implemented, tested | `services/reports.py`, `hallab report` |
| UI-independent application services (workspace, dashboard, reports) | ✅ implemented, tested | `src/hallucination_lab/services/` |
| Labelled benchmark and metrics | ⏳ Phase 6 | — |
| Docker, CI, release | ⏳ Phase 8 | — |

## Integrity rules enforced in code

These rules are checked by schema validation, so invalid records cannot be created or
saved. Each one has a test.

- A `SUPPORTED` verdict must cite supporting evidence, and a `CONTRADICTED` verdict must
  cite contradicting evidence. Missing evidence can never turn into a contradiction.
- A human verification must carry a `reviewer_id`. Automated or model verifications
  must not carry one, so they can never pass as human review.
- A model-generated suggestion must name the model in its provenance.
- `extraction_confidence` can only be stored together with a description of how it was
  calibrated. The rule-based extractor stores no confidence at all.
- An AI response's ID is a SHA-256 hash of its exact text. If a stored response is
  altered, loading it fails.
- A session is referentially consistent. Evidence and verdicts can only point at claims
  and evidence in the same session, and a verdict cannot cite evidence that was
  retrieved for a different claim.
- Evidence counts as retrieved until a human verdict cites it. `human_cited_evidence_ids()`
  tracks the difference.
- Removed, split, and merged claims are deactivated, never deleted. Every change is
  logged in `reviewer_edits`.
- Retrieved passages store their SHA-256. A session whose quoted passage was edited
  fails to load.
- Reference documents are content-addressed. A changed or missing document blocks
  indexing and search until it is restored or deliberately re-registered.
- A SUPPORTED or CONTRADICTED verdict from the verification workflow must cite passages
  the reviewer **explicitly assessed** as supporting or contradicting. A high retrieval
  score can never stand in for that judgement.
- Rationales that are empty, placeholders ("n/a", "tbd"), bare scores ("4/5"), or a
  single verdict word are rejected by the schema.
- Saved verifications and evidence are append-only. A correction is a new record linked
  by `revision_of`, histories cannot branch or loop, and revising another reviewer's
  record requires a recorded authorization.

## Quick start

Requirements: Python 3.10 or newer and git. No API keys, no paid services, no network
access at runtime.

```bash
git clone <your-fork-url> ai-hallucination-verification-lab
cd ai-hallucination-verification-lab

# Option A: uv (fast)
uv venv .venv
uv pip install -e ".[ui,dev]"      # [ui] = web interface (Streamlit); omit for CLI only

# Option B: standard library venv + pip
python -m venv .venv
# Linux/macOS:   source .venv/bin/activate
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
pip install -e ".[ui,dev]"
```

### Web interface

```bash
streamlit run app/streamlit_app.py      # or: hallab ui
```

Then open http://localhost:8501. The interface covers the whole workflow: paste a
response, extract and edit claims, retrieve evidence, record human verdicts (with
drafts, confirmation, and revisions), and download reports. See
[docs/WEB_INTERFACE.md](docs/WEB_INTERFACE.md).

Try it:

```bash
# Extract candidate claims from text
hallab extract --text "The Eiffel Tower was completed in 1889. Is it in Paris? I think it is beautiful."

# Same, as JSON (shows spans, features, and which rule fired)
hallab extract --file my_response.txt --json

# Run extraction over the bundled synthetic fixtures and save them as sessions
hallab demo --save
hallab sessions list
hallab sessions show <session_id>
```

### Complete offline example: claim → evidence candidates → saved session

```bash
# 1. Look at the packaged reference collection (10 documents, no network needed)
hallab sources list --collection demo

# 2. Search one claim (the index is built automatically on first use)
hallab search --collection demo --top-k 2 --claim "Mount Everest has an elevation of about 9,848 metres."

# 3. Extract claims from a response and save a session
hallab extract --save --text "The Eiffel Tower was completed in 1899 for the World's Fair in Paris. A leap year has 365 days."

# 4. Retrieve candidates for every checkable claim in that session, then inspect it
hallab retrieve --session <session_id printed in step 3> --collection demo --top-k 2
hallab sessions show <session_id>

# 5. Measure retrieval quality on the labelled query set
hallab eval retrieval

# 6. Human verification: assess passages, choose verdicts, write rationales
hallab verify --session <session_id> --reviewer-id reviewer_01   # q pauses, rerun resumes
hallab verify status  --session <session_id>
hallab verify summary --session <session_id>                     # add --json for structured output
hallab verify --session <session_id> --reviewer-id reviewer_01 --revise <verification_id>

# 7. Export a traceable report (same generator as the web interface)
hallab report --session <session_id> --format md --output report.md   # or json / csv
```

How to judge claims, write rationales, and correct verdicts is covered in
[docs/HUMAN_VERIFICATION_GUIDE.md](docs/HUMAN_VERIFICATION_GUIDE.md).

Excerpt of step 2's output:

```
RETRIEVAL CANDIDATES: passages that share words with the claim. They are NOT verified
evidence and may support, contradict, or be irrelevant to the claim. BM25 scores measure
lexical overlap only; they are not probabilities, confidence, or truth estimates.

Claim:       Mount Everest has an elevation of about 9,848 metres.
Query terms: mount, everest, elevation, 9848, metre
Note:        terms not found anywhere in the collection: 9848

  [1] BM25 score 13.338   matched terms: elevation, mount, everest, metre
      Source:   Mount Everest (Wikipedia)  (packaged_demo; metadata captured_at_download)
      URL:      https://en.wikipedia.org/w/index.php?title=Mount_Everest&oldid=1376942985
      Location: wikipedia_mount_everest.txt lines 3-4
      Passage:  psg_62832ca3b3fee844   source: src_c86df31b4d4f351e
```

The note that `9848` appears nowhere in the collection is a cue for the reviewer. It is
not proof of a hallucination. Adding your own documents is covered in
[docs/RETRIEVAL.md](docs/RETRIEVAL.md#using-it).

Sessions are saved as JSON under `./lab_data/sessions/`. Set `HALLAB_DATA_DIR` to change
the location and `HALLAB_LOG_LEVEL=INFO` for more logging.

Example output (from `hallab demo`):

```
== Mathematics and calendar facts (4 candidates, 4 to check)
  1. [CHECK] The square root of 144 is 12.
       numbers: 144, 12
  2. [CHECK] A leap year has 365 days.
       numbers: 365 | measurements: 365 days
  3. [CHECK] Pi is approximately 3.14159
       numbers: 3.14159 | entities: Pi
  4. [CHECK] It is an irrational number.
```

Extraction only produces *candidates*. Claim 4 starts with a pronoun, and the claim
record notes that it needs context. Claim 2 is wrong, but deciding that is the job of
the verification phases, not extraction.

## Development

```bash
pytest            # run the test suite
ruff check .      # lint
ruff format .     # format
```

Platform note: development and testing so far have been on Linux (WSL2) with Python 3.12
and 3.13. The web interface was smoke-tested in headless Chromium on the same system.
Windows and macOS instructions above use standard Python tooling but **have not yet been
verified**. Cross-platform CI is planned for Phase 8.

## How claim extraction works (and its limits)

1. **Segmentation.** Text is split into sentences with transparent rules: a fixed
   abbreviation list ("Dr.", "e.g.", "U.S."), decimal handling, bullet and heading
   detection, and soft-wrapped line joining. Each segment keeps exact character offsets
   into the original response.
2. **Compound splitting.** A sentence is split at `;` or `, and / but / while / whereas`
   only when both sides look like full clauses (at least three words and a verb).
3. **Classification.** Ordered rules flag questions, filler, instructions,
   hypotheticals, and opinion markers. The rule that fired is stored in
   `extraction_notes`.
4. **Features.** Regexes detect numbers, dates, measurements, capitalised-name runs,
   citation patterns, quotations, and time-sensitive wording.

**This is sentence segmentation, not true factual claim extraction.** Known limitations:

- A sentence may contain several facts that are not split, for example "X, born in 1879
  in Ulm, developed relativity".
- Claims that start with a pronoun ("It opened in 1889") need context. They are flagged
  but not resolved.
- "I think Paris is the capital of France" is flagged as opinion even though it contains a
  checkable fact. Reviewers can override this.
- Entity detection is just runs of capitalised words, and spelled-out numbers are not
  detected.
- The rules are tuned for English.

The design response to these limits is human review. Reviewers can edit, split, merge,
add, and remove claims before verification, and every change is recorded. An optional
LLM-assisted extractor may be added later as a separate, clearly labelled module.

## Project layout

```
src/hallucination_lab/
  models.py          Pydantic schemas + integrity rules
  ids.py             deterministic content-hashed IDs
  storage.py         atomic, validated JSON session store
  sessions.py        session creation
  examples.py        loader for the synthetic fixtures
  cli.py             `hallab` command-line interface
  config.py          settings from environment variables
  logging_utils.py   logging setup
  errors.py          exception hierarchy
  extraction/
    segmenter.py     sentence + clause segmentation with offsets
    rules.py         non-checkable rules, normalisation
    features.py      numbers/dates/entities/citations/... detection
    extractor.py     pipeline producing candidate Claim records
    editing.py       human edit/split/merge/add/remove operations
  retrieval/
    models.py        source metadata, documents, passages, collections
    ingest.py        loading, validation, canonical text, passage segmentation
    tokenizer.py     lexical tokenizer (numbers, units, dates)
    bm25.py          Okapi BM25 with per-term contributions
    search.py        top-k search, de-duplication, per-source cap
    catalog.py       collections on disk, integrity checks, index lifecycle
    session_retrieval.py  attach candidates to sessions (no verdicts)
    evaluation.py    Recall@k / Precision@k / MRR
  verification/
    rationale.py     rationale error/warning checks
    integrity.py     re-check evidence against its source before citing
    workflow.py      verdict building, revisions, drafts, progress, summary
    interactive.py   terminal review flow (hallab verify)
  migrations.py      versioned session-schema migrations
  cli_retrieval.py   sources / index / search / retrieve / eval commands
  cli_verify.py      verify / verify status / verify summary
  cli_report.py      report / ui
  services/
    workspace.py     conflict-safe load/save, claim edits, retrieval, verification
    dashboard.py     totals across sessions (active verdicts only)
    reports.py       JSON / CSV / Markdown reports
  ui/                Streamlit pages (call services only)
app/streamlit_app.py   web interface launcher
data/examples/       synthetic response fixtures (hand-written, labelled as such)
data/references/demo/  packaged reference collection + CONTENT_LICENSE.md
data/evaluation/     labelled retrieval queries
scripts/             dataset build scripts
tests/               pytest suite (fixtures/ holds real schema-1.0.0 and 1.1.0 sessions)
docs/                architecture, retrieval guide, human verification guide, roadmap
```

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
[docs/RETRIEVAL.md](docs/RETRIEVAL.md) ·
[docs/HUMAN_VERIFICATION_GUIDE.md](docs/HUMAN_VERIFICATION_GUIDE.md) ·
[docs/WEB_INTERFACE.md](docs/WEB_INTERFACE.md) · [docs/ROADMAP.md](docs/ROADMAP.md)

## Retrieval is not verification

BM25 finds passages that **share words** with a claim. A top-ranked passage may
contradict the claim (the Eiffel Tower "1899" claim retrieves a passage saying it was
built 1887–1889), and "no candidates" only means no word overlap. On the labelled
evaluation set (19 positive and 5 negative queries), BM25 returned candidates for every
negative query. The highest negative score (12.7) was above some correct hits, so **no
score threshold distinguishes "evidence exists" from "nothing relevant exists"**. Scores
are never converted into confidence or hallucination probabilities.

Retrieval results: Recall@5 0.789, MRR 0.728. These describe passage retrieval only;
see [docs/RETRIEVAL.md](docs/RETRIEVAL.md#retrieval-evaluation) for how the labels were
made, the failure cases, and limitations. **They are not hallucination-detection
accuracy.** No verdict accuracy has been measured yet, because that needs the
human-labelled benchmark planned for Phase 6.

## Human verification

Verdicts come only from a person using `hallab verify`:

1. Label each retrieved passage as SUPPORTING, CONTRADICTING, IRRELEVANT, or UNCERTAIN.
   The BM25 score is shown as "lexical overlap, NOT support".
2. Choose SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, or NOT_CHECKABLE.
   SUPPORTED and CONTRADICTED require passages you labelled accordingly. Missing
   evidence leads to INSUFFICIENT_EVIDENCE, never CONTRADICTED.
3. Write a rationale. Placeholders are rejected and generic text is flagged.
4. Review the summary and save explicitly.

Each source shows whether it still matches what was retrieved and a credibility note.
Demo sources are marked "credibility not independently established", and that caveat
is stored with any verdict that cites them.

## Data and content policy

- `data/examples/synthetic_responses.json` contains **hand-written fixtures**. None of
  them were produced by an AI model. Some contain deliberate errors. Their
  `author_notes` describe intent and are **not** verified ground-truth labels.
- `data/references/demo/` contains real source texts: Wikipedia/Wikiquote plain-text
  extracts pinned to exact revision IDs (**CC BY-SA 4.0**, not MIT) and a NASA fact
  sheet (U.S. Government work; licence status to be confirmed). Attribution and the
  changes made are in [CONTENT_LICENSE.md](data/references/demo/CONTENT_LICENSE.md).
  `scripts/build_demo_collection.py` documents exactly how they were captured.
- Wikipedia is a tertiary source. Demo documents are reference material for retrieval,
  not ground truth.
- Tests use invented text about a fictional place ("Lake Verenna"), clearly labelled as
  a synthetic test fixture.

## Licence

Code: MIT. See [LICENSE](LICENSE). Packaged reference documents keep their own licences
(see above).
