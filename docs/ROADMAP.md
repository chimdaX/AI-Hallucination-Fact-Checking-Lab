# Development roadmap

Each phase ends with a working, tested milestone and a review before the next begins.
Nothing is pushed, tagged, or released without explicit approval.

## ✅ Milestone 1: Foundation + extraction prototype (Phases 1 and 2, initial)

- `pyproject.toml` (hatchling, src layout), Pytest, Ruff
- Pydantic schemas: response, claim, source, evidence, verification, session
- Validation-enforced integrity rules; schema versioning; deterministic IDs
- Atomic validated JSON store; error hierarchy; logging setup
- Rule-based segmentation, compound splitting, non-checkable rules, feature detection
- Human claim editing operations with audit trail
- 6 synthetic fixtures (labelled), `hallab` CLI
- Test suite: IDs, schema rules, storage and recovery, segmentation, extraction,
  editing, fixtures, and CLI

## Phase 2 (remaining)

- Evaluate the extractor on the fixtures and record its failure cases
- Optional, separately documented LLM-assisted extractor interface (no default provider)

## ✅ Milestone 2: Evidence retrieval (Phase 3)

- Reference collections: manifest + documents, packaged read-only `demo` and user
  collections, declared vs computed metadata with status
- Ingestion of `.txt`/`.md`: validation, encodings, canonical text, duplicate and
  modification detection, warnings
- Passage segmentation: sections, sentence packing, configurable size and overlap,
  offsets, line numbers, stable IDs
- BM25 (direct implementation) with lexical tokenizer, per-term explanations,
  de-duplication, per-source cap, deterministic ranking
- Session integration: schema 1.1.0 (optional fields), migrations, retrieval runs with
  explicit outcomes, passage hashes
- CLI: `sources add/list/show/verify`, `index build`, `search`, `retrieve`,
  `eval retrieval`
- Demo collection of 10 real source texts with attribution; 24-query labelled retrieval
  evaluation

**Open follow-ups found during Milestone 2:**

- Human review of the retrieval relevance labels (currently a single AI annotator)
- Tokenizer: optional `-ing`/`-ed` stemming; evaluate on a *fresh* query set
- Decide the default `max_per_source` (3 vs 5); see the trade-off in docs/RETRIEVAL.md
- Manual confirmation of the NASA fact sheet's licence status

## ✅ Milestone 3: Human verification

- Methodology: `docs/HUMAN_VERIFICATION_GUIDE.md` (verdict criteria, insufficient vs
  contradicted, rationale guidance, limitations)
- Schema 1.2.0: `EvidenceAssessment`, verification `session_id`, revision chain fields,
  noted limitations, `VerificationDraft` (all optional; 1.0/1.1 sessions load)
- Interactive `hallab verify` with passage labels, verdict validation, rationale checks,
  explicit save confirmation, drafts, pause and resume
- Revisions via `--revise`, with cross-reviewer authorization and linear history
- Source re-check before citing; append-only, conflict-safe storage
- `verify status` and `verify summary [--json]`

**Open follow-ups:**

- Multi-reviewer workflows: independent double review, agreement statistics,
  adjudication
- Authenticated reviewer identity (currently self-declared)
- Attaching manually found evidence (outside retrieval) with provenance

## Phase 4: Automated baseline (deterministic suggestions only)

- Numeric comparison between a claim and a curated reference record (with tolerance
  and unit normalisation), and exact match against curated fact records
- Output stored as `automated_rule` verifications: never counted as human verdicts,
  always citing evidence, and anything uncertain returns `INSUFFICIENT_EVIDENCE`
- Use `RetrievalRun` outcomes so "searched, nothing found" maps to
  `INSUFFICIENT_EVIDENCE`, never `CONTRADICTED`
- Tests: missing evidence ≠ contradiction, numeric edge cases, human override

## ✅ Milestone 4: Streamlit interface and reports (Phase 5)

- Local Streamlit app: Dashboard, Analyze response, Human verification, Reports,
  Reference collections, Getting started
- Services layer (workspace, dashboard, reports) shared with the CLI
- JSON / CSV / Markdown reports with active verdicts, revision history, cited evidence,
  unresolved claims, integrity warnings, limitations, and a deterministic content
  fingerprint; `hallab report`, `hallab ui`
- State safety: reload per run, stale-view refusal, two-step confirmed saves, drafts
- Upload of `.txt`/`.md` documents into user collections via the existing ingestion

**Open follow-ups:**

- Verify the interface on Windows and macOS browsers (Phase 8 CI)
- Merge of concurrent edits (currently detected and refused)
- Optional authentication if the app is ever shared beyond one machine

## Phase 6: Benchmark

- 30–50 claim-level examples across the specified categories
- Each label backed by a cited reference; items needing manual verification marked
- Metrics (per-class counts, confusion matrix, precision/recall/F1) computed only on
  the labelled set, with the evaluation script included

## Phase 7: Testing and reproducibility hardening

- Coverage review against the spec's test list; offline-run check

## Phase 8: Packaging and release

- Dockerfile; GitHub Actions (tests, lint, build) on Linux/Windows/macOS
- Package the example data properly (currently resolved from the source tree)
- Verified installation docs per OS; quick-start guide; versioned release (on approval)
