# Architecture and design decisions

This document explains *why* the system is built the way it is. Each decision below
can be discussed in an interview.

## Pipeline

```
AI response (stored verbatim, hash-identified)
   │
   ▼
Claim extraction  ── rule-based candidates ──►  Human claim review
   (Phase 2)                                     edit / split / merge / add / remove
   │                                                    │
   ▼                                                    ▼
Evidence retrieval (Phase 3) ── BM25 over curated reference passages
   │        records: source, passage offsets, rank, raw score, parameters
   ▼
Verification (Phase 4)
   ├─ automated baseline: deterministic checks only, always cites evidence
   ├─ (optional, later) model suggestion: labelled as a suggestion
   └─ human verdict: authoritative, requires reviewer_id + reasoning + evidence
   │
   ▼
Report (Phase 5): per-claim findings, citations, limitations, unchecked items
```

Everything for one investigated response lives in a single `VerificationSession` JSON
file.

## Key decisions

### 1. Integrity rules live in the schema, not in the UI

Rules such as "SUPPORTED must cite evidence" and "human verdicts need a reviewer" are
Pydantic validators. They therefore apply everywhere: the CLI, the future Streamlit UI,
tests, and hand-edited JSON files. A UI bug cannot produce an unsupported verdict,
because the model will not validate and the store will not save it.

### 2. Retrieved ≠ verified

An `Evidence` record means "a ranking algorithm thought this passage was relevant". It
says nothing about truth. Evidence only becomes *relied-upon* when a human verification
cites it. Retrieval scores are stored as raw ranking scores (for example BM25), never as
percentages or probabilities.

### 3. No uncalibrated confidence numbers

A number like "87% confident" implies calibration that rule-based or untested systems
don't have. The schema lets `extraction_confidence` be stored only together with a
`confidence_calibration` description. The rule-based extractor leaves both empty.

### 4. Human judgments and automated outputs are structurally distinct

`VerificationMethod` is `human`, `automated_rule`, or `model_suggestion`. Only `human`
may carry a `reviewer_id`, and only human verdicts count in `human_verdict()` and
`pending_review()`. A claim with only an automated verdict therefore still shows as
pending human review.

### 5. Deterministic, content-derived identifiers

IDs are `prefix_` plus the first 16 hex characters of a SHA-256 hash of the defining
content. For a response this is its text. For a claim it is the response ID, span, and
extractor version. Re-running extraction gives identical IDs, which makes runs
reproducible and lets tests assert exact values. The response ID also works as a
tamper check. Sessions use random IDs because two sessions on the same text are
legitimately different investigations. 64 bits of hash makes collisions negligible at
this scale.

### 6. Original wording is never lost

`AIResponse.text` is never stripped or normalised. Each rule-based claim stores its
exact `original_text` and character `span`, and the test suite asserts that
`response.text[span.start:span.end] == original_text`. Reviewer edits only change
`normalized_claim`, and each edit is appended to `reviewer_edits`.

### 7. Soft deletion and an audit trail for claim edits

Removing, splitting, or merging a claim deactivates the old record and links the new
ones through `related_claim_ids`. The path from AI wording to the final verified claim
can always be reconstructed.

### 8. Atomic, validated persistence

Sessions are written to a temporary file and then moved into place with `os.replace`,
so a crash cannot leave a half-written file. Every save re-validates the whole session.
Every load validates against the schema and reports a readable error. Session IDs must
match a strict pattern before they are used as file names, which prevents path
traversal.

### 9. Schema versioning

Each session carries `schema_version` (semver). Loading rejects an unknown major
version instead of guessing. Minor versions can add optional fields without breaking
older files.

### 10. Rule-based extraction first, LLM optional later

Rules are offline, free, deterministic, fast, and explainable: each claim records which
rule fired. Their limits are documented rather than hidden. An LLM extractor could find
more claims, but it would add cost and non-determinism, so it will be an optional module
whose outputs are tagged `llm_assisted`.

### 11. Minimal dependencies

The core needs only Pydantic. Retrieval (`rank-bm25`) and the UI (`streamlit`) will be
added when those phases are built.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Rule-based extraction misses or mis-splits claims | Human claim editing; documented limitations; tests pin known behaviour |
| Users read retrieval relevance as truth | Separate concepts in schema and UI; raw scores only; explicit labels |
| Automated verdicts mistaken for human review | Schema-level separation; pending-review logic ignores automated verdicts |
| Benchmark labels fabricated or unverified | Phase 6 labels require cited sources; fixtures are labelled as fixtures |
| Copyright issues in reference collections | Only original or openly licensed documents, with licence recorded per source |
| Session file corruption | Atomic writes; validation on load; corrupt files reported, not fatal |
| Time-sensitive facts going stale | `temporal_context` captured at extraction; publication dates on evidence |
| Schema evolution breaks saved sessions | `schema_version` with major-version gate |
| Examples path only resolves in a source checkout | Package data properly in Phase 8 (known gap) |
