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

The core, including retrieval, needs only Pydantic. Streamlit will be added in Phase 5.

## Milestone 2 decisions: evidence retrieval

### 12. BM25 implemented directly instead of using `rank-bm25`

BM25 is about 60 lines. Writing it directly:

- adds no dependency (`rank-bm25` pulls in numpy and has not been released since 2022)
- uses the Lucene IDF variant, which never goes negative. The classic formula can
  penalise a passage for matching a very common word.
- returns each query term's contribution, so the CLI can show *why* a passage ranked
  where it did

A unit test checks the score against a hand calculation.

### 13. Content-addressed sources, hash-checked documents

`source_id = hash(file bytes)`. Two consequences:

- the same document gets the same ID everywhere, so duplicates are detected for free
- an edited document gets a *new* ID, so evidence saved in an old session can never be
  silently re-pointed at changed text

Indexing and search refuse to run while a document differs from its registered hash. A
deliberate change requires `--replace`, which is visible and recorded.

### 14. Canonical text and exact offsets

The only normalisations applied to stored text are BOM removal and newline unification.
Case folding, accent folding, and unit rewriting happen only inside the tokenizer. Every
passage is therefore an exact slice of the canonical text, with character offsets and
line numbers a reviewer can open. Tokenizer normalisation affects ranking, never the
evidence text.

### 15. Declared vs computed metadata

Title, author, URL, and dates are declarations that cannot be checked offline, so they
carry a status (`user_supplied`, `captured_at_download`, or `manually_confirmed`).
Hashes, size, and encoding are computed from the file and never accepted from input.
Nothing is ever filled in with guessed values.

### 16. "Searched, nothing found" is recorded explicitly

A `RetrievalRun` stores an outcome for every searched claim: `candidates_found`,
`no_candidates`, or `no_searchable_terms`. Later phases need this to tell
`INSUFFICIENT_EVIDENCE` (looked, found nothing) apart from an unchecked claim. Missing
candidates never become a contradiction.

### 17. Additive, idempotent session updates

Evidence IDs are derived from (claim, passage, passage hash). Re-running retrieval
reuses existing records instead of duplicating or overwriting them. A new run only
appends. Schema 1.1.0 adds only optional fields, and `migrations.py` upgrades 1.0.0
files explicitly. The backward-compatibility test uses a session file generated by the
actual Milestone 1 code.

### 18. Evaluation labels independent of the system under test

Relevance is defined as (file, exact phrase), not as passage IDs, and the labels were
written before the retriever was run. The labels therefore survive re-chunking and are
not biased toward what the retriever happened to return. Defaults were not tuned on the
evaluation set.

## Milestone 3 decisions: human verification

### 19. Four separate concepts, four separate records

A claim (`Claim`), an evidence candidate (`Evidence`, with its BM25 score), a reviewer's
judgement of one passage (`EvidenceAssessment`, which deliberately has **no score
field**), and a verdict (`Verification`) are stored separately. Records created by the
workflow carry `session_id`. For those, the schema requires every cited passage to have
an explicit matching assessment. "The passage ranked first" can therefore never turn
into "the passage supports the claim" without a person saying so.

### 20. Corrections are new records (revision chains)

A saved verdict is never edited. A correction is a new `Verification` with
`revision_of`, a required `revision_reason`, and, when correcting someone else's work,
a required `revision_authorization`. The session validator enforces that a revision
targets the same claim, is later in time, comes from a human, and does not branch or
loop. The *active* verdict is the newest unsuperseded record, and superseded records
are not counted as additional verifications. This is how audit-grade systems handle
corrections: the history is the data.

### 21. One active verdict per claim, with legacy tolerance

Workflow-created records may have only one active human verdict per claim. Sessions
from schema 1.0/1.1, where several human records for a claim were allowed, still load:
their records have no `session_id` and keep the older rule that the latest one wins.

### 22. Append-only and conflict-safe storage

`JsonSessionStore.save` compares the new session with the file on disk and refuses to
modify or drop any existing verification or evidence record (`AuditTrailError`). The
interactive workflow saves with an expected file version (a SHA-256 of the file
contents), so a concurrent change raises `SessionConflictError` instead of being
overwritten. There is no OS-level lock, so this is detection rather than prevention,
which is enough for a single-user local tool.

### 23. Drafts live inside the session

Work in progress is a `VerificationDraft` in the same session file. It is validated
and written atomically like everything else, so recovery needs no second file. Drafts
are saved after every step, so an interruption loses at most the current answer.
Drafts are never counted as verifications.

### 24. Re-check sources before relying on them

Before evidence can be cited, `verification/integrity.py` confirms that the source is
still registered, the file hash matches the one recorded at retrieval, and the text at
the stored offsets still hashes to the passage. Changed or missing sources block
citation until retrieval is re-run. An unavailable collection or legacy evidence
without hashes is allowed, but it is recorded as a limitation on the verdict.

### 25. Rationale checks: hard floor, soft guidance

Placeholders, bare scores, empty text, and single verdict words are rejected in the
schema, so no code path can store them. Generic phrases, rationales unrelated to the
evidence, and NOT_CHECKABLE used for "couldn't find it" produce warnings that a
reviewer may override. There is deliberately no minimum length.

### 26. UI-independent workflow core

`verification/workflow.py` holds all rules and has no I/O. `interactive.py` is a thin
terminal layer with an injectable prompter, so tests script complete reviewer sessions,
including quitting mid-way and resuming. The Phase 5 Streamlit UI will reuse the same
core.

## Milestone 4 decisions: web interface and reports

### 27. Three layers: engines, services, UI

```
ui/ (Streamlit pages)  ->  services/ (workspace, dashboard, reports)  ->  engines
                                                                          extraction/, retrieval/,
                                                                          verification/, storage
```

Pages contain layout and input handling only. Every rule lives in the engines, which
the CLI also uses, so the web interface cannot bypass an integrity check. Services
contain no Streamlit code and are tested directly.

### 28. Streamlit state: saved data is always reloaded

Streamlit reruns the whole script on every interaction. Each run therefore reloads the
current session from disk, and `st.session_state` holds only interface state: the
selected session, the reviewer ID, the file version last displayed, and a pending
confirmation. Three layers stop duplicate or conflicting writes:

1. **View freshness:** before a write, the page compares the file version on disk with
   the one it displayed. If they differ (another tab, the CLI, or a repeated click),
   it refuses and refreshes.
2. **Store:** saves pass `expected_version`, and the file must still have that hash.
3. **Schema:** one active verdict per claim. A repeated commit of the same record is
   a no-op.

Repeated extraction of the same text in one browser session reopens the existing
session unless the user asks for a new one.

### 29. Two-step save for verdicts

**Check and review** saves a draft and validates through `build_verification`. The
confirmation panel shows the verdict, cited sources, assessments, limitations and
rationale warnings. **Save verification** is disabled until the reviewer confirms, and
also acknowledges any warnings. At commit time the verification is rebuilt and
re-validated against the current file, so the stored timestamp and checks reflect
the moment of saving.

### 30. Reports are a pure function of validated records

`build_report(session, checks)` re-validates the session and derives everything else
from it. JSON is the canonical form, and CSV and Markdown are renderings of it.
`content_sha256` hashes the report without `generated_at`, so identical data gives
identical fingerprints. Active verdicts and revision history are separate fields
(no double counting), and candidates and assessed evidence are separate lists. The
report includes unresolved claims, integrity warnings and limitations. It contains no
accuracy or hallucination-rate figure.

### 31. Charts that cannot mislead

Dashboard and report charts are single-series horizontal bars with direct labels,
tooltips and a table view. They use one hue, with no red/green verdict colouring that
would imply a score. Each chart sits next to a caption saying the counts are not an
accuracy metric or a hallucination rate.

### 32. Upload through the existing ingestion

Uploaded files are reduced to a portable file name (directory parts from both POSIX
and Windows paths are stripped). They are written to a temporary folder and
registered with `CollectionCatalog.add_document`, so encoding, size, binary and
duplicate checks are unchanged. Packaged collections stay read-only, and typed
metadata is recorded as `user_supplied`.

## Milestone 5 decisions: ground-truth benchmark

### 33. Proposed labels and human labels are different data types

`ProposedLabel` (with a fixed "not ground truth" notice) is stored on the item. Human
judgements are separate write-once `Annotation` / `Adjudication` files, and the official
label exists only inside a human `Approval`. `decide_label` never reads the proposed
label, and a test proves that a wrong AI proposal cannot leak into the label. Reviewer
IDs reserved for automated authors are rejected by the schema.

### 34. Blind review

The proposed label is hidden by default in the CLI and the web interface. Revealing it
is possible but is recorded on the subsequent review
(`proposed_label_seen_before_review`), so anchoring bias can be audited.

### 35. Evidence as captured, hashed passages

Each passage has a stable ID (source + passage hash), a locator, its anchors and the
capture method. Text is stored only when the licence allows. Re-running the same
extraction on the same source must reproduce the hash. That is how
`check-sources --online` detects changes, and how the candidate build was verified to be
reproducible.

### 36. Item content fingerprint ties reviews to what was reviewed

Reviews and approvals store the item's `content_sha256` (claim, context, scope,
evidence). New evidence makes earlier reviews non-current instead of silently carrying
them over. Exports refuse approvals whose content changed.

### 37. Review kinds make the strength of a label explicit

First review, independent second review (different person) and delayed repeat review
(same person, at least 24 h later) are distinct types. The approved label carries its
`review_level`, so a one-person project is honest about what it is, rather than
presenting self-review as inter-annotator agreement.

### 38. Evaluation refuses what it cannot measure

Evaluation requires an immutable approved export, and predictions must match its
version and `dataset_sha256`. Coverage, abstention and missing counts are reported apart
from conditional and strict accuracy, and precision is *undefined* rather than 0 when a
class is never predicted. Test-partition runs are logged, and repeats are flagged.

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
| Examples, demo collection, and evaluation paths only resolve in a source checkout | Package data properly in Phase 8 (known gap) |
| Reviewers treat a high BM25 score as support | Banner on every result; no score-to-confidence conversion; docs show a contradicting top hit |
| Reference document silently edited | Content-addressed IDs, registration hashes, passage hashes, refusal to index |
| Licence violations in reference text | Only CC BY-SA and U.S. Government sources, attribution file, per-document licence fields |
| A retrieval score treated as a verdict | Schema requires explicit per-passage assessments for cited evidence; score shown as "NOT support" |
| Silent edits to past judgements | Append-only storage check; corrections only via revision records with reasons |
| Lost work on interruption | Drafts saved after every step inside the session file |
| Reviewer cites a passage whose source changed | Source re-check blocks citation until retrieval is re-run |
| Self-declared reviewer IDs | Documented limitation; cross-reviewer rule prevents accidents, not impersonation |
| Streamlit reruns cause duplicate or stale writes | Reload-per-run, view-freshness check, store version check, idempotent commit |
| Dashboard read as a hallucination rate | Captions on every count; split by data provenance; no rates computed |
| Tracebacks shown to users | Per-page error boundary; `showErrorDetails = "none"` |
| AI-authored labels mistaken for ground truth | Separate types; hidden during review; "provisional" notices; 0 approved at delivery |
| Tuning on the test split | Deterministic dev/test split; every test evaluation logged; repeat warning |
| Evidence source changes after capture | Passage hashes; online re-check; pinned Wikipedia revisions |
| Over-claiming from a tiny benchmark | Counts with every rate; small-sample note; review level per item |
