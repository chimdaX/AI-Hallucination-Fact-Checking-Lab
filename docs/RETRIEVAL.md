# Evidence retrieval guide

## What retrieval is for

After claims are extracted from an AI response, a reviewer needs to find the passages in
trusted reference documents that address each claim. Retrieval does that search. It
returns **candidate passages** with their exact source location, so a human can read them
and decide whether the claim is supported, contradicted, or unresolved.

> **Retrieval is not verification.** A retrieved passage shares words with the claim.
> It may support the claim, contradict it, or be irrelevant. This milestone produces no
> verdicts.

For example, for the claim *"The Eiffel Tower was completed in 1899"* the top candidate
says the company "designed and built the tower from 1887 to 1889". The passage is
relevant because it **contradicts** the claim, but only a reader can tell. BM25 ranks it
highly because it shares the words *Eiffel*, *Tower*, *Paris*, and *World's Fair*.

## Supported documents

| Format | Extensions | Notes |
|---|---|---|
| Plain text | `.txt` | |
| Markdown | `.md`, `.markdown` | `#` headings become passage sections |
| MediaWiki plain-text | `.txt` | `== Heading ==` lines become passage sections |

- **Encodings:** UTF-8 (default), UTF-16 (detected via byte-order mark), Latin-1, and
  CP1252 (declare with `--encoding`).
- **Size limit:** 5 MB per document by default.
- **Rejected with a clear error:** empty or whitespace-only files, binary content (NUL
  bytes), undecodable text, unsupported extensions, and files over the size limit.
- **Not supported yet:** PDF, Word, and HTML. Convert them to text first. Adding them
  would bring heavy dependencies and layout-dependent extraction errors.

## Reference collections

A collection is a folder:

```
<collection>/
  collection.json      manifest: one entry per document (metadata + registered hash)
  documents/           the original files, never modified by indexing
```

- **`demo`** is packaged with the repository (`data/references/demo/`) and is read-only.
  Its origin is `packaged_demo`.
- **Your collections** live in `lab_data/collections/<name>/` (or under
  `$HALLAB_DATA_DIR`) and are created by `hallab sources add`. Their documents are marked
  `user_added`.

### Metadata: declared vs computed

| Declared (`SourceMetadata`) | Computed by the system (`SystemFacts`) |
|---|---|
| title, author/organization, publisher | SHA-256 of the raw file bytes |
| publication date, last updated, URL | SHA-256 of the canonical text |
| access date, document type, licence | byte size, character count, line count |
| source version, notes | detected encoding |

Declared metadata cannot be checked offline, so every record carries a `status`:

- `user_supplied` (default): typed in by whoever added the document, unchecked.
- `captured_at_download`: recorded automatically from the publisher's API or page when
  the document was obtained. Requires a `status_note` saying how.
- `manually_confirmed`: a named person checked it against the source. Requires a
  `status_note`.

Leave a field empty rather than guess. The schema never fills in dates, authors, or URLs.

Metadata file example (`--metadata meta.json`):

```json
{
  "title": "Example agency fact sheet",
  "author_or_organization": "Example Agency",
  "url": "https://www.example.gov/factsheet",
  "accessed_on": "2026-10-11",
  "document_type": "fact_sheet",
  "license": "Public domain (U.S. Government work)"
}
```

### Integrity

- **Content-addressed source IDs.** `source_id = hash(file bytes)`. The same document
  always gets the same ID in every collection. An edited document gets a new ID, so old
  evidence can never silently point at new text.
- **Duplicate detection.** Adding a file whose bytes match an already registered
  document is refused, even under a different file name. Near-duplicates (for example,
  the same text with different line endings) are not detected.
- **Modification detection.** `hallab sources verify` re-hashes every file. Indexing and
  searching refuse to run while any document differs from its registered hash. To accept
  a changed file deliberately, re-register it with `hallab sources add --replace`.
- **Stale index detection.** The saved index stores a fingerprint of the manifest
  (documents, hashes, metadata), the chunking settings, and the tokenizer version. If
  any of these changes, the index is rebuilt (CLI) or `StaleIndexError` is raised
  (`auto_build=False`).
- **Passage hashes.** Every passage, and every evidence record saved in a session,
  stores the SHA-256 of its text. A session whose quoted passage was edited fails to
  load.

## Ingestion and passage segmentation

1. **Canonical text:** decode, remove a byte-order mark, and convert `\r\n`/`\r` to `\n`.
   Nothing else changes: no case folding, no Unicode rewriting, no whitespace
   collapsing. All offsets refer to this text.
2. **Sections:** Markdown `#` headings and MediaWiki `== … ==` headings split the
   document. Each passage records its section path, for example
   `Surveys > 20th century`.
3. **Sentences:** each section is split with the same rule-based segmenter used for
   claims.
4. **Packing:** consecutive sentences are grouped into passages of at most `max_chars`
   (default 800). The next passage repeats the last `overlap_sentences` sentences
   (default 1), so a fact that straddles a boundary is still retrievable. Passages never
   cross sections.
5. **Long unpunctuated text** (tables, lists) is cut at line breaks or spaces, and an
   `long_unpunctuated_text` warning is recorded.

Each passage stores: `passage_id`, `source_id`, character offsets, 1-based line numbers,
section path, exact text, and text hash. `passage_id = hash(source_id, start, end)`, so
it is stable for the same document and settings. It changes if the chunking settings
change, and that is expected.

## BM25 retrieval

BM25 ranks passages by how many query words they contain, weighted by how rare each word
is in the collection and normalised for passage length:

```
score = Σ over query terms t found in the passage:
        idf(t) · tf · (k1 + 1) / (tf + k1 · (1 − b + b · len / avg_len))
idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5))
```

- **Defaults:** `k1 = 1.5`, `b = 0.75`, `top_k = 5`, `max_per_source = 3`.
- **Implemented directly** in `retrieval/bm25.py` (~60 lines), not with a library. This
  avoids a dependency, keeps IDF always positive (the classic formula can go negative),
  and lets each result show which terms contributed to its score.
- **Deterministic:** terms are processed in sorted order, ties are broken by passage
  position, and stored scores are rounded to 6 decimals.

### Tokenization

The same tokenizer is applied to passages and queries:

- lower-casing and accent folding (`São` → `sao`)
- numbers kept whole, with thousands separators removed (`8,848.86` → `8848.86`, plus
  `8848`)
- years and day numbers kept (`3 May 1921` → `3 may 1921`)
- symbols and units spelled out (`°C` → `celsius`, `%` → `percent`, `8,848 m` →
  `8848 metre`, `ft` → `foot`)
- a small stopword list removed. Numbers are never removed, and "may" is kept because
  it is also a month.
- light plural stemming (`cities` → `city`), with exceptions such as `mars` and
  `species`

### Result filtering

- Only passages with score > 0 (at least one shared term) are returned. Nothing is
  returned when there is no overlap.
- Identical passage text from different documents appears once.
- A passage that overlaps an already selected passage from the same document by more
  than half is skipped.
- At most `max_per_source` passages are taken from one document.

### Recorded for reproducibility

Every search records: method (`bm25`), `k1`, `b`, `top_k`, `max_per_source`,
`min_score`, tokenizer version, chunking settings, collection name, and collection
fingerprint. In a session these are stored on each `Evidence` record and on the
`RetrievalRun`.

## Interpreting results

| You see | It means | It does **not** mean |
|---|---|---|
| A high BM25 score | Many rare query words appear in the passage | The claim is true, or the passage supports it |
| A low score | Few or common words overlap | The passage is irrelevant (synonyms are missed) |
| No candidates | No passage shares a searchable word | The claim is false. The collection may simply not cover it. |
| `matched terms` | Which query words produced the score | That the passage's facts agree with those words |
| `terms not found anywhere` | E.g. the claim's number `9848` appears in no document | Evidence of a hallucination. It is a cue for the reviewer to look. |

BM25 scores are not comparable between different queries or collections, are not
probabilities, and must not be converted into "hallucination likelihood".

## Lexical search limitations

These are confirmed by the evaluation below:

- **Vocabulary mismatch:** "a year on Mars lasts 687 days" does not match the fact
  sheet's "Sidereal orbit period (days) | 686.980". There is no synonym handling.
- **Morphology:** only plurals are stemmed. "boils" does not match "boiling".
- **Numbers must match exactly:** `687` ≠ `686.980`, and a wrong number in a claim
  (`9,848`) matches nothing.
- **Pronouns:** "It is an irrational number" has no subject, so retrieval depends on the
  remaining words.
- **Topic crowding:** if a document has many passages on the topic, the passage holding
  the decisive fact can rank 4th or lower and be cut by `top_k` or `max_per_source`.
- **No phrase or proximity matching:** word order is ignored.

## Using it

```bash
# Inspect the packaged collection
hallab sources list --collection demo

# Build (or rebuild) the index; settings are configurable
hallab index build --collection demo --max-chars 800 --overlap 1

# Search one claim
hallab search --collection demo --claim "The Eiffel Tower was completed in 1889."
hallab search --collection demo --claim "Pi is 3.14159" --top-k 3 --json

# Your own collection
hallab sources add --collection my-refs --file notes.md --title "My notes" --license "CC BY 4.0"
hallab sources add --collection my-refs --file report.txt --metadata report.json
hallab sources verify --collection my-refs
hallab index build --collection my-refs

# Attach candidates to a saved session (only claims flagged for verification by default)
hallab extract --save --text "The Eiffel Tower was completed in 1899."
hallab retrieve --session <session_id> --collection demo
hallab sessions show <session_id>
```

`search` and `retrieve` rebuild a missing or stale index automatically and say so. They
never rebuild over a modified document: that is an error.

## How retrieval is saved in a session (schema 1.1.0)

- Each candidate becomes an `Evidence` record with source title and URL,
  `document_reference` (`demo/wikipedia_eiffel_tower.txt#L1-L2`), offsets, line numbers,
  section, `passage_id`, `passage_sha256`, source hash, origin (packaged/user), rank, raw
  score, matched terms, parameters, and run ID.
- Each run is a `RetrievalRun` that lists, per claim, the outcome (`candidates_found`,
  `no_candidates`, or `no_searchable_terms`) and the ranked evidence IDs. This separates
  "searched, nothing found" from "never searched".
- Re-running retrieval never modifies or deletes existing records. If the same passage is
  found again for the same claim, the existing evidence record is reused, because its ID
  is deterministic.
- Sessions saved by Milestone 1 (schema 1.0.0) load unchanged through an explicit
  migration. The new fields are optional.

## Retrieval evaluation

`hallab eval retrieval` runs the labelled set in
`data/evaluation/retrieval_relevance_demo.json`.

**How labels were made:** for each query, the relevant information is specified as
exact phrases from a named document. A retrieved passage is relevant if it comes from
that document and contains the phrase. These criteria:

- were written from the document text **before** the retriever was run on the queries
- do not depend on passage IDs or chunking
- are checked automatically against the collection (`validate_criteria`)

The labels were written by the AI coding assistant during development (a single
annotator) and **have not yet been independently reviewed**, so spot-check them.
Negative queries were confirmed by text search to have no relevant passage.

**Metrics** (positive queries only):

- Recall@k: share of a query's evidence criteria found in the top k
- Precision@k: relevant passages in the top k divided by k
- MRR: mean reciprocal rank of the first relevant passage

Negative queries are reported by their top scores.

**Results** (19 positive and 5 negative queries; defaults `top_k=5`, `max_per_source=3`,
800-character passages, 1-sentence overlap):

| Metric | Value |
|---|---|
| Recall@1 / @3 / @5 | 0.658 / 0.789 / 0.789 |
| Precision@1 / @3 / @5 | 0.684 / 0.316 / 0.189 |
| MRR | 0.728 |
| Mean top score, positive vs negative queries | 22.2 vs 9.9 |
| Highest top score on a negative query | 12.7 |
| Negative queries returning zero candidates | 0 of 5 |

How to read these numbers:

- **Four positive queries fail at k=5.** R16 (vocabulary mismatch) and R07 ("boils" vs
  "boiling") are tokenizer limits. R09 and R13 are retrieved at ranks 5 and 4 when
  `max_per_source` is 5 (recall@5 then rises to 0.895), so the per-source cap trades
  topic recall for source diversity.
- **Defaults were not tuned on this set.** With 19 queries, tuning would overfit.
  Improvements should be evaluated on a fresh query set.
- **BM25 returned candidates for every negative query.** The highest negative top score
  (12.7) exceeds some correct top hits (e.g. 11.3), so **no score threshold separates
  "evidence exists" from "nothing relevant exists"**. That judgement needs a human
  reviewer or a later verification step.
- These are passage-retrieval metrics on a 10-document collection. They are **not**
  hallucination-detection accuracy and do not generalise to other collections.
