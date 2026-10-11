# Human verification guide

This guide covers how to judge a claim from an AI-generated response against retrieved
evidence, and how to record that judgement with `hallab verify`.

## Four different things

| Thing | What it is | Who produces it |
|---|---|---|
| **Claim** | One factual statement extracted from the AI response | Rule-based extractor, edited by reviewers |
| **Evidence candidate** | A passage that shares words with the claim (BM25) | Retrieval. Automated, says nothing about truth |
| **Evidence assessment** | Your label for one passage: SUPPORTING, CONTRADICTING, IRRELEVANT, or UNCERTAIN | You |
| **Verification verdict** | Your conclusion about the claim, with cited passages and a rationale | You |

A retrieval score is never an assessment, and an assessment is never automatic. The
schema refuses a SUPPORTED or CONTRADICTED verdict unless every cited passage carries
your explicit, matching assessment.

## Step 1: Understand the claim

- Read the **normalized claim** and the **original wording** in its **response
  context** (marked `>>> <<<`). The context often supplies a missing subject, e.g.
  "It was designed by…".
- Work out what exactly would have to be true. Note the specifics: dates, numbers,
  names, places, and qualifiers such as "about", "first", or "currently".
- Check the **time context**. Time-sensitive claims ("currently the tallest…") must be
  judged against the evidence's date, which is shown as the revision or last-updated
  date.
- If the claim mixes several facts, judge the one stated, or ask for it to be split
  before verification (see claim editing).

## Step 2: Assess each passage

| Label | Use when the passage… | Example for "The Eiffel Tower was completed in 1899" |
|---|---|---|
| **SUPPORTING** | establishes the claim, or the specific part you are checking | — |
| **CONTRADICTING** | establishes that the claim is false or materially wrong | "The main structural work was completed at the end of March 1889" |
| **IRRELEVANT** | shares words but does not bear on the claim | "After it was completed, some have put forward various mathematical hypotheses…" |
| **UNCERTAIN** | might bear on it but is ambiguous, partial, or unclear | a passage giving a year without saying what was completed |

Points to keep in mind:

- **A high score is not support.** In the demo, "An unknown research group published a
  revolutionary energy discovery in 2040" scores 8.26 against a passage about computing
  π. It shares the words *discovery* and *published*, and that is all.
- **Read the whole passage.** Partial matches are common, and the key qualifier may be
  in the next sentence.
- **Check the source line.** It shows the URL, revision, licence, a **source check**
  (has the document changed since retrieval?), and a **credibility note**. The tool
  cannot establish whether a source is reliable. Every demo source is marked
  "credibility not independently established", and Wikipedia is a tertiary source. If
  reliability matters for your conclusion, say so in the rationale.
- **Blocked passages:** if the source document changed or disappeared after
  retrieval, the passage can only be marked IRRELEVANT or UNCERTAIN until you re-run
  `hallab retrieve`.
- Add a short note when your label is not obvious, e.g. "gives 1889 for completion".

## Step 3: Choose a verdict

### SUPPORTED

Use when available, reliable evidence **adequately establishes** the claim.

- You must cite at least one passage you assessed as SUPPORTING.
- The rationale names the passage and explains the connection, e.g. "The lead passage
  gives π ≈ 3.14159, the same value and precision as the claim."
- Approximate claims are supported by precise evidence within the claim's stated
  precision ("about 8,849 m" vs "8,848.86 m"). A *different* value is not support.
- If you also marked contradicting passages, explain why they do not outweigh the
  support, or choose INSUFFICIENT_EVIDENCE.

### CONTRADICTED

Use when available, reliable evidence **establishes that the claim is false or
materially incorrect**.

- You must cite at least one passage you assessed as CONTRADICTING.
- The rationale states what the evidence says and how it conflicts, e.g. "Source says
  the main structure was completed in March 1889; the claim says 1899."
- "Materially incorrect" means the error changes the meaning: a wrong year, a wrong
  person, a figure off by an order of magnitude. A trivial rounding difference within
  the claim's stated precision is not a contradiction.

### INSUFFICIENT_EVIDENCE

Use when there is **not enough reliable information** to decide. This covers:

- retrieval returned nothing, or only irrelevant passages
- the evidence is weak, partial, outdated, or ambiguous
- reliable sources conflict and you cannot resolve the conflict
- the only relevant source is of doubtful reliability

The rationale explains **what is missing or inconclusive**, e.g. "No retrieved passage
mentions this research group or any 2040 publication; the collection does not cover
it."

### NOT_CHECKABLE

Use when the statement is **fundamentally not an objective factual proposition**: an
opinion ("the most beautiful city"), a subjective evaluation, a normative statement, or
something too vague to test without clarification ("prices are high").

- The rationale explains **why** it cannot be checked objectively.
- **Do not** use NOT_CHECKABLE because a claim is hard to research or no evidence was
  found. That is INSUFFICIENT_EVIDENCE. The tool warns if your rationale suggests this
  mix-up.

### Insufficient evidence vs contradiction

| Situation | Verdict | Why |
|---|---|---|
| The sources say the tower was completed in 1889; the claim says 1899 | CONTRADICTED | Evidence positively establishes a different value |
| Nothing retrieved mentions the claimed 2040 discovery | INSUFFICIENT_EVIDENCE | **Absence of evidence is not evidence of falsehood.** The collection may simply not cover it |
| One source says 1887, another 1889, and you cannot tell which is right | INSUFFICIENT_EVIDENCE | An unresolved conflict establishes neither value |
| A passage gives the year but for a different event (e.g. construction start) | UNCERTAIN passage, then probably INSUFFICIENT_EVIDENCE | The evidence does not address the claim as stated |
| "73% of coastal cities will be underwater by 2050" with no source covering it | INSUFFICIENT_EVIDENCE | Even if it sounds implausible, plausibility is not evidence |

## Step 4: Write the rationale

A good rationale lets someone else check your work. It says **which evidence**, **what
it says**, and **how that relates to the claim**. It can be short:

- ✅ "Source dates the main structure to March 1889, not 1899."
- ✅ "No passage gives any population figure for Brindle."
- ✅ "Subjective aesthetic judgement; no objective criterion."
- ❌ "n/a", "tbd", "4/5", "correct": **rejected**
- ⚠️ "Looks fine.", "The claim is correct.", "The AI is wrong.", "Good evidence.":
  **warned**, because they explain nothing. You may keep them after the warning, but
  you should not.

The checks are simple heuristics. They catch empty or placeholder text and flag
generic text. They **cannot** tell whether your reasoning is sound. That responsibility
stays with you.

## Using `hallab verify`

```bash
# Prerequisite: a saved session with retrieval results
hallab extract --save --text "..."            # or: hallab demo --save
hallab retrieve --session SESSION_ID --collection demo

# Review every claim awaiting verification
hallab verify --session SESSION_ID --reviewer-id reviewer_01

# Review specific claims only
hallab verify --session SESSION_ID --reviewer-id reviewer_01 --claim-id CLAIM_ID

# Progress and results
hallab verify status  --session SESSION_ID
hallab verify summary --session SESSION_ID
hallab verify summary --session SESSION_ID --json
```

At every prompt, `?` shows help and `q` pauses. For each claim, the tool:

1. shows the claim, its context, how it was extracted, and the retrieval run (method,
   parameters, query terms)
2. shows each passage with source, URL, version date, location (file, lines,
   section), retrieval time, matched terms, source check, credibility note, and licence
3. asks for a label: `s`/`c`/`i`/`u`, `k` to skip, `f` to show the full passage
4. asks for a verdict: `s`/`c`/`i`/`n`. SUPPORTED or CONTRADICTED are refused if you
   have not marked a matching passage.
5. asks which marked passages to cite
6. asks for a rationale, then runs the checks above
7. shows a **review summary**, and saves **only if you type `y`**. `e` goes back to the
   verdict, and `n` keeps your answers as a draft and moves on.

### Pausing and resuming

Your work is saved as a **draft** after every step. Typing `q`, pressing Ctrl-D or
Ctrl-C, or losing the terminal leaves the draft in the session file. Run the same
command again and the tool offers to resume it: earlier passage labels are kept, and
you confirm or change each one.

Drafts are never counted as verifications. `verify status` lists them separately.
Saves fail with a clear message if the session file was changed by another process
since you loaded it.

### Correcting a verdict

Saved verifications are never edited or deleted. To correct one:

```bash
hallab verify --session SESSION_ID --reviewer-id reviewer_01 --revise VERIFICATION_ID
```

- You must give a **correction reason**.
- The tool shows the previous verdict and rationale. Your passage labels start from the
  previous ones.
- The result is a **new record** with `revision_of` pointing at the old one. The old
  record stays in the file unchanged.
- The newest record in the chain is the **active** verdict. Superseded records are kept
  for audit and are not counted as extra verifications.
- Only the latest record can be revised. Revision histories cannot branch or loop.
- Revising **another reviewer's** record requires
  `--authorization "who approved this and why"`. The authorization is stored on the
  record.
- A claim can have only one active verdict. A second review of the same claim must be
  a revision.

`verify summary` shows the revision number, the record it revises, and the reason.
`summary --json` includes the full chain information.

## Worked demonstration (three cases)

These are **illustrative scenarios** for trying the workflow. No judgements are
prefilled. The verdicts are yours to make.

```bash
hallab extract --synthetic --save --text "Pi is approximately 3.14159. The Eiffel Tower was completed in 1899. An unknown research group published a revolutionary energy discovery in 2040."
hallab retrieve --session SESSION_ID --collection demo --top-k 3
hallab verify --session SESSION_ID --reviewer-id YOUR_ID
```

`--synthetic` labels the text as an illustrative fixture rather than a real model
output.

**Case A: "Pi is approximately 3.14159."** The top candidate is the lead of the *Pi*
Wikipedia article ("…a mathematical constant, approximately equal to 3.14159…"). The
other candidates match only "3.14159" or "pi" in unrelated contexts, such as Pi Day
trivia. Decide whether the lead passage establishes the claim, whether a tertiary
source is adequate for a well-known constant, and cite what you rely on.

**Case B: "The Eiffel Tower was completed in 1899."** Candidates include "The Eiffel
Tower was the world's tallest structure when completed in 1889…" and "The main
structural work was completed at the end of March 1889…". Consider whether this
positively establishes a different year, which would make the claim CONTRADICTED.
Explain the conflict concretely.

**Case C: "An unknown research group published a revolutionary energy discovery in
2040."** The candidates are about computing π, a TV documentary on Everest, and a NASA
table footnote. They share only words like *discovery* or *published*. Consider whether
anything addresses the claim at all. If not, the defensible verdict is about *missing*
evidence, not falsehood. Also note that 2040 is in the future relative to every source.

## Known methodological limitations

- **Single reviewer.** There is no inter-annotator agreement yet, and one person's
  verdict is recorded as the active verdict. Multi-reviewer adjudication is future work.
- **Evidence is limited to the collection.** Verdicts are relative to the retrieved
  passages from the chosen reference documents. INSUFFICIENT_EVIDENCE often means "not
  covered", not "unknowable".
- **Retrieval misses things.** Lexical search misses synonyms and paraphrases (see
  `docs/RETRIEVAL.md`). A reviewer who never sees the decisive passage may reach
  INSUFFICIENT_EVIDENCE when better evidence exists.
- **Source reliability is not assessed by the tool.** The tool shows metadata status
  and source type. Judging reliability is the reviewer's responsibility and should be
  mentioned in the rationale when it matters.
- **Rationale checks are heuristics.** They catch empty or placeholder text and flag
  generic phrasing. They do not measure reasoning quality.
- **Reviewer identity is self-declared.** `--reviewer-id` is not authenticated, so the
  cross-reviewer rule prevents accidents, not deliberate impersonation.
- **Concurrent editing is detected, not merged.** If two people edit the same session
  file, the second save is refused and must be redone after reloading.
