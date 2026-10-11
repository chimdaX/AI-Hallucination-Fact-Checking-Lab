# Web interface guide

The Streamlit interface runs **locally**. It is not deployed anywhere. It uses the same
extraction, retrieval, verification and storage code as the `hallab` CLI, and the same
session files, so you can switch between the CLI and the web interface at any time.

## Install and start

```bash
# from the repository root, inside your virtual environment
pip install -e ".[ui]"          # or: uv pip install -e ".[ui]"

streamlit run app/streamlit_app.py
# or
hallab ui                       # same thing; --port 8502 to change the port
```

Open **http://localhost:8501** in your browser.

- Data is read from and written to `./lab_data/` by default. To use another folder,
  set `HALLAB_DATA_DIR` before starting, for example
  `HALLAB_DATA_DIR=/path/to/folder streamlit run app/streamlit_app.py`. The sidebar
  shows which folder is in use.
- Start Streamlit from the repository root, so `.streamlit/config.toml` applies. That
  file hides Python tracebacks from the browser.
- Without Streamlit installed, the CLI still works fully, and `hallab ui` explains what
  to install.

## Pages

| Page | What you can do |
|---|---|
| **Dashboard** | Totals across all saved sessions: claims, flagged, human-verified, pending, active verdicts, revisions. Split by data type (illustrative fixture / user-submitted / authentic). Charts of verdicts and status. |
| **Analyze response** | Paste text, extract claims, see why each claim was selected or skipped, edit / split / remove / restore / add claims, mark claims for verification, retrieve evidence from a collection, inspect passages. Open saved sessions. |
| **Human verification** | Per claim: context, retrieval parameters, passages with source, URL, dates, licence, location and integrity status. Label passages, choose a verdict, cite passages, write a rationale, save a draft, review and confirm. Correct a verdict through a revision. |
| **Reports** | Session findings, charts (verdicts, status, evidence usage), unresolved claims, integrity warnings, limitations. Download **JSON / CSV / Markdown**. |
| **Reference collections** | Browse collections (packaged vs user-supplied), document metadata, licences, registration dates, SHA-256 hashes, integrity check, index status, document preview. Add `.txt`/`.md` documents to your own collections. |
| **Getting started** | Short explanation and an example to try. |

## Example workflow

1. In the sidebar, enter a **Reviewer ID** (e.g. `reviewer_01`). It is recorded on
   every edit and verdict.
2. **Analyze response**: paste
   `Pi is approximately 3.14159. The Eiffel Tower was completed in 1899. An unknown research group published a revolutionary energy discovery in 2040.`
   Tick *illustrative example* and click **Extract claims**.
3. Review the claims. Each one shows why it was kept or skipped. Edit any that are
   wrong.
4. Choose the `demo` collection and click **Retrieve evidence**. Expand a claim to see
   its candidate passages. The BM25 score is labelled *lexical overlap only*.
5. **Human verification**: pick a claim, label each passage, choose a verdict, select
   the cited passages, write a rationale, then click **Check and review**.
6. Read the summary. Tick the confirmation box, plus the rationale-warning box if
   warnings were shown, and click **Save verification**.
7. **Reports**: download the JSON, CSV or Markdown report.

## Saving, drafts and safety

- **Form inputs are temporary** until you click **Save draft** or confirm a
  verification. **Check and review** also saves a draft first.
- **Drafts** are stored in the session file. Close the browser, come back, and the
  page says *Resuming your saved draft*.
- **Nothing is saved without confirmation.** Save stays disabled until you tick the
  confirmation box.
- **Stale pages are refused.** If the session changed after the page was displayed
  (another tab, the CLI, or a double click), the action is not applied and the page
  refreshes. Underneath, the store refuses saves on a changed file, and the schema
  allows only one active verdict per claim.
- **No silent edits.** Saved verifications and evidence cannot be modified or removed.
  Claims with a verdict or a draft cannot be edited. Removing a claim keeps it as
  inactive, after a confirmation.
- **Repeated extraction:** pasting the same text twice in one browser session reopens
  the existing session unless you explicitly ask for a new one.

## Correcting a verdict

On **Human verification**, a claim with a verdict shows the active record and its
revision history. Click **Correct this verdict** and enter a **correction reason**.
When correcting someone else's verdict, you also need an **authorization** note. Then
redo the assessment, which starts from the previous labels, and confirm. The old
record stays in the file as *superseded*. Reports show both records, and counts
include only the active one.

## Reports

Reports are generated from schema-validated session records:

- session ID, original response (verbatim) and its provenance
- every claim with its review status
- the active human verdict, rationale, reviewer and timestamp
- cited passages with evidence IDs, source IDs, URLs, file locations, passage hashes
  and source-check status
- full revision history (superseded and active)
- retrieval candidates, kept separate from reviewer-assessed evidence
- unresolved claims, document integrity warnings, methodological limitations

`content_sha256` is identical for repeated exports of unchanged data, because it
excludes `generated_at`. The CLI equivalent is
`hallab report --session ID --format json|csv|md [--output FILE]`.

Reports contain **no accuracy figures and no hallucination rate**. Verdict counts
describe the claims reviewed in one session, using the chosen reference documents.

## Reference collections in the web interface

- **Packaged** collections (`demo`) are read-only.
- **Add a document** copies a `.txt` or `.md` file into a collection you own, under
  your data folder. It goes through the same validation as `hallab sources add`:
  encoding, size limit, binary detection, duplicate content, and filename
  sanitisation. Metadata you type is stored as `user_supplied` and is **not verified**.
- The integrity check re-hashes every document. A modified or missing document is
  shown in red, and searching that collection is blocked until it is restored or
  re-registered with `hallab sources add --replace`.

## Known limitations

- Single-user, local tool. There is no login, and reviewer IDs are self-declared.
- Concurrent edits are detected and refused, not merged.
- Large collections are re-hashed on every integrity check, which is fine for the
  small collections this tool targets.
- Streamlit reruns the page on each interaction, so very long sessions (hundreds of
  claims) will feel slower.
- The interface has been tested in headless Chromium on Linux (WSL). Other browsers and
  operating systems have not been verified.
