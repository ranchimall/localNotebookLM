This project falls under the ProjectAI Token System of RanchiMall Artificial Intelligence Blockchain Contract AIBC. This semantically searches your ingested notes and documents (Keep, PDF, DOCX, TXT, images) and exports results as PPTX/PDF.

# Local NotebookLM (Chroma + sentence-transformers)

A local, private, NotebookLM-style RAG pipeline. Point it at a folder of
your own files — Google Keep exports, PDFs, Word docs, plain text, and
images — and it builds a searchable index you can query in plain English,
with an optional generated answer or a slide/PDF-deck summary of the
results.

Everything runs on-device — embeddings via `sentence-transformers`, storage
and vector search via `ChromaDB` (its built-in HNSW index is plenty fast for
a personal notes collection, so there's no separate FAISS step). The only
thing that ever leaves your machine is the question text itself, and only
if you opt in to the Claude-generated answer step.

## What it can ingest

Drop any mix of these into one folder and `ingest.py` will pick them all up:

| Type | Source | Notes |
|---|---|---|
| `.json` | Google Keep Takeout notes | Text notes and checklists (`listContent`). Trashed notes are skipped; archived notes are skipped by default. |
| `.pdf` | Any PDF | Extracts text per page; skipped if a PDF has no extractable text (e.g. a pure image scan with no OCR pass). |
| `.docx` | Word documents | Extracts paragraph text. |
| `.txt` | Plain text files | Read as UTF-8; files that fail to decode are skipped with a warning. |
| `.png` / `.jpg` / `.jpeg` | Images | Run through Tesseract OCR; images with no detectable text are skipped. |

All formats are normalized into the same shape internally (title + body +
metadata), so they're searched and displayed consistently regardless of
where they came from.

## 1. Set up the environment

```bash
cd rag_basic
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You'll also need the **Tesseract OCR binary** installed separately (it's
not pip-installable) if you want image ingestion:
- macOS: `brew install tesseract`
- Windows: [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt install tesseract-ocr`

> **Heads up:** `ingest.py` currently hardcodes the Tesseract path for
> Windows (`pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`).
> On macOS/Linux, comment out or remove that line (Tesseract just needs to
> be on your `PATH`), or point it at your actual binary location.

On Apple Silicon (M1/M2/M3) everything else installs and runs natively — no
special flags needed. First run of `ingest.py` will download the embedding
model (~80MB for the default `all-MiniLM-L6-v2`).

## 2. Organize your source files

`ingest.py` takes a single `--input-dir` and globs for every supported
extension inside it — it doesn't need separate folders per type. If you'd
rather keep things tidy (e.g. `Keep/`, `PDF/`, `Docs/`, `Txt/`, `Images/`),
just run `ingest.py` once per subfolder pointing `--input-dir` at each, or
flatten them into one folder first — either works, since ingestion is
additive and upserts by ID.

## 3. Ingest your files

```bash
python ingest.py --input-dir "/path/to/your/files"
```

This creates a persistent database in `./chroma_db`. Useful flags:

- `--include-archived` — by default archived Keep notes are skipped
- `--db-path ./my_db` — custom storage location
- `--model all-mpnet-base-v2` — a larger, more accurate (slower) embedding model
- `--max-chars 800` — chunk size for long documents (pasted articles, long PDFs)

Re-running `ingest.py` on the same folder is safe — it upserts by a hash of
the file path, so it won't create duplicates.

## 4. Query it

Retrieval only (no API key needed):

```bash
python query.py "what was that recipe with lentils I saved?" --no-llm
```

With a generated answer (uses the Claude API — set `ANTHROPIC_API_KEY` in
your shell first):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python query.py "gift ideas I noted down for mom"
```

Other flags:

- `--top-k 8` — retrieve more/fewer chunks
- `--label Recipes` — restrict to results tagged with that exact label (Keep notes only)
- `--full` — show each result's full original document instead of just the matched chunk

If `anthropic` isn't installed or `ANTHROPIC_API_KEY` isn't set, `query.py`
still works — it just prints the retrieved results without a generated
answer on top.

## 5. Turn results into a deck or PDF

`present.py` reuses the same retrieval logic as `query.py`, but instead of
printing text it lays out one slide (or page) per retrieved result — raw
text, no summarization — styled with an accent color per note.

```bash
# PowerPoint deck
python present.py pptx "openclaw project notes" --top-k 20 --output openclaw_deck

# PDF booklet
python present.py pdf "openclaw project notes" --top-k 20 --output openclaw_deck
```

Shared flags (same as `query.py`): `--db-path`, `--collection`, `--model`,
`--top-k`, `--label`, `--full`, plus `--output` for the filename (no
extension — `.pptx`/`.pdf` is added automatically).

Notes on styling:
- Each slide/page gets an accent color pulled from the Keep note's `color`
  metadata (mapped in `KEEP_COLOR_HEX` in `present.py`); non-Keep sources
  and unset colors fall back to a default slate blue.
- Checklist lines (`[x] item` / `[ ] item`) render as actual checkboxes,
  not raw brackets.
- Long PDF pages auto-continue onto a "(cont.)" page rather than
  truncating.

## Notes on the design

- **Chunking**: most notes are short, so they're stored whole. Long
  documents (pasted articles, long PDFs, long lists) get split into
  ~800-character chunks with overlap so context isn't lost at boundaries.
- **Checklists**: Keep's checklist notes (`listContent`) are flattened into
  `[x] item` / `[ ] item` lines so they embed, read, and render sensibly.
- **Metadata kept per chunk**: title, labels, pinned/archived flags, color,
  creation timestamp, source filename, and (for PDFs) page count — useful
  for filtering or displaying results. Non-Keep sources leave the
  Keep-specific fields (labels, pinned, color, etc.) blank/default.
- **Full-text fallback**: every chunk also carries the full original
  document's text in its metadata, so a search hit on one chunk can still
  show (or export) the whole document with `--full`.
- **No FAISS**: Chroma's own index already gives you fast approximate
  nearest-neighbor search out of the box; adding FAISS on top would just be
  a second index doing the same job.
