# Pharma Supply Chain Copilot

IITKGP - Team Velocity Capstone Project repository.

The `RAG/` project directory builds a hybrid index from pharmaceutical
regulatory PDFs and searches it using FAISS dense vectors and BM25 sparse
retrieval. Project files, scripts, inputs, secrets, and generated indexes are
kept together under `RAG/`.

The Git repository metadata remains at the repository root; the working
directory for the commands below is `RAG/`.

## Project layout

| Path | Purpose |
|---|---|
| `input/` | Maintained source metadata used to build the manifest |
| `docs/` | Source PDF documents |
| `index/` | Generated manifest, chunks, reports, caches, and retrieval artifacts |
| `scripts/build_manifest.py` | Validates PDFs and creates the enriched document manifest |
| `scripts/parse_and_chunk.py` | Extracts text and creates section-aware chunks |
| `scripts/embed_and_index.py` | Creates dense and sparse indexes |
| `scripts/retriever.py` | Loads the index and performs hybrid retrieval |
| `scripts/local_secrets.py` | Loads local secrets from `.env` |
| `RAG.ipynb` | Production notebook for running and inspecting the complete pipeline |
| `requirements.txt` | Python dependencies for the scripts and notebook |
| `.env.example` | Secret-file template |

## Prerequisites

Use Python 3.14.6 (the version used to validate the pinned dependencies) and
install the project dependencies from the `RAG/` project directory:

```powershell
python -m pip install -r requirements.txt
```

Create the local secret file:

```powershell
Copy-Item .env.example .env
```

Set the actual key in `.env`:

```text
EMBEDDINGS_KEY=your_embeddings_key
```

The default CA certificate path is `secrets/vm02-ca.crt`. The certificate is
kept in the ignored `secrets/` directory and is not committed. If you need a
different certificate path, set `EMBED_CERT` in the environment or in `.env`,
for example `EMBED_CERT=C:\path\to\certificate.crt`. TLS certificate
verification remains enabled; do not disable it to work around certificate
errors.

## End-to-end pipeline

Run commands from the `RAG/` project directory. The order is required because each step
consumes output from the previous step.

```powershell
# 1. Read sources and validate their PDFs
python scripts/build_manifest.py

# 2. Parse PDFs and create chunks
python scripts/parse_and_chunk.py

# 3. Embed chunks and build FAISS/BM25 indexes
python scripts/embed_and_index.py

# 4. Search the completed index
python scripts/retriever.py --index-dir index --bm25-only --query "warehousing temperature 211.142"
```

The [RAG.ipynb](RAG.ipynb) notebook executes the same steps in this order.

## Inputs and generated files

### Source inputs

| File or directory | Produced by | Consumed by | Description |
|---|---|---|---|
| `input/sources.csv` | Maintainer/source setup | `build_manifest.py` | Source document IDs, titles, URLs, publication dates, use cases, and PDF bucket values |
| `docs/<doc_id>.pdf` | External source/download | `build_manifest.py`, `parse_and_chunk.py` | Regulatory source PDF. The file name must match `doc_id` |
| `.env` | Local operator | All embedding and retrieval code | Optional local `EMBEDDINGS_KEY` and `EMBED_CERT` configuration |
| `secrets/vm02-ca.crt` | Local operator | Embedding API client | Default CA certificate used to verify the embeddings service's TLS connection; ignored by Git |

### Manifest and chunk generation

| File | Producer | Input | Description |
|---|---|---|---|
| `index/manifest.csv` | `build_manifest.py` | `input/sources.csv`, `docs/*.pdf` | Enriched source metadata with SHA-256 hash, size, page count, text statistics, token count, and first-seen date |
| `index/chunks.jsonl` | `parse_and_chunk.py` | `index/manifest.csv`, `docs/*.pdf` | One JSON chunk per line. Fields include chunk ID, document ID, section metadata, text, embedding text, and token count |
| `index/chunk_report.json` | `parse_and_chunk.py` | `index/manifest.csv`, `docs/*.pdf` | Chunking configuration, per-document statistics, warnings, and errors |
| `index/debug/<doc_id>.txt` | `parse_and_chunk.py` | Extracted PDF text | Optional cleaned text for diagnosing parsing and boilerplate removal |

### Retrieval index artifacts

| File | Producer | Input | Description |
|---|---|---|---|
| `index/faiss.index` | `embed_and_index.py` | `index/chunks.jsonl` | Dense FAISS `IndexFlatIP` vector index with L2-normalized embeddings |
| `index/bm25.pkl` | `embed_and_index.py` | `index/chunks.jsonl` | Serialized BM25 index using the repository's fixed tokenizer |
| `index/indexed_chunks.jsonl` | `embed_and_index.py` | `index/chunks.jsonl` | Aligned chunk metadata; line `i` corresponds to FAISS row and BM25 document `i` |
| `index/index_manifest.json` | `embed_and_index.py` | Build configuration and source chunks | Embedding model, dimension, chunk count, chunk file hash, tokenizer version, and index type |
| `index/embed_cache.jsonl` | `embed_and_index.py` | Embedding API responses | Append-only resume cache for chunk embeddings |
| `index/query_cache.jsonl` | `retriever.py` | Query embedding API responses | Append-only cache for previously embedded queries |

## Safe rebuild and cache behavior

- `manifest.csv` preserves a document's original `first_seen` date and reports hash changes.
- `embed_cache.jsonl` allows embedding work to resume after interruption.
- `query_cache.jsonl` avoids repeated API calls for identical queries.
- Re-run the pipeline from `build_manifest.py` when source PDFs or source metadata change.
- Re-run `parse_and_chunk.py` and `embed_and_index.py` when chunking or embedding inputs change.
- Do not manually edit `faiss.index`, `bm25.pkl`, or `indexed_chunks.jsonl`; rebuild them together.

## Secrets

- `.env` is local and ignored by Git.
- `secrets/` is local and ignored by Git.
- `.env.example` contains a placeholder only and is safe to commit.
- If a key is accidentally committed, revoke and rotate it immediately.

## Chunk JSONL Field Reference

Each line of `index/chunks.jsonl` is one JSON object (one chunk), produced by
`parse_and_chunk.py`. Fields are listed in the order they are written.

| Field | Description |
|---|---|
| `chunk_id` | Unique ID like `DOC_c0007` for text chunks or `DOC_t0002` for table chunks. |
| `doc_id` | ID of the source document, matching `index/manifest.csv` and the PDF file name. |
| `chunk_type` | `text` for normal prose or `table` for chunks built from extracted tables. |
| `section` | Title of the deepest heading the chunk starts under, e.g. `§ 211.22 Responsibilities of quality control unit.` |
| `section_path` | Full heading hierarchy joined with ` > `, e.g. `Subpart B > § 211.22 …`, or `General / Introduction` if no heading was found. |
| `section_number` | Number pulled from the heading (`211.22`, `5.1`, `Annex 9`), or empty if none. Useful for citations and exact-match filters. |
| `heading_level` | Depth of that heading (1 = top level, 0 = no heading). |
| `sections_covered` | List of section paths whose text is in the chunk. It has more than one entry only when a tiny section was merged forward. |
| `page_start` | First PDF page (1-based) the chunk's text comes from. |
| `page_end` | Last PDF page (1-based) the chunk's text comes from. |
| `use_case` | Value copied from the manifest, for filtering by use case. |
| `published_date` | Publication date copied from the manifest. |
| `title`, `authority`, `jurisdiction`, `doc_type`, `version`, `source_url` | Optional manifest columns, copied only when the column exists and the value is non-empty. |
| `token_count` | Number of tokens in `embed_text`, measured with the BGE tokenizer without special tokens. Always under the 400 hard cap. |
| `content_hash` | Short SHA-1 of the lowercased, whitespace-normalized `text`, used for de-duplication and change detection on re-runs. |
| `text` | Clean chunk text for display, citations and LLM context. It has no prefix. |
| `embed_text` | `text` with a `[doc_id \| section_path]` prefix. This is the string to embed, so section context is part of the vector. |

### Using chunk fields

- **Embed** `embed_text`.
- **Show** `text` to the user and pass it to the LLM as context.
- **Keep the rest as metadata** for filtering and citations, e.g. `doc_id`,
  `section_number`, `page_start`, and `page_end`.

### Related chunk report

`index/chunk_report.json` is written alongside `index/chunks.jsonl`. It contains
the chunking configuration and per-document QA statistics and warnings.

- `boilerplate_patterns` lists repeating header and footer text detected and
  removed. Up to 20 patterns are shown per document.
- Digits are replaced with `#`, so `Page 1 of 29` becomes `page # of #`.
- A pattern counts as boilerplate when it appears in the top or bottom 7% of at
  least 30% of pages (minimum 3 pages).
- If real content appears in the list, raise `boilerplate_page_ratio` or narrow
  `header_margin` / `footer_margin` in `Config`.
- If headers or footers survive in `index/debug/<doc_id>.txt`, lower
  `boilerplate_page_ratio` or widen the margins.
- Documents under 3 pages skip detection, so their list is empty.
