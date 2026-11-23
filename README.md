# pharma-supply-chain-copilot
IITKGP - Team Velocity Capstone Project Repo

This repository contains the regulatory-document indexing workflow and a
grounded question-answering pipeline for pharmaceutical supply-chain compliance.

## Working with an AI agent

The skill in [.github/skills/velocity-capstone/SKILL.md](.github/skills/velocity-capstone/SKILL.md)
covers onboarding, debugging, PR review and project status. Any coding agent
can be pointed at that file (for example "follow the instructions in
.github/skills/velocity-capstone/SKILL.md"); GitHub Copilot in VS Code loads it
automatically.

## Pipelines

### Build the regulatory corpus and search index

The [RAG/](RAG/) folder contains the source PDFs, metadata, scripts, notebook,
and generated FAISS/BM25 indexes. Follow [RAG/README.md](RAG/README.md) to
build or query the index.

### Ask a grounded compliance question

[`RAG_PIPELINE/scripts/rag_pipeline.py`](RAG_PIPELINE/scripts/rag_pipeline.py)
uses the built hybrid index to retrieve regulatory excerpts, applies a
confidence gate, calls an OpenAI-compatible Qwen3 chat endpoint, and verifies
citations, section identifiers, and numeric claims against the retrieved
context. It can attempt one self-repair when verification fails.

The pipeline abstains when retrieval is weak. By default, it blocks answers
that fail its verifier; `--allow-unverified` returns such answers clearly
flagged instead. The verifier checks grounding markers and numbers, not whether
the answer faithfully interprets the source, so human review is still needed.

### Evaluate the copilot

The [EVAL/](EVAL/) folder holds the 64-query evaluation set from Deliverable 1
and the scripts that run and score it.

| File | Purpose |
|---|---|
| `EVAL/eval_set_v1.jsonl` | The 64 queries with type and gold behaviour (A 14, B 13, C 13, D 12, E 12). |
| `EVAL/build_eval_set.py` | Rebuilds the JSONL from the Deliverable 1 `.docx` and prints its SHA-256 for the freeze record. |
| `EVAL/run_retrieval.py` | Retrieval-only run (embeddings API only): retrieved documents, scores, and whether the confidence gate would abstain. |
| `EVAL/run_eval.py` | Full run through `rag_pipeline.py`: one trace per query (decision, latency, tokens, retrieved documents, verifier result), decision accuracy and first-pass failure labels. |

Results are written to `EVAL/results/<run-name>/`.

```powershell
# Retrieval only (needs EMBEDDINGS_KEY and secrets/vm02-ca.crt)
python EVAL/run_retrieval.py --run-name retrieval-v1

# Full run (also needs LLM_URL, LLM_KEY and secrets/vm04-ca.crt)
python EVAL/run_eval.py --run-name smoke --limit 3
python EVAL/run_eval.py --run-name baseline-rag
```

Current limits: only the decision is scored (key facts, numbers and citation
validity need an LLM judge or manual review); ESCALATE and REFUSE items count as
wrong until the agent layer returns those decisions; PI, PII and out-of-scope
items are logged but not scored here. The retrieval-only run shows the
confidence gate never abstains on this set, so abstention needs a scope check in
the agent layer.

## Setup

Use Python 3.14.6 and install dependencies from the repository root:

```powershell
python -m pip install -r RAG/requirements.txt
```

Create the root `.env` from `.env.example` if it does not already exist. Set:

- `EMBEDDINGS_KEY` for the embeddings service.
- `LLM_URL` for the OpenAI-compatible chat service (base `/v1` URL or full
  `/chat/completions` URL).
- `LLM_KEY` if the chat service uses a separate key; otherwise it falls back to
  `EMBEDDINGS_KEY`.
- `LLM_MODEL` to override the default `Qwen/Qwen3-8B`.

The embeddings and chat TLS certificates default to the ignored root files
`secrets/vm02-ca.crt` and `secrets/vm04-ca.crt`. Override them with `EMBED_CERT`
and `LLM_CERT` if necessary. TLS verification remains enabled.

The single root `.env` and root `.gitignore` keep local credentials and
certificates out of Git. Do not commit credentials.

## Build and run

If the index has not been built yet, run these steps from the repository root:

```powershell
python RAG/scripts/build_manifest.py
python RAG/scripts/parse_and_chunk.py
python RAG/scripts/embed_and_index.py
```

Then ask a question:

```powershell
python RAG_PIPELINE/scripts/rag_pipeline.py --query "Which temperature controls apply under 21 CFR 211.142?"
```

Useful options:

```powershell
# Filter retrieval and return structured JSON, including retrieved context
python RAG_PIPELINE/scripts/rag_pipeline.py --query "What controls apply under 211.142?" --use-case C --json --show-context

# Skip the repair call or allow a failed verification as a flagged answer
python RAG_PIPELINE/scripts/rag_pipeline.py --query "What controls apply under 211.142?" --no-repair
python RAG_PIPELINE/scripts/rag_pipeline.py --query "What controls apply under 211.142?" --allow-unverified
```

Other controls include `--top-k`, `--token-budget`, `--max-tokens`, `--min-cosine`,
`--min-bm25`, `--doc-id`, `--think`, and `--verbose`. Confidence thresholds
should be calibrated against in-scope and out-of-scope questions.

The answer pipeline reports a status (`answered`, `abstained`,
`blocked_unverified`, `unverified_answer`, or `error`), citations, confidence
signals, verification details, and latency. Use `--json` for a structured
response.
