---
name: velocity-capstone
description: Onboard to, debug, review PRs for, and report outcomes of the Team Velocity pharma supply chain copilot (IITKGP Capstone Track B). Use when a teammate asks how to get started, why a RAG or eval run fails, to review a PR, or for project status against the capstone requirements.
---

# Velocity capstone: onboarding, debugging, PR review, outcomes

Repo: pharmaceutical supply chain AI copilot (RAG + planned Planner/Executor/Critic agent) on a hosted 8B model.
Treat the code on `origin/main` as the source of truth. This file can be stale: check the code before relying on any statement below.

## Ground rules for any agent
- Run `git fetch origin` and read `origin/main` (or update local `main`) before analysing code.
- Never print, log, commit or paste keys. `.env` and `secrets/` are git-ignored; report only whether a variable is set.
- Never disable TLS verification (no `verify=False`, `curl -k`). Fix the certificate instead.
- Do not approve or merge a PR for the user; summarise findings and let a human decide.
- Do not invent facts, identifiers or numbers. Cite the file and line, or say "not found".

## Project map
| Path | Purpose |
|---|---|
| `RAG/` | Source PDFs, chunking, FAISS + BM25 index, `retriever.py` (hybrid retrieval, RRF) |
| `RAG_PIPELINE/scripts/rag_pipeline.py` | Confidence gate, Qwen3-8B answer, citation/number verifier, one repair attempt |
| `RAG_PIPELINE/tests/` | Unit tests for the pipeline |
| `EVAL/` | Eval set and gold docs, `run_retrieval.py` (recall@k), `run_eval.py` (full run + spans), `tracing.py`, `trace_summary.py`, `build_dashboard.py` (HTML dashboard for a run), `build_failure_log.py` (failure_log.csv to review by hand), `demo.py` (fake-data demo run, no keys needed; output in git-ignored `EVAL/demo_output/`), `safety.py` (PII and injection checks), `tests/test_safety.py`, results under `EVAL/results/` |
| `.env.example` | Names of the settings (placeholders only) |

Capstone requirements (Track B): 60-100 query gold eval set frozen in Week 1, three prompting strategies, hybrid retrieval with a reranker and citations, recall@k measured separately, an agent with 3+ tools and a planner/executor/critic loop (10-step cap), tracing, a failure taxonomy, a UI with rate limiting and caching, prompt-injection and PII tests, a model card and report.

## 1. Getting started
1. Use Python 3.14 (the version the pinned requirements target). Newer versions may lack wheels.
2. `python3.14 -m venv .venv && source .venv/bin/activate && pip install -r RAG/requirements.txt`
3. `cp .env.example .env`, then set:
   - `EMBEDDINGS_KEY` (shared key from the coordinator)
   - `LLM_URL`, and `LLM_KEY` if the chat service uses a separate key
4. Put the CA certificates in `secrets/`: `vm02-ca.crt` (embeddings) and `vm04-ca.crt` (chat). Ask the coordinator if missing.
5. Check the embeddings service: POST `/v1/embeddings` with model `bge-small-en-v1.5`; expect HTTP 200 and 384 numbers.
6. Run: `python RAG_PIPELINE/scripts/rag_pipeline.py --query "<question>" --json`
7. Run the offline tests (no network needed): `python RAG_PIPELINE/tests/test_rag_pipeline.py` (27 tests). If `LLM_URL` is set in `.env` but `secrets/vm04-ca.crt` is missing, one test errors with "LLM certificate not found"; add the certificate or unset `LLM_URL`.
8. Run the eval offline tests (PII, injection, tracing, trace summary, dashboard, failure log, demo): `python EVAL/tests/test_safety.py` (18 tests).
9. Never commit `.env`, `secrets/`, or `.venv/`.

## 2. Debugging playbook
Work from the outside in and change one thing at a time.

| Symptom | Likely cause | Check |
|---|---|---|
| `Set LLM_URL in the repo-root .env` | Chat endpoint not configured | `.env` has `LLM_URL` and `LLM_KEY` |
| `TLS verification failed` / curl exit 60 / HTTP 000 | Wrong or missing CA for that port | Embeddings and chat use different certs (`vm02` vs `vm04`); confirm the file in `secrets/` |
| `EMBEDDINGS_KEY is not configured` | Placeholder left in `.env` | Value is not `YOUR_EMBEDDINGS_KEY` |
| HTTP 401 | Wrong key for that service | Test the service directly with curl before blaming code |
| HTTP 429 or slow | Rate limit (1 active request, 60/min on chat) | Pipeline already paces calls; do not parallelise |
| `No matching distribution` on install | Python too new | Recreate the venv with 3.14 |
| Status `blocked_unverified` | Verifier found an uncited number/section ID | Read `verification.issues` in the `--json` output |
| Answer says not found though text exists | Retrieval miss | Run with `--show-context`; check `retrieved_chunks` and scores |
| Status `error` | LLM call failed | See the issue text; retries are already built in |
| Slow queries | Which step is slow? | `python EVAL/trace_summary.py EVAL/results/<run>/spans.jsonl` shows average time per span (retrieve vs llm_call) |
| PII or injection concern | Output or index may carry unsafe text | `redact_pii()` / `find_injection()` in `EVAL/safety.py`; `EVAL/tests/test_safety.py` scans the indexed chunks |

Isolate the layer: retrieval (`EVAL/run_retrieval.py`) -> gate -> generation -> verifier. A failure belongs to the earliest layer that is wrong.

Known behaviour: the confidence gate never abstains on the eval set (scores for answerable, should-abstain and out-of-scope queries overlap), so abstention depends on the model and verifier. Do not treat a gate threshold change as a fix without re-running `EVAL/run_retrieval.py`.

Known data gap: 15 of the 22 ANSWER items in the eval set need data that is not in the 6-PDF corpus (labels, Orange Book, shortages, recalls, NDC). A wrong or abstained answer on these is expected until the corpus grows or the items are relabelled ABSTAIN. Check `EVAL/gold_docs_v1.json` (only supported queries are listed) before calling it a retrieval bug.

## 3. PR review checklist
Review the diff against `origin/main`. Report findings grouped as blocking / should fix / nit, each with file and line.

- Security: no keys, certificates, tokens, hostnames of private services, or `.env` content; no `verify=False`; no secrets in notebook outputs; no new tracked cache or index files that contain user queries.
- Correctness: matches the Deliverable 1 rules (R1-R8) and the four outcomes (ANSWER, ESCALATE, REFUSE, ABSTAIN); numbers come from tools, not the model; citations point to real retrieved chunks.
- Evaluation integrity: `EVAL/eval_set_v1.jsonl` is frozen once its hash is recorded; a change needs a new version (v1.1) and a logged reason. Results folders should name the eval set version they used.
- Tests: new logic has tests; existing tests pass; offline tests do not call the live services. Run `python RAG_PIPELINE/tests/test_rag_pipeline.py` and `python EVAL/tests/test_safety.py`.
- Privacy: queries and answers written to traces or logs go through `redact_pii()`; new log fields must not store raw user text.
- Hygiene: dependencies pinned in `RAG/requirements.txt`; no large generated files added unnecessarily; README and docs updated when behaviour changes.
- Scope: one concern per PR; description states what is not covered.

## 4. Project outcome and status
To report status, check the repo and compare with the capstone requirements above. Use only evidence from the repo or run results.
- Count what exists: documents in `RAG/docs/` (target 500-2,000), tools, agent loop, eval runs in `EVAL/results/`.
- Quote numbers from `summary.json`, `retrieval_summary.json` and `trace_summary.json` files with the run name and date. Do not estimate.
- Retrieval and answer quality are reported separately: recall@k and MRR come from `run_retrieval.py` (document level, only queries listed in `gold_docs_v1.json`); decision accuracy comes from `run_eval.py`.
- Separate: done and verified, done but not run against the live model, not started.
- Failure counts for the taxonomy come from `confirmed_label` in `failure_log.csv` (reviewed by a person). `auto_label` is only a first guess; do not report it as a result.
- Never quote numbers from `EVAL/demo_output/` (fake data from `demo.py`) as results.
- List blockers with an owner (for example the chat certificate, gold document IDs for queries beyond the 7 already mapped, missing `decision` and `reason_code` output fields).
- Flag risks early: scope (five use cases in four weeks), corpus size, abstention behaviour, and the Week 1 gate (eval set frozen and baseline recorded before agent code).

Suggested report layout: scope, what was built, evidence (metrics with run names), failures by taxonomy label (retrieval miss, reasoning error, tool-call or format error, hallucination, over-refusal, under-refusal, injection compliance, PII leak), blockers, next steps.
