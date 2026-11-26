---
name: velocity-capstone
description: Onboard to, test, debug, review PRs for, and report outcomes of the Team Velocity pharma supply chain copilot (IITKGP Capstone Track B). Use when a teammate asks how to get started, how to test a part or the whole system, why a run fails, to review a PR, or for project status against the capstone requirements.
---

# Velocity capstone: onboarding, testing, debugging, PR review, outcomes

Repo: a pharma supply-chain copilot on one hosted 8B model. Question -> safety rules -> hybrid retrieval over an
openFDA corpus -> Planner (LLM) -> deterministic tools -> Critic -> output guard -> observability.
Every reply is ANSWER, ESCALATE, REFUSE or ABSTAIN.
Treat the code on `origin/main` as the source of truth. This file can be stale: check the code before relying on it.

## Ground rules for any agent
- Run `git fetch origin` and read `origin/main` (or update local `main`) before analysing code.
- Never print, log, commit or paste keys. Keys are passed at run time (`--api-key-file` or a hidden prompt); `.env` ignores `*_API_KEY`. Report only whether a value is set.
- Never commit `.env`, `secrets/`, `*.crt`, `*.pem`, key files, `data/raw/` or `data/processed/`.
- Never disable TLS verification (no `verify=False`, `curl -k`). Fix the certificate instead.
- Do not approve or merge a PR for the user; summarise findings and let a human decide.
- Do not invent facts, identifiers or numbers. Cite the file and line, or say "not found".

## Project map
| Path | Purpose |
|---|---|
| `src/copilot/safety/` | `rules.py`: rule pre-check R1-R8, redaction (phones, emails, dates of birth, labelled names), output guard |
| `src/copilot/rag/` | `corpus.py`: openFDA corpus in 800-char passages; `retrieval.py`: BM25 + dense + RRF, optional reranker (`rerank: true`) |
| `src/copilot/agent/` | `engine.py`: Planner -> Executor -> Critic loop, 10-step cap, token budget, per-query metrics in `last_metrics`; `tools.py`: deterministic tools; `llm.py`: model clients (per-port CA, retries, token counts) |
| `src/copilot/observability/` | `tracing.py` (`trace.jsonl`), `metrics.py` (result rows, summary, recall@k), `failure_log.py`, `dashboard.py`. How it is built into the engine (three layers: `last_metrics`, trace, run reports): README section 5.1 |
| `src/copilot/runtime.py`, `config.py`, `schemas.py`, `service.py` | Endpoint flags and key files; settings; answer schema; builds the engine |
| `scripts/` | `demo.py` (per part or `all`), `run_eval.py`, `eval_retrieval.py`, `rebuild_report.py`, `ask.py`, `build_corpus.py`, `check_setup.py`, `freeze_eval.py` |
| `app/app.py` | Streamlit app; rate limit from `configs/settings.yaml`; caching |
| `eval/gold/` | Frozen 64-query set + SHA-256, retrieval gold set |
| `eval/results/` | Reported runs (force-added; the folder is otherwise git-ignored) |
| `tests/` | 77 offline tests (fake model and embeddings, no keys) |

Course endpoints (verified 2026-11-25): chat VM04 `https://<COURSE_HOST>:1034/v1`, model `Qwen/Qwen3-8B`, cert `vm04-ca.crt`;
embeddings VM02 `https://<COURSE_HOST>:1035/v1`, `bge-small-en-v1.5`, cert `vm02-ca.crt`. The host, certs and keys come
from the course coordinator and must never be committed. Each server has its own key
and its own CA; a cert only verifies its own port. Ports 1031-1033 belong to other servers.

Capstone requirements (Track B): 60-100 query gold eval set frozen in Week 1, three prompting strategies, hybrid
retrieval with a reranker and citations, recall@k measured separately, an agent with 3+ tools and a
planner/executor/critic loop (10-step cap), tracing, a failure taxonomy, a UI with rate limiting and caching,
prompt-injection and PII tests, a model card and report.

## 1. Getting started and testing
The full testing guide is section 2 of `README.md`: three levels (offline, with corpus, live model), a
per-component table (unit-test command, demo command, expected output) and an 8-step whole-system checklist.
Point teammates there. Short version:
1. Python 3.11+: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
2. Offline: `python -m pytest -q` (77 passed); `python scripts/demo.py safety`, `tools`, `dashboard --open` (fake data).
3. Live: `python scripts/check_setup.py --live <flags>` where `<flags>` are `--chat-endpoint`, `--model`,
   `--chat-ca-cert`, `--api-key-file`, `--embed-endpoint`, `--embed-model`, `--embed-ca-cert`, `--embed-api-key-file`.
4. Corpus once: `python scripts/build_corpus.py --fetch --embed <flags>` (about 4 minutes, git-ignored).
5. Whole system: `python scripts/demo.py all <flags>`; evaluation: `python scripts/run_eval.py <flags>` (about 6 minutes).

## 2. Debugging playbook
Work from the outside in and change one thing at a time.

| Symptom | Likely cause | Check |
|---|---|---|
| `CERTIFICATE_VERIFY_FAILED` | Cert and port don't match | VM04 cert with port 1034, VM02 cert with 1035; `check_setup.py --live` prints a hint |
| HTTP 401 | Wrong key for that server | Each server has its own key; test with `check_setup.py --live` |
| HTTP 429 or slow | Busy server | 429/5xx are retried 3 times with backoff; do not parallelise calls |
| Embeddings HTTP 413 / 422 "512 tokens" | Request too big | At most 16 texts per request; passages under 512 tokens (`chunk_chars: 800`) |
| "Model endpoint error" in an answer | Model call or JSON parse failed | `error` in `results.jsonl` / dashboard; `ValueError` = invalid JSON (retried once) |
| Critic rejects repeatedly / step cap | Schema problems in the model's JSON | Trace shows `critic ... problems`; the problem text is fed back to the model |
| Answer ABSTAIN though the fact exists | Retrieval miss or data gap | `python scripts/demo.py retrieval "<question>"`; check the passage is in `data/processed/corpus.jsonl` |
| Slow or expensive query | Which stage? | Dashboard "Where the time goes"; `retrieve_ms`, `llm_ms`, `tool_ms`, tokens in `results.jsonl` |
| `No corpus yet` | Corpus not built | `python scripts/build_corpus.py --fetch` |
| `ModuleNotFoundError: copilot` | Not installed | `pip install -e ".[dev]"` from the repo root |

Isolate the layer: safety rules (`demo.py safety`) -> retrieval (`demo.py retrieval`, `eval_retrieval.py`) ->
agent (`demo.py agent` shows every step). A failure belongs to the earliest layer that is wrong.

Known data gap: most wrong ANSWER items need data the corpus does not have (Orange Book, products outside the
20 drugs in `configs/drug_names.txt`, shortages beyond the first 100). Check the corpus before calling it a bug.

## 3. PR review checklist
Review the diff against `origin/main`. Report findings grouped as blocking / should fix / nit, each with file and line.

- Security: no keys, certificates, tokens or `.env` content; no `verify=False`; no data files or embeddings committed.
- Correctness: matches rules R1-R8 (`configs/refusal_rules.yaml`, `src/copilot/safety/rules.py`) and the four outcomes; numbers come from tools, not the model; citations point to retrieved passages.
- Evaluation integrity: `eval/gold/eval_set_v1.0.jsonl` is frozen (hash in `eval_set_v1.0.sha256`); a change needs a new version and a logged reason.
- Tests: new logic has tests; `python -m pytest -q` passes; offline tests never call the live services.
- Privacy: anything written to traces, results or logs goes through `redact_pii()`.
- Docs: `README.md` (testing guide, expected outputs) updated when behaviour or commands change.
- Scope: one concern per PR; the description states what is not covered.

## 4. Project outcome and status
Use only evidence from the repo or run results.
- Quote numbers from `eval/results/run_*/summary.json` and `eval/results/retrieval_*/retrieval_summary.json`, with the run name. Do not estimate.
  Latest: `run_20261125_201203` (decision accuracy 53%, avg latency 5.5 s) and `retrieval_20261125_201202` (recall@5 BM25 0.82, hybrid 1.00).
- Report retrieval quality (`eval_retrieval.py`) separately from decision accuracy (`run_eval.py`).
- Failure taxonomy counts come from the `label` column of `failure_log.csv`, filled by a person. `auto_label` is only a first guess.
- Never quote numbers from `eval/demo_output/` (fake data).
- Separate: done and verified live, done but only tested offline, not started. List blockers with an owner.

Suggested report layout: scope, what was built, evidence (metrics with run names), failures by taxonomy label
(retrieval miss, reasoning error, tool or format error, hallucination, over-refusal, under-refusal, injection
compliance, PII leak), blockers, next steps.
