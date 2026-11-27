---
title: "Pharmaceutical Supply Chain AI Copilot"
subtitle: "Capstone Track B — Technical Report"
author: "Team Velocity"
date: "27 November 2026"
---

# 1. Problem and scope

Supply chain and quality staff in a pharmaceutical company answer many operational questions every day.
Was a cold-chain excursion acceptable? Is an alternate supplier qualified? Will a shipment clear customs?
Is there a shortage, and what can replace the product? Which lots does a recall touch? The answers sit in
regulations, guidance documents, product labels and public FDA data. Some questions also need a calculation,
such as mean kinetic temperature or a cost difference. Some must not be answered by a machine at all: batch
release, supplier approval and patient dosing belong to a named person.

Track B asks for an assistant built on **one shared small model (≤8B parameters) that cannot be changed**.
We may not fine-tune it or swap it for a larger model. Accuracy has to come from the system around the model.

**Our copilot** answers questions in five use cases (Deliverable 1, `docs/01_spec.md`):

| Code | Use case | What the copilot does |
|---|---|---|
| A | Cold-chain excursion triage and GDP release | Checks temperature data against limits, computes MKT, escalates release to the QP |
| B | Alternate API / generic supplier qualification | Looks up therapeutic-equivalence (TE) codes, compares impurity results to limits, flags Warning Letters |
| C | Customs and import pre-clearance | Checks manifest counts, import alerts and import regulations |
| D | Drug shortage and therapeutic substitution | Reports shortage status, label excipients and unit-cost differences |
| E | Pharmacovigilance and recall triage | Summarises adverse-event and recall records by lot, estimates quarantine exposure |

**Every answer is exactly one of four decisions**, chosen before any text is written:

- **ANSWER:** supported by a cited source or a tool result.
- **ESCALATE:** a person (QP, QA, Regulatory, Pharmacist or Prescriber) must decide.
- **REFUSE:** a forbidden request, such as fabricating a record, clinical dosing or following an injected instruction.
- **ABSTAIN:** the evidence is missing.

Eight rules (R1–R8) define the boundaries. Examples: never decide a release (R1); never invent an identifier
(R5); never follow instructions found inside documents (R6).

**Evaluation set.** 64 questions, written and frozen before any agent code (SHA-256 `8c9a5205…5d2ce2`):
22 answer, 11 calculation, 9 escalation, 5 refusal, 8 abstain, 4 prompt-injection, 3 personal-data and
2 out-of-scope items, spread across the five use cases.

**Models.** Only the two course models are used, both behind OpenAI-compatible endpoints:

- Chat: Qwen3-8B.
- Embeddings: bge-small-en-v1.5 (384 dimensions, 512-token input limit, at most 16 texts per request).

Nothing else is downloaded or called.

# 2. Architecture

![Figure 1. Request pipeline. Blue steps are deterministic code; orange steps call the course chat model.
Source: `docs/report/architecture.mmd`.](architecture.png){width=62%}

The answer is one JSON object: decision, answer, citations, computed, escalate_to, missing_evidence and
reason_code.

**Code layout.** One folder per component under `src/copilot/`: `safety/`, `rag/`, `agent/`,
`observability/`, plus `schemas.py` for the output contract. The web app is `app/app.py`.

## 2.1 Safety layer (`safety/rules.py`)

The rules run before any model call, in this order:

1. Injection (R6)
2. Record fabrication (R2)
3. Clinical dosing and patient switching (R3)
4. Predictions (R7)
5. Release or approval requests (R1, escalated to QP or QA)
6. Scope

Questions that match a rule never reach the model. In the latest run, 15 of the 64 questions were decided
this way, in under 10 ms each.

Personal data is redacted at three points: before retrieval, again on the output, and before anything is
logged.

**Only the rules may refuse as out of scope.** When the model itself labelled an in-scope question as off-topic,
it was usually wrong, so a model OOS verdict now becomes ABSTAIN (decision log #1).

## 2.2 Retrieval (`rag/`)

The corpus holds 57,731 passages of up to 800 characters. They come from 12,613 source records:

- **614 text documents:**
  - 200 PDFs from the data packs (FDA reviews, labels, approval letters and regulations);
  - 66 regulations and guidance documents (21 CFR, 19 CFR, ICH, WHO);
  - 348 openFDA drug labels.
- **11,999 table rows**, each indexed as its own small passage:
  - establishment registrations;
  - recalls and shortages;
  - tariff lines;
  - Drugs@FDA and TE codes;
  - NADAC prices;
  - Purple Book entries;
  - import-alert firms;
  - FAERS lot counts.

Retrieval combines two searches:

- **BM25 keyword search:** a small pure-Python implementation.
- **Dense search:** cosine similarity over bge-small embeddings, computed once and stored.

The two ranked lists are merged with reciprocal rank fusion. The **reranker** sends the top 20 fused passages
(300 characters each) to the course chat model and asks for their order as JSON. If that call fails, the fused
order is kept.

## 2.3 Agent loop (`agent/engine.py`)

Each planner step returns one JSON object: either a tool call or an answer.

- **Executor:** runs the named tool in Python and feeds the result back to the planner.
- **Critic:** validates the answer against the schema and sends specific errors back, such as
  "citations.0.doc_id: field required".
- **Engine:** attaches the tool results to the final answer itself, so values are exact and the model never
  re-types them (decision log #7).

Hard limits:

- 10 steps per question;
- an 8,000-token budget per question;
- one retry on invalid JSON;
- any endpoint error becomes ABSTAIN with the reason recorded, never an invented answer.

**Tools** (`agent/tools.py`) are deterministic Python functions. The model only chooses a tool and its arguments:

| Tool | Use case | What it returns |
|---|---|---|
| `mkt` | A | Haynes mean kinetic temperature from a list of readings |
| `impurity_passes` | B | Pass/fail of an impurity result against a limit |
| `lookup_approved_products` | B, D | SQL lookup in the FDA Drugs@FDA table: applicant, TE code, reference drug, marketing status |
| `manifest_mismatch`, `demurrage` | C | Manifest minus invoice units; port fees |
| `cost_delta`, `markup_percent` | D | Brand vs generic totals; percent above a reference price |
| `quarantine_cost` | E | Value of quarantined stock |

The table lookup loads the CSV into an in-memory SQLite table and runs **fixed SQL**. The model's text only
fills two `?` placeholders, so it cannot change the query; a test checks that an injected `' OR '1'='1`
returns no rows.

**Multi-turn memory.** The last three turns of a conversation are kept, after redaction. Refused turns are
never kept. A follow-up such as "What is its TE code?" is searched together with the previous question, and
the planner sees the earlier turns only to work out what "it" refers to. Every safety rule still judges the
current question on its own. The one exception is the scope check: it may borrow the previous topic, because
it runs last, after all the other rules have passed the question.

## 2.4 Interfaces and deployment

- **Command line:** `ask.py` answers one question; `demo.py` runs the demo cases.
- **Web app:** a Streamlit app with a per-session rate limit (10 questions a minute), answer caching keyed on
  the question and the conversation, and conversation memory.
- **Docker:** a Dockerfile builds the app image. Keys and certificates are passed at run time, never baked in.

API keys are never stored in the repository or in `.env`; the loader ignores any `*_API_KEY` line. Each team
member passes a key at run time through a hidden prompt, a key file or a shell export.

# 3. Baseline results (Deliverable 2)

Before building the agent we measured what prompting alone achieves. Each of five strategies received the same
question and the same top-5 hybrid passages, with one model call (five for S3) and no rules, tools or loop.

| Strategy | Decision accuracy | Valid JSON | Latency avg / p95 | Tokens / query |
|---|---|---|---|---|
| S0 zero-shot | **64%** (41/64) | 100% | 2.7 s / 4.8 s | 1,511 |
| S1 structured + few-shot | 55% (35/64) | 100% | 2.0 s / 4.9 s | 1,774 |
| S2 chain-of-thought | 58% (37/64) | 100% | 4.0 s / 6.3 s | 1,678 |
| S3 self-consistency (5 samples) | 58% (37/64) | 100% | 22.3 s / 35.7 s | 8,383 |
| S4 decomposition | 58% (37/64) | 100% | 7.9 s / 10.9 s | 1,794 |

What the baseline taught us:

- **Format was never the problem.** Every strategy returned valid JSON 100% of the time. The errors were in
  the decisions.
- **More reasoning made the model cautious, not correct.** Chain-of-thought and self-consistency turned
  answerable questions into ABSTAIN: 16 and 15 such errors, against 5 for S0.
- **Few-shot examples were copied, not generalised.** S1 scored 0/11 on calculations, because none of its
  examples contained arithmetic.
- **Prompting did not stop prompt injection.** At best 2 of the 4 injection questions were refused.
- **Refusals came out as escalations.** "Write a certificate" or "change the lot number" was handed to a
  person instead of refused.
- **Arithmetic without tools was unreliable.** The best strategy got 6/11 calculations right.
- **Thirteen questions failed under every strategy.** Eleven of them were calculations, refusals, injections
  or personal data: exactly the cases code can decide.

**Decision for the build:** keep S0-style direct prompting inside the agent, and move each kind of decision the
model gets wrong into code: rules for refusal, injection and escalation; tools for arithmetic and table
lookups; a critic for citations.

# 4. Retrieval results

Retrieval is measured separately from answer quality, on 11 gold questions (`eval/gold/retrieval_gold.json`). A
question counts as found when a passage from the right source, containing the expected text, is in the top 5.

| Method | recall@5 | MRR |
|---|---|---|
| BM25 only | 0.455 | 0.326 |
| Hybrid (BM25 + embeddings, RRF) | 0.909 | 0.689 |
| **Hybrid + rerank (course chat model)** | **1.00** | **0.886** |

Source: `eval/results/retrieval_20261127_104859`.

- **Keyword search alone misses paraphrased questions.** With 57,731 passages, many of them near-identical
  table rows, common words match too many passages.
- **Embeddings double recall.**
- **The reranker finds the last missing passage and moves the right one to the top.** MRR 0.886 means the
  first relevant passage is usually at rank 1.
- **Cost:** one extra model call per question.

Two retrieval problems were fixed during the build:

- **Long passages broke embedding.** Table passages with dot leaders exceeded the embedding model's 512-token
  limit (HTTP 422). The embeddings client now embeds such a batch one text at a time and shortens any text that
  is too long by at least 10% per try.
- **Passage size.** It was set to 800 characters, because 1,200-character label passages hit the same limit.

# 5. Agent results

Full live run on the frozen set: `eval/results/run_20261127_110934`.

| Type | Correct | Type | Correct |
|---|---|---|---|
| Answer | 16/22 | Prompt injection | 4/4 |
| Calculation | 10/11 | Personal data | 1/3 |
| Escalate | 8/9 | Out of scope | 2/2 |
| Refuse | 4/5 | Abstain | 5/8 |

- **Decision accuracy: 78% (50/64).** That is 14 points above the best single prompt (64%).
- **By use case:** A 12/14, B 12/13, C 10/13, D 8/12, E 8/12.
- **Latency:** average 8.1 s, median 9.3 s, p95 14.3 s, maximum 33.0 s.
- **Cost per question that reaches the agent:** 2.2 model calls (one of them the reranker), 1.2 planner steps,
  about 4,360 prompt and 380 completion tokens. The whole run used 232,291 tokens.
- **15 questions were decided by the rules,** with no model call and under 10 ms each.

**Progress during the build** (same 64 questions, live model):

| Run | Accuracy | What changed |
|---|---|---|
| 25 Nov | 53% (34/64) | First full agent run, before the data-pack corpus was loaded |
| 27 Nov 09:01 | 72% (46/64) | Full corpus from all five data packs, embedding fix |
| 27 Nov 11:09 | **78% (50/64)** | Only rules may refuse as out of scope, reranker, table lookup tool |

**Against the baseline**, the agent gains most where code replaces the model:

- calculations: 6/11 → 10/11;
- prompt injection: 1/4 → 4/4;
- refusals: 2/5 → 4/5;
- abstain: 2/8 → 5/8.

It **loses on plain answers** (19/22 for S0 → 16/22). The agent abstains more readily: the critic rejects
answers without a citation, and the output guard turns unsupported answers into ABSTAIN. Section 6 shows that
part of this is correct behaviour scored against a disputed label.

**Calculations are now exact.** In the one calculation the agent missed (C01, a manifest count), the model
escalated instead of calling the tool. When the tool is called, its result is right by construction.

# 6. Failure taxonomy

Every wrong decision is written to `failure_log.csv` with a first-guess label. A person then reviews each one.
The 14 failures of the latest run, reviewed by hand:

| Label | Count | Items | What happened |
|---|---|---|---|
| Over-refusal | 4 | C04, C05, D01, E07 | Abstained when evidence was absent instead of saying "none listed" (C04, D01); abstained on redaction requests instead of returning the redacted summary (C05, E07) |
| Label mismatch | 4 | A04, B01, D04, D05 | The behaviour matches the gold text, but the decision label differs. Example: B01's gold says "abstain if the product is absent"; amlodipine is absent from the FDA table and the copilot abstained, but the item is labelled ANSWER |
| Hallucination | 2 | A13, D09 | Placeholder names ("product Y", "product Z") were mapped to real products (filgrastim, clonazepam) |
| Reasoning error | 2 | C01, E08 | Escalated instead of calling `manifest_mismatch`; discussed causality instead of abstaining |
| Endpoint error | 2 | E09, E10 | HTTP 429: the shared course key allows only a few requests at once |

The run had **no under-refusals, no injection compliance and no personal-data leaks.** A scan of all 64
results finds none of the test personal data, not even in the stored questions.

**What the counts say:**

- **Only 4 of 14 failures are wrong in substance:** the 2 hallucinations and the 2 reasoning errors.
- **The other 10 are cautious answers, disputed labels or infrastructure errors.** If the 4 label mismatches
  were scored by their gold text, accuracy would be 54/64 (84%). We report 78%, because the frozen set may only
  change through a new version (v1.1) with a logged reason.

**Planned fixes**, not built yet, from `docs/failure_taxonomy.md`:

- a planner rule that a placeholder product with no matching evidence means ABSTAIN;
- "none listed" answers when the snapshot has no record;
- returning redacted text for redaction requests;
- retry with back-off on HTTP 429.

# 7. What did we do to compensate for a small model?

Each measure below answers a weakness we measured, not one we assumed:

| Weakness (evidence) | Compensation |
|---|---|
| Hands refusals to a person; follows injected text (baseline: 1/4 injections refused) | Rule pre-check R1–R8 in code, before any model call; injections 4/4 |
| Arithmetic and identifiers unreliable (baseline: 6/11 calculations) | Deterministic tools and a fixed-SQL table lookup; the model only picks the tool; calculations 10/11 |
| Breaks long nested JSON when copying tool results (cisplatin lookup, PR #8) | The engine attaches tool results itself; the model's reply stays short |
| Labels in-scope questions as off-topic (PR #5) | Only the rules may refuse as out of scope; a model OOS verdict becomes ABSTAIN |
| Keyword search misses paraphrases (recall@5 0.455) | Hybrid retrieval, then a reranker using the same chat model (recall@5 1.00) |
| Can claim facts without support | The critic requires a citation or tool result; the output guard drops citations to passages that were not retrieved |
| More reasoning made it more cautious, not more accurate (S2, S3) | Direct S0-style prompting; reasoning moved into code |
| Can loop or run long | 10-step cap, 8,000-token budget, one JSON retry, ABSTAIN on any endpoint error |
| Cannot resolve "it" in follow-ups | Short, redacted conversation memory, with safety rules still applied per question |

The common idea: **let the model do what it does reliably** (choose among a few options, read short evidence,
write a short JSON answer), and **move every decision that must be exact or safe into code.**

# 8. Responsible AI and safety

The full card is in `docs/responsible_ai_card.md`. In summary:

- **Human in the loop:** release, disposition and approval questions are escalated to a named role. The
  copilot never states such a decision.
- **Personal data:** redacted before the model, on output and in logs.
- **Grounding:** every factual claim needs a citation that was actually retrieved.
- **Secrets:** no API key in the repository or in `.env`. The course host address is never committed. The
  committed results were scanned for keys, the host address and personal data.
- **Models:** only the course models are used, and no data is sent anywhere else.

# 9. Observability

Every question produces:

- **A metrics record:** latency, steps, model calls, tokens, tool calls, time per stage, and any error.
- **One trace line per stage:** precheck, retrieve, planner, executor, critic, plus error stages.

Each evaluation run writes `results.jsonl`, `summary.json`, `trace.jsonl`, `failure_log.csv` and an HTML
dashboard. The dashboard shows accuracy by type and use case, latency, tokens, and a table of questions whose
rows open to show each agent step. `rebuild_report.py` recounts the reviewed failure labels and flags any label
not in the taxonomy.

![Figure 2. Dashboard of the reported run (`eval/results/run_20261127_110934/dashboard.html`), top of the
page. The "first-pass label" panel shows the automatic first guesses; the reviewed labels are in section 6.](dashboard.png){width=100%}

# 10. Limitations

- **Small evaluation sets.** 64 decisions and 11 retrieval questions give coarse numbers: one question is
  1.6 points of accuracy.
- **Decisions are scored automatically; content is not.** The run scores the decision only. Key facts and
  numeric tolerance are checked by hand on a sample.
- **The corpus is a snapshot.** Shortage and recall status can change after the snapshot date.
- **The 2,000-document guideline.** The corpus holds 614 text documents and 11,999 table rows, each indexed as
  its own passage. Counted as documents, the rows exceed the guideline. They are structured records, not
  free text.
- **Name redaction.** Names are redacted only when a label word ("driver", "contact", "patient") comes before
  them. Catching all names needs a named-entity model.
- **Placeholder products.** Names like "product Y" can still be matched to a real product (2 items).
- **Shared endpoint.** The course key limits concurrent requests. Under load some questions fail with HTTP 429
  and abstain (2 items).
- **Run-to-run variation.** The model is not fully deterministic, and individual items can change between
  runs. For example, E08 passed after the out-of-scope fix and failed in the latest run.
- **No ERP data.** The planned `query_erp_inventory` tool was not built, because no ERP data is available.

# 11. How to reproduce

```
python -m pytest -q                                   # 118 tests, offline
python scripts/check_setup.py --live "${FLAGS[@]}"    # both course endpoints answer
python scripts/eval_retrieval.py "${FLAGS[@]}"        # section 4
python scripts/run_eval.py "${FLAGS[@]}"              # section 5 (about 9 minutes)
python scripts/demo.py all "${FLAGS[@]}"              # demo cases
python scripts/demo.py conversation "${FLAGS[@]}"     # multi-turn demo
python -m streamlit run app/app.py                    # web app
```

`FLAGS` holds the course endpoints, model names and CA certificates (README section 3). Keys are given at run
time.
