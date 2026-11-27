# Decision log

Short record of the design choices, why they were made, and the evidence. Newest last.

| # | Decision | Why | Evidence |
|---|---|---|---|
| 1 | Safety rules run before any model call, and only the rules may refuse a question as out of scope | The small model refused in-scope questions as out of scope | PR #5: a model OOS verdict now becomes ABSTAIN; C12, D06 and E08 then passed on re-run |
| 2 | Hybrid retrieval: BM25 plus dense embeddings, fused with reciprocal rank fusion | Keyword search alone misses paraphrased questions | `retrieval_20261127_104859`: recall@5 0.455 (BM25) vs 0.909 (hybrid) |
| 3 | Reranker = one call to the course chat model, not a downloaded cross-encoder | Track B allows only the course models; no new dependency | Hybrid + rerank recall@5 1.00, MRR 0.886 (from 0.689). Falls back to the fused order on any error |
| 4 | Embeddings client shortens passages that exceed the 512-token limit, one text at a time | HTTP 422 on long table passages stopped corpus embedding | PR #3: the full corpus (57,731 passages) embeds |
| 5 | Numbers and identifiers come from deterministic tools; the model only picks the tool and its arguments | The model's own arithmetic and identifiers are unreliable | Calculation items 10/11 in `run_20261127_110934` |
| 6 | Table lookup = fixed SQL over an in-memory SQLite copy of the FDA Drugs@FDA CSV, committed in `data/tables/` | Course blueprint asks for a SQL lookup (use case B); fixed SQL with `?` placeholders blocks SQL injection | `tests/test_tools.py` (SQL text in the argument returns 0 rows); live cisplatin and amlodipine checks in PR #8 |
| 7 | The engine attaches tool results to the answer; the model no longer copies them | Copying nested lookup rows made the model drop a `}` and break the JSON | PR #8: the failing cisplatin question answered correctly after the change |
| 8 | Conversation memory: last 3 turns, redacted; refused turns not kept; only the scope rule may use the previous turn | Follow-ups ("what is its TE code?") need context, but history must not weaken the safety rules | PR #9 tests: injection and release requests are still refused or escalated with history present |
| 9 | The frozen eval set is not edited, even where a gold label looks wrong | Freeze rule: changes need a new version (v1.1) with a logged reason | 4 items labelled `label_mismatch` in `docs/failure_taxonomy.md`, kept for v1.1 |
