# Build plan (7 days)

Start here. The full plan is in the Word document shared with the team; this file is the working copy.

| Day | Focus | Done when |
|---|---|---|
| 1 | Baseline: endpoint client, S0 to S4 runs, failure labels | Baseline table filled; eval hash recorded |
| 2 | Corpus: download sources, chunk, BM25 and dense indexes | Indexes answer a test query |
| 3 | Hybrid retrieval, reranker, citations, recall@k | Retrieval metrics recorded |
| 4 | Tools and agent loop (planner, executor, critic), step cap, recovery | Three tools called end to end |
| 5 | Evaluation run, safety tests, tracing, failure taxonomy | Injection and PII tests pass |
| 6 | Streamlit app, rate limit, cache, Docker, Responsible AI card | App runs from Docker |
| 7 | Report, demo, tag v1.0-final | Repo tagged; demo rehearsed |

Freeze first: the evaluation file is frozen before agent code is merged.
