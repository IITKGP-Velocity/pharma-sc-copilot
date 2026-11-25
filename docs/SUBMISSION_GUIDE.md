# Submission guide

## What to submit
1. **Code:** this repository (zip or Git link). Include `eval/results/` for the run you report. Do not include `.env` or `data/raw/`.
2. **Technical report:** 12 pages. Write it in `docs/report/` (Word or LaTeX). Sections to include:
   - Problem and scope (from Deliverable 1)
   - Architecture (Planner → Executor → Critic, tools, retrieval)
   - Baseline results (from Deliverable 2)
   - Retrieval results (recall@k, MRR), measured separately from answer quality
   - Agent results (`eval/results/`), with tool calls per query, latency, and cost
   - Failure taxonomy with counts
   - "What did we do to compensate for a small model?"
   - Limitations
3. **Responsible AI card:** `docs/responsible_ai_card.md` (complete every section).
4. **Demo:** 10 minutes. Show a cited answer, a tool calculation, an escalation, an injection refusal, and a PII redaction.
5. **Release tag:** `v1.0-final`.

## Before you submit (checklist)
- [ ] `python -m pytest -q` passes on a fresh clone
- [ ] `python scripts/check_setup.py` shows the endpoint as configured
- [ ] `python scripts/run_eval.py` ran in agent mode, and the summary is in `eval/results/`
- [ ] The evaluation file hash matches `eval/gold/eval_set_v1.0.sha256`
- [ ] `.env` is not in the repository and no key appears in any file (search for the key prefix)
- [ ] The report numbers match `summary.json`
- [ ] The app starts with `python -m streamlit run app/app.py`
- [ ] Demo rehearsed twice; backup recording saved

## Items the university must supply
- Chat endpoint, API key, and model name (`COPILOT_CHAT_*`)
- Embeddings endpoint, API key, and model name (`COPILOT_EMBED_*`)

Enter these only in your local `.env`. This project ships with them blank.
