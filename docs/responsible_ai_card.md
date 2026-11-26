# Responsible AI card

**System:** Pharmaceutical Supply Chain Copilot (Track B)

## Intended use
Answers to operational questions about cold-chain excursions, alternate API suppliers, customs and import alerts, drug shortages, and recall triage. Every answer cites its source. Calculations use deterministic tools.

## Out of scope
Batch release, disposition, quarantine, supplier approval, patient dosing, therapy switching, predicting regulator or customs outcomes, and any question the corpus does not support.

## Human in the loop
Release, disposition, and approval outputs are escalated to a named authority: QP, QA, Regulatory, Pharmacist, or Prescriber. The system never states those decisions.

## Safety measures
- Rule pre-check for R1–R8 before any model call (`src/copilot/safety/rules.py`).
- Scope check: questions outside the five supply-chain use cases are refused with a fixed "Out of scope." message
  (reason `OOS`); clinical dosing questions are refused under R3 and pointed to the prescriber or pharmacist.
- Phone numbers, emails, dates of birth, and names that follow a label word (driver, contact, patient)
  are redacted before the model sees them, before output is returned, and before anything is logged.
- Instructions inside questions or documents are treated as data (injection tests in `tests/test_safety.py`).
- Citations must match retrieved documents; uncited answers become ABSTAIN.
- Step cap and token budget per query.

## Known limitations
- A small model makes arithmetic and format errors. Calculations go through tools; malformed output is retried once.
- Corpus is a snapshot. Shortage and recall status can change after the snapshot date.
- Retrieval can miss relevant documents. Recall is measured separately from answer quality.
- Names without a label word in front are not redacted; that needs a named-entity model.

## Privacy and security
No API keys in the repository. Keys are read from `.env` only. Logs hold queries and decisions; do not log secrets.

## Evaluation
64-query frozen set (`eval/gold/eval_set_v1.0.jsonl`). Results in `eval/results/`. Failure taxonomy in `docs/failure_taxonomy.md`.

## Owner and review
[Name], reviewed by [Name], date [date].
