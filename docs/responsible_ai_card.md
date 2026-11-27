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
- Table lookups run fixed SQL; the model only fills `?` placeholders, so it cannot change the query.
- Tool results are attached to the answer by the engine, not re-typed by the model.
- Conversation memory keeps only the last 3 turns of a session, stored after redaction; refused turns are never
  kept, and every safety rule except the scope check judges each question on its own.
- Only the two course models are used (chat and embeddings); nothing is downloaded or sent elsewhere.

## Known limitations
- A small model makes arithmetic and format errors. Calculations go through tools; malformed output is retried once.
- Corpus is a snapshot. Shortage and recall status can change after the snapshot date.
- Retrieval can miss relevant documents. Recall is measured separately from answer quality.
- Names without a label word in front are not redacted; that needs a named-entity model.
- Placeholder product names ("product Y") can be mapped to a real product by the model (2 items in the latest run).
- The shared course key limits concurrent requests; under load some queries fail with HTTP 429 and abstain.

## Privacy and security
No API keys in the repository. Keys are passed at run time (hidden prompt, key file, or a shell export);
`.env` holds endpoints only and any `*_API_KEY` line in it is ignored. Logs hold redacted queries and decisions,
never keys.

## Evaluation
64-query frozen set (`eval/gold/eval_set_v1.0.jsonl`). Latest run `eval/results/run_20261127_110934`: decision
accuracy 78% (50/64); prompt injection 4/4, out of scope 2/2, no PII leak. Failure taxonomy with counts in
`docs/failure_taxonomy.md`.

## Owner and review
[Name], reviewed by [Name], date [date].
