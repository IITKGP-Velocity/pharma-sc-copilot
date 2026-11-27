# Deliverable 2: Baseline

Single-prompt baseline on the frozen 64-query eval set (`eval/gold/eval_set_v1.0.jsonl`,
SHA-256 `8c9a5205…5d2ce2`). One call to Qwen3-8B per question (S3: five), **no rule pre-check, no tools,
no agent loop**. Every strategy gets the same question and the same top-5 passages from hybrid retrieval
(BM25 + embeddings) over the full corpus (57,731 passages), so only the prompt changes.

Run: `python scripts/run_baseline.py "${FLAGS[@]}"` (about 50 minutes for all five; `--strategies S0 S1`,
`--ids A01 A02` for a subset). Results: `eval/results/baseline_20261127_092148/` (one `.jsonl` per strategy
and `summary.json`). Prompts: `prompts/strategies/` (`base.txt` is shared; `S1.txt`, `S2.txt`, `S4.txt` add the
technique; S3 samples the S2 prompt five times at temperature 0.7 and takes the majority decision).

## Results (2026-11-27, live)

| Strategy | Decision accuracy | Format validity | Latency avg / p95 (ms) | Tokens / query |
|---|---|---|---|---|
| S0 control (zero-shot) | **64%** (41/64) | 100% | 2,685 / 4,830 | 1,511 |
| S1 structured output + few-shot | 55% (35/64) | 100% | 2,046 / 4,894 | 1,774 |
| S2 chain-of-thought | 58% (37/64) | 100% | 3,959 / 6,310 | 1,678 |
| S3 self-consistency (5 samples) | 58% (37/64) | 100% | 22,332 / 35,749 | 8,383 |
| S4 decomposition | 58% (37/64) | 100% | 7,902 / 10,855 | 1,794 |
| *Agent (rules + tools + critic), same corpus* | *72% (46/64), 77% after #5* | — | 5,901 / 11,164 | — |

Accuracy by question type (correct / total):

| Type | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| ANS answer with citation (22) | 19 | 19 | 15 | 16 | 18 |
| CALC needs arithmetic (11) | 6 | 0 | 3 | 3 | 5 |
| ESC escalate to a person (9) | 8 | 8 | 6 | 6 | 8 |
| ABS abstain (8) | 2 | 3 | 7 | 6 | 3 |
| REF forbidden request (5) | 2 | 2 | 2 | 2 | 1 |
| PI prompt injection (4) | 1 | 1 | 2 | 2 | 0 |
| PII personal data (3) | 1 | 1 | 1 | 1 | 2 |
| OOS out of scope (2) | 2 | 1 | 1 | 1 | 0 |

## How the small model breaks

1. **Format is not the problem.** Every strategy returned valid JSON with a valid decision 100% of the time.
   Qwen3-8B follows a JSON contract reliably; the errors are in the decision.
2. **More reasoning makes it cautious, not correct.** Chain-of-thought (S2) and self-consistency (S3) turn
   answerable questions into ABSTAIN (16 and 15 "ANSWER→ABSTAIN" errors vs 5 for S0). They win on the ABSTAIN
   questions (7/8 and 6/8) for the same reason. Net accuracy drops.
3. **Few-shot examples were copied, not generalised.** S1 scored 0/11 on calculations (8 abstained, 3 escalated):
   its examples show only answer, escalate, refuse and abstain, none with arithmetic.
4. **Refusals look like escalations.** The most common error for S0, S1 and S4 is REFUSE→ESCALATE (6, 5, 8):
   on "write a certificate", "change the lot number" or injected instructions the model hands over to a person
   instead of refusing. Prompt wording alone did not fix it.
5. **Prompt injection is not stopped by prompting.** At best 2/4 injection questions were refused (S4: 0/4).
6. **Arithmetic without tools is unreliable.** Best case 6/11 (S0); MKT and cost calculations are wrong or
   abstained.
7. **Self-consistency is expensive for nothing here.** S3 equals S2 in accuracy at 5.6× the latency and 5×
   the tokens; the five samples mostly agree with each other, including when they are wrong.
8. **13 questions fail under every strategy** (A02, A04, A08, A13, B01, B06, B11, C01, C05, C08, D07, E05, E12).
   11 of them are calculations (5), refusals (3), injections (2) or a personal-data case (1): exactly what the
   agent's deterministic parts handle (tools for MKT, cost and demurrage; the rule pre-check R1–R8, OOS and
   injection; redaction). The other two are retrieval or data gaps (A13 needs a stability monograph that is not
   in the corpus; B01 asks about amlodipine, which has no label or therapeutic-equivalence data in the corpus; it appears only in a tariff line).

**Decision for the build:** keep S0-style direct prompting inside the agent, and move the decisions the model
gets wrong into code (rules for refusal and injection, tools for arithmetic, a critic for citations). That is
why the agent (72–77%) beats the best single prompt (64%).
