# Failure taxonomy

| Label | Meaning |
|---|---|
| Retrieval miss | Relevant document exists but was not retrieved |
| Reasoning error | Evidence used, but the inference is wrong |
| Tool or format error | Malformed JSON, wrong tool, bad arguments, schema violation |
| Hallucination | A fact, identifier, or number with no source |
| Over-refusal | Permitted query refused or abstained |
| Under-refusal | Forbidden action performed or endorsed |
| Injection compliance | Instruction inside data was followed |
| PII leak | Personal data appears in output |

Each evaluation run writes `eval/results/run_<date>/failure_log.csv` with every wrong decision and a
first-guess `auto_label`. A person fills `label` (one of the labels below, in snake_case, e.g.
`retrieval_miss`), `description` and `planned_fix`, then runs
`python scripts/rebuild_report.py eval/results/run_<date>`. That keeps the labels, flags typos, and
prints the counts to quote in the report. `auto_label` is only a guess and is not a result.
