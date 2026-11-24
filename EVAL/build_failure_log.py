#!/usr/bin/env python3
"""
build_failure_log.py - Create the failure log for a run: one row per wrong decision.

Why: Week 4 needs a failure taxonomy built from real traces. run_eval.py gives every wrong
decision a first-pass label (auto_label); a person must confirm or correct it. This script
lists those failures in a CSV so the team can review them in a spreadsheet.

Input : EVAL/results/<run-name>/traces.jsonl  (written by run_eval.py)
Output: EVAL/results/<run-name>/failure_log.csv

Columns the script fills : run, id, use_case, gold_type, gold_decision, pred_decision, auto_label,
                           verifier_issues, query, answer_excerpt
Columns people fill      : confirmed_label, notes
Re-running keeps what people already typed (matched by id), so it is safe to run again after a new run.

Usage (from the repo root)
  python EVAL/build_failure_log.py EVAL/results/baseline-rag
"""
import csv
import json
import sys
from pathlib import Path

from safety import redact_pii

# Labels from the taxonomy in Deliverable 1, section 6. confirmed_label must be one of these.
TAXONOMY = ["retrieval_miss", "reasoning_error", "tool_call_or_format_error", "hallucination",
            "over_refusal", "under_refusal", "injection_compliance", "pii_leak"]

AUTO_COLUMNS = ["run", "id", "use_case", "gold_type", "gold_decision", "pred_decision",
                "auto_label", "verifier_issues", "query", "answer_excerpt"]
HUMAN_COLUMNS = ["confirmed_label", "notes"]


def read_existing(path: Path) -> dict:
    """Return {id: {confirmed_label, notes}} typed by people in an earlier version of the file."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: {c: r.get(c, "") for c in HUMAN_COLUMNS} for r in csv.DictReader(f)}


def failure_rows(run_name: str, traces: list[dict]) -> list[dict]:
    """One row per scored query whose decision did not match the gold decision."""
    rows = []
    for t in traces:
        if t["decision_correct"] is not False:      # None = safety item, True = correct
            continue
        rows.append({
            "run": run_name, "id": t["id"], "use_case": t["use_case"], "gold_type": t["gold_type"],
            "gold_decision": t["gold_decision"], "pred_decision": t["pred_decision"],
            "auto_label": t["auto_label"],
            "verifier_issues": "; ".join(t["verifier_issues"]),
            "query": redact_pii(t["query"]),
            # Short and redacted: the full answer stays in traces.jsonl for anyone who needs it.
            "answer_excerpt": redact_pii(t["answer"])[:200],
        })
    return rows


def write_log(path: Path, rows: list[dict], existing: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUTO_COLUMNS + HUMAN_COLUMNS)
        writer.writeheader()
        for r in rows:
            r.update(existing.get(r["id"], {c: "" for c in HUMAN_COLUMNS}))
            writer.writerow(r)


def unknown_labels(path: Path) -> list[str]:
    """Rows whose confirmed_label is filled in but is not in the taxonomy (usually a typo)."""
    return [f"{i}: {v['confirmed_label']}" for i, v in read_existing(path).items()
            if v["confirmed_label"] and v["confirmed_label"] not in TAXONOMY]


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = Path(sys.argv[1])
    with open(run_dir / "traces.jsonl", encoding="utf-8") as f:
        traces = [json.loads(line) for line in f if line.strip()]

    out = run_dir / "failure_log.csv"
    existing = read_existing(out)                  # read before the file is overwritten
    rows = failure_rows(run_dir.name, traces)
    write_log(out, rows, existing)

    print(f"{len(rows)} failures out of {len(traces)} queries -> {out}")
    bad = unknown_labels(out)
    if bad:
        print("confirmed_label not in the taxonomy:", ", ".join(bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
