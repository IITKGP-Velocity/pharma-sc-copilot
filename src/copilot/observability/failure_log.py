"""Failure log: every wrong decision of a run in failure_log.csv, for a person to label.

The script fills the facts and a first-guess label; people fill label (one of TAXONOMY), description and
planned_fix. Re-running keeps what people typed, and label_counts() gives the taxonomy counts for the report.
"""
import csv
from collections import Counter
from pathlib import Path

# Failure taxonomy (docs/failure_taxonomy.md). People pick one of these in failure_log.csv.
TAXONOMY = ["retrieval_miss", "reasoning_error", "tool_or_format_error", "hallucination",
            "over_refusal", "under_refusal", "injection_compliance", "pii_leak",
            "endpoint_error", "label_mismatch"]


# ---- failure log ---------------------------------------------------------------------------

AUTO_COLUMNS = ["item_id", "run", "type", "expected", "pred", "auto_label", "error", "query", "answer"]
HUMAN_COLUMNS = ["label", "description", "planned_fix"]      # filled in by a person


def write_failure_log(run_dir: Path, rows: list[dict]) -> Path:
    """One CSV row per wrong decision. Re-running keeps what people already typed (matched by item_id)."""
    path = run_dir / "failure_log.csv"
    kept = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            kept = {r["item_id"]: {c: r.get(c, "") for c in HUMAN_COLUMNS} for r in csv.DictReader(f)}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUTO_COLUMNS + HUMAN_COLUMNS)
        writer.writeheader()
        for r in rows:
            if r["ok"]:
                continue
            writer.writerow({"item_id": r["id"], "run": run_dir.name, "type": r["type"], "expected": r["expected"],
                             "pred": r["pred"], "auto_label": r["auto_label"], "error": r["error"] or "",
                             "query": r["query"], "answer": r["answer"][:200],
                             **kept.get(r["id"], {c: "" for c in HUMAN_COLUMNS})})
    return path


def unknown_labels(path: Path) -> list[str]:
    """Labels people typed that are not in the taxonomy (usually typos)."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [f"{r['item_id']}: {r['label']}" for r in csv.DictReader(f) if r["label"] and r["label"] not in TAXONOMY]


def label_counts(path: Path) -> dict:
    """Failure taxonomy counts from the reviewed labels (the numbers the report should quote)."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return dict(Counter(r["label"] for r in csv.DictReader(f) if r["label"]))
