#!/usr/bin/env python3
"""
run_eval.py - Run the frozen eval set through the copilot, log a trace per query, and score it.

What it does
  1. Reads EVAL/eval_set_v1.jsonl (built by build_eval_set.py).
  2. Sends each query to the current system (today: the RAG pipeline) and records a trace:
     decision, latency, tokens, LLM calls, retrieved documents, retrieval scores, issues.
  3. Compares the system's decision with the gold outcome (ANSWER / ESCALATE / REFUSE / ABSTAIN).
  4. Writes traces.jsonl + summary.json into EVAL/results/<run-name>/.

What it does NOT do yet (needs later work, see the notes in each function)
  - Key-fact / numeric / citation-validity scoring: gold answers are free text, so this needs
    an LLM judge or manual review. Traces already hold the answers for that.
  - Scoring PI / PII / OOS items: they are safety tests with zero tolerance, to be checked
    in the safety tests, so they are logged but left unscored here.
  - ESCALATE / REFUSE: the RAG pipeline can only answer or abstain, so those items count as
    wrong until the agent layer produces those decisions. That is the honest baseline.

Usage (from the repo root, with .env and secrets/ set up as in the README)
  python EVAL/run_eval.py --run-name baseline-rag          # all 64 queries
  python EVAL/run_eval.py --run-name smoke --limit 5       # quick check
  python EVAL/run_eval.py --run-name a-only --ids A01 A02  # chosen queries
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
EVAL_FILE = EVAL_DIR / "eval_set_v1.jsonl"

# Gold type -> the decision the system should produce.
# CALC expects an ANSWER (the number comes from a tool). Safety types have no single decision.
GOLD_DECISION = {"ANSWER": "ANSWER", "CALC": "ANSWER", "ESCALATE": "ESCALATE",
                 "REFUSE": "REFUSE", "ABSTAIN": "ABSTAIN"}
SAFETY_TYPES = {"PI", "PII", "OOS"}

# RAG pipeline status -> decision label. Replace this when the agent returns a `decision` field.
STATUS_TO_DECISION = {"answered": "ANSWER", "unverified_answer": "ANSWER",
                      "abstained": "ABSTAIN", "blocked_unverified": "ABSTAIN", "error": "ERROR"}


def load_eval_set() -> list[dict]:
    with open(EVAL_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def auto_label(gold_decision: str | None, pred: str, status: str) -> str:
    """First-pass failure label for the failure log. A human confirms or corrects it later.
    Labels follow the taxonomy in Deliverable 1, section 6."""
    if gold_decision is None or gold_decision == pred:
        return ""
    if pred == "ERROR":
        return "tool_call_or_format_error"
    if gold_decision == "ANSWER" and pred == "ABSTAIN":
        # Either the evidence was not retrieved or the system was too cautious.
        return "retrieval_miss_or_over_refusal"
    if gold_decision in ("ESCALATE", "REFUSE") and pred == "ANSWER":
        return "under_refusal"
    return "wrong_decision"


def make_trace(item: dict, resp) -> dict:
    """Build one trace record from the RAG pipeline response for one eval item."""
    pred = STATUS_TO_DECISION.get(resp.status, "ERROR")
    gold = GOLD_DECISION.get(item["type"])          # None for PI / PII / OOS
    return {
        "id": item["id"],
        "use_case": item["use_case"],
        "gold_type": item["type"],
        "gold_decision": gold,
        "pred_decision": pred,
        "status": resp.status,
        # None means "not scored here" (safety types), so they stay out of the accuracy number.
        "decision_correct": None if gold is None else gold == pred,
        "auto_label": auto_label(gold, pred, resp.status),
        "latency_ms": resp.latency_ms,
        "llm_calls": resp.llm_calls,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        # Which documents were retrieved: the input for recall@k once gold doc IDs exist.
        "retrieved_doc_ids": [h.get("doc_id") for h in resp.retrieved_chunks],
        "top_dense_cosine": resp.confidence.get("top_dense_cosine"),
        "top_bm25_score": resp.confidence.get("top_bm25_score"),
        "verifier_passed": resp.verification.passed,
        "verifier_issues": resp.verification.issues,
        "repair_attempted": resp.verification.repair_attempted,
        "query": item["query"],
        "answer": resp.answer,
    }


def summarise(traces: list[dict]) -> dict:
    """Aggregate the traces into the headline numbers for the report."""
    scored = [t for t in traces if t["decision_correct"] is not None]

    def accuracy(rows):
        return round(sum(t["decision_correct"] for t in rows) / len(rows), 3) if rows else None

    # Accuracy split by gold type and by use case shows where the system is weak.
    by_type, by_case = defaultdict(list), defaultdict(list)
    for t in scored:
        by_type[t["gold_type"]].append(t)
        by_case[t["use_case"]].append(t)

    n = len(traces) or 1
    labels = defaultdict(int)
    for t in traces:
        if t["auto_label"]:
            labels[t["auto_label"]] += 1

    return {
        "queries_run": len(traces),
        "queries_scored": len(scored),
        "decision_accuracy": accuracy(scored),
        "accuracy_by_type": {k: accuracy(v) for k, v in sorted(by_type.items())},
        "accuracy_by_use_case": {k: accuracy(v) for k, v in sorted(by_case.items())},
        "error_rate": round(sum(t["pred_decision"] == "ERROR" for t in traces) / n, 3),
        "avg_latency_ms": round(sum(t["latency_ms"] for t in traces) / n),
        "avg_llm_calls": round(sum(t["llm_calls"] for t in traces) / n, 2),
        "total_prompt_tokens": sum(t["prompt_tokens"] for t in traces),
        "total_completion_tokens": sum(t["completion_tokens"] for t in traces),
        "auto_failure_labels": dict(labels),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the eval set and score decisions.")
    ap.add_argument("--run-name", required=True, help="folder name under EVAL/results/")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N queries")
    ap.add_argument("--ids", nargs="+", default=None, help="only run these query IDs")
    args = ap.parse_args()

    items = load_eval_set()
    if args.ids:
        items = [i for i in items if i["id"] in set(args.ids)]
    if args.limit:
        items = items[: args.limit]

    # Import here so build/score helpers can be used without the index and API keys.
    sys.path.insert(0, str(REPO_ROOT / "RAG_PIPELINE" / "scripts"))
    from rag_pipeline import RegulatoryRAGPipeline
    pipeline = RegulatoryRAGPipeline()

    out_dir = EVAL_DIR / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = []
    # Traces are appended one by one, so a crash part-way keeps the queries already done.
    with open(out_dir / "traces.jsonl", "w", encoding="utf-8") as f:
        for n, item in enumerate(items, 1):
            resp = pipeline.ask(item["query"])
            trace = make_trace(item, resp)
            traces.append(trace)
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(items)}] {item['id']}: gold={trace['gold_decision']} "
                  f"pred={trace['pred_decision']} {trace['latency_ms']} ms")

    summary = summarise(traces)
    summary["run_name"] = args.run_name
    summary["run_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
