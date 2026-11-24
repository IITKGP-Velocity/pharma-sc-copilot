#!/usr/bin/env python3
"""
trace_summary.py - Turn a trace file into the numbers the report and dashboard need.

Input : a JSONL file written by tracing.traced_ask() (for example
        EVAL/results/<run-name>/spans.jsonl, written by run_eval.py).
Output: printed table + a summary JSON next to the input (trace_summary.json).

Reports per-query cost and latency (the brief asks for both), the decision mix, how often
the verifier failed or needed a repair, and where the time goes (retrieve vs llm_call).

Usage
  python EVAL/trace_summary.py EVAL/results/baseline-rag/spans.jsonl
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    """Simple nearest-rank percentile; fine for the small samples we have."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))]


def summarise(traces: list[dict]) -> dict:
    n = len(traces)
    if n == 0:
        return {"queries": 0}

    latencies = [t["latency_ms"] for t in traces]

    # Total time and call count per span name, to show where the time goes.
    span_ms, span_n = defaultdict(int), defaultdict(int)
    for t in traces:
        for s in t["spans"]:
            span_ms[s["name"]] += s["duration_ms"]
            span_n[s["name"]] += 1

    return {
        "queries": n,
        "decisions": dict(Counter(t["decision"] for t in traces)),
        "latency_ms": {"avg": round(sum(latencies) / n), "p50": percentile(latencies, 50),
                       "p95": percentile(latencies, 95), "max": max(latencies)},
        "avg_llm_calls": round(sum(t["llm_calls"] for t in traces) / n, 2),
        "avg_prompt_tokens": round(sum(t["prompt_tokens"] for t in traces) / n),
        "avg_completion_tokens": round(sum(t["completion_tokens"] for t in traces) / n),
        "verifier_failed_rate": round(sum(not t["verifier_passed"] for t in traces) / n, 3),
        "repair_rate": round(sum(t["repair_attempted"] for t in traces) / n, 3),
        "error_rate": round(sum(t["decision"] == "ERROR" for t in traces) / n, 3),
        "avg_ms_per_span": {k: round(span_ms[k] / span_n[k]) for k in span_ms},
        "share_of_span_time": {k: round(span_ms[k] / (sum(span_ms.values()) or 1), 3) for k in span_ms},
    }


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    with open(path, encoding="utf-8") as f:
        traces = [json.loads(line) for line in f if line.strip()]
    summary = summarise(traces)
    out = path.with_name("trace_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
