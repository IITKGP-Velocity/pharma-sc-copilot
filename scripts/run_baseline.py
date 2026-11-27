"""Deliverable 2 baseline: compare prompting strategies S0-S4 on the frozen 64-query eval set.

  python scripts/run_baseline.py <endpoint flags>                    # all five strategies
  python scripts/run_baseline.py --strategies S0 S1 --ids A01 A02 <flags>

Single prompt per question: no rules pre-check, no tools, no agent loop. Every strategy gets the same
top-5 retrieved passages, so only the prompt changes. Writes eval/results/baseline_<date>/:
  <strategy>.jsonl  one line per query     summary.json  accuracy, format validity, latency, tokens
and prints a table to paste into docs/02_baseline.md.
"""
import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime

import _paths
from copilot.agent.strategies import STRATEGIES, run_strategy
from copilot.observability.metrics import EXPECTED
from copilot.rag.retrieval import hybrid_search
from copilot.runtime import add_endpoint_args, apply_endpoint_args
from copilot.safety import redact_pii
from copilot.service import build_engine

ROOT = _paths.ROOT
EVAL = ROOT / "eval" / "gold" / "eval_set_v1.0.jsonl"


def evidence_for(engine, question: str) -> list[dict]:
    """Top-5 passages from the same hybrid retrieval the agent uses."""
    qvec = engine.embed.embed([question])[0] if engine.dense is not None else None
    hits = hybrid_search(question, engine.corpus, engine.bm25, engine.dense, qvec, k=5)
    return [{"doc_id": d, "text": engine.texts[d][:800]} for d, _ in hits]


def summarise(rows: list[dict]) -> dict:
    """Headline numbers for one strategy."""
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["use_case"]].append(r["ok"])
    lat = [r["latency_ms"] for r in rows]
    return {"queries": len(rows),
            "decision_accuracy": round(sum(r["ok"] for r in rows) / len(rows), 3),
            "format_validity": round(sum(r["format_ok"] for r in rows) / len(rows), 3),
            "by_use_case": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_case.items())},
            "latency_ms_avg": int(statistics.mean(lat)), "latency_ms_p95": int(sorted(lat)[int(0.95 * (len(lat) - 1))]),
            "llm_calls": sum(r["llm_calls"] for r in rows),
            "tokens_per_query": int(statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in rows)),
            "errors": sum(1 for r in rows if r["error"])}


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare prompting strategies S0-S4.")
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    ap.add_argument("--ids", nargs="+", help="only run these query ids")
    add_endpoint_args(ap)
    args = ap.parse_args()
    apply_endpoint_args(args)

    engine = build_engine()
    if engine.chat is None or not engine.corpus:
        print("Needs the chat endpoint and a built corpus (see README sections 3 and 4).")
        return 1
    items = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        items = [i for i in items if i["id"] in set(args.ids)]
    run_dir = ROOT / "eval" / "results" / datetime.now().strftime("baseline_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Retrieval: {engine.mode.split('|')[-1].strip()}. Writing to {run_dir.relative_to(ROOT)}")
    questions = {i["id"]: redact_pii(i["query"]) for i in items}
    evidence = {i["id"]: evidence_for(engine, questions[i["id"]]) for i in items}

    summary = {}
    for strategy in args.strategies:
        rows = []
        for item in items:
            before, start, error = dict(engine.chat.usage), time.monotonic(), None
            try:
                res = run_strategy(engine.chat, strategy, questions[item["id"]], evidence[item["id"]])
            except Exception as err:  # endpoint failure: count it, keep going
                res, error = {"decision": None, "format_ok": False, "answer": ""}, f"{type(err).__name__}: {str(err)[:150]}"
            expected = EXPECTED[item["type"]]
            rows.append({"id": item["id"], "use_case": item["use_case"], "type": item["type"], "expected": expected,
                         "pred": res["decision"], "ok": res["decision"] == expected, "format_ok": res["format_ok"],
                         "latency_ms": int((time.monotonic() - start) * 1000),
                         **{k: engine.chat.usage[u] - before[u] for k, u in (("llm_calls", "calls"),
                            ("prompt_tokens", "prompt_tokens"), ("completion_tokens", "completion_tokens"))},
                         "error": error, "answer": redact_pii(res.get("answer", ""))})
        (run_dir / f"{strategy}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        summary[strategy] = {"name": STRATEGIES[strategy], **summarise(rows)}
        s = summary[strategy]
        print(f"{strategy} {s['name']:30} accuracy {s['decision_accuracy']:.0%}  valid JSON {s['format_validity']:.0%}  "
              f"{s['latency_ms_avg']} ms  {s['tokens_per_query']} tokens/query", flush=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n| Strategy | Decision accuracy | Format validity | Latency avg / p95 (ms) | Tokens / query | Errors |")
    print("|---|---|---|---|---|---|")
    for k, s in summary.items():
        print(f"| {k} {s['name']} | {s['decision_accuracy']:.0%} | {s['format_validity']:.0%} | "
              f"{s['latency_ms_avg']} / {s['latency_ms_p95']} | {s['tokens_per_query']} | {s['errors']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
