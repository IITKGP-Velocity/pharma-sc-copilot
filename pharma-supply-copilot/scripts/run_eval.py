"""Run the 64-query evaluation set and write everything needed to report and debug it.

  python scripts/run_eval.py                                   # whole set
  python scripts/run_eval.py --ids A02 C03 E12                 # chosen queries only
  python scripts/run_eval.py --chat-endpoint https://HOST:PORT/v1 --model MODEL --chat-ca-cert PATH \
                             --api-key-file PATH               # live model (or leave out the key file to be prompted)

Output folder eval/results/run_<date>/:
  results.jsonl  one line per query (decision, expected, latency, tokens, steps, tool calls, error)
  trace.jsonl    one line per agent stage, tagged with the query id
  summary.json   accuracy by type and use case, latency, cost, errors, plus mode and eval-file hash
  failure_log.csv  wrong decisions to review and label
  dashboard.html   open in a browser
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime

import _paths
from copilot.observability import build_dashboard, result_row, summarise, write_failure_log, write_jsonl
from copilot.runtime import add_endpoint_args, apply_endpoint_args
from copilot.service import build_engine

ROOT = _paths.ROOT
EVAL = ROOT / "eval" / "gold" / "eval_set_v1.0.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the 64-query evaluation set.")
    ap.add_argument("--ids", nargs="+", help="only run these query ids")
    add_endpoint_args(ap)
    args = ap.parse_args()
    apply_endpoint_args(args)

    items = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        items = [i for i in items if i["id"] in set(args.ids)]
    engine = build_engine()
    run_dir = ROOT / "eval" / "results" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Mode: {engine.mode}\nWriting to {run_dir.relative_to(ROOT)}")

    rows = []
    for n, item in enumerate(items, 1):
        resp = engine.answer(item["query"], run_dir=run_dir, query_id=item["id"])
        rows.append(result_row(item, resp, engine.last_metrics))
        r = rows[-1]
        print(f"[{n}/{len(items)}] {r['id']} {r['pred']:8s} {'ok ' if r['ok'] else 'NO '} {r['latency_ms']} ms"
              + (f"  error: {r['error'][:80]}" if r["error"] else ""))
        write_jsonl(run_dir / "results.jsonl", rows)       # rewritten each time, so a crash keeps what is done

    summary = {"mode": engine.mode, **summarise(rows),
               "eval_sha256": hashlib.sha256(EVAL.read_bytes()).hexdigest(),
               "finished_at": datetime.now().isoformat(timespec="seconds"),
               "note": "Decision accuracy only. Key-fact and numeric scoring are done by hand on a sample."}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_failure_log(run_dir, rows)
    build_dashboard(run_dir)
    print(json.dumps({k: summary[k] for k in ("decision_accuracy", "by_type", "latency_ms", "errors")}, indent=2))
    print(f"Dashboard: {run_dir / 'dashboard.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
