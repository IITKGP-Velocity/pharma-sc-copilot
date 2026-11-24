#!/usr/bin/env python3
"""
demo.py - Build a demo run with FAKE data so anyone can see the dashboard and failure log.

No keys, certificates or network are needed. It runs the real eval set through the real
tracing, scoring, summary, dashboard and failure-log code, but the retriever and the LLM
are fakes with random outcomes. The numbers mean nothing; only the layout does.

Output: EVAL/demo_output/DEMO_fake_data/   (git-ignored; the folder name says it is fake)
  traces.jsonl, spans.jsonl, summary.json, trace_summary.json, dashboard.html, failure_log.csv

Usage (from the repo root)
  python EVAL/demo.py            # build it
  python EVAL/demo.py --open     # build it and open the dashboard in the browser
"""
import argparse
import json
import random
import sys
import time
import webbrowser
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
# The fake retriever, fake LLM and pipeline builder already exist in the pipeline tests; reuse them.
sys.path.insert(0, str(REPO_ROOT / "RAG_PIPELINE" / "tests"))
sys.path.insert(0, str(REPO_ROOT / "RAG_PIPELINE" / "scripts"))

import test_rag_pipeline as fakes                                   # noqa: E402
from build_dashboard import build                                   # noqa: E402
from build_failure_log import failure_rows, write_log               # noqa: E402
from run_eval import load_eval_set, make_trace, summarise           # noqa: E402
from trace_summary import summarise as summarise_spans              # noqa: E402
from tracing import traced_ask                                      # noqa: E402

DOCS = ["WHO_TRS961_Annex9", "EU_GDP_2013_C343_01", "ICH_Q7", "ICH_Q9", "ICH_Q10", "CFR21_Part211"]
OUTPUT_DIR = EVAL_DIR / "demo_output" / "DEMO_fake_data"

# Chance of each fake outcome for the gold type of a query:
#   good  = retrieval is strong and the model answers with a grounded citation
#   weak  = retrieval is weak, so the confidence gate abstains without calling the model
#   bad   = the model cites a number that is not in the context, so the verifier blocks it
OUTCOME_WEIGHTS = {"ANSWER": (0.6, 0.25, 0.15), "CALC": (0.6, 0.25, 0.15),
                   "ABSTAIN": (0.7, 0.2, 0.1), "ESCALATE": (0.8, 0.05, 0.15),
                   "REFUSE": (0.8, 0.05, 0.15)}
DEFAULT_WEIGHTS = (0.8, 0.1, 0.1)       # PI, PII, OOS


def with_delay(fn, rng, low=0.005, high=0.05):
    """Make a fake call take a random few milliseconds, so the latency bars have something to show."""
    def slow(*args, **kwargs):
        time.sleep(rng.uniform(low, high))
        return fn(*args, **kwargs)
    return slow


def fake_pipeline(item: dict, rng: random.Random):
    """Build a pipeline whose fake retriever and fake LLM behave as one random outcome."""
    outcome = rng.choices(["good", "weak", "bad"], OUTCOME_WEIGHTS.get(item["type"], DEFAULT_WEIGHTS))[0]
    doc = rng.choice(DOCS)
    if outcome == "weak":
        hits = [fakes.make_hit(1, doc, "5", "Unrelated text.", cos=0.2, bm25=0.5)]
        replies = []                                              # the gate stops before any LLM call
    else:
        hits = [fakes.make_hit(1, doc, "211.142", "Store at 2 to 8 degrees.", ref=1)]
        answer = "Store at 2 to 8 degrees [1]." if outcome == "good" else "Store at 99 degrees [1]."
        replies = [fakes.reply(answer)] * 2                       # second reply is used for the repair attempt
    p = fakes.make_pipeline(hits, replies)
    p.retriever.search = with_delay(p.retriever.search, rng)
    p.llm.chat = with_delay(p.llm.chat, rng, 0.02, 0.15)
    return p


def run_demo(out_dir: Path = OUTPUT_DIR, limit: int | None = None, seed: int = 42) -> Path:
    """Run the eval set through the fakes and write every output file. Returns the run folder."""
    rng = random.Random(seed)               # fixed seed: everyone sees the same demo
    out_dir.mkdir(parents=True, exist_ok=True)
    spans_file = out_dir / "spans.jsonl"
    spans_file.unlink(missing_ok=True)      # traced_ask appends, so start clean

    items = load_eval_set()[:limit]
    traces = []
    for item in items:
        resp, _ = traced_ask(fake_pipeline(item, rng), item["query"], trace_file=spans_file)
        traces.append(make_trace(item, resp))

    with open(out_dir / "traces.jsonl", "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summarise(traces), indent=2), encoding="utf-8")

    spans = [json.loads(line) for line in spans_file.read_text(encoding="utf-8").splitlines()]
    (out_dir / "trace_summary.json").write_text(json.dumps(summarise_spans(spans), indent=2), encoding="utf-8")
    (out_dir / "dashboard.html").write_text(build(out_dir), encoding="utf-8")
    write_log(out_dir / "failure_log.csv", failure_rows(out_dir.name, traces), {})
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a demo run with fake data.")
    ap.add_argument("--open", action="store_true", help="open the dashboard in the browser")
    args = ap.parse_args()
    out_dir = run_demo()
    print(f"Demo run written to {out_dir}")
    print(f"  dashboard   : {out_dir / 'dashboard.html'}")
    print(f"  failure log : {out_dir / 'failure_log.csv'}")
    print("The data is FAKE (random fake retriever and LLM). Only the layout is real.")
    if args.open:
        webbrowser.open((out_dir / "dashboard.html").as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
