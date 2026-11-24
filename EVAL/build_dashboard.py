#!/usr/bin/env python3
"""
build_dashboard.py - Build a single-file HTML dashboard from a finished eval run.

Why: the Track B brief asks for a trace dashboard showing real query executions. This makes
one self-contained page (no server, no internet, no JavaScript) that can be opened in a
browser, committed with the results, and shown in the demo.

Input : a run folder, EVAL/results/<run-name>/, containing
          spans.jsonl   (required) one trace per query, written by tracing.traced_ask()
          traces.jsonl  (optional) scored decisions written by run_eval.py
          summary.json  (optional) accuracy numbers written by run_eval.py
Output: EVAL/results/<run-name>/dashboard.html

Usage (from the repo root)
  python EVAL/build_dashboard.py EVAL/results/baseline-rag
"""
import html
import json
import sys
import time
from pathlib import Path

from trace_summary import summarise


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def esc(value) -> str:
    """Escape anything that came from a trace before it goes into HTML (queries are user text)."""
    return html.escape(str(value), quote=True)


def bar(value: float, maximum: float) -> str:
    """A horizontal bar whose width is value/maximum of the available space."""
    pct = 0 if maximum <= 0 else min(100, round(100 * value / maximum))
    return f'<div class="bar"><div class="fill" style="width:{pct}%"></div></div>'


def kpi(label: str, value) -> str:
    return f'<div class="kpi"><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div></div>'


def table(headers: list[str], rows: list[list[str]]) -> str:
    """Rows hold ready-made HTML cells; callers escape their own values."""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def counts_section(title: str, counts: dict) -> str:
    """One labelled bar per key, for the decision mix and the failure labels."""
    if not counts:
        return ""
    top = max(counts.values())
    rows = [[esc(k), esc(v), bar(v, top)] for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    return f"<h2>{esc(title)}</h2>" + table(["", "queries", ""], rows)


def query_rows(traces: list[dict]) -> list[list[str]]:
    """One row per query. Spans are listed in a collapsible cell so the table stays short."""
    slowest = max((t["latency_ms"] for t in traces), default=0)
    rows = []
    for t in traces:
        spans = "".join(
            f"<div>{esc(s['name'])}: {esc(s['duration_ms'])} ms"
            + (f" <b>(error: {esc(s['error'])})</b>" if s.get("error") else "") + "</div>"
            for s in t["spans"])
        rows.append([
            esc(t["query"][:90]),
            esc(t["decision"]),
            esc(t["latency_ms"]) + bar(t["latency_ms"], slowest),
            esc(t["prompt_tokens"] + t["completion_tokens"]),
            "yes" if t["verifier_passed"] else "<b>no</b>",
            f"<details><summary>{len(t['spans'])} span{'s' if len(t['spans']) != 1 else ''}</summary>{spans}</details>",
        ])
    return rows


STYLE = """
body{font-family:system-ui,sans-serif;margin:24px;color:#1b1f23}
h1{margin-bottom:4px} h2{margin-top:28px;font-size:18px} .sub{color:#57606a}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin-top:16px}
.kpi{border:1px solid #d0d7de;border-radius:8px;padding:12px 18px;min-width:120px}
.kpi .v{font-size:24px;font-weight:600} .kpi .l{color:#57606a;font-size:13px}
table{border-collapse:collapse;width:100%;margin-top:8px;font-size:14px}
th,td{border-bottom:1px solid #eaeef2;padding:6px 8px;text-align:left;vertical-align:top}
.bar{background:#eaeef2;height:8px;border-radius:4px;min-width:80px}
.fill{background:#0969da;height:8px;border-radius:4px}
details summary{cursor:pointer;color:#0969da}
"""


def build(run_dir: Path) -> str:
    traces = read_jsonl(run_dir / "spans.jsonl")
    if not traces:
        raise SystemExit(f"No traces found in {run_dir / 'spans.jsonl'}")
    s = summarise(traces)

    cards = [kpi("queries", s["queries"]),
             kpi("avg latency (ms)", s["latency_ms"]["avg"]),
             kpi("p95 latency (ms)", s["latency_ms"]["p95"]),
             kpi("avg LLM calls", s["avg_llm_calls"]),
             kpi("avg prompt tokens", s["avg_prompt_tokens"]),
             kpi("avg completion tokens", s["avg_completion_tokens"]),
             kpi("verifier failed", f"{round(100 * s['verifier_failed_rate'])}%"),
             kpi("repair attempted", f"{round(100 * s['repair_rate'])}%"),
             kpi("errors", f"{round(100 * s['error_rate'])}%")]

    # Scored numbers exist only when run_eval.py produced them for this run.
    scored = {}
    summary_file = run_dir / "summary.json"
    if summary_file.exists():
        scored = json.loads(summary_file.read_text(encoding="utf-8"))
        if scored.get("decision_accuracy") is not None:
            cards.insert(1, kpi("decision accuracy", f"{round(100 * scored['decision_accuracy'])}%"))

    time_rows = [[esc(k), esc(s["avg_ms_per_span"][k]) + " ms", esc(f"{round(100 * v)}%"), bar(v, 1)]
                 for k, v in s["share_of_span_time"].items()]

    parts = [
        f"<h1>Copilot run: {esc(run_dir.name)}</h1>",
        f'<div class="sub">Built {time.strftime("%Y-%m-%d %H:%M")} from spans.jsonl. '
        "Queries and answers are redacted for personal data.</div>",
        f'<div class="kpis">{"".join(cards)}</div>',
        counts_section("Decisions", s["decisions"]),
        counts_section("Failure labels (first-pass, to be confirmed by a person)",
                       scored.get("auto_failure_labels", {})),
        "<h2>Where the time goes</h2>" + table(["step", "avg per call", "share of step time", ""], time_rows),
        "<h2>Queries</h2>" + table(["query", "decision", "latency (ms)", "tokens", "verifier ok", "steps"],
                                   query_rows(traces)),
    ]
    return ("<!doctype html><html><head><meta charset='utf-8'><title>Copilot run</title>"
            f"<style>{STYLE}</style></head><body>{''.join(parts)}</body></html>")


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = Path(sys.argv[1])
    out = run_dir / "dashboard.html"
    out.write_text(build(run_dir), encoding="utf-8")
    print("Wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
