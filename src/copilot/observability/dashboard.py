"""Dashboard: one self-contained HTML page per run (no server, no internet, no JavaScript).

Reads results.jsonl, summary.json and trace.jsonl from the run folder and writes dashboard.html.
"""
import html
import json
from collections import defaultdict
from pathlib import Path

from copilot.observability.metrics import read_jsonl, summarise


# ---- dashboard -----------------------------------------------------------------------------

_STYLE = """
body{font-family:system-ui,sans-serif;margin:24px;color:#1b1f23}
h1{margin-bottom:4px} h2{margin-top:28px;font-size:18px} .sub{color:#57606a}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin-top:16px}
.kpi{border:1px solid #d0d7de;border-radius:8px;padding:12px 18px;min-width:120px}
.kpi .v{font-size:24px;font-weight:600} .kpi .l{color:#57606a;font-size:13px}
table{border-collapse:collapse;width:100%;margin-top:8px;font-size:14px}
th,td{border-bottom:1px solid #eaeef2;padding:6px 8px;text-align:left;vertical-align:top}
.bar{background:#eaeef2;height:8px;border-radius:4px;min-width:80px}
.fill{background:#0969da;height:8px;border-radius:4px} .bad{color:#cf222e;font-weight:600}
details summary{cursor:pointer;color:#0969da} code{font-size:12px}
"""


def _e(value) -> str:
    """Escape every value taken from a run file before it goes into HTML (queries are user text)."""
    return html.escape(str(value), quote=True)


def _bar(value: float, top: float) -> str:
    """HTML bar whose filled width is value / top."""
    pct = 0 if top <= 0 else min(100, round(100 * value / top))
    return f'<div class="bar"><div class="fill" style="width:{pct}%"></div></div>'


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """HTML table; the cells must already be escaped."""
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    return f"<table><tr>{head}</tr>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                                                for r in rows) + "</table>"


def _fraction_rows(fractions: dict) -> list[list[str]]:
    """Turn {'ANS': '4/22', ...} into table rows with a bar each."""
    out = []
    for key, frac in fractions.items():
        ok, total = (int(x) for x in frac.split("/"))
        out.append([_e(key), _e(frac), _bar(ok, total)])
    return out


def build_dashboard(run_dir: Path) -> Path:
    """Write run_dir/dashboard.html from results.jsonl, summary.json and trace.jsonl."""
    rows = read_jsonl(run_dir / "results.jsonl")
    if not rows:
        raise SystemExit(f"No results.jsonl in {run_dir}")
    s = summarise(rows)
    meta = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if (run_dir / "summary.json").exists() else {}
    steps_by_query = defaultdict(list)
    for ev in read_jsonl(run_dir / "trace.jsonl"):
        if ev.get("query_id"):
            steps_by_query[ev["query_id"]].append(ev)

    def kpi(label, value):
        """One headline-number card."""
        return f'<div class="kpi"><div class="v">{_e(value)}</div><div class="l">{_e(label)}</div></div>'

    a = s["agent_avg"]
    cards = [kpi("queries", s["items"]), kpi("decision accuracy", f"{round(100 * s['decision_accuracy'])}%"),
             kpi("avg latency (ms)", s["latency_ms"]["avg"]), kpi("p95 latency (ms)", s["latency_ms"]["p95"]),
             kpi("decided by rules", s["decided_by_rules"]), kpi("avg LLM calls (agent)", a["llm_calls"]),
             kpi("avg tool calls (agent)", a["tool_calls"]),
             kpi("avg tokens (agent)", a["prompt_tokens"] + a["completion_tokens"]),
             kpi("total tokens", s["total_tokens"]), kpi("errors", s["errors"])]

    top_ms = max(a["retrieve_ms"], a["llm_ms"], a["tool_ms"], 1)
    stage_rows = [[_e(k), _e(f"{a[k]} ms"), _bar(a[k], top_ms)] for k in ("retrieve_ms", "llm_ms", "tool_ms")]
    labels = s["auto_labels"]
    label_rows = [[_e(k), _e(v), _bar(v, max(labels.values()))] for k, v in sorted(labels.items(), key=lambda kv: -kv[1])]

    slowest = max((r["latency_ms"] or 0 for r in rows), default=1)
    query_rows = []
    for r in rows:
        trail = "".join(f"<div><code>{_e(ev.get('ms'))} ms</code> {_e(ev.get('stage'))} "
                        f"{_e({k: v for k, v in ev.items() if k not in ('ts', 'ms', 'stage', 'query_id')})}</div>"
                        for ev in steps_by_query.get(r["id"], []))
        decision = _e(r["pred"]) if r["ok"] else f'<span class="bad">{_e(r["pred"])}</span> (expected {_e(r["expected"])})'
        query_rows.append([_e(r["id"]), _e(r["query"][:90]), decision, _e(r["latency_ms"]) + _bar(r["latency_ms"] or 0, slowest),
                           _e((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)), _e(r["tool_calls"]),
                           _e(r["error"] or ""),
                           f"<details><summary>{len(steps_by_query.get(r['id'], []))} steps</summary>{trail}</details>"])

    parts = [
        f"<h1>Copilot run: {_e(run_dir.name)}</h1>",
        f'<div class="sub">Mode: {_e(meta.get("mode", "unknown"))}. Queries and answers are redacted for personal data.</div>',
        f'<div class="kpis">{"".join(cards)}</div>',
        "<h2>Decision accuracy by question type</h2>" + _table(["type", "correct", ""], _fraction_rows(s["by_type"])),
        "<h2>Decision accuracy by use case</h2>" + _table(["use case", "correct", ""], _fraction_rows(s["by_use_case"])),
        "<h2>Where the time goes (average per agent query)</h2>" + _table(["stage", "time", ""], stage_rows),
        ("<h2>Failures by first-pass label (confirm in failure_log.csv)</h2>" + _table(["label", "queries", ""], label_rows))
        if label_rows else "",
        "<h2>Queries</h2>" + _table(["id", "query", "decision", "latency (ms)", "tokens", "tools", "error", "trace"],
                                    query_rows),
    ]
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>Copilot run</title>"
            f"<style>{_STYLE}</style></head><body>{''.join(parts)}</body></html>")
    out = run_dir / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    return out
