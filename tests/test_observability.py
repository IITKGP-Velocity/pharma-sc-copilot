"""Offline tests for the observability layer, agent metrics, retries, redaction and the demo."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from conftest import ScriptedChat
from copilot import observability as obs
from copilot.agent import llm
from copilot.agent import Engine
from copilot.safety import redact_pii
from copilot.schemas import CopilotResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def metrics(**kw):
    base = {"latency_ms": 10, "steps": 1, "llm_calls": 1, "prompt_tokens": 100, "completion_tokens": 20,
            "tool_calls": 0, "retrieve_ms": 1, "llm_ms": 8, "tool_ms": 0, "stopped_by": None, "error": None,
            "retrieved": ["d1"]}
    return {**base, **kw}


def row(qid, qtype, decision, **kw):
    item = {"id": qid, "use_case": qid[0], "type": qtype, "query": "Driver Ravi Kumar phone 9876543210?"}
    return obs.result_row(item, CopilotResponse(decision=decision), metrics(**kw))


# ---- redaction -----------------------------------------------------------------------------

def test_names_after_labels_and_dates_of_birth_are_redacted():
    out = redact_pii("Include patient Asha Nair, date of birth 12/03/1985; contact Anita Rao.")
    assert "Asha Nair" not in out and "12/03/1985" not in out and "Anita Rao" not in out


def test_other_dates_and_drug_names_are_kept():
    text = "Shipment 4471 left on 2026-11-21. Store Humira at 2 C to 8 C."
    assert redact_pii(text) == text


# ---- rows, labels, summary -----------------------------------------------------------------

def test_result_row_scores_and_redacts():
    r = row("A01", "ANS", "ABSTAIN")
    assert r["expected"] == "ANSWER" and r["ok"] is False
    assert r["auto_label"] == "retrieval_miss_or_over_refusal"
    assert "9876543210" not in r["query"] and "Ravi Kumar" not in r["query"]


@pytest.mark.parametrize("expected, pred, error, stopped, label", [
    ("ANSWER", "ANSWER", None, None, ""),
    ("ANSWER", "ABSTAIN", "ValueError: bad JSON", None, "tool_or_format_error"),
    ("ANSWER", "ABSTAIN", None, "step_cap", "tool_or_format_error"),
    ("ESCALATE", "ANSWER", None, None, "under_refusal"),
    ("REFUSE", "ESCALATE", None, None, "wrong_decision"),
])
def test_auto_label(expected, pred, error, stopped, label):
    assert obs.auto_label(expected, pred, error, stopped) == label


def test_summary_counts_accuracy_latency_and_rules():
    rows = [row("A01", "ANS", "ANSWER", latency_ms=100), row("A02", "ESC", "ESCALATE", stopped_by="precheck",
            latency_ms=1, steps=0, llm_calls=0), row("A03", "ANS", "ABSTAIN", error="HTTPError", latency_ms=300)]
    s = obs.summarise(rows)
    assert s["decision_accuracy"] == pytest.approx(0.667, abs=0.001)
    assert s["by_type"] == {"ANS": "1/2", "ESC": "1/1"}
    assert s["decided_by_rules"] == 1 and s["errors"] == 1
    assert s["agent_avg"]["llm_calls"] == 1.0                     # the rule-decided query is left out
    assert s["latency_ms"]["max"] == 300


# ---- failure log ---------------------------------------------------------------------------

def test_failure_log_lists_wrong_rows_and_keeps_human_labels(tmp_path):
    rows = [row("A01", "ANS", "ANSWER"), row("A02", "ANS", "ABSTAIN")]
    path = obs.write_failure_log(tmp_path, rows)
    with path.open(newline="") as f:
        data = list(csv.DictReader(f))
    assert [r["item_id"] for r in data] == ["A02"]
    data[0]["label"], data[0]["description"] = "retrieval_mis", "typo on purpose"   # a person edits it
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    obs.write_failure_log(tmp_path, rows)                               # re-run must keep the edit
    assert obs.unknown_labels(path) == ["A02: retrieval_mis"]
    assert obs.label_counts(path) == {"retrieval_mis": 1}


# ---- dashboard -----------------------------------------------------------------------------

def test_dashboard_escapes_and_shows_trace(tmp_path):
    rows = [row("A01", "ANS", "ANSWER")]
    rows[0]["query"] = "<script>alert(1)</script>"
    obs.write_jsonl(tmp_path / "results.jsonl", rows)
    (tmp_path / "trace.jsonl").write_text(json.dumps({"query_id": "A01", "ms": 5, "stage": "planner"}) + "\n")
    page = obs.build_dashboard(tmp_path).read_text()
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    assert "1 steps" in page and "decision accuracy" in page


# ---- retrieval metrics ---------------------------------------------------------------------

def test_recall_and_mrr():
    passages = {"p1": {"source": "s", "text": "vancomycin store at 20 C"},
                "p2": {"source": "s", "text": "heparin"}, "p3": {"source": "t", "text": "vancomycin store"}}
    gold = [{"id": "q1", "source": "s", "contains": ["vancomycin", "store"]},
            {"id": "q2", "source": "s", "contains": ["lidocaine"]}]
    m = obs.retrieval_metrics({"q1": ["p3", "p1"], "q2": ["p2"]}, gold, passages, k=5)
    assert m["recall_at_5"] == 0.5 and m["mrr"] == 0.25               # q1 found at rank 2, q2 missed


# ---- agent metrics -------------------------------------------------------------------------

def test_agent_records_metrics_and_query_id(corpus, tmp_path):
    chat = ScriptedChat([
        {"action": "tool", "tool": "mkt", "args": {"temps_c": [4, 5, 6, 14]}},
        {"action": "answer", "response": {"decision": "ANSWER", "answer": "x",
                                          "citations": [{"doc_id": "label-adalimumab", "span": "2 C"}]}}])
    e = Engine(corpus, chat=chat)
    e.answer("storage temperature unopened adalimumab pens", run_dir=tmp_path, query_id="Q1")
    m = e.last_metrics
    assert (m["steps"], m["tool_calls"], m["llm_calls"]) == (2, 1, 2) and m["latency_ms"] >= 0
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert {ev["query_id"] for ev in events} == {"Q1"} and all("ms" in ev for ev in events)


def test_agent_error_detail_is_kept(corpus):
    class Broken:
        def complete_json(self, *a, **k):
            raise ValueError("Expecting ',' delimiter")
    e = Engine(corpus, chat=Broken())
    assert e.answer("storage temperature unopened adalimumab pens").decision == "ABSTAIN"
    assert e.last_metrics["error"].startswith("ValueError: Expecting")


def test_rules_decision_is_marked(corpus):
    e = Engine(corpus)
    e.answer("Can we release it to the customer?")
    assert e.last_metrics["stopped_by"] == "precheck"


# ---- client retries ------------------------------------------------------------------------

def test_busy_server_is_retried(monkeypatch):
    replies = [SimpleNamespace(status_code=429, text="busy"),
               SimpleNamespace(status_code=200, text="", json=lambda: {"ok": True})]
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    client = llm._OpenAICompatible("https://h/v1", "k")
    assert client._post("/x", {}) == {"ok": True}


def test_client_error_is_not_retried_and_names_the_reason(monkeypatch):
    calls = []
    def post(*a, **k):
        calls.append(1)
        return SimpleNamespace(status_code=422, text='{"message":"inputs must have less than 512 tokens"}')
    monkeypatch.setattr(llm.requests, "post", post)
    with pytest.raises(requests.HTTPError, match="512 tokens"):
        llm._OpenAICompatible("https://h/v1", "k")._post("/embeddings", {})
    assert len(calls) == 1


# ---- demo ----------------------------------------------------------------------------------

def test_fake_demo_run_writes_every_report_file(tmp_path, monkeypatch):
    import demo
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    gold = tmp_path / "eval" / "gold"
    gold.mkdir(parents=True)
    src = Path(__file__).resolve().parents[1] / "eval" / "gold" / "eval_set_v1.0.jsonl"
    gold.joinpath("eval_set_v1.0.jsonl").write_text("\n".join(src.read_text().splitlines()[:6]) + "\n")
    out = demo.demo_dashboard(open_page=False)
    assert {p.name for p in out.iterdir()} >= {"results.jsonl", "trace.jsonl", "summary.json",
                                               "failure_log.csv", "dashboard.html"}
