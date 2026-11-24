"""
tracing.py - Record a trace (timed spans) for every copilot query, with personal data redacted.

Why: the Week 3-4 deliverables need a trace of real query executions and per-query cost and
latency. This works with no extra service: it writes one JSON line per query to a file that
a dashboard (or Langfuse / Phoenix import) can read later.

How: traced_ask() temporarily wraps the retriever's search() and the LLM client's chat() on
the given pipeline, times each call as a span, then restores them. rag_pipeline.py is not
changed. Queries and answers are passed through redact_pii() before they are written (R4).

Usage
  from tracing import traced_ask
  resp, trace = traced_ask(pipeline, "What does 211.142 cover?", trace_file="EVAL/traces/run1.jsonl")
"""
import json
import time
import uuid
from pathlib import Path
from unittest import mock

from run_eval import STATUS_TO_DECISION
from safety import redact_pii


def _timed(name, fn, spans, t0, describe_result):
    """Return a copy of fn that adds one span to `spans` every time it is called.

    name            label of the span, e.g. "retrieve"
    t0              start time of the whole query, so spans show when they began
    describe_result function that turns fn's return value into extra span fields
    """
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        span = {"name": name, "start_ms": int((start - t0) * 1000), "error": None}
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            span["error"] = type(e).__name__      # keep the failure in the trace, then re-raise
            raise
        else:
            span.update(describe_result(result))  # only possible when the call succeeded
            return result
        finally:
            span["duration_ms"] = int((time.monotonic() - start) * 1000)
            spans.append(span)
    return wrapper


def traced_ask(pipeline, query: str, trace_file=None, **ask_kwargs):
    """Run pipeline.ask(query) and return (response, trace). Appends the trace to trace_file."""
    spans: list[dict] = []
    t0 = time.monotonic()

    retrieve = _timed("retrieve", pipeline.retriever.search, spans, t0,
                      lambda hits: {"n_hits": len(hits),
                                    "doc_ids": [h.get("doc_id") for h in hits]})
    llm_call = _timed("llm_call", pipeline.llm.chat, spans, t0,
                      lambda r: {"prompt_tokens": r.prompt_tokens,
                                 "completion_tokens": r.completion_tokens,
                                 "finish_reason": r.finish_reason})

    # mock.patch.object puts the original methods back even if ask() raises.
    with mock.patch.object(pipeline.retriever, "search", retrieve), \
         mock.patch.object(pipeline.llm, "chat", llm_call):
        resp = pipeline.ask(query, **ask_kwargs)

    trace = {
        "trace_id": uuid.uuid4().hex,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query": redact_pii(query),
        "status": resp.status,
        "decision": STATUS_TO_DECISION.get(resp.status, "ERROR"),
        "latency_ms": resp.latency_ms,
        "llm_calls": resp.llm_calls,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "verifier_passed": resp.verification.passed,
        "verifier_issues": resp.verification.issues,
        "repair_attempted": resp.verification.repair_attempted,
        "spans": sorted(spans, key=lambda s: s["start_ms"]),
        "answer": redact_pii(resp.answer),
    }
    if trace_file:
        path = Path(trace_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return resp, trace
