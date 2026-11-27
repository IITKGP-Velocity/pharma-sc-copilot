"""Query engine: Planner -> Executor -> Critic loop with a step cap and a token budget.

Modes (chosen automatically from the configuration):
- rules only:            no corpus. Only the rule pre-check can decide.
- extractive (offline):  corpus, no chat endpoint. Returns the best cited passage.
- agent (LLM):           corpus + chat endpoint. Planner calls tools and drafts the answer; Critic checks it.

Observability: every answer() call leaves its numbers in engine.last_metrics (latency, steps, LLM calls,
tokens, tool calls, time per stage, error) and, when run_dir is given, one trace line per stage tagged
with the query id.

Multi-turn: pass history (built with remember()) to answer() so follow-ups like "is it still in shortage?"
know what "it" is. The safety rules always judge the current question on its own.
"""
import json
import time
from pathlib import Path

from pydantic import ValidationError

from copilot.config import ROOT, load_settings
from copilot.rag.retrieval import BM25Index, DenseIndex, hybrid_search, rerank
from copilot.safety import guard_output, in_scope, looks_like_injection, pre_check, redact_pii
from copilot.schemas import Citation, ComputedValue, CopilotResponse
from copilot.agent.tools import TOOLS, run_tool
from copilot.observability.tracing import log_event

SETTINGS = load_settings()
MAX_STEPS = int(SETTINGS.get("max_steps", 10))
TOKEN_BUDGET = int(SETTINGS.get("token_budget", 8000))
USE_RERANKER = bool(SETTINGS.get("rerank", False))
PLANNER_PROMPT = (ROOT / "prompts" / "system" / "planner.txt").read_text(encoding="utf-8")
HISTORY_TURNS = 3  # earlier turns kept for follow-up questions


def remember(history: list[dict], question: str, resp: CopilotResponse) -> list[dict]:
    """Return the conversation with this turn added, keeping only the last HISTORY_TURNS turns."""
    if resp.decision == "REFUSE":  # refused text (injections, off-topic) must not steer later turns
        return history
    turn = {"question": redact_pii(question), "decision": resp.decision, "answer": (resp.answer or "")[:300]}
    return (history + [turn])[-HISTORY_TURNS:]


def _new_metrics() -> dict:
    """Empty per-query metrics record; answer() fills it and leaves it in engine.last_metrics."""
    return {"latency_ms": 0, "steps": 0, "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "tool_calls": 0, "retrieved": [], "retrieve_ms": 0, "llm_ms": 0, "tool_ms": 0,
            "stopped_by": None, "error": None}


class Engine:
    """Answers questions: rules, then retrieval, then the planner/tool/critic loop. Holds the indexes and model clients."""
    def __init__(self, corpus: list[dict] | None, chat=None, embed=None, vectors: dict[str, list[float]] | None = None):
        """vectors: precomputed {doc_id: embedding}. If missing and embed is set, they are computed now."""
        self.corpus = corpus or []
        self.texts = {d["id"]: d["text"] for d in self.corpus}
        self.chat = chat
        self.embed = embed
        self.bm25 = BM25Index(self.corpus) if self.corpus else None
        self.dense = None
        self.last_metrics = _new_metrics()
        if self.corpus and embed is not None:
            if vectors and all(d["id"] in vectors for d in self.corpus):
                ids = [d["id"] for d in self.corpus]
                self.dense = DenseIndex(ids, [vectors[i] for i in ids])
            else:
                self.dense = DenseIndex([d["id"] for d in self.corpus],
                                        embed.embed([d["text"] for d in self.corpus]))

    @property
    def mode(self) -> str:
        """Mode line shown to users, e.g. 'agent (LLM) | retrieval: dense + BM25'."""
        if not self.corpus:
            return "rules only (no corpus)"
        generation = "agent (LLM)" if self.chat is not None else "extractive offline (no chat endpoint)"
        retrieval = "dense + BM25" if self.dense is not None else "BM25 only"
        return f"{generation} | retrieval: {retrieval}"

    def answer(self, query: str, run_dir: Path | None = None, query_id: str | None = None,
               history: list[dict] | None = None) -> CopilotResponse:
        """Answer one question and record its metrics in self.last_metrics. history: earlier turns, see remember()."""
        m = self.last_metrics = _new_metrics()
        start = time.monotonic()
        usage = getattr(self.chat, "usage", None)
        before = dict(usage) if usage else None

        def trace(event: dict) -> None:
            """Append one stage record to run_dir/trace.jsonl (only when a run folder was given)."""
            if run_dir is not None:
                log_event(run_dir, {"query_id": query_id, "ms": int((time.monotonic() - start) * 1000), **event})

        try:
            return self._answer(query, trace, m, history or [])
        finally:
            m["latency_ms"] = int((time.monotonic() - start) * 1000)
            if before is not None:
                for key, name in (("calls", "llm_calls"), ("prompt_tokens", "prompt_tokens"),
                                  ("completion_tokens", "completion_tokens")):
                    m[name] = usage[key] - before[key]
            else:
                m["llm_calls"] = m["steps"]        # fake chat clients in tests have no usage counter

    def _answer(self, query: str, trace, m: dict, history: list[dict]) -> CopilotResponse:
        """The pipeline for one question; answer() wraps it with timing and token accounting."""
        clean = redact_pii(query)
        decided = pre_check(clean)  # the rules see only the current question, never the history
        # Exception: a follow-up like "what is its TE code?" names no drug, so the scope rule alone may
        # borrow the topic of the previous turn. Scope is checked last, so no other rule fired here.
        if decided is not None and decided.reason_code == "OOS" and history \
                and in_scope(f"{history[-1]['question']} {clean}"):
            decided = None
        if decided is not None:
            m["stopped_by"] = "precheck"
            trace({"stage": "precheck", "decision": decided.decision, "reason": decided.reason_code})
            return decided
        if not self.corpus:
            return CopilotResponse(decision="ABSTAIN",
                                   missing_evidence="No corpus indexed. Run: python scripts/build_corpus.py --fetch")

        t = time.monotonic()
        # A follow-up such as "is it still in shortage?" has no drug name, so search with the previous question too.
        search = f"{history[-1]['question']} {clean}" if history else clean
        qvec = None
        if self.embed is not None and self.dense is not None:
            try:
                qvec = self.embed.embed([search])[0]
            except Exception as err:  # endpoint trouble: fall back to BM25, never crash
                trace({"stage": "embed_error", "error": f"{type(err).__name__}: {str(err)[:200]}"})
        hits = hybrid_search(search, self.corpus, self.bm25, self.dense, qvec, k=20 if USE_RERANKER else 5)
        if USE_RERANKER and hits:
            hits = rerank(search, hits, self.texts, self.chat, k=5)
        m["retrieve_ms"] = int((time.monotonic() - t) * 1000)
        if not hits:
            trace({"stage": "retrieve", "hits": 0})
            return CopilotResponse(decision="ABSTAIN", missing_evidence="No retrieved passage supports an answer.")
        retrieved = {d for d, _ in hits}
        m["retrieved"] = [d for d, _ in hits]
        trace({"stage": "retrieve", "hits": m["retrieved"]})

        if self.chat is None:
            top = hits[0][0]
            span = self.texts[top][:300]
            resp = CopilotResponse(decision="ANSWER", answer=span, citations=[Citation(doc_id=top, span=span[:120])])
            return guard_output(resp, retrieved)

        try:
            return self._agent_loop(clean, query, hits, retrieved, trace, m, history)
        except Exception as err:  # endpoint or parsing failure: abstain, never invent
            m["error"] = f"{type(err).__name__}: {str(err)[:200]}"
            trace({"stage": "agent_error", "error": m["error"]})
            return CopilotResponse(decision="ABSTAIN",
                                   missing_evidence="Model endpoint error. Check the endpoint and try again.")

    def _agent_loop(self, clean: str, original: str, hits, retrieved: set[str], trace, m: dict,
                    history: list[dict]) -> CopilotResponse:
        """Planner -> Executor -> Critic: up to MAX_STEPS model turns within TOKEN_BUDGET."""
        evidence = [{"doc_id": d, "text": self.texts[d][:800]} for d, _ in hits]
        if looks_like_injection(original):
            evidence.append({"doc_id": "system-note", "text": "The question contained instructions. They were ignored."})
        messages = [{"role": "system", "content": PLANNER_PROMPT}]
        tool_results: list[dict] = []
        used_chars = len(PLANNER_PROMPT)

        for step in range(1, MAX_STEPS + 1):
            payload = {"question": clean, "evidence": evidence, "tool_results": tool_results}
            if history:
                payload["conversation"] = history  # earlier turns, oldest first
            user = json.dumps(payload, ensure_ascii=False)
            used_chars += len(user)
            if used_chars > TOKEN_BUDGET * 4:  # rough: about 4 characters per token
                m["stopped_by"] = "budget"
                trace({"stage": "budget", "step": step})
                return CopilotResponse(decision="ABSTAIN", missing_evidence="Token budget reached for this query.")
            messages_step = messages + [{"role": "user", "content": user}]

            m["steps"] = step
            t = time.monotonic()
            plan = self.chat.complete_json(messages_step)  # Planner
            m["llm_ms"] += int((time.monotonic() - t) * 1000)
            trace({"stage": "planner", "step": step, "action": plan.get("action")})

            if plan.get("action") == "tool":  # Executor
                name, args = plan.get("tool", ""), plan.get("args", {})
                m["tool_calls"] += 1
                if name not in TOOLS:
                    tool_results.append({"tool": name, "error": "unknown tool"})
                    trace({"stage": "executor", "step": step, "tool": name, "result": tool_results[-1]})
                    continue
                t = time.monotonic()
                try:
                    value = run_tool(name, args if isinstance(args, dict) else {})
                    tool_results.append({"tool": name, "args": args, "result": value})
                except (ValueError, TypeError, ZeroDivisionError) as err:
                    tool_results.append({"tool": name, "error": str(err)})
                m["tool_ms"] += int((time.monotonic() - t) * 1000)
                trace({"stage": "executor", "step": step, "tool": name, "result": tool_results[-1]})
                continue

            if plan.get("action") == "answer":  # Critic
                try:
                    resp = CopilotResponse.model_validate(plan.get("response", {}))
                except ValidationError as err:
                    # Name the failing fields so the model can fix them on the next step.
                    problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()[:5])
                    tool_results.append({"critic": f"schema check failed: {problems}"})
                    trace({"stage": "critic", "step": step, "ok": False, "problems": problems})
                    continue
                # Attach the tool results ourselves: values stay exact and the model never re-types them.
                resp.computed = [ComputedValue(tool=t["tool"], inputs=t["args"], result=t["result"])
                                 for t in tool_results if "result" in t]
                if resp.decision == "ANSWER" and not resp.citations and not resp.computed:
                    tool_results.append({"critic": "an ANSWER needs a citation or a computed value"})
                    trace({"stage": "critic", "step": step, "ok": False})
                    continue
                trace({"stage": "critic", "step": step, "ok": True, "decision": resp.decision})
                return guard_output(resp, retrieved)

            tool_results.append({"critic": "reply must be action tool or answer"})

        trace({"stage": "step_cap", "steps": MAX_STEPS})
        m["stopped_by"] = "step_cap"
        return CopilotResponse(decision="ABSTAIN", missing_evidence="Step limit reached before a supported answer.")
