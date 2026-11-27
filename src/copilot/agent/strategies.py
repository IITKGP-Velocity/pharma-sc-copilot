"""Single-prompt baselines S0-S4 (Deliverable 2): one model call per question, no tools, no agent loop.

Every strategy gets the same question and the same retrieved passages; only the prompt changes.
  S0 control: base prompt only          S1 structured output + few-shot examples
  S2 chain-of-thought                   S3 self-consistency: S2 sampled 5 times, majority vote
  S4 decomposition into sub-questions
"""
import json
from collections import Counter

from copilot.agent.llm import parse_json_object
from copilot.config import ROOT

PROMPTS = ROOT / "prompts" / "strategies"
STRATEGIES = {"S0": "control (zero-shot)", "S1": "structured output + few-shot", "S2": "chain-of-thought",
              "S3": "self-consistency (5 samples)", "S4": "decomposition"}
DECISIONS = {"ANSWER", "ESCALATE", "REFUSE", "ABSTAIN"}
SAMPLES = 5


def system_prompt(strategy: str) -> str:
    """Base prompt plus the strategy's addition (S3 reuses the chain-of-thought prompt)."""
    extra = {"S1": "S1", "S2": "S2", "S3": "S2", "S4": "S4"}.get(strategy)
    base = (PROMPTS / "base.txt").read_text(encoding="utf-8")
    return base + ((PROMPTS / f"{extra}.txt").read_text(encoding="utf-8") if extra else "")


def ask_once(chat, system: str, question: str, evidence: list[dict], temperature: float = 0.0) -> dict:
    """One model call. format_ok is False when the reply is not JSON or has no valid decision."""
    user = json.dumps({"question": question, "evidence": evidence}, ensure_ascii=False)
    text = chat.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                     temperature=temperature, max_tokens=1024)
    try:
        reply = parse_json_object(text)
    except ValueError:
        return {"decision": None, "format_ok": False, "answer": text[:300], "citations": []}
    decision = str(reply.get("decision", "")).upper()
    ok = decision in DECISIONS
    return {"decision": decision if ok else None, "format_ok": ok,
            "answer": str(reply.get("answer", ""))[:300], "citations": reply.get("citations") or []}


def run_strategy(chat, strategy: str, question: str, evidence: list[dict]) -> dict:
    """Answer with one strategy. S3 samples at temperature 0.7 and keeps the most common valid decision."""
    system = system_prompt(strategy)
    if strategy != "S3":
        return ask_once(chat, system, question, evidence)
    votes = [ask_once(chat, system, question, evidence, temperature=0.7) for _ in range(SAMPLES)]
    valid = [v for v in votes if v["format_ok"]]
    if not valid:
        return {**votes[0], "valid_samples": 0}
    winner = Counter(v["decision"] for v in valid).most_common(1)[0][0]
    return {**next(v for v in valid if v["decision"] == winner), "valid_samples": len(valid)}
