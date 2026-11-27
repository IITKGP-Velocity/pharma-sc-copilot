"""Prompting strategies S0-S4: prompt assembly, JSON checking and the self-consistency vote (no network)."""
import pytest

from copilot.agent.strategies import SAMPLES, STRATEGIES, ask_once, run_strategy, system_prompt


class TextChat:
    """Fake chat endpoint returning raw text replies in order; records temperatures."""
    def __init__(self, replies):
        self.replies, self.temperatures = list(replies), []

    def chat(self, messages, temperature=0.0, max_tokens=600):
        self.temperatures.append(temperature)
        return self.replies.pop(0)


EVIDENCE = [{"doc_id": "label-x#c0", "text": "Store at 2 C to 8 C."}]


@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_every_strategy_has_a_prompt_with_the_json_contract(strategy):
    assert '"decision"' in system_prompt(strategy)


def test_prompts_differ_by_technique():
    assert "Examples" in system_prompt("S1") and "step by step" in system_prompt("S2")
    assert "sub-questions" in system_prompt("S4") and system_prompt("S3") == system_prompt("S2")
    assert "Examples" not in system_prompt("S0")


def test_valid_reply_is_parsed():
    chat = TextChat(['<think>x</think>{"decision": "answer", "answer": "2-8 C", "citations": ["label-x#c0"]}'])
    out = ask_once(chat, "sys", "How to store?", EVIDENCE)
    assert out["decision"] == "ANSWER" and out["format_ok"] and out["citations"] == ["label-x#c0"]


@pytest.mark.parametrize("reply", ["Store it cold.", '{"decision": "MAYBE"}'])
def test_invalid_reply_counts_as_format_failure(reply):
    out = ask_once(TextChat([reply]), "sys", "q", EVIDENCE)
    assert out["format_ok"] is False and out["decision"] is None


def test_self_consistency_takes_the_majority_of_valid_samples():
    chat = TextChat(['{"decision": "ANSWER"}', '{"decision": "ABSTAIN"}', "not json",
                     '{"decision": "ANSWER"}', '{"decision": "ANSWER"}'])
    out = run_strategy(chat, "S3", "q", EVIDENCE)
    assert out["decision"] == "ANSWER" and out["valid_samples"] == 4
    assert len(chat.temperatures) == SAMPLES and set(chat.temperatures) == {0.7}


def test_other_strategies_make_one_deterministic_call():
    chat = TextChat(['{"decision": "REFUSE"}'])
    assert run_strategy(chat, "S1", "Edit the record", EVIDENCE)["decision"] == "REFUSE"
    assert chat.temperatures == [0.0]


def test_p95_uses_nearest_rank():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("run_baseline", pathlib.Path(__file__).parents[1] / "scripts" / "run_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.p95([100, 2000]) == 2000 and mod.p95(list(range(1, 101))) == 95
