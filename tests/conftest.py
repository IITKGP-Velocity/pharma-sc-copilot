"""Shared test fakes: a scripted chat model, keyword-bag embeddings and a three-passage corpus. No network."""
import sys
from pathlib import Path

import pytest

# Scripts import their helper module _paths from scripts/; make it importable for every test file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class ScriptedChat:
    """Fake chat endpoint: returns scripted JSON replies, one per call. No network."""

    def __init__(self, replies):
        """replies: the JSON objects to return, in order."""
        self.replies = list(replies)
        self.calls = 0

    def complete_json(self, messages, retries=1):
        """Return the next scripted reply, like ChatClient.complete_json."""
        self.calls += 1
        if not self.replies:
            raise RuntimeError("script exhausted")
        return self.replies.pop(0)


class FakeEmbed:
    """Keyword-bag embeddings so dense search is testable offline."""

    VOCAB = ["storage", "temperature", "adalimumab", "shortage", "recall", "cisplatin"]

    def embed(self, texts):
        """One vector per text: how often each VOCAB word appears."""
        out = []
        for t in texts:
            t = t.lower()
            out.append([float(t.count(w)) for w in self.VOCAB] or [0.0])
        return out


@pytest.fixture
def corpus():
    """Three passages: an adalimumab label, a WHO mapping note and a cisplatin shortage."""
    return [
        {"id": "label-adalimumab", "text": "Storage temperature: store unopened adalimumab pens refrigerated at 2 C to 8 C. Do not freeze."},
        {"id": "who-trs961", "text": "Temperature mapping of transport and storage is a qualification activity."},
        {"id": "shortage-cisplatin", "text": "Drug shortage: cisplatin injection. Status: Currently in shortage."},
    ]


@pytest.fixture(autouse=True)
def no_rerank(monkeypatch):
    """Scripted chat replies count every model call, so keep the reranker call out of agent tests.
    The reranker has its own tests in test_retrieval.py."""
    monkeypatch.setattr("copilot.agent.engine.USE_RERANKER", False)
