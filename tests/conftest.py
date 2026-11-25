import pytest


class ScriptedChat:
    """Fake chat endpoint: returns scripted JSON replies, one per call. No network."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete_json(self, messages, retries=1):
        self.calls += 1
        if not self.replies:
            raise RuntimeError("script exhausted")
        return self.replies.pop(0)


class FakeEmbed:
    """Keyword-bag embeddings so dense search is testable offline."""

    VOCAB = ["storage", "temperature", "adalimumab", "shortage", "recall", "cisplatin"]

    def embed(self, texts):
        out = []
        for t in texts:
            t = t.lower()
            out.append([float(t.count(w)) for w in self.VOCAB] or [0.0])
        return out


@pytest.fixture
def corpus():
    return [
        {"id": "label-adalimumab", "text": "Storage temperature: store unopened adalimumab pens refrigerated at 2 C to 8 C. Do not freeze."},
        {"id": "who-trs961", "text": "Temperature mapping of transport and storage is a qualification activity."},
        {"id": "shortage-cisplatin", "text": "Drug shortage: cisplatin injection. Status: Currently in shortage."},
    ]
