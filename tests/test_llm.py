"""Model client: JSON parsing, blank settings and CA certificate handling (no network)."""
import pytest
import requests

from copilot.config import env, llm_configured
from copilot.agent.llm import ChatClient, EmbedClient, EndpointNotConfigured, ca_bundle, parse_json_object


def test_parses_plain_and_fenced_json():
    assert parse_json_object('{"action": "answer"}') == {"action": "answer"}
    assert parse_json_object('Here:\n```json\n{"action": "tool"}\n```')["action"] == "tool"


def test_trailing_prose_is_rejected():
    with pytest.raises(ValueError):
        parse_json_object('{"action": "answer"} thanks!')


def test_blank_endpoint_means_not_configured(monkeypatch):
    for name in ["COPILOT_CHAT_ENDPOINT", "COPILOT_CHAT_API_KEY", "COPILOT_MODEL_NAME"]:
        monkeypatch.setenv(name, "   ")  # blank counts as missing
    assert llm_configured() is False
    with pytest.raises(EndpointNotConfigured):
        ChatClient()


def test_env_treats_blank_as_none(monkeypatch):
    monkeypatch.setenv("SOME_VALUE", "  ")
    assert env("SOME_VALUE") is None


def test_ca_bundle_blank_uses_system_cas(monkeypatch):
    monkeypatch.setenv("COPILOT_CHAT_CA_CERT", "")
    assert ca_bundle("COPILOT_CHAT_CA_CERT") is True


def test_ca_bundle_missing_file_fails_early(monkeypatch):
    monkeypatch.setenv("COPILOT_CHAT_CA_CERT", "secrets/does-not-exist.crt")
    with pytest.raises(FileNotFoundError):
        ca_bundle("COPILOT_CHAT_CA_CERT")


def test_ca_bundle_resolves_relative_path(monkeypatch, tmp_path):
    cert = tmp_path / "ca.crt"
    cert.write_text("x")
    monkeypatch.setenv("COPILOT_CHAT_CA_CERT", str(cert))
    assert ca_bundle("COPILOT_CHAT_CA_CERT") == str(cert)


def test_think_block_before_json_is_ignored():
    assert parse_json_object('<think>plan...</think>\n{"action": "answer"}') == {"action": "answer"}


class _TokenLimitedServer(EmbedClient):
    """Fake embeddings server that rejects any input longer than 50 characters, like the 512-token limit."""
    def __init__(self):
        self.model, self.calls = "fake", []

    def _post(self, path, payload):
        self.calls.append(list(payload["input"]))
        too_long = [t for t in payload["input"] if len(t) > 50]
        if too_long:
            raise requests.HTTPError(f"HTTP 422 from /embeddings: `inputs` must have less than 50 tokens. Given: {len(too_long[0])}")
        return {"data": [{"index": i, "embedding": [float(len(t))]} for i, t in enumerate(payload["input"])]}


def test_embed_shortens_only_texts_over_the_token_limit():
    client = _TokenLimitedServer()
    vectors = client.embed(["short", "x" * 200, "also short"], batch=16)
    assert len(vectors) == 3
    assert vectors[0] == [5.0] and vectors[2] == [10.0]   # untouched, order kept
    assert vectors[1][0] <= 50                          # the long text was shortened to fit


def test_embed_reraises_other_http_errors():
    class Broken(_TokenLimitedServer):
        def _post(self, path, payload):
            raise requests.HTTPError("HTTP 401 from /embeddings: bad key")
    with pytest.raises(requests.HTTPError):
        Broken().embed(["a"])
