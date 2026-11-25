import pytest

from copilot.config import env, llm_configured
from copilot.agent.llm import ChatClient, EndpointNotConfigured, ca_bundle, parse_json_object


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
