import argparse

import pytest

from copilot import config
from copilot.runtime import add_endpoint_args, apply_endpoint_args, read_key_file


def parse(*argv):
    ap = argparse.ArgumentParser()
    add_endpoint_args(ap)
    return ap.parse_args(list(argv))


def test_dotenv_never_loads_api_keys(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("COPILOT_CHAT_API_KEY=from-file\nCOPILOT_MODEL_NAME=m1\n")
    monkeypatch.delenv("COPILOT_CHAT_API_KEY", raising=False)
    monkeypatch.delenv("COPILOT_MODEL_NAME", raising=False)
    config._load_dotenv(env_file)
    assert config.env("COPILOT_CHAT_API_KEY") is None
    assert config.env("COPILOT_MODEL_NAME") == "m1"


def test_flags_set_endpoint_model_and_cert(monkeypatch):
    for n in ["COPILOT_CHAT_ENDPOINT", "COPILOT_MODEL_NAME", "COPILOT_CHAT_CA_CERT"]:
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    apply_endpoint_args(parse("--chat-endpoint", "https://h:1/v1", "--model", "m", "--chat-ca-cert", "c.crt"))
    assert (config.env("COPILOT_CHAT_ENDPOINT"), config.env("COPILOT_MODEL_NAME"),
            config.env("COPILOT_CHAT_CA_CERT")) == ("https://h:1/v1", "m", "c.crt")


def test_missing_key_is_asked_for_in_a_terminal(monkeypatch):
    monkeypatch.setenv("COPILOT_CHAT_ENDPOINT", "https://h:1/v1")
    monkeypatch.setenv("COPILOT_EMBED_ENDPOINT", "https://h:2/v1")
    monkeypatch.delenv("COPILOT_CHAT_API_KEY", raising=False)
    monkeypatch.delenv("COPILOT_EMBED_API_KEY", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["typed-key", ""])                    # Enter on the second prompt reuses the chat key
    apply_endpoint_args(parse(), prompt=lambda _msg: next(answers))
    assert config.env("COPILOT_CHAT_API_KEY") == "typed-key"
    assert config.env("COPILOT_EMBED_API_KEY") == "typed-key"


def test_no_prompt_when_not_interactive(monkeypatch):
    monkeypatch.setenv("COPILOT_CHAT_ENDPOINT", "https://h:1/v1")
    monkeypatch.delenv("COPILOT_CHAT_API_KEY", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    def fail(_msg):
        raise AssertionError("must not prompt")
    apply_endpoint_args(parse(), prompt=fail)
    assert config.env("COPILOT_CHAT_API_KEY") is None


def test_key_file_sets_both_keys_without_prompting(monkeypatch, tmp_path):
    key_file = tmp_path / "key.txt"
    key_file.write_text("sk-file-key\n")
    monkeypatch.setenv("COPILOT_CHAT_ENDPOINT", "https://h:1/v1")
    monkeypatch.setenv("COPILOT_EMBED_ENDPOINT", "https://h:2/v1")
    for n in ["COPILOT_CHAT_API_KEY", "COPILOT_EMBED_API_KEY"]:
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def fail(_msg):
        raise AssertionError("must not prompt")
    apply_endpoint_args(parse("--api-key-file", str(key_file)), prompt=fail)
    assert config.env("COPILOT_CHAT_API_KEY") == "sk-file-key"
    assert config.env("COPILOT_EMBED_API_KEY") == "sk-file-key"


def test_missing_or_empty_key_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_key_file(str(tmp_path / "nope.txt"))
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n")
    with pytest.raises(ValueError):
        read_key_file(str(empty))
