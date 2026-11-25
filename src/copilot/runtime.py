"""Endpoint settings passed by the person running a script, not stored in files.

- Non-secret values (endpoint URLs, model names, CA certificate paths) can be given as command-line
  flags, as environment variables, or in .env.
- API keys are never read from .env and should not be typed on the command line (they would end up in
  shell history). Export them in the shell for this session, pass a file that holds the key
  (--api-key-file, for non-interactive runs), or leave them blank and the script asks with a hidden prompt.
"""
import argparse
import getpass
import os
import sys
from pathlib import Path

# flag name -> environment variable it sets
FLAGS = {
    "chat_endpoint": "COPILOT_CHAT_ENDPOINT",
    "model": "COPILOT_MODEL_NAME",
    "chat_ca_cert": "COPILOT_CHAT_CA_CERT",
    "embed_endpoint": "COPILOT_EMBED_ENDPOINT",
    "embed_model": "COPILOT_EMBED_MODEL",
    "embed_ca_cert": "COPILOT_EMBED_CA_CERT",
}


def add_endpoint_args(parser: argparse.ArgumentParser) -> None:
    """Add the endpoint, model, certificate and key-file flags to a script's argument parser."""
    group = parser.add_argument_group("university endpoints (optional; keys are asked for, never passed here)")
    group.add_argument("--chat-endpoint", help="chat base URL ending in /v1")
    group.add_argument("--model", help="chat model name")
    group.add_argument("--chat-ca-cert", help="CA certificate file for the chat endpoint")
    group.add_argument("--embed-endpoint", help="embeddings base URL ending in /v1")
    group.add_argument("--embed-model", help="embeddings model name")
    group.add_argument("--embed-ca-cert", help="CA certificate file for the embeddings endpoint")
    group.add_argument("--api-key-file", help="file holding the chat API key (also used for embeddings)")
    group.add_argument("--embed-api-key-file", help="file holding a separate embeddings API key")


def read_key_file(path: str) -> str:
    """Return the key stored in a file (its first word). Never logs the value."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"API key file not found: {p}")
    lines = p.read_text(encoding="utf-8").split()
    if not lines:
        raise ValueError(f"API key file is empty: {p}")
    return lines[0]


def _blank(name: str) -> bool:
    """True when the environment variable is unset or only spaces."""
    return not (os.environ.get(name) or "").strip()


def apply_endpoint_args(args: argparse.Namespace, prompt=getpass.getpass) -> None:
    """Copy given flags into the environment, then ask for any missing key of a configured endpoint."""
    for attr, name in FLAGS.items():
        value = getattr(args, attr, None)
        if value:
            os.environ[name] = value

    if getattr(args, "api_key_file", None):
        os.environ["COPILOT_CHAT_API_KEY"] = read_key_file(args.api_key_file)
    if getattr(args, "embed_api_key_file", None):
        os.environ["COPILOT_EMBED_API_KEY"] = read_key_file(args.embed_api_key_file)
    elif getattr(args, "api_key_file", None) and _blank("COPILOT_EMBED_API_KEY"):
        # The course uses one shared key for both services.
        os.environ["COPILOT_EMBED_API_KEY"] = os.environ["COPILOT_CHAT_API_KEY"]

    # Only prompt in an interactive terminal; in CI or Docker a missing key just turns the feature off.
    if not sys.stdin.isatty():
        return
    if not _blank("COPILOT_CHAT_ENDPOINT") and _blank("COPILOT_CHAT_API_KEY"):
        os.environ["COPILOT_CHAT_API_KEY"] = prompt("Chat API key (hidden, Enter to skip): ")
    if not _blank("COPILOT_EMBED_ENDPOINT") and _blank("COPILOT_EMBED_API_KEY"):
        # The course uses one shared key for both services, so Enter reuses the chat key.
        key = prompt("Embeddings API key (hidden, Enter = same as chat key): ")
        os.environ["COPILOT_EMBED_API_KEY"] = key or os.environ.get("COPILOT_CHAT_API_KEY", "")
