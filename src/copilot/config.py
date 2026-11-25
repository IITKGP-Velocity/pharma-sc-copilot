"""Settings and secrets.

- Non-secret settings: configs/settings.yaml
- Secrets and endpoints: .env (copied from .env.example) or real environment variables.
- Blank values mean "not configured". The code then runs in offline mode and says so.
"""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"


def _load_dotenv(path: Path = ENV_FILE) -> None:
    """Minimal .env loader: KEY=VALUE lines, # comments. Never overrides real environment variables.
    API keys are skipped on purpose: the person running passes them at run time (see copilot/runtime.py)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip().endswith("_API_KEY"):
            continue
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def load_settings(path: str | Path | None = None) -> dict:
    """Read the non-secret settings from configs/settings.yaml."""
    path = Path(path) if path else ROOT / "configs" / "settings.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def env(name: str, default: str | None = None) -> str | None:
    """Read a value from the environment. Blank counts as missing. Never print secret values."""
    value = os.environ.get(name, default)
    return value.strip() if value and value.strip() else None


def llm_configured() -> bool:
    """True when the chat endpoint, key and model name are all set."""
    return all(env(n) for n in ["COPILOT_CHAT_ENDPOINT", "COPILOT_CHAT_API_KEY", "COPILOT_MODEL_NAME"])


def embeddings_configured() -> bool:
    """True when the embeddings endpoint, key and model name are all set."""
    return all(env(n) for n in ["COPILOT_EMBED_ENDPOINT", "COPILOT_EMBED_API_KEY", "COPILOT_EMBED_MODEL"])
