from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def load_secret_env(env_file: str | Path = DEFAULT_ENV_FILE) -> None:
    """Load KEY=value entries from a local env file without overriding existing values."""
    path = Path(env_file)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_embeddings_key(env_file: str | Path = DEFAULT_ENV_FILE) -> str:
    load_secret_env(env_file)
    key = os.environ.get("EMBEDDINGS_KEY", "").strip()
    if not key or key == "YOUR_EMBEDDINGS_KEY":
        raise RuntimeError(
            f"EMBEDDINGS_KEY is not configured. Add it to {env_file} or set the environment variable."
        )
    return key
