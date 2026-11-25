"""Client for the university OpenAI-compatible endpoints (chat and embeddings).

Keys come from the environment only. JSON replies are checked, and a malformed reply gets one retry
with the parse error fed back to the model.

The university endpoints use their own (private) CA certificates. Point COPILOT_CHAT_CA_CERT and
COPILOT_EMBED_CA_CERT at those files; TLS verification always stays on."""
import json
import re
import time
from pathlib import Path

import requests

from copilot.config import ROOT, env

RETRYABLE_STATUS = {429, 500, 502, 503, 504}   # busy or temporarily failing server
MAX_ATTEMPTS = 3


class EndpointNotConfigured(RuntimeError):
    pass


def ca_bundle(name: str) -> str | bool:
    """Certificate file named by env var `name`, or True (use the system CAs) when it is blank."""
    value = env(name)
    if not value:
        return True
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path            # relative paths are relative to the project folder
    if not path.is_file():
        raise FileNotFoundError(f"{name} points to a file that does not exist: {path}")
    return str(path)


class _OpenAICompatible:
    def __init__(self, endpoint: str, api_key: str, verify: str | bool = True, timeout: int = 60):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.verify = verify             # True or a CA file path; never False
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = requests.post(f"{self.endpoint}{path}",
                                     headers={"Authorization": f"Bearer {self.api_key}"},
                                     json=payload, timeout=self.timeout, verify=self.verify)
            except (requests.ConnectionError, requests.Timeout):
                if attempt == MAX_ATTEMPTS:
                    raise
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                time.sleep(2 ** attempt)                       # 2 s, then 4 s
                continue
            if resp.status_code >= 400:
                # Include the server's reason (e.g. "input too long"); it never echoes the key.
                raise requests.HTTPError(f"HTTP {resp.status_code} from {path}: {resp.text[:300]}", response=resp)
            return resp.json()
        raise RuntimeError("unreachable")


class ChatClient(_OpenAICompatible):
    def __init__(self, endpoint: str | None = None, api_key: str | None = None, model: str | None = None):
        endpoint = endpoint or env("COPILOT_CHAT_ENDPOINT")
        api_key = api_key or env("COPILOT_CHAT_API_KEY")
        self.model = model or env("COPILOT_MODEL_NAME")
        if not (endpoint and api_key and self.model):
            raise EndpointNotConfigured("Chat endpoint not configured. Fill COPILOT_CHAT_* in .env (see README).")
        super().__init__(endpoint, api_key, verify=ca_bundle("COPILOT_CHAT_CA_CERT"))
        # Running totals, read by the engine and the evaluation to report cost per query.
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 600) -> str:
        data = self._post("/chat/completions", {"model": self.model, "messages": messages,
                                                "temperature": temperature, "max_tokens": max_tokens})
        usage = data.get("usage") or {}
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        return data["choices"][0]["message"]["content"]

    def complete_json(self, messages: list[dict], retries: int = 1, max_tokens: int = 1024) -> dict:
        # 1024 is the endpoint's output limit; 600 cut long answers off mid-JSON in the first live run.
        for attempt in range(retries + 1):
            text = self.chat(messages, max_tokens=max_tokens)
            try:
                return parse_json_object(text)
            except ValueError as err:
                if attempt == retries:
                    raise
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"Invalid JSON ({err}). Reply with one JSON object only, no other text."},
                ]
        raise RuntimeError("unreachable")


class EmbedClient(_OpenAICompatible):
    def __init__(self, endpoint: str | None = None, api_key: str | None = None, model: str | None = None):
        endpoint = endpoint or env("COPILOT_EMBED_ENDPOINT")
        api_key = api_key or env("COPILOT_EMBED_API_KEY")
        self.model = model or env("COPILOT_EMBED_MODEL")
        if not (endpoint and api_key and self.model):
            raise EndpointNotConfigured("Embeddings endpoint not configured. Fill COPILOT_EMBED_* in .env.")
        super().__init__(endpoint, api_key, verify=ca_bundle("COPILOT_EMBED_CA_CERT"))

    def embed(self, texts: list[str], batch: int = 16) -> list[list[float]]:
        # The course embeddings API accepts at most 16 texts per request (HTTP 413 above that).
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            data = self._post("/embeddings", {"model": self.model, "input": texts[i:i + batch]})
            out.extend(item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"]))
        return out


def parse_json_object(text: str) -> dict:
    """Accept plain JSON or JSON in a code fence. Reject anything else."""
    # Reasoning models (e.g. Qwen3) may put a <think>...</think> block before the JSON.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text.strip()
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as err:
        raise ValueError(str(err)) from err
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj
