"""Build the corpus from openFDA (public API, no key needed; an optional key raises the rate limit).

Each record becomes one document: {"id", "text", "source", "url"}.
Raw JSON responses are saved under data/raw with SHA-256 hashes in data/raw/manifest.json."""
import hashlib
import json
import time
from pathlib import Path

import requests

from copilot.config import ROOT, env

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
BASE = "https://api.fda.gov"


def _get(path: str, params: dict, retries: int = 3) -> dict:
    key = env("OPENFDA_API_KEY")
    if key:
        params = {**params, "api_key": key}
    for attempt in range(retries):
        resp = requests.get(f"{BASE}{path}", params=params, timeout=30)
        if resp.status_code == 404:  # openFDA returns 404 when a search has no matches
            return {"results": []}
        if resp.status_code == 429 and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp.json()
    return {"results": []}


def save_raw(name: str, payload: dict) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[path.name] = {"sha256": digest, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _first(value) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value) if value else ""


def label_to_doc(rec: dict) -> dict | None:
    """Turn one openFDA drug label into a document. Keeps storage, excursion, and excipient text."""
    openfda = rec.get("openfda", {})
    names = openfda.get("generic_name", []) or openfda.get("brand_name", [])
    set_id = rec.get("set_id") or rec.get("id")
    if not set_id:
        return None
    sections = {
        "Storage and handling": rec.get("storage_and_handling") or rec.get("how_supplied"),
        "Inactive ingredients": rec.get("inactive_ingredient"),
        "Indications": rec.get("indications_and_usage"),
        "Warnings": rec.get("warnings_and_cautions") or rec.get("warnings"),
    }
    body = "\n".join(f"{title}: {_first(text)}" for title, text in sections.items() if text)
    if not body.strip():
        return None
    title = _first(names)[:120] or "Drug label"
    return {"id": f"label-{set_id}", "text": f"{title}\n{body}",
            "source": "openFDA drug label", "url": f"https://open.fda.gov/apis/drug/label/"}


def shortage_to_doc(rec: dict) -> dict | None:
    key = rec.get("generic_name") or rec.get("package_ndc") or rec.get("ndc")
    if not key:
        return None
    text = (f"Drug shortage: {rec.get('generic_name', '')} {rec.get('dosage_form', '')}. "
            f"Status: {rec.get('status', '')}. Company: {rec.get('company_name', '')}. "
            f"Reason: {rec.get('shortage_reason', '')}. Resupply: {rec.get('estimated_resupply', rec.get('update_type', ''))}.")
    return {"id": f"shortage-{hashlib.md5(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:12]}",
            "text": text, "source": "openFDA drug shortages", "url": "https://open.fda.gov/apis/drug/drugshortages/"}


def enforcement_to_doc(rec: dict) -> dict | None:
    if not rec.get("recall_number"):
        return None
    text = (f"Recall {rec['recall_number']}: class {rec.get('classification', '')}. "
            f"Product: {rec.get('product_description', '')}. Reason: {rec.get('reason_for_recall', '')}. "
            f"Firm: {rec.get('recalling_firm', '')}. Status: {rec.get('status', '')}.")
    return {"id": f"recall-{rec['recall_number']}", "text": text,
            "source": "openFDA drug enforcement", "url": "https://open.fda.gov/apis/drug/enforcement/"}


def fetch_labels(names: list[str], per_name: int = 25) -> list[dict]:
    docs = []
    for name in names:
        data = _get("/drug/label.json", {"search": f'openfda.generic_name:"{name}"', "limit": per_name})
        save_raw(f"label_{name.replace(' ', '_')}", data)
        docs += [d for d in (label_to_doc(r) for r in data.get("results", [])) if d]
        time.sleep(0.3)  # be polite to the public API
    return docs


def fetch_shortages(limit: int = 100) -> list[dict]:
    data = _get("/drug/shortages.json", {"limit": limit})
    save_raw("shortages", data)
    return [d for d in (shortage_to_doc(r) for r in data.get("results", [])) if d]


def fetch_enforcement(limit: int = 100) -> list[dict]:
    data = _get("/drug/enforcement.json", {"search": "classification:\"Class I\"", "limit": limit})
    save_raw("enforcement_class1", data)
    return [d for d in (enforcement_to_doc(r) for r in data.get("results", [])) if d]


def chunk_document(doc: dict, chunk_chars: int = 1200) -> list[dict]:
    """Split one document into passages of about chunk_chars characters.

    Labels run to 30,000 characters, far past the embedding model's 512-token limit, and a whole label
    is too coarse to cite. Each chunk keeps the document title (first line) so it still names the drug,
    and gets the id '<doc id>#c<n>'. Documents that are already chunks are returned unchanged."""
    if "parent" in doc:
        return [doc]
    title, _, body = doc["text"].partition("\n")
    if len(doc["text"]) <= chunk_chars:
        return [{**doc, "id": f"{doc['id']}#c0", "parent": doc["id"]}]
    chunks, start = [], 0
    while start < len(body):
        end = min(start + chunk_chars, len(body))
        if end < len(body):                       # cut at the last sentence or space, not mid-word
            cut = max(body.rfind(". ", start, end), body.rfind(" ", start, end))
            end = cut + 1 if cut > start else end
        chunks.append(body[start:end].strip())
        start = end
    return [{**doc, "id": f"{doc['id']}#c{n}", "parent": doc["id"], "text": f"{title}\n{text}"}
            for n, text in enumerate(chunks) if text]


def write_corpus(docs: list[dict]) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path


def load_corpus() -> list[dict] | None:
    path = PROCESSED / "corpus.jsonl"
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
