#!/usr/bin/env python3
"""
embed_and_index.py - build the dense (FAISS) and sparse (BM25) indexes for RAG.

Input : chunks.jsonl produced by parse_and_chunk.py (needs chunk_id, text, embed_text)
Output: faiss.index, bm25.pkl, indexed_chunks.jsonl (row i == vector i == BM25 doc i),
        index_manifest.json (model, dim, counts, hash of the chunks file),
        embed_cache.jsonl (resume cache - safe to keep, re-runs only embed new/changed chunks)

Environment
  EMBEDDINGS_KEY   API key for the embeddings service (required)
  EMBED_URL        default https://<COURSE_HOST>:1035/v1/embeddings
  EMBED_CERT       default <project-root>/secrets/vm02-ca.crt

Run
  python embed_and_index.py --chunks index/chunks.jsonl --out-dir index/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import requests
from rank_bm25 import BM25Okapi

try:
    from .local_secrets import get_embeddings_key
except ImportError:
    from local_secrets import get_embeddings_key

MODEL_NAME = "bge-small-en-v1.5"
EXPECTED_DIM = 384
BATCH_SIZE = 16                      # coordinator hard limit per request
RETRYABLE = {429, 500, 502, 503, 504}
TOKENIZER_VERSION = 2                # bump when tokenize_for_bm25 changes (forces a rebuild check)


# --------------------------------------------------------------------------- #
# Embedding client
# --------------------------------------------------------------------------- #
class EmbeddingClient:
    def __init__(self, url: str, key: str, cert: Path, batch_size: int = BATCH_SIZE,
                 max_retries: int = 6, min_interval: float = 0.2):
        self.url, self.batch_size = url, batch_size
        self.max_retries, self.min_interval = max_retries, min_interval
        self._last_call = 0.0
        self.s = requests.Session()                      # reuse the TLS connection
        self.s.headers.update({"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"})
        self.s.verify = str(cert)                        # keep HTTPS verification ON

    def _pace(self):
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        if retry_after and retry_after.isdigit():
            return min(60.0, float(retry_after))
        return min(60.0, 2 ** attempt) + random.uniform(0, 0.5)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not 0 < len(texts) <= self.batch_size:
            raise ValueError(f"batch must have 1..{self.batch_size} texts, got {len(texts)}")
        payload = {"model": MODEL_NAME, "input": texts}
        last_err = "unknown"

        for attempt in range(self.max_retries):
            self._pace()
            try:
                resp = self.s.post(self.url, json=payload, timeout=(10, 60))
            except requests.exceptions.SSLError as e:
                # wrong/missing CA file will never succeed on retry
                raise RuntimeError(f"TLS verification failed - check EMBED_CERT path/file: {e}") from e
            except requests.exceptions.RequestException as e:
                last_err = f"{type(e).__name__}"
                wait = self._backoff(attempt, None)
                print(f"  network error ({last_err}); retry in {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue

            if resp.status_code == 200:
                try:
                    return self._parse(resp, len(texts))
                except RuntimeError as e:                # e.g. HTML gateway page with HTTP 200
                    last_err = str(e)
                    wait = self._backoff(attempt, None)
                    print(f"  bad response ({last_err}); retry in {wait:.1f}s", flush=True)
                    time.sleep(wait)
                    continue
            if resp.status_code in RETRYABLE:
                last_err = f"HTTP {resp.status_code}"
                wait = self._backoff(attempt, resp.headers.get("Retry-After"))
                print(f"  {last_err}; retry in {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue
            # 400 / 401 / 403 / 404 ... : retrying cannot help
            raise RuntimeError(f"Embeddings API HTTP {resp.status_code}: {resp.text[:200]}")

        raise RuntimeError(f"Embeddings API failed after {self.max_retries} attempts ({last_err})")

    @staticmethod
    def _parse(resp: requests.Response, n: int) -> np.ndarray:
        try:
            data = resp.json()["data"]
            data.sort(key=lambda d: d["index"])           # order must match the inputs
            if len(data) != n:
                raise RuntimeError(f"API returned {len(data)} vectors for {n} inputs")
            vecs = np.asarray([d["embedding"] for d in data], dtype="float32")
        except (ValueError, KeyError, TypeError) as e:     # not JSON, missing fields, wrong types
            raise RuntimeError(f"Malformed embeddings response: {type(e).__name__}: {e}") from e
        if vecs.ndim != 2 or not np.isfinite(vecs).all():
            raise RuntimeError("API returned malformed or non-finite embeddings")
        return vecs


# --------------------------------------------------------------------------- #
# BM25 tokenizer  (import and reuse this EXACT function at query time)
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_NUMERIC_DOTTED = re.compile(r"\d+(?:\.\d+)+")
_STOP = frozenset("""a an and are as at be by for from has have in is it its of on or that the
this to was were will with shall should may must not no any all each such than then these those
which who whom into per""".split())


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase tokens. Dotted numbers keep their full form AND their parents, so
    '211.142' is found by queries for '211.142' and for 'part 211'. Hyphenated words
    also emit their parts ('non-conforming' -> non, conforming)."""
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        out.append(tok)
        if _NUMERIC_DOTTED.fullmatch(tok):
            parts = tok.split(".")
            out.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        elif "-" in tok:
            out.extend(p for p in tok.split("-") if len(p) > 1 and p not in _STOP)
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{n}: invalid JSON ({e})")
            for k in ("chunk_id", "text"):
                if not c.get(k):
                    raise SystemExit(f"{path}:{n}: missing '{k}'")
            c.setdefault("embed_text", c["text"])
            chunks.append(c)
    if not chunks:
        raise SystemExit(f"{path} has no chunks")
    dups = [i for i, k in Counter(c["chunk_id"] for c in chunks).items() if k > 1][:5]
    if dups:
        raise SystemExit(f"duplicate chunk_id values, e.g. {dups}")
    return chunks


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_key(text: str) -> str:
    return hashlib.sha1(f"{MODEL_NAME}\n{text}".encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    cache[r["h"]] = np.asarray(r["v"], dtype="float32")
                except (ValueError, KeyError):
                    continue                              # tolerate a half-written last line
    return cache


def atomic_write(path: Path, write_fn, mode: str = "w"):
    tmp = path.with_suffix(path.suffix + ".tmp")
    kwargs = {"encoding": "utf-8"} if "b" not in mode else {}
    with open(tmp, mode, **kwargs) as f:
        write_fn(f)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parent.parent
    index_dir = project_root / "index"
    ap.add_argument("--chunks", default=str(index_dir / "chunks.jsonl"))
    ap.add_argument("--out-dir", default=str(index_dir))
    ap.add_argument("--min-interval", type=float, default=0.2,
                    help="minimum seconds between API calls (raise it if you see HTTP 429)")
    ap.add_argument("--force", action="store_true", help="ignore the embedding cache")
    args = ap.parse_args()

    try:
        key = get_embeddings_key()
    except RuntimeError as e:
        raise SystemExit(str(e)) from e
    url = os.environ.get("EMBED_URL", "https://<COURSE_HOST>:1035/v1/embeddings")
    cert = Path(os.environ.get("EMBED_CERT", str(project_root / "secrets" / "vm02-ca.crt")))
    if not cert.exists():
        raise SystemExit(f"Certificate not found: {cert} (set EMBED_CERT)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chunks_path = Path(args.chunks)
    chunks = load_chunks(chunks_path)
    n = len(chunks)
    print(f"Loaded {n} chunks from {chunks_path}")

    long_chunks = [c["chunk_id"] for c in chunks if c.get("token_count", 0) + 2 > 512]
    if long_chunks:
        print(f"WARNING: {len(long_chunks)} chunks exceed 512 tokens and may be truncated, e.g. {long_chunks[:3]}")

    # ---- 1. dense embeddings (resumable) ---------------------------------- #
    texts = [c["embed_text"] for c in chunks]
    keys = [cache_key(t) for t in texts]
    cache_path = out / "embed_cache.jsonl"
    cache = {} if args.force else load_cache(cache_path)

    todo, seen = [], set()
    for i, k in enumerate(keys):
        if k not in cache and k not in seen:
            seen.add(k)
            todo.append(i)
    print(f"{n - len(todo)} chunks already embedded (cache), {len(todo)} to embed")

    if todo:
        client = EmbeddingClient(url, key, cert, min_interval=args.min_interval)
        t0 = time.time()
        with open(cache_path, "a", encoding="utf-8") as cf:
            for b in range(0, len(todo), BATCH_SIZE):
                idxs = todo[b:b + BATCH_SIZE]
                vecs = client.embed([texts[i] for i in idxs])
                if vecs.shape[1] != EXPECTED_DIM:           # fail on batch 1, not after hours
                    raise SystemExit(f"API returned {vecs.shape[1]}-d vectors, expected {EXPECTED_DIM}")
                for i, v in zip(idxs, vecs):
                    cache[keys[i]] = v
                    cf.write(json.dumps({"h": keys[i], "v": v.tolist()}) + "\n")
                cf.flush()                                # progress survives a crash
                done = b + len(idxs)
                if (b // BATCH_SIZE) % 10 == 0 or done == len(todo):
                    rate = done / max(1e-9, time.time() - t0)
                    print(f"Embedded {done}/{len(todo)}  (~{(len(todo) - done) / rate:.0f}s left)", flush=True)

    vectors = np.stack([cache[k] for k in keys]).astype("float32")
    if vectors.shape != (n, EXPECTED_DIM):
        raise SystemExit(f"Unexpected embedding shape {vectors.shape}, expected ({n}, {EXPECTED_DIM})")
    faiss.normalize_L2(vectors)                           # inner product == cosine
    index = faiss.IndexFlatIP(EXPECTED_DIM)
    index.add(vectors)

    # self-check: every vector should find itself first (duplicates may tie)
    _, top1 = index.search(vectors[: min(50, n)], 1)
    misses = int(sum(1 for r, i in enumerate(top1[:, 0]) if i != r))
    if misses > 2:
        raise SystemExit(f"Index self-check failed: {misses}/{len(top1)} vectors do not retrieve themselves")

    tmp_idx = str(out / "faiss.index.tmp")
    faiss.write_index(index, tmp_idx)
    os.replace(tmp_idx, out / "faiss.index")
    print(f"Saved FAISS index ({index.ntotal} x {EXPECTED_DIM}) to {out / 'faiss.index'}")

    # ---- 2. BM25 over embed_text (section path + numbers are searchable) --- #
    bm25 = BM25Okapi([tokenize_for_bm25(t) for t in texts])
    atomic_write(out / "bm25.pkl", lambda f: pickle.dump(bm25, f), "wb")
    print(f"Saved BM25 index to {out / 'bm25.pkl'}")

    # ---- 3. aligned metadata + manifest ----------------------------------- #
    def dump_chunks(f):
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    atomic_write(out / "indexed_chunks.jsonl", dump_chunks)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": MODEL_NAME,
        "dim": EXPECTED_DIM,
        "num_chunks": n,
        "chunks_file": str(chunks_path.relative_to(out))
        if chunks_path.is_relative_to(out)
        else str(chunks_path),
        "chunks_sha256": sha256_file(chunks_path),
        "bm25_tokenizer_version": TOKENIZER_VERSION,
        "faiss_index_type": "IndexFlatIP (L2-normalized)",
    }
    atomic_write(out / "index_manifest.json", lambda f: json.dump(manifest, f, indent=2))
    print("Done. Row i in faiss.index == BM25 doc i == line i of indexed_chunks.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())