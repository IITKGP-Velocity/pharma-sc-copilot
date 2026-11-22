#!/usr/bin/env python3
"""
retriever.py - Hybrid retriever: FAISS dense search + BM25 keyword search, fused with
Reciprocal Rank Fusion (RRF), plus an exact section-reference boost.

Loads (from --index-dir, produced by embed_and_index.py):
    faiss.index, bm25.pkl, indexed_chunks.jsonl, index_manifest.json

Use as a library:
    from retriever import HybridRetriever, format_context, confidence_signal
    r = HybridRetriever("index")
    hits = r.search("quarantine storage conditions 211.142", top_k=5, use_case="A")
    context = format_context(hits)            # ready to paste into an LLM prompt
    many = r.search_batch(list_of_questions)  # fast path for evaluation runs

Or from the command line:
    python retriever.py --index-dir index --query "warehousing temperature 211.142" --top-k 5

How one search works
    1. Embed the query (BGE query instruction added; vectors cached on disk).
    2. Dense search : cosine similarity against every chunk vector      -> ranked list A
    3. Sparse search: BM25 keyword score against every chunk            -> ranked list B
    4. Reference boost: if the query names e.g. "4.5", "211.142" or "Annex 9", chunks whose
       section_number matches (exact first, then sub-sections like 4.5.1) form a
       third ranked list                                                -> ranked list C
    5. RRF: score(chunk) = sum over lists of weight / (rrf_k + rank). Ranks, not raw
       scores, are fused, so cosine and BM25 scales never need normalizing.
    6. Post-process: drop duplicates, cap chunks per section, apply a token budget.

NOTE: RRF scores only say how chunks rank against EACH OTHER, not whether the corpus can
answer the question. Use confidence_signal(hits) (cosine / BM25 / reference match) for that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from pathlib import Path

import faiss
import numpy as np

try:
    from .local_secrets import get_embeddings_key
except ImportError:
    from local_secrets import get_embeddings_key

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index"
DEFAULT_EMBED_CERT = REPOSITORY_ROOT / "secrets" / "vm02-ca.crt"

# Reuse the EXACT embedding client and BM25 tokenizer used at indexing time.
# If the tokenizer differed, keyword matching would silently break.
from embed_and_index import (
    BATCH_SIZE,
    EXPECTED_DIM,
    MODEL_NAME,
    TOKENIZER_VERSION,
    EmbeddingClient,
    tokenize_for_bm25,
)

# BGE models are trained to see this prefix on short SEARCH QUERIES (never on passages).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Up to this many chunks we score every chunk on every query (exact, simple, fast for a
# flat index). That also makes metadata filters exact. Above it we only fetch a pool.
FULL_SCAN_LIMIT = 20_000

# Errors that mean "the embeddings call did not give us a usable vector".
EMBED_ERRORS = (RuntimeError, ValueError, KeyError, TypeError)

# ---- query reference patterns -------------------------------------------------------
# "211.142", "4.5.1", "211.82a"  (dotted section numbers, optional trailing letter;
# also found inside "21 CFR 211.142(a)", where the "(a)" is not part of the number)
_REF_DOTTED = re.compile(r"(?<![\d.])(\d+(?:\.\d+)+[a-z]?)", re.I)
# "Annex 9", "Chapter IV", "Subpart B", "Appendix A"  (digits, Roman numerals or a letter)
_REF_NAMED = re.compile(r"\b(annex|appendix|chapter|subpart)\s+(\d+|[ivxlcdm]+|[a-z])\b", re.I)
# A chunk section_number that is a dotted number, e.g. "4.5.1" or "211.82a"
_NUMERIC_SECTION = re.compile(r"\d+(?:\.\d+)*[a-z]?")


def extract_references(query: str) -> list[str]:
    """Return lowercase section references mentioned in the query, in order, no duplicates.
    A trailing letter ('211.82a') also yields its base ('211.82') so either form matches.
    These are compared with each chunk's `section_number` (also lowercased)."""
    refs: list[str] = []
    for m in _REF_DOTTED.finditer(query):
        ref = m.group(1).lower()
        refs.append(ref)
        if ref[-1].isalpha():
            refs.append(ref[:-1])
    refs += [f"{m.group(1).lower()} {m.group(2).lower()}" for m in _REF_NAMED.finditer(query)]
    return list(dict.fromkeys(refs))


def make_citation(chunk: dict) -> str:
    """Short human-readable source label, e.g. 'FDA_211 | 211.142 | p.14' ."""
    ps, pe = chunk.get("page_start"), chunk.get("page_end")
    if ps is None:
        pages = ""
    elif pe is None or pe == ps:
        pages = f"p.{ps}"
    else:
        pages = f"pp.{ps}-{pe}"
    parts = [chunk.get("doc_id") or "?", chunk.get("section_number") or chunk.get("section") or "", pages]
    return " | ".join(p for p in parts if p)


def format_context(hits: list[dict], max_chars_per_chunk: int | None = None) -> str:
    """Numbered, citation-labelled context block for an LLM prompt."""
    blocks = []
    for i, h in enumerate(hits, 1):
        text = h["text"] if not max_chars_per_chunk else h["text"][:max_chars_per_chunk]
        blocks.append(f"[{i}] ({h['citation']})\n{text}")
    return "\n\n".join(blocks)


def confidence_signal(hits: list[dict]) -> dict:
    """Summarise how strongly the results match the query. Use it to decide whether to
    answer or abstain. Thresholds are corpus-specific: calibrate them on out-of-scope
    questions (what cosine / BM25 do unanswerable queries get?) before hard-coding any."""
    if not hits:
        return {"n_hits": 0, "top_dense_cosine": None, "top_bm25_score": None,
                "has_ref_match": False, "mode": None}
    cos = [h["retrieval"]["dense_cosine"] for h in hits if h["retrieval"]["dense_cosine"] is not None]
    bm = [h["retrieval"]["bm25_score"] for h in hits if h["retrieval"]["bm25_score"] is not None]
    return {
        "n_hits": len(hits),
        "top_dense_cosine": max(cos) if cos else None,
        "top_bm25_score": max(bm) if bm else None,
        "has_ref_match": any(h["retrieval"]["ref_rank"] for h in hits),
        "mode": hits[0]["retrieval"]["mode"],
    }


class HybridRetriever:
    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        embed_url: str | None = None,
        embed_key: str | None = None,
        embed_cert: str | Path | None = None,
        rrf_k: int = 60,                       # standard RRF constant; larger = flatter fusion
        use_query_instruction: bool = True,    # BGE query prefix; A/B test it in your evaluation
        allow_sparse_fallback: bool = True,    # if the embeddings API fails, keep going with BM25
        cache_queries: bool = True,            # persist query embeddings (saves API calls in evals)
    ):
        self.index_dir = Path(index_dir)
        self.rrf_k = rrf_k
        self.query_instruction = BGE_QUERY_INSTRUCTION if use_query_instruction else ""
        self.allow_sparse_fallback = allow_sparse_fallback

        # ---- 1. Validate the manifest written by embed_and_index.py -------------------
        manifest_path = self.index_dir / "index_manifest.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if self.manifest.get("embedding_model") != MODEL_NAME:
                raise RuntimeError(
                    f"Model mismatch: index built with {self.manifest.get('embedding_model')}, "
                    f"code expects {MODEL_NAME}. Re-run embed_and_index.py.")
            if self.manifest.get("bm25_tokenizer_version") != TOKENIZER_VERSION:
                raise RuntimeError(
                    f"BM25 tokenizer mismatch: index={self.manifest.get('bm25_tokenizer_version')}, "
                    f"code={TOKENIZER_VERSION}. Re-run embed_and_index.py.")
            if self.manifest.get("dim") not in (None, EXPECTED_DIM):
                raise RuntimeError(f"Index dimension {self.manifest.get('dim')} != {EXPECTED_DIM}")
        else:
            self.manifest = {}
            print("WARNING: index_manifest.json missing - model/tokenizer checks skipped.", file=sys.stderr)

        # ---- 2. Load the three aligned artifacts (row i == FAISS vector i == BM25 doc i)
        self.faiss_index = faiss.read_index(str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "bm25.pkl", "rb") as f:     # only load files YOU built
            self.bm25 = pickle.load(f)
        self.chunks: list[dict] = []
        with open(self.index_dir / "indexed_chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))
        self.n = len(self.chunks)

        if not (self.faiss_index.ntotal == self.n == self.bm25.corpus_size):
            raise RuntimeError(
                f"Index alignment error: faiss={self.faiss_index.ntotal}, "
                f"chunks={self.n}, bm25={self.bm25.corpus_size}. Rebuild the index.")
        if self.manifest.get("num_chunks") not in (None, self.n):
            raise RuntimeError(f"Manifest says {self.manifest['num_chunks']} chunks, found {self.n}.")

        # ---- 3. Precompute lookup structures so filtering is fast and vectorised ------
        self._doc_ids = np.array([c.get("doc_id") or "" for c in self.chunks])
        self._chunk_types = np.array([c.get("chunk_type") or "text" for c in self.chunks])
        self._use_case_sets = [self._parse_use_cases(c.get("use_case")) for c in self.chunks]
        self._build_section_indexes()

        # ---- 4. Embedding client for QUERIES (key from env, never hard-coded) ---------
        if embed_key:
            key = embed_key
        else:
            try:
                key = get_embeddings_key()
            except RuntimeError as e:
                raise ValueError(str(e)) from e
        url = embed_url or os.environ.get("EMBED_URL", "https://<COURSE_HOST>:1035/v1/embeddings")
        cert = Path(embed_cert or os.environ.get("EMBED_CERT", str(DEFAULT_EMBED_CERT)))
        if not cert.exists():
            raise FileNotFoundError(f"Certificate not found: {cert}")
        self.client = EmbeddingClient(url=url, key=key, cert=cert, min_interval=0.05)

        # ---- 5. Query-embedding cache (memory + append-only file) ---------------------
        self._qcache: dict[str, np.ndarray] = {}
        self._qcache_path = self.index_dir / "query_cache.jsonl" if cache_queries else None
        if self._qcache_path and self._qcache_path.exists():
            with open(self._qcache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self._qcache[r["h"]] = np.asarray(r["v"], dtype="float32").reshape(1, -1)
                    except (ValueError, KeyError):
                        continue                       # ignore a torn last line

    # ------------------------------------------------------------------ helpers ----
    def _build_section_indexes(self):
        """_sec_index   : section_number -> chunk rows            ('4.5'   -> [rows of 4.5])
        _sec_children: ancestor number  -> rows of ALL its sub-sections
                       ('4.5' -> rows of 4.5.1, 4.5.2, ...), so a query for section 4.5
                       also reaches its sub-sections."""
        self._sec_index: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            sn = (c.get("section_number") or "").strip().lower()
            if sn:
                self._sec_index.setdefault(sn, []).append(i)
        self._sec_children: dict[str, list[int]] = {}
        for sn, rows in self._sec_index.items():
            if not _NUMERIC_SECTION.fullmatch(sn):
                continue                               # named sections ('annex 9') have no numeric parents
            parts = sn.split(".")
            for j in range(1, len(parts)):
                self._sec_children.setdefault(".".join(parts[:j]), []).extend(rows)

    @staticmethod
    def _parse_use_cases(value: str | None) -> set[str]:
        """Manifest stores use cases like 'A;B;E' - turn into {'A','B','E'}."""
        return {u.strip().upper() for u in re.split(r"[;,]", value or "") if u.strip()}

    def _build_mask(self, use_case, doc_ids, chunk_type) -> np.ndarray | None:
        """Boolean array over all chunks: True = allowed by the filters. None = no filters.
        Filters are applied BEFORE ranking, so a filter can never leave you with fewer
        results just because the top-N happened to belong to other documents."""
        if not (use_case or doc_ids or chunk_type):
            return None
        mask = np.ones(self.n, dtype=bool)
        if doc_ids:
            mask &= np.isin(self._doc_ids, list(doc_ids))
        if chunk_type:
            mask &= self._chunk_types == chunk_type
        if use_case:
            uc = use_case.strip().upper()
            mask &= np.fromiter((uc in s for s in self._use_case_sets), dtype=bool, count=self.n)
        return mask

    def _query_key(self, query: str) -> tuple[str, str]:
        """(cache key, text actually sent to the API) for a query."""
        text = self.query_instruction + query
        return hashlib.sha1(f"{MODEL_NAME}\n{text}".encode("utf-8")).hexdigest(), text

    def _cache_put(self, pairs: list[tuple[str, np.ndarray]]):
        """Remember query vectors in memory and append them to the cache file.
        (The `with` block closes - and therefore flushes - the file on exit.)"""
        for key, vec in pairs:
            self._qcache[key] = vec
        if self._qcache_path:
            with open(self._qcache_path, "a", encoding="utf-8") as f:
                for key, vec in pairs:
                    f.write(json.dumps({"h": key, "v": vec[0].tolist()}) + "\n")

    def _embed_query(self, query: str) -> np.ndarray:
        """Return a normalized (1, 384) float32 query vector (cached)."""
        key, text = self._query_key(query)
        if key in self._qcache:
            return self._qcache[key].copy()
        vec = self.client.embed([text]).astype("float32")
        if vec.shape != (1, EXPECTED_DIM):                 # explicit check (assert can be disabled)
            raise RuntimeError(f"Unexpected query embedding shape {vec.shape}")
        faiss.normalize_L2(vec)                            # cosine == inner product
        self._cache_put([(key, vec)])
        return vec.copy()

    # --------------------------------------------------------------------- search ---
    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 30,            # how many results each retriever contributes
        use_case: str | None = None,         # e.g. "A"
        doc_ids: list[str] | None = None,
        chunk_type: str | None = None,       # "text" | "table"
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        ref_weight: float = 1.5,             # boost for section-number matches; 0 disables
        min_dense_cosine: float | None = None,  # drop dense candidates below this similarity
        max_per_section: int | None = 3,     # diversity: at most N chunks from one section
        token_budget: int | None = None,     # stop adding chunks once this many tokens are used
        use_dense: bool = True,              # False = BM25 only (no embeddings API call)
    ) -> list[dict]:
        """Return up to top_k chunk dicts, best first. Each has:
            citation  - short source label for answers
            retrieval - {mode, rrf_score, dense_rank, dense_cosine, bm25_rank, bm25_score,
                         ref_rank, matched_ref, ref_match_type}  (use these to debug)
        """
        if not query or not query.strip():                 # nothing to search for; skip the API call
            return []

        n = self.n
        pool = min(n, max(candidate_pool, top_k * 5))
        mask = self._build_mask(use_case, doc_ids, chunk_type)
        if mask is not None and not mask.any():
            return []

        # ---- A. Dense search ---------------------------------------------------------
        mode = "hybrid"
        dense_scores = None                  # array of n cosines, -inf = not scored
        if use_dense:
            try:
                q = self._embed_query(query)
                # Flat index: scanning all n costs about the same as scanning fewer, and
                # scoring everything makes filters / reference boosts exact.
                k = n if (n <= FULL_SCAN_LIMIT or mask is not None) else pool
                D, I = self.faiss_index.search(q, k)
                dense_scores = np.full(n, -np.inf, dtype=np.float32)
                ok = I[0] >= 0
                dense_scores[I[0][ok]] = D[0][ok]
            except EMBED_ERRORS as e:        # API down, HTML gateway page, bad JSON, wrong shape...
                if not self.allow_sparse_fallback:
                    raise
                print(f"WARNING: query embedding failed ({type(e).__name__}: {e}); using BM25 only.",
                      file=sys.stderr)
        if dense_scores is None:
            mode = "bm25_only"

        dense_rank: dict[int, int] = {}
        if dense_scores is not None:
            ds = dense_scores.copy()
            if mask is not None:
                ds[~mask] = -np.inf
            if min_dense_cosine is not None:
                ds[ds < min_dense_cosine] = -np.inf
            for r, i in enumerate(np.argsort(-ds)[:pool], 1):
                if np.isfinite(ds[i]):
                    dense_rank[int(i)] = r

        # ---- B. Sparse (BM25) search -------------------------------------------------
        q_tokens = tokenize_for_bm25(query)
        bm = (np.asarray(self.bm25.get_scores(q_tokens), dtype=np.float32).copy()
              if q_tokens else np.zeros(n, dtype=np.float32))
        if mask is not None:
            bm[~mask] = 0.0
        sparse_rank: dict[int, int] = {}
        for r, i in enumerate(np.argsort(-bm)[:pool], 1):
            if bm[i] > 0:                    # zero score = no keyword overlap, not a match
                sparse_rank[int(i)] = r

        # ---- C. Section-reference boost (exact match first, then sub-sections) -------
        refs = extract_references(query)
        ref_rank: dict[int, int] = {}
        ref_type: dict[int, str] = {}
        if refs and ref_weight > 0:
            allowed = (lambda i: True) if mask is None else (lambda i: bool(mask[i]))
            exact: set[int] = set()
            child: set[int] = set()
            for ref in refs:
                exact.update(i for i in self._sec_index.get(ref, []) if allowed(i))
                child.update(i for i in self._sec_children.get(ref, []) if allowed(i))
            child -= exact
            if exact or child:
                order_by = dense_scores if dense_scores is not None else bm
                ordered = (sorted(exact, key=lambda i: -order_by[i])
                           + sorted(child, key=lambda i: -order_by[i]))
                for r, i in enumerate(ordered[:pool], 1):
                    ref_rank[i] = r
                    ref_type[i] = "exact" if i in exact else "child"

        # ---- D. Reciprocal Rank Fusion -----------------------------------------------
        fused: dict[int, float] = {}
        for ranks, weight in ((dense_rank, dense_weight), (sparse_rank, sparse_weight), (ref_rank, ref_weight)):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + weight / (self.rrf_k + r)

        # ---- E. Post-process: dedupe, per-section cap, token budget ------------------
        results: list[dict] = []
        seen_hash: set[str] = set()
        per_section: dict[tuple, int] = {}
        used_tokens = 0
        for i, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            c = self.chunks[i]
            h = c.get("content_hash")
            if h and h in seen_hash:
                continue
            sec_key = (c.get("doc_id"), c.get("section_path"))
            if max_per_section and per_section.get(sec_key, 0) >= max_per_section:
                continue
            tok = c.get("token_count") or 0
            if token_budget and results and used_tokens + tok > token_budget:
                continue                     # skip this one; a smaller chunk may still fit
            if h:
                seen_hash.add(h)
            per_section[sec_key] = per_section.get(sec_key, 0) + 1
            used_tokens += tok

            item = dict(c)
            item["citation"] = make_citation(c)
            item["retrieval"] = {
                "mode": mode,
                "rrf_score": round(score, 6),
                "dense_rank": dense_rank.get(i),
                "dense_cosine": round(float(dense_scores[i]), 4)
                if dense_scores is not None and np.isfinite(dense_scores[i]) else None,
                "bm25_rank": sparse_rank.get(i),
                "bm25_score": round(float(bm[i]), 4) if i in sparse_rank else None,
                "ref_rank": ref_rank.get(i),
                "matched_ref": (c.get("section_number") or "") if i in ref_rank else None,
                "ref_match_type": ref_type.get(i),
            }
            results.append(item)
            if len(results) >= top_k:
                break
        return results

    def search_batch(self, queries: list[str], **search_kwargs) -> list[list[dict]]:
        """Search many queries (evaluation runs). Uncached queries are embedded 16 at a time
        instead of one HTTP call each, then every query is searched locally from the cache.
        Takes the same keyword arguments as search(). Returns one hit list per query."""
        use_dense = search_kwargs.get("use_dense", True)
        pending: dict[str, str] = {}
        if use_dense:
            for q in queries:
                if q and q.strip():
                    key, text = self._query_key(q)
                    if key not in self._qcache:
                        pending[key] = text

        warmup_failed = False
        items = list(pending.items())
        try:
            for b in range(0, len(items), BATCH_SIZE):
                batch = items[b:b + BATCH_SIZE]
                vecs = self.client.embed([t for _, t in batch]).astype("float32")
                if vecs.shape != (len(batch), EXPECTED_DIM):
                    raise RuntimeError(f"Unexpected batch embedding shape {vecs.shape}")
                faiss.normalize_L2(vecs)
                self._cache_put([(k, v.reshape(1, -1).copy()) for (k, _), v in zip(batch, vecs)])
        except EMBED_ERRORS as e:
            if not self.allow_sparse_fallback:
                raise
            warmup_failed = True
            print(f"WARNING: batch query embedding failed ({type(e).__name__}: {e}); "
                  "uncached queries will use BM25 only.", file=sys.stderr)

        out = []
        for q in queries:
            kw = dict(search_kwargs)
            if warmup_failed and q and q.strip() and self._query_key(q)[0] not in self._qcache:
                kw["use_dense"] = False                       # don't retry a dead endpoint per query
            out.append(self.search(q, **kw))
        return out


# ------------------------------------------------------------------------------ CLI --
def main() -> int:
    ap = argparse.ArgumentParser(description="Test the hybrid retriever from the command line.")
    ap.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--use-case", default=None, help="filter by use case letter: A, B, C, D or E")
    ap.add_argument("--doc-id", action="append", default=None, help="filter to a doc_id (repeatable)")
    ap.add_argument("--chunk-type", default=None, choices=["text", "table"])
    ap.add_argument("--max-per-section", type=int, default=3)
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--min-cosine", type=float, default=None)
    ap.add_argument("--ref-weight", type=float, default=1.5)
    ap.add_argument("--bm25-only", action="store_true", help="skip the embeddings API")
    ap.add_argument("--no-instruction", action="store_true", help="embed the query without the BGE prefix")
    ap.add_argument("--json", action="store_true", help="print full JSON instead of a summary")
    args = ap.parse_args()

    retriever = HybridRetriever(index_dir=args.index_dir, use_query_instruction=not args.no_instruction)
    hits = retriever.search(
        query=args.query, top_k=args.top_k, use_case=args.use_case, doc_ids=args.doc_id,
        chunk_type=args.chunk_type, max_per_section=args.max_per_section,
        token_budget=args.token_budget, min_dense_cosine=args.min_cosine, ref_weight=args.ref_weight,
        use_dense=not args.bm25_only,
    )

    if args.json:
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return 0

    print(f"\nQuery: {args.query}")
    if args.use_case:
        print(f"Filter use_case={args.use_case}")
    print("-" * 80)
    if not hits:
        print("No results (empty query, filters too strict, or nothing matched).")
    for rank, h in enumerate(hits, start=1):
        r = h["retrieval"]
        print(
            f"#{rank} [{h['chunk_id']}] {h['citation']} | {h.get('section_path') or h.get('section')}\n"
            f"   RRF={r['rrf_score']:.5f}  dense=#{r['dense_rank']}/{r['dense_cosine']}  "
            f"bm25=#{r['bm25_rank']}/{r['bm25_score']}  ref=#{r['ref_rank']}({r['ref_match_type']})  mode={r['mode']}"
        )
        print(f"   {h['text'][:240].replace(chr(10), ' ')}...\n")
    print("confidence:", confidence_signal(hits))
    return 0


if __name__ == "__main__":
    sys.exit(main())