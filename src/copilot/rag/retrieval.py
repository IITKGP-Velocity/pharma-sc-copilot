"""Retrieval: BM25 (always on), dense cosine search (when the embeddings endpoint is set),
reciprocal rank fusion of both, and an optional reranker that uses the course chat model."""
import json
import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "by", "with", "from",
    "is", "are", "was", "be", "it", "its", "this", "that", "what", "which", "how", "do", "does",
    "can", "we", "my", "your", "i", "if", "as",
}


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens without common stopwords; used by BM25."""
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """Keyword search with BM25 scoring (k1 and b are the standard BM25 constants)."""
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        """Pre-compute tokens, average passage length and inverse document frequency."""
        self.docs = docs
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d["text"]) for d in docs]
        self.avgdl = (sum(len(t) for t in self.tokens) / len(self.tokens)) if self.tokens else 0.0
        df: Counter = Counter()
        for toks in self.tokens:
            df.update(set(toks))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Top k (passage id, score) pairs; passages sharing no word with the query are left out."""
        q_terms = set(tokenize(query))
        scores = []
        for doc, toks in zip(self.docs, self.tokens):
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for t in q_terms:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scores.append((doc["id"], s))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]


class DenseIndex:
    """Cosine similarity over embeddings from the university endpoint. Pure Python, no numpy needed."""

    def __init__(self, ids: list[str], vectors: list[list[float]]):
        """Store normalised vectors so a dot product equals cosine similarity."""
        self.ids = ids
        self.vectors = [self._normalise(v) for v in vectors]

    @staticmethod
    def _normalise(v: list[float]) -> list[float]:
        """Scale a vector to length 1."""
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def search(self, query_vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        """Top k (passage id, cosine similarity) pairs for a query vector."""
        q = self._normalise(query_vector)
        scores = [(doc_id, sum(a * b for a, b in zip(q, vec))) for doc_id, vec in zip(self.ids, self.vectors)]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Merge ranked lists: each list gives a passage 1/(k + rank); k=60 is the usual value."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


def hybrid_search(query: str, docs: list[dict], bm25: BM25Index, dense: DenseIndex | None,
                  query_vector: list[float] | None, k: int = 5) -> list[tuple[str, float]]:
    """BM25 plus dense (if available), fused with RRF. Returns (doc_id, fused score) pairs."""
    lists = [[d for d, _ in bm25.search(query, k=20)]]
    if dense is not None and query_vector is not None:
        lists.append([d for d, _ in dense.search(query_vector, k=20)])
    fused = reciprocal_rank_fusion(lists)
    return [(d, 1.0 / (rank + 1)) for rank, d in enumerate(fused[:k])]


RERANK_PROMPT = ("Rank the passages by how well they answer the question, most relevant first. "
                 'Reply with one JSON object and nothing else: {"ranking": [passage numbers]}.')


def rerank(query: str, hits: list[tuple[str, float]], texts: dict[str, str], chat, k: int = 5) -> list[tuple[str, float]]:
    """Reorder the fused hits with one call to the course chat model; keep the fused order if that fails.
    Turn it on with `rerank: true` in configs/settings.yaml."""
    if chat is None or len(hits) <= 1:
        return hits[:k]

    # 1. Number the passages and send the model a short preview of each.
    ids = [d for d, _ in hits]
    listing = [{"n": n, "text": texts[d][:300]} for n, d in enumerate(ids, start=1)]
    question = json.dumps({"question": query, "passages": listing})

    # 2. Ask for the passage numbers in order of relevance.
    try:
        reply = chat.complete_json([{"role": "system", "content": RERANK_PROMPT},
                                    {"role": "user", "content": question}])
    except Exception:  # endpoint or JSON trouble: never lose the retrieval result
        return hits[:k]

    # 3. Keep only valid passage numbers; passages the model left out follow in their fused order.
    numbers = reply.get("ranking", []) if isinstance(reply, dict) else []
    picked = [ids[n - 1] for n in numbers if isinstance(n, int) and 1 <= n <= len(ids)]
    order = list(dict.fromkeys(picked + ids))  # dict.fromkeys drops repeats and keeps order
    return [(d, 1.0 / (rank + 1)) for rank, d in enumerate(order[:k])]
