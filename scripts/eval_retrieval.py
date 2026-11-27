"""Measure retrieval on its own: recall@k and MRR for BM25 only, hybrid, and hybrid + rerank.

Uses eval/gold/retrieval_gold.json. A query counts as a hit when a passage from the right source with
the expected text is in the top k. Dense retrieval needs the embeddings endpoint and data/processed/embeddings.json;
the rerank row needs the chat endpoint (the reranker is the course chat model).

  python scripts/eval_retrieval.py
  python scripts/eval_retrieval.py <endpoint flags>
"""
import argparse
import json
import sys
from datetime import datetime

import _paths
from copilot.rag import load_corpus
from copilot.observability import retrieval_metrics
from copilot.rag import hybrid_search, rerank
from copilot.runtime import add_endpoint_args, apply_endpoint_args
from copilot.service import build_engine

ROOT = _paths.ROOT
GOLD = ROOT / "eval" / "gold" / "retrieval_gold.json"
K = 5


def main() -> int:
    """Rank passages for each gold query three ways and report recall@k and MRR for each."""
    ap = argparse.ArgumentParser(description="Retrieval-only evaluation.")
    add_endpoint_args(ap)
    apply_endpoint_args(ap.parse_args())
    if not load_corpus():
        sys.exit("No corpus yet. Run: python scripts/build_corpus.py --fetch")

    gold = json.loads(GOLD.read_text(encoding="utf-8"))["items"]
    engine = build_engine()
    passages = {d["id"]: d for d in engine.corpus}

    results = {}
    for name in ("bm25_only", "hybrid", "hybrid_rerank"):
        use_dense = name != "bm25_only"
        if use_dense and engine.dense is None:
            print(f"{name}: skipped (embeddings endpoint or embeddings file missing).")
            continue
        if name == "hybrid_rerank" and engine.chat is None:
            print(f"{name}: skipped (needs the chat endpoint).")
            continue
        ranked = {}
        for g in gold:
            qvec = engine.embed.embed([g["query"]])[0] if use_dense else None
            depth = 20 if name == "hybrid_rerank" else K
            hits = hybrid_search(g["query"], engine.corpus, engine.bm25, engine.dense if use_dense else None, qvec, k=depth)
            if name == "hybrid_rerank":
                hits = rerank(g["query"], hits, engine.texts, engine.chat, k=K)
            ranked[g["id"]] = [d for d, _ in hits]
        results[name] = retrieval_metrics(ranked, gold, passages, K)
        print(f"{name:14s} recall@{K}={results[name][f'recall_at_{K}']}  MRR={results[name]['mrr']}")

    out_dir = ROOT / "eval" / "results" / datetime.now().strftime("retrieval_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Written to {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
