"""Measure retrieval on its own (no chat model): recall@k and MRR, BM25 only vs hybrid.

Uses eval/gold/retrieval_gold.json. A query counts as a hit when a passage from the right source with
the expected text is in the top k. Dense retrieval needs the embeddings endpoint and data/processed/embeddings.json.

  python scripts/eval_retrieval.py
  python scripts/eval_retrieval.py --embed-endpoint https://HOST:1035/v1 --embed-model bge-small-en-v1.5 \
                                   --embed-ca-cert PATH --embed-api-key-file PATH
"""
import argparse
import json
import sys
from datetime import datetime

import _paths
from copilot.corpus import load_corpus
from copilot.observability import retrieval_metrics
from copilot.retrieval import hybrid_search
from copilot.runtime import add_endpoint_args, apply_endpoint_args
from copilot.service import build_engine

ROOT = _paths.ROOT
GOLD = ROOT / "eval" / "gold" / "retrieval_gold.json"
K = 5


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieval-only evaluation.")
    add_endpoint_args(ap)
    apply_endpoint_args(ap.parse_args())
    if not load_corpus():
        sys.exit("No corpus yet. Run: python scripts/build_corpus.py --fetch")

    gold = json.loads(GOLD.read_text(encoding="utf-8"))["items"]
    engine = build_engine()
    passages = {d["id"]: d for d in engine.corpus}

    results = {}
    for name, use_dense in (("bm25_only", False), ("hybrid", True)):
        if use_dense and engine.dense is None:
            print("Dense retrieval not available (embeddings endpoint or embeddings file missing); skipping hybrid.")
            continue
        ranked = {}
        for g in gold:
            qvec = engine.embed.embed([g["query"]])[0] if use_dense else None
            hits = hybrid_search(g["query"], engine.corpus, engine.bm25, engine.dense if use_dense else None, qvec, k=K)
            ranked[g["id"]] = [d for d, _ in hits]
        results[name] = retrieval_metrics(ranked, gold, passages, K)
        print(f"{name:10s} recall@{K}={results[name][f'recall_at_{K}']}  MRR={results[name]['mrr']}")

    out_dir = ROOT / "eval" / "results" / datetime.now().strftime("retrieval_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Written to {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
