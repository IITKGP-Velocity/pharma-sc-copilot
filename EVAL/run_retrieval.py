#!/usr/bin/env python3
"""
run_retrieval.py - Retrieval-only check of the eval set (no LLM, only the embeddings API).

Why: lets us test retrieval quality and the pipeline's abstain gate before the chat
endpoint is available, and separates retrieval problems from generation problems.

For each query it records the top retrieved documents, the retrieval scores, and whether
the confidence gate in rag_pipeline.py would abstain. Safety items (PI / PII / OOS) are
included in the trace but not in the summary.

Usage (from the repo root; needs EMBEDDINGS_KEY in .env and secrets/vm02-ca.crt)
  python EVAL/run_retrieval.py --run-name retrieval-v1
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
EVAL_FILE = EVAL_DIR / "eval_set_v1.jsonl"
GOLD_FILE = EVAL_DIR / "gold_docs_v1.json"   # query id -> source documents that support the answer

# Same defaults as RegulatoryRAGPipeline (RAG_MIN_COSINE / RAG_MIN_BM25).
MIN_COSINE, MIN_BM25 = 0.48, 2.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieval-only evaluation run.")
    ap.add_argument("--run-name", required=True, help="folder name under EVAL/results/")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    with open(EVAL_FILE, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    # Keys starting with "_" are notes, not query IDs.
    gold = {k: v for k, v in json.loads(GOLD_FILE.read_text(encoding="utf-8")).items()
            if not k.startswith("_")}

    # Import after the path is set; rag_pipeline also loads the repo-root .env at import time.
    sys.path.insert(0, str(REPO_ROOT / "RAG_PIPELINE" / "scripts"))
    from rag_pipeline import RegulatoryRAGPipeline
    from retriever import HybridRetriever, confidence_signal

    retriever = HybridRetriever(index_dir=REPO_ROOT / "RAG" / "index",
                                embed_cert=REPO_ROOT / "secrets" / "vm02-ca.crt")
    # Reuse the pipeline's gate logic with just the two thresholds it reads, so we
    # don't need the LLM settings and the rule is not copied here.
    gate_cfg = SimpleNamespace(min_cosine=MIN_COSINE, min_bm25=MIN_BM25)

    out_dir = EVAL_DIR / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, item in enumerate(items, 1):
        hits = retriever.search(query=item["query"], top_k=args.top_k)
        sig = confidence_signal(hits)
        would_abstain, _ = RegulatoryRAGPipeline._should_abstain_early(gate_cfg, sig)
        doc_ids = [h.get("doc_id") for h in hits]
        # Rank (1-based) of the first gold document in the results; None = not retrieved.
        gold_rank = None
        if item["id"] in gold:
            gold_rank = next((i for i, d in enumerate(doc_ids, 1) if d in gold[item["id"]]), None)
        rows.append({
            "id": item["id"], "use_case": item["use_case"], "gold_type": item["type"],
            "would_abstain": would_abstain,
            "has_gold": item["id"] in gold,
            "gold_rank": gold_rank,
            "retrieved_doc_ids": doc_ids,
            "top_dense_cosine": sig["top_dense_cosine"],
            "top_bm25_score": sig["top_bm25_score"],
            "has_ref_match": sig["has_ref_match"],
            "query": item["query"],
        })
        print(f"[{n}/{len(items)}] {item['id']} ({item['type']}): abstain={would_abstain} "
              f"docs={sorted(set(r for r in rows[-1]['retrieved_doc_ids'] if r))}")

    with open(out_dir / "retrieval.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Headline numbers: ANSWER/CALC items should NOT abstain, ABSTAIN items should.
    def rate(types):
        sel = [r for r in rows if r["gold_type"] in types]
        return sel, (sum(r["would_abstain"] for r in sel) / len(sel) if sel else None)

    ans, ans_rate = rate({"ANSWER", "CALC"})
    abs_, abs_rate = rate({"ABSTAIN"})

    # Document-level recall@k and MRR over queries that have gold documents.
    # With only 6 PDFs this is a weak test; chunk-level gold IDs would be stricter.
    gold_rows = [r for r in rows if r["has_gold"]]
    recall = (sum(r["gold_rank"] is not None for r in gold_rows) / len(gold_rows)) if gold_rows else None
    mrr = (sum(1 / r["gold_rank"] for r in gold_rows if r["gold_rank"]) / len(gold_rows)) if gold_rows else None
    summary = {
        "queries": len(rows),
        "gold_queries": len(gold_rows),
        f"doc_recall_at_{args.top_k}": None if recall is None else round(recall, 3),
        "doc_mrr": None if mrr is None else round(mrr, 3),
        "answerable_items": len(ans),
        "answerable_wrongly_gated_rate": None if ans_rate is None else round(ans_rate, 3),
        "abstain_items": len(abs_),
        "abstain_items_correctly_gated_rate": None if abs_rate is None else round(abs_rate, 3),
    }
    (out_dir / "retrieval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
