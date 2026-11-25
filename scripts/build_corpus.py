"""Build the corpus and (optionally) the embeddings.

  python scripts/build_corpus.py --fetch            # download openFDA labels, shortages, Class I recalls
  python scripts/build_corpus.py --fetch --embed    # also compute embeddings (needs COPILOT_EMBED_* in .env)
  python scripts/build_corpus.py                    # rebuild from already downloaded raw files
"""
import argparse
import json

import _paths
from copilot.rag import corpus as C
from copilot.config import embeddings_configured, load_settings
from copilot.agent.llm import EmbedClient
from copilot.runtime import add_endpoint_args, apply_endpoint_args
from copilot.service import EMBEDDINGS_FILE

ROOT = _paths.ROOT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download from openFDA first")
    ap.add_argument("--embed", action="store_true", help="compute embeddings with the university endpoint")
    ap.add_argument("--per-name", type=int, default=25, help="labels per drug name")
    add_endpoint_args(ap)
    args = ap.parse_args()
    apply_endpoint_args(args)

    docs: list[dict] = []
    if args.fetch:
        names = [n.strip() for n in (ROOT / "configs" / "drug_names.txt").read_text(encoding="utf-8").splitlines()
                 if n.strip() and not n.startswith("#")]
        print(f"Fetching labels for {len(names)} drug names...")
        docs += C.fetch_labels(names, per_name=args.per_name)
        print("Fetching shortages...")
        docs += C.fetch_shortages()
        print("Fetching Class I recalls...")
        docs += C.fetch_enforcement()
    else:
        existing = C.load_corpus()
        if existing is None:
            print("No corpus yet. Run with --fetch.")
            return 1
        docs = existing

    # Remove duplicate IDs, keep first
    seen, unique = set(), []
    for d in docs:
        if d["id"] not in seen:
            seen.add(d["id"])
            unique.append(d)
    n_docs = len({d.get("parent", d["id"]) for d in unique})
    chunk_chars = int(load_settings().get("chunk_chars", 1200))
    unique = [chunk for d in unique for chunk in C.chunk_document(d, chunk_chars)]
    path = C.write_corpus(unique)
    print(f"Wrote {len(unique)} passages from {n_docs} documents to {path.relative_to(ROOT)}")

    if args.embed:
        if not embeddings_configured():
            print("Embeddings not configured (COPILOT_EMBED_* blank). Skipping. BM25 search still works.")
        else:
            print(f"Embedding {len(unique)} passages (16 per request)...")
            vectors = EmbedClient().embed([d["text"] for d in unique])
            EMBEDDINGS_FILE.write_text(json.dumps({d["id"]: v for d, v in zip(unique, vectors)}), encoding="utf-8")
            print(f"Wrote embeddings for {len(vectors)} passages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
