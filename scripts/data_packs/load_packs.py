"""Append every data pack's corpus_<X>.jsonl to the copilot corpus, skipping documents already present.

Usage: python scripts/data_packs/load_packs.py [packs folder] [corpus.jsonl]
Defaults: data/raw/packs and data/processed/corpus.jsonl. Then run build_corpus.py --embed."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, ROOT  # noqa: E402

PACKS = ["UseCaseA_ColdChain_GDP_PDFs", "UseCaseA_ColdChain_GDP_Guidance_and_Data", "UseCaseB_Supplier_Qualification_PDFs",
         "UseCaseB_Supplier_Qualification_Guidance_and_Data", "UseCaseC_Customs_Import_PreClearance",
         "UseCaseD_Shortage_Therapeutic_Substitution", "UseCaseE_Pharmacovigilance_Recall_Triage"]


def main() -> int:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    corpus = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "processed" / "corpus.jsonl"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    present = set()
    if corpus.exists():
        present = {d.get("parent") or d["id"] for d in map(json.loads, corpus.open(encoding="utf-8"))}
    total = 0
    with corpus.open("a", encoding="utf-8") as out:
        for p in PACKS:
            n = 0
            for f in sorted((base / p).glob("corpus_*.jsonl")):
                for line in f.open(encoding="utf-8"):
                    d = json.loads(line)
                    if d["id"] in present or not d.get("text", "").strip():
                        continue
                    present.add(d["id"])
                    out.write(json.dumps({k: d[k] for k in ("id", "text", "source", "url")}, ensure_ascii=False) + "\n")
                    n += 1
            print(f"{p:52} +{n}" if (base / p).exists() else f"{p:52} (not found, skipped)")
            total += n
    print(f"added {total} documents to {corpus}. Next: python scripts/build_corpus.py --embed <flags>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
