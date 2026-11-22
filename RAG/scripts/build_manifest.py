import csv
import hashlib
from datetime import date
from pathlib import Path
import fitz  # pymupdf
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")  # downloads once

def count_tokens(text: str) -> int:
    # No special tokens, so this measures raw text length in embedding tokens
    return len(tok(text, add_special_tokens=False)["input_ids"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "index"
INPUT_DIR = PROJECT_ROOT / "input"
DOCS = PROJECT_ROOT / "docs"
SRC = INPUT_DIR / "sources.csv"
OUT = INDEX_DIR / "manifest.csv"

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Keep first_seen dates and hashes from a previous run
    old = {}
    if Path(OUT).exists():
        with open(OUT, encoding="utf-8") as f:
            old = {r["doc_id"]: r for r in csv.DictReader(f)}

    rows = []
    with open(SRC, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            path = DOCS / f"{r['doc_id']}.pdf"
            if not path.exists():
                print(f"MISSING: {r['doc_id']} -> {path}")
                continue

            digest = sha256(path)
            prev = old.get(r["doc_id"])
            if prev and prev["sha256"] != digest:
                print(f"HASH CHANGED: {r['doc_id']} (file was modified!)")

            pages = chars = tokens = ""
            if path.suffix.lower() == ".pdf":
                doc = fitz.open(path)
                pages = len(doc)
                page_texts = [pg.get_text() for pg in doc]
                chars = sum(len(t) for t in page_texts)
                tokens = sum(count_tokens(t) for t in page_texts)
                if pages and chars / pages < 200:
                    print(f"WARNING: {r['doc_id']} may be scanned (little text)")

            r.update(
                sha256=digest,
                size_bytes=path.stat().st_size,
                pages=pages,
                text_chars=chars,
                tokens=tokens,
                first_seen=prev["first_seen"] if prev else date.today().isoformat(),
            )
            rows.append(r)

    if not rows:
        print("No matching files found. Check that your PDFs are in docs/ and named <doc_id>.pdf.")
        return

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} documents written to {OUT}")

if __name__ == "__main__":
    main()