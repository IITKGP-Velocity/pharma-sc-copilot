"""Make one zip the team can unpack into the repo: the ready corpus + embeddings and the pack metadata/datasets.

Usage: python scripts/data_packs/bundle.py [output.zip]   (default: data/raw/Velocity_corpus_bundle_<date>.zip, git-ignored)
Unpack with: unzip -o <zip> -d <repo folder>   (files land in git-ignored data/processed/ and data/raw/packs/)"""
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, ROOT  # noqa: E402
from load_packs import PACKS  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / f"Velocity_corpus_bundle_{time.strftime('%Y%m%d')}.zip"
    corpus, emb = PROCESSED / "corpus.jsonl", PROCESSED / "embeddings.json"
    ids = [json.loads(line)["id"] for line in corpus.open(encoding="utf-8")]
    vectors = json.loads(emb.read_text())
    missing = [i for i in ids if i not in vectors]
    if missing:
        print(f"{len(missing)} of {len(ids)} passages have no embedding. Run build_corpus.py --embed first.")
        return 1
    docs = len({i.split("#")[0] for i in ids})
    readme = f"""Team Velocity - pharma supply-chain copilot: ready-to-use corpus bundle
Built {time.strftime('%Y-%m-%d')} from the repo at {ROOT.name}.

What is inside
  data/processed/corpus.jsonl     {len(ids)} passages from {docs} documents (openFDA base corpus + all 7 data packs)
  data/processed/embeddings.json  one bge-small-en-v1.5 vector (384 numbers) per passage
  data/raw/packs/<pack>/          README.txt, manifest.csv (source, URL, licence, sha256) and datasets/ (CSV tables)
                                  for each pack. Source PDFs are not included; rebuild them with make_all.py if needed.

Install (from the repo folder)
  unzip -o {out.name} -d .
  python scripts/check_setup.py                 # shows the passage count
  python scripts/demo.py retrieval "Which firms are on import alert 66-40?"
  Then follow README section 2 (testing guide); skip `build_corpus.py --fetch --embed`, the bundle replaces it.

Checksums
  corpus.jsonl     sha256 {sha(corpus)}
  embeddings.json  sha256 {sha(emb)}

Licences
  US government data (FDA, CMS, CFR, USITC, NIST) is public domain. WHO documents are CC BY-NC-SA 3.0 IGO,
  ICH and EU documents need attribution. Share this bundle inside the team only; never commit it or
  data/processed/ to the public repo (both are git-ignored).
"""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("BUNDLE_README.txt", readme)
        z.write(corpus, "data/processed/corpus.jsonl")
        z.write(emb, "data/processed/embeddings.json")
        for p in PACKS:
            folder = DEFAULT_OUT / p
            if not folder.exists():
                continue
            for f in [folder / "README.txt", folder / "manifest.csv", *sorted((folder / "datasets").glob("*"))]:
                if f.is_file():
                    z.write(f, f"data/raw/packs/{p}/{f.relative_to(folder)}")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.0f} MB): {len(ids)} passages, {docs} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
