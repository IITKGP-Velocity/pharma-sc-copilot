"""Build all seven data packs (A-E) into data/raw/packs/ (git-ignored), each as a folder and a zip.

Usage: python scripts/data_packs/make_all.py [out folder]
Needs internet and `pdftotext` (poppler: brew install poppler / apt install poppler-utils). Takes 1-2 hours,
mostly PDF downloads; downloads are cached in data/raw/packs_work/ so a re-run is faster."""
import collections
import csv
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT  # noqa: E402

HERE = Path(__file__).resolve().parent
PDF_PACKS = {
    "B": ("UseCaseB_Supplier_Qualification_PDFs", "Use case B - Alternate API / generic supplier qualification (owner: Shrilesh)"),
    "A": ("UseCaseA_ColdChain_GDP_PDFs", "Use case A - Cold-chain excursion triage and GDP release (owner: Siddhesh)"),
}


def run(*args) -> None:
    print("$", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def pdf_pack_readme(folder: Path, code: str, title: str) -> None:
    """README for a PDF pack, written from its manifest."""
    rows = list(csv.DictReader(open(folder / "manifest.csv", encoding="utf-8")))
    cats = collections.Counter(r["category"] for r in rows)
    with_tables = sum(r["has_tables"] == "yes" for r in rows)
    lines = [title, f"{len(rows)} PDFs, all US government works (public domain).", "",
             "Contents",
             f"  pdfs/            the PDFs, named {code}<nnn>_<category>_<application>_<drug>.pdf",
             "  manifest.csv     one row per PDF: category, drug, application, sponsor, date, pages, tables, URL, licence, sha256",
             f"  corpus_{code}.jsonl   extracted text in the copilot corpus format {{\"id\",\"text\",\"source\",\"url\"}}",
             "  tables/          each detected table as CSV: <pdf name>_p<page>_t<n>.csv", "",
             "Categories"] + [f"  {k}: {v}" for k, v in sorted(cats.items())] + [
             f"  {with_tables} of the {len(rows)} PDFs contain at least one detected table.", "",
             "Load into the copilot: python scripts/data_packs/load_packs.py, then python scripts/build_corpus.py --embed <flags>",
             "Table extraction is layout-based: numeric tables are clean, cells that wrap over lines may be split.",
             "Reviews and letters contain names of FDA officials and company contacts (public records); keep them out of reports."]
    (folder / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def zip_folder(base: Path, name: str) -> None:
    (base / f"{name}.zip").unlink(missing_ok=True)
    subprocess.run(["zip", "-qr", "-X", f"{name}.zip", name, "-x", "*.DS_Store"], cwd=base, check=True)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    if not shutil.which("pdftotext"):
        print("pdftotext not found: install poppler (brew install poppler / apt install poppler-utils)")
        return 1
    b_name = PDF_PACKS["B"][0]
    run(HERE / "build_pdfs_b.py", out / b_name)
    run(HERE / "build_pdfs_a.py", out / PDF_PACKS["A"][0], out / b_name / "manifest.csv")  # A skips B's files
    for code, (name, title) in PDF_PACKS.items():
        pdf_pack_readme(out / name, code, title)
        zip_folder(out, name)
    run(HERE / "build_packs.py", out, "a", "b", "c", "d", "e")
    run(HERE / "redact_freemail.py", out, "UseCaseC_Customs_Import_PreClearance",
        "UseCaseD_Shortage_Therapeutic_Substitution", "UseCaseE_Pharmacovigilance_Recall_Triage")
    print(f"\nDone. Packs in {out}. Next:\n  python scripts/data_packs/load_packs.py\n"
          "  python scripts/build_corpus.py --embed <flags>\n  python scripts/data_packs/bundle.py   # one zip for the team")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
