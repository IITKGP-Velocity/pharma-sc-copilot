"""Download 100 public-domain PDFs for use case B (alternate API / generic supplier qualification).

Sources: govinfo.gov (21 CFR, 2025 edition) and the Drugs@FDA document store on accessdata.fda.gov
(listed by openFDA drugsfda for the project's drug names). Keeps only real PDFs with a text layer.
Outputs, under OUT: pdfs/, tables/ (CSV per detected table), manifest.csv, corpus_B.jsonl.
Usage: python build_pdfs_b.py <out folder>"""
import csv, hashlib, json, re, subprocess, sys, time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import drugsfda_docs, project_drugs  # noqa: E402

OUT = Path(sys.argv[1])
DOCS = drugsfda_docs(project_drugs(), "drugsfda_project_drugs")  # rows: drug, prefix, appno, sponsor, type, subtype, date, url
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) IITKGP-capstone-research"}
MAX_BYTES = 40_000_000
TARGET = 100
QUOTA = {"regulation": 11, "quality_review": 30, "approval_letter": 22, "biosimilar_approval_letter": 15, "label": 100}

CFR = [  # (volume, part, title)
    (1, 11, "Electronic records; electronic signatures"),
    (4, 203, "Prescription drug marketing"),
    (4, 207, "Establishment registration and drug listing"),
    (4, 210, "cGMP in manufacturing, processing, packing or holding of drugs: general"),
    (4, 211, "cGMP for finished pharmaceuticals (incl. 211.84 component testing and supplier approval)"),
    (4, 251, "Section 804 importation program"),
    (5, 314, "Applications for FDA approval to market a new drug (NDA, ANDA, DMF references)"),
    (5, 320, "Bioavailability and bioequivalence requirements"),
    (7, 600, "Biological products: general"),
    (7, 601, "Licensing (biologics and biosimilars)"),
    (7, 610, "General biological products standards"),
]

session = requests.Session()
session.headers.update(UA)
OUT.joinpath("pdfs").mkdir(parents=True, exist_ok=True)
OUT.joinpath("tables").mkdir(exist_ok=True)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:40]


def fetch(url: str) -> bytes | None:
    try:
        r = session.get(url.replace("http://", "https://"), timeout=120, stream=True)
        if r.status_code != 200:
            return None
        data = b""
        for part in r.iter_content(1 << 16):
            data += part
            if len(data) > MAX_BYTES:
                return None
        return data if data[:5] == b"%PDF-" else None
    except requests.RequestException:
        return None
    finally:
        time.sleep(0.4)


def pdf_text(path: Path) -> list[str]:
    """Text of each page in layout mode (keeps table columns aligned)."""
    out = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"], capture_output=True, text=True)
    return out.stdout.split("\f") if out.returncode == 0 else []


CELL_SPLIT = re.compile(r"\s{2,}")


def page_to_text_and_tables(page: str):
    """Rows with 3+ columns separated by 2+ spaces, in runs of 3+ rows, are treated as a table."""
    lines = page.splitlines()
    cells = [CELL_SPLIT.split(l.strip()) if l.strip() else [] for l in lines]
    is_row = [len(c) >= 3 for c in cells]
    tables, out, i = [], [], 0
    while i < len(lines):
        if is_row[i]:
            j = i
            while j < len(lines) and (is_row[j] or (not lines[j].strip() and j + 1 < len(lines) and is_row[j + 1])):
                j += 1
            block = [cells[k] for k in range(i, j) if cells[k]]
            if len(block) >= 3:
                tables.append(block)
                out.append("[table]")
                out += [" | ".join(r) for r in block]
                out.append("[/table]")
                i = j
                continue
        line = re.sub(r"\s+", " ", lines[i]).strip()
        if line:
            out.append(line)
        i += 1
    return "\n".join(out), tables


def candidates():
    for vol, part, title in CFR:
        url = f"https://www.govinfo.gov/content/pkg/CFR-2025-title21-vol{vol}/pdf/CFR-2025-title21-vol{vol}-part{part}.pdf"
        yield {"category": "regulation", "drug": "", "application": f"21 CFR Part {part}", "sponsor": "",
               "doc_type": "Regulation", "date": "2025-04-01", "title": f"21 CFR Part {part} - {title}",
               "url": url, "publisher": "US GPO govinfo (eCFR annual edition 2025)"}
    seen_toc = set()
    # Review packages: biosimilars (BLA 761xxx) and ANDAs; pick product-quality and facility-list PDFs.
    for drug, prefix, app, sponsor, typ, sub, date, url in DOCS:
        if not (typ or "").endswith("Review") or not url.endswith(".html") or url in seen_toc:
            continue
        if not (app.startswith("BLA761") or prefix == "ANDA"):
            continue
        seen_toc.add(url)
        try:
            html = session.get(url.replace("http://", "https://"), timeout=60).text
        except requests.RequestException:
            continue
        time.sleep(0.3)
        base = url.replace("http://", "https://").rsplit("/", 1)[0]
        names = sorted(set(re.findall(r'"(\d{6}Orig\w*?(?:ChemR|ProductQualityR|QualityR)\.pdf)"', html)))
        for name in names:
            yield {"category": "quality_review", "drug": drug, "application": app, "sponsor": sponsor,
                   "doc_type": "Product quality / chemistry review", "date": date,
                   "title": f"{drug} - product quality review - {sponsor} ({app})",
                   "url": f"{base}/{name}", "publisher": "FDA Drugs@FDA"}
    for drug, prefix, app, sponsor, typ, sub, date, url in DOCS:
        if prefix == "ANDA" and typ == "Letter" and url.lower().endswith(".pdf"):
            yield {"category": "approval_letter", "drug": drug, "application": app, "sponsor": sponsor,
                   "doc_type": "ANDA approval letter", "date": date,
                   "title": f"{drug} - generic (ANDA) approval letter - {sponsor} ({app})", "url": url, "publisher": "FDA Drugs@FDA"}
    seen_app = set()
    for drug, prefix, app, sponsor, typ, sub, date, url in DOCS:
        if app.startswith("BLA761") and typ == "Letter" and sub == "ORIG" and url.lower().endswith(".pdf") and app not in seen_app:
            seen_app.add(app)
            yield {"category": "biosimilar_approval_letter", "drug": drug, "application": app, "sponsor": sponsor,
                   "doc_type": "Biosimilar (BLA 761) original approval letter", "date": date,
                   "title": f"{drug} - biosimilar approval letter - {sponsor} ({app})", "url": url, "publisher": "FDA Drugs@FDA"}
    labels = [r for r in DOCS if r[4] == "Label" and r[7].lower().endswith(".pdf")
              and (r[1] == "ANDA" or r[2].startswith("BLA761"))]
    labels.sort(key=lambda r: r[6], reverse=True)  # newest first
    for drug, prefix, app, sponsor, typ, sub, date, url in labels:
        yield {"category": "label", "drug": drug, "application": app, "sponsor": sponsor,
               "doc_type": "Generic/biosimilar label (composition, how supplied, storage)", "date": date,
               "title": f"{drug} - {'biosimilar' if app.startswith('BLA761') else 'generic'} label - {sponsor} ({app})",
               "url": url, "publisher": "FDA Drugs@FDA"}


rows, corpus, taken, seen_url = [], [], {k: 0 for k in QUOTA}, set()
labels_per_app = {}
for c in candidates():
    if len(rows) >= TARGET:
        break
    if c["url"] in seen_url or taken[c["category"]] >= QUOTA[c["category"]]:
        continue
    if c["category"] == "label":  # one label per application keeps variety
        if labels_per_app.get(c["application"]):
            continue
    seen_url.add(c["url"])
    data = fetch(c["url"])
    if data is None:
        print("skip (not a PDF / too big / error):", c["url"])
        continue
    n = len(rows) + 1
    stem = f"B{n:03d}_{c['category']}_{slug(c['application'])}_{slug(c['drug'] or 'cfr')}"
    path = OUT / "pdfs" / f"{stem}.pdf"
    path.write_bytes(data)
    pages = pdf_text(path)
    chars = sum(len(p.strip()) for p in pages)
    if chars < 1500:  # scanned image with no text layer
        print("skip (no text layer):", c["url"])
        path.unlink()
        continue
    texts, ntables = [], 0
    for pno, page in enumerate(pages, 1):
        text, tables = page_to_text_and_tables(page)
        if text:
            texts.append(f"[page {pno}]\n{text}")
        for t_i, tbl in enumerate(tables, 1):
            ntables += 1
            with open(OUT / "tables" / f"{stem}_p{pno}_t{t_i}.csv", "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(tbl)
    if c["category"] == "label":
        labels_per_app[c["application"]] = True
    taken[c["category"]] += 1
    doc_id = f"pdfB-{stem}"
    corpus.append({"id": doc_id, "text": f"{c['title']}\n" + "\n".join(texts),
                   "source": f"{c['publisher']} - {c['doc_type']}", "url": c["url"]})
    rows.append({"file": f"pdfs/{stem}.pdf", "doc_id": doc_id, "use_case": "B", "category": c["category"],
                 "doc_type": c["doc_type"], "drug": c["drug"], "application": c["application"], "sponsor": c["sponsor"],
                 "date": c["date"], "pages": len([p for p in pages if p.strip()]), "text_chars": chars,
                 "tables_detected": ntables, "has_tables": "yes" if ntables else "no", "title": c["title"],
                 "url": c["url"], "publisher": c["publisher"], "licence": "US government work - public domain",
                 "sha256": hashlib.sha256(data).hexdigest(), "downloaded_on": time.strftime("%Y-%m-%d")})
    print(f"{n:3d} {c['category']:16} pages={rows[-1]['pages']:4} tables={ntables:4} {stem}")

with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
with open(OUT / "corpus_B.jsonl", "w", encoding="utf-8") as f:
    for d in corpus:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print("TOTAL", len(rows), "by category", taken, "with tables", sum(r["has_tables"] == "yes" for r in rows))
