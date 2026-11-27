"""Download 100 public-domain PDFs for use case A (cold-chain excursion triage and GDP release).

Sources: govinfo.gov (21 CFR 2025 and 49 CFR 2024 DOT dry-ice/air rules), NIST vaccine-storage studies,
Drugs@FDA (labels and product quality reviews of refrigerated products, via openFDA drugsfda).
Keeps only real PDFs with a text layer whose text covers cold storage. Skips anything in the use case B set.
Outputs, under OUT: pdfs/, tables/, manifest.csv, corpus_A.jsonl.
Usage: python build_pdfs_a.py <out folder> <use case B manifest.csv>"""
import csv, hashlib, json, re, subprocess, sys, time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import drugsfda_docs  # noqa: E402

COLD_DRUGS = ["adalimumab", "insulin glargine", "insulin lispro", "insulin aspart", "insulin detemir", "insulin degludec",
              "insulin human", "oxytocin", "filgrastim", "pegfilgrastim", "rituximab", "trastuzumab", "infliximab",
              "amphotericin b", "etanercept", "semaglutide", "liraglutide", "dulaglutide", "epoetin alfa", "darbepoetin alfa",
              "bevacizumab", "ranibizumab", "aflibercept", "denosumab", "teriparatide", "octreotide", "glucagon",
              "carboprost tromethamine", "dinoprostone", "succinylcholine chloride", "rocuronium bromide",
              "cisatracurium besylate", "latanoprost", "vasopressin", "nivolumab", "pembrolizumab", "tocilizumab",
              "ustekinumab", "eculizumab", "alteplase", "exenatide", "tirzepatide"]  # refrigerated / cold-chain products
OUT = Path(sys.argv[1])
DOCS = drugsfda_docs(COLD_DRUGS, "drugsfda_cold_chain")  # rows: drug, prefix, appno, sponsor, type, subtype, date, url
B_URLS = {r["url"] for r in csv.DictReader(open(sys.argv[2]))}
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) IITKGP-capstone-research"}
MAX_BYTES = 40_000_000
TARGET = 100
QUOTA = {"regulation": 10, "study": 2, "quality_review": 28, "label": 100}
PER_DRUG = {"label": 3, "quality_review": 3}
COLD = re.compile(r"refrigerat|2\s*°?\s*C?\s*(to|-|–)\s*8\s*°\s*C|36\s*°?\s*F?\s*(to|-|–)\s*46\s*°\s*F|excursion|cold chain", re.I)
SHIP = re.compile(r"shipping|shipment|transport|excursion|temperature cycling|freeze[- ]thaw", re.I)

REGS = [  # (url, application, title)
    ("https://www.govinfo.gov/content/pkg/CFR-2025-title21-vol4/pdf/CFR-2025-title21-vol4-part203.pdf", "21 CFR Part 203", "Prescription drug marketing (storage and handling of drug samples)"),
    ("https://www.govinfo.gov/content/pkg/CFR-2025-title21-vol4/pdf/CFR-2025-title21-vol4-part205.pdf", "21 CFR Part 205", "Wholesale drug distributors: storage and handling requirements"),
    ("https://www.govinfo.gov/content/pkg/CFR-2025-title21-vol4/pdf/CFR-2025-title21-vol4-part211.pdf", "21 CFR Part 211", "cGMP for finished pharmaceuticals (211.142 warehousing, 211.150 distribution, 211.166 stability)"),
    ("https://www.govinfo.gov/content/pkg/CFR-2025-title21-vol7/pdf/CFR-2025-title21-vol7-part600.pdf", "21 CFR Part 600", "Biological products: general (600.15 shipment temperatures)"),
    ("https://www.govinfo.gov/content/pkg/CFR-2025-title21-vol7/pdf/CFR-2025-title21-vol7-part606.pdf", "21 CFR Part 606", "cGMP for blood and blood components (storage and distribution)"),
    ("https://www.govinfo.gov/content/pkg/CFR-2025-title21-vol7/pdf/CFR-2025-title21-vol7-part610.pdf", "21 CFR Part 610", "General biological products standards (610.53 dating periods and storage temperatures)"),
    ("https://www.govinfo.gov/content/pkg/CFR-2025-title21-vol7/pdf/CFR-2025-title21-vol7-part640.pdf", "21 CFR Part 640", "Additional standards for human blood and blood products (storage temperatures)"),
    ("https://www.govinfo.gov/content/pkg/CFR-2025-title21-vol8/pdf/CFR-2025-title21-vol8-part1271.pdf", "21 CFR Part 1271", "Human cells, tissues and HCT/Ps (storage, shipping, temperature limits)"),
    ("https://www.govinfo.gov/content/pkg/CFR-2024-title49-vol2/pdf/CFR-2024-title49-vol2-sec173-217.pdf", "49 CFR 173.217", "DOT hazardous materials: carbon dioxide, solid (dry ice) packaging"),
    ("https://www.govinfo.gov/content/pkg/CFR-2024-title49-vol2/pdf/CFR-2024-title49-vol2-part175.pdf", "49 CFR Part 175", "DOT carriage by aircraft (dry ice quantities and marking)"),
]
STUDIES = [
    ("https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir7656.pdf", "NISTIR 7656", "Thermal analysis of refrigeration systems used for vaccine storage"),
    ("https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir7753.pdf", "NISTIR 7753", "Vaccine storage: pharmaceutical-grade vs household refrigerator thermal analysis"),
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
    for url, app, title in REGS:
        yield {"category": "regulation", "drug": "", "application": app, "sponsor": "", "doc_type": "Regulation",
               "date": url.split("CFR-")[1][:4], "title": f"{app} - {title}", "url": url, "publisher": "US GPO govinfo (CFR annual edition)"}
    for url, app, title in STUDIES:
        yield {"category": "study", "drug": "", "application": app, "sponsor": "", "doc_type": "NIST technical report",
               "date": "", "title": f"{app} - {title}", "url": url, "publisher": "NIST (US Department of Commerce)"}
    # Product quality reviews (stability, shipping studies) of refrigerated products, newest first.
    tocs = sorted({(r[6] or "", r[0], r[2], r[3], r[7]) for r in DOCS
                   if (r[4] or "").endswith("Review") and r[7].endswith(".html")}, reverse=True)
    for date, drug, app, sponsor, url in tocs:
        try:
            html = session.get(url.replace("http://", "https://"), timeout=60).text
        except requests.RequestException:
            continue
        time.sleep(0.3)
        base = url.replace("http://", "https://").rsplit("/", 1)[0]
        for name in sorted(set(re.findall(r'"(\d{6}Orig\w*?(?:ChemR|ProductQualityR|QualityR)\.pdf)"', html))):
            yield {"category": "quality_review", "drug": drug, "application": app, "sponsor": sponsor,
                   "doc_type": "Product quality review (stability, shipping, storage)", "date": date,
                   "title": f"{drug} - product quality review - {sponsor} ({app})", "url": f"{base}/{name}", "publisher": "FDA Drugs@FDA"}
    labels = sorted((r for r in DOCS if r[4] == "Label" and r[7].lower().endswith(".pdf")), key=lambda r: r[6] or "", reverse=True)
    for drug, prefix, app, sponsor, typ, sub, date, url in labels:
        yield {"category": "label", "drug": drug, "application": app, "sponsor": sponsor,
               "doc_type": "Prescribing information (storage and handling, in-use and excursion limits)", "date": date,
               "title": f"{drug} - label (storage and handling) - {sponsor} ({app})", "url": url, "publisher": "FDA Drugs@FDA"}


rows, corpus, taken, seen_url = [], [], {k: 0 for k in QUOTA}, set(B_URLS)
per_app, per_drug = set(), {}
for c in candidates():
    if len(rows) >= TARGET:
        break
    cat = c["category"]
    if c["url"] in seen_url or taken[cat] >= QUOTA[cat]:
        continue
    if cat in PER_DRUG:
        if (cat, c["application"]) in per_app or per_drug.get((cat, c["drug"]), 0) >= PER_DRUG[cat]:
            continue
    seen_url.add(c["url"])
    data = fetch(c["url"])
    if data is None:
        print("skip (not a PDF / too big / error):", c["url"])
        continue
    n = len(rows) + 1
    stem = f"A{n:03d}_{cat}_{slug(c['application'])}_{slug(c['drug'] or 'ref')}"
    path = OUT / "pdfs" / f"{stem}.pdf"
    path.write_bytes(data)
    pages = pdf_text(path)
    full = "\n".join(pages)
    chars = sum(len(p.strip()) for p in pages)
    if chars < 1500:
        print("skip (no text layer):", c["url"])
        path.unlink()
        continue
    if cat == "label" and not COLD.search(full):
        print("skip (label not cold-chain):", c["url"])
        path.unlink()
        continue
    if cat == "quality_review" and not (COLD.search(full) and SHIP.search(full)):
        print("skip (review has no cold storage/shipping content):", c["url"])
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
    if cat in PER_DRUG:
        per_app.add((cat, c["application"]))
        per_drug[(cat, c["drug"])] = per_drug.get((cat, c["drug"]), 0) + 1
    taken[cat] += 1
    doc_id = f"pdfA-{stem}"
    corpus.append({"id": doc_id, "text": f"{c['title']}\n" + "\n".join(texts),
                   "source": f"{c['publisher']} - {c['doc_type']}", "url": c["url"]})
    rows.append({"file": f"pdfs/{stem}.pdf", "doc_id": doc_id, "use_case": "A", "category": cat,
                 "doc_type": c["doc_type"], "drug": c["drug"], "application": c["application"], "sponsor": c["sponsor"],
                 "date": c["date"], "pages": len([p for p in pages if p.strip()]), "text_chars": chars,
                 "tables_detected": ntables, "has_tables": "yes" if ntables else "no", "title": c["title"],
                 "url": c["url"], "publisher": c["publisher"], "licence": "US government work - public domain",
                 "sha256": hashlib.sha256(data).hexdigest(), "downloaded_on": time.strftime("%Y-%m-%d")})
    print(f"{n:3d} {cat:15} pages={rows[-1]['pages']:4} tables={ntables:4} {stem}", flush=True)

with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
with open(OUT / "corpus_A.jsonl", "w", encoding="utf-8") as f:
    for d in corpus:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print("TOTAL", len(rows), "by category", taken, "with tables", sum(r["has_tables"] == "yes" for r in rows))
