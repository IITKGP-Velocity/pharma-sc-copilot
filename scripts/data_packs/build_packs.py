"""Build the guidance and data packs for use cases A-E (documents + tabular datasets).

Usage: python build_packs.py <out folder> [a b c d e]
Each zip folder gets: documents/, datasets/, tables/, manifest.csv, corpus_<X>.jsonl, README.txt.
corpus_<X>.jsonl uses the copilot corpus format {"id","text","source","url"}."""
import csv, hashlib, html, io, json, re, subprocess, sys, time, zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, project_drugs  # noqa: E402

BASE = Path(sys.argv[1]); BASE.mkdir(parents=True, exist_ok=True)
DRUGS = project_drugs()
CACHE = WORK / "http_cache"; CACHE.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "text/html,application/pdf,application/json,text/csv,*/*"}
S = requests.Session(); S.headers.update(UA)
csv.field_size_limit(10**9)

PD = "US government work - public domain"
WHO_LIC = "WHO - CC BY-NC-SA 3.0 IGO (non-commercial, attribute, share-alike)"
ICH_LIC = "ICH - copyright ICH; may be reproduced with acknowledgement of ICH copyright"
EU_LIC = "European Union - reuse authorised with acknowledgement (Commission Decision 2011/833/EU)"


# ---------- download helpers ----------
def get(url: str, tries: int = 4, stream_to: Path | None = None) -> bytes | None:
    """GET with retries (EUR-Lex answers 202 + challenge page on first hit); cached by URL hash."""
    key = CACHE / hashlib.sha1(url.encode()).hexdigest()
    if key.exists():
        return key.read_bytes()
    for i in range(tries):
        try:
            r = S.get(url, timeout=300)
            if r.status_code == 200 and not (r.content[:15].lower().startswith(b"<!doctype") and url.lower().endswith("pdf")):
                key.write_bytes(r.content)
                time.sleep(0.4)
                return r.content
        except requests.RequestException:
            pass
        time.sleep(3 * (i + 1))
    return None


def is_pdf(b: bytes | None) -> bool:
    return bool(b) and b[:5] == b"%PDF-"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def slug(s: str, n: int = 60) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:n]


# Some import-alert entries are individuals listed with a national ID number (e.g. Vietnamese CCCD).
ID_RE = re.compile(r"\b(CCCD|CMND|CNIC|CPF|CURP|DNI|AADHAAR|PASSPORT)(\s*(?:NO\.?|NUMBER)?\s*[#:.]?\s*)[A-Z]?\d[\d\-]{4,}", re.I)


def redact_ids(s: str) -> str:
    return ID_RE.sub(r"\1\2[ID REDACTED]", s or "")


# ---------- PDF text + tables (same method as the A and B packs) ----------
CELL = re.compile(r"\s{2,}")


def pdf_pages(path: Path) -> list[str]:
    out = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"], capture_output=True, text=True)
    return out.stdout.split("\f") if out.returncode == 0 else []


def page_text_tables(page: str):
    lines = page.splitlines()
    cells = [CELL.split(l.strip()) if l.strip() else [] for l in lines]
    row = [len(c) >= 3 for c in cells]
    tables, out, i = [], [], 0
    while i < len(lines):
        if row[i]:
            j = i
            while j < len(lines) and (row[j] or (not lines[j].strip() and j + 1 < len(lines) and row[j + 1])):
                j += 1
            block = [cells[k] for k in range(i, j) if cells[k]]
            if len(block) >= 3:
                tables.append(block); out.append("[table]"); out += [" | ".join(r) for r in block]; out.append("[/table]")
                i = j
                continue
        t = re.sub(r"\s+", " ", lines[i]).strip()
        if t:
            out.append(t)
        i += 1
    return "\n".join(out), tables


def html_text(h: str) -> str:
    h = re.sub(r"(?is)<(script|style).*?</\1>", " ", h)
    h = re.sub(r"(?i)<br\s*/?>|</(p|div|h\d|li|tr)>", "\n", h)
    t = html.unescape(re.sub(r"<[^>]+>", " ", h))
    return "\n".join(re.sub(r"[ \t\xa0]+", " ", l).strip() for l in t.splitlines() if l.strip())


# ---------- pack ----------
class Pack:
    def __init__(self, code: str, folder: str, owner: str, purpose: str):
        self.code, self.owner, self.purpose = code, owner, purpose
        self.dir = BASE / folder
        if self.dir.exists():
            subprocess.run(["rm", "-rf", str(self.dir)])
        for d in ("documents", "datasets", "tables"):
            (self.dir / d).mkdir(parents=True, exist_ok=True)
        self.rows, self.corpus, self.n, self.notes = [], [], 0, []

    def _row(self, file, kind, category, title, url, publisher, licence, data: bytes, size_note, tables=0):
        self.rows.append({"file": file, "kind": kind, "category": category, "title": title, "publisher": publisher,
                          "url": url, "licence": licence, "pages_or_rows": size_note, "tables_detected": tables,
                          "sha256": sha(data), "downloaded_on": time.strftime("%Y-%m-%d")})

    def pdf(self, url, category, title, publisher, licence) -> bool:
        data = get(url)
        if not is_pdf(data):
            print(f"  [{self.code}] MISSING pdf: {title} <- {url}")
            self.notes.append(f"Not downloadable at build time: {title} ({url})")
            return False
        self.n += 1
        stem = f"{self.code}{self.n:03d}_{category}_{slug(title, 50)}"
        path = self.dir / "documents" / f"{stem}.pdf"
        path.write_bytes(data)
        pages = pdf_pages(path)
        texts, nt = [], 0
        for p, page in enumerate(pages, 1):
            t, tabs = page_text_tables(page)
            if t:
                texts.append(f"[page {p}]\n{t}")
            for k, tb in enumerate(tabs, 1):
                nt += 1
                with open(self.dir / "tables" / f"{stem}_p{p}_t{k}.csv", "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(tb)
        if sum(len(t) for t in texts) < 500:
            self.notes.append(f"No text layer (scanned) - kept as PDF only: {title}")
        else:
            self.corpus.append({"id": f"doc{self.code}-{stem}", "text": f"{title}\n" + "\n".join(texts),
                                "source": f"{publisher} - {category}", "url": url})
        self._row(f"documents/{stem}.pdf", "document (PDF)", category, title, url, publisher, licence, data,
                  f"{len([p for p in pages if p.strip()])} pages", nt)
        print(f"  [{self.code}] {self.n:3d} pdf {category:14} tables={nt:4} {title[:70]}", flush=True)
        return True

    def html_doc(self, url, category, title, publisher, licence, text: str, raw: bytes):
        self.n += 1
        stem = f"{self.code}{self.n:03d}_{category}_{slug(title, 50)}"
        (self.dir / "documents" / f"{stem}.html").write_bytes(raw)
        (self.dir / "documents" / f"{stem}.txt").write_text(text, encoding="utf-8")
        self.corpus.append({"id": f"doc{self.code}-{stem}", "text": f"{title}\n{text}", "source": f"{publisher} - {category}", "url": url})
        self._row(f"documents/{stem}.html", "document (HTML + .txt)", category, title, url, publisher, licence, raw, f"{len(text)} chars")
        print(f"  [{self.code}] {self.n:3d} html {category:13} {title[:70]}", flush=True)
        return stem

    def table(self, name, rows: list[dict], category, title, url, publisher, licence, note=""):
        if not rows:
            self.notes.append(f"Empty dataset skipped: {title}")
            return
        path = self.dir / "datasets" / f"{name}.csv"
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        self._row(f"datasets/{name}.csv", "dataset (CSV)", category, title + (f" - {note}" if note else ""), url, publisher,
                  licence, path.read_bytes(), f"{len(rows)} rows")
        print(f"  [{self.code}] csv {name} rows={len(rows)}", flush=True)

    def raw(self, name, data: bytes, category, title, url, publisher, licence, note=""):
        (self.dir / "datasets" / name).write_bytes(data)
        self._row(f"datasets/{name}", "dataset (original file)", category, title, url, publisher, licence, data, note)

    def passages(self, docs: list[dict]):
        self.corpus += docs

    def finish(self, readme_extra: str):
        with open(self.dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys())); w.writeheader(); w.writerows(self.rows)
        with open(self.dir / f"corpus_{self.code}.jsonl", "w", encoding="utf-8") as f:
            for d in self.corpus:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        docs = sum(1 for r in self.rows if r["kind"].startswith("document"))
        sets = len(self.rows) - docs
        withtab = sum(1 for r in self.rows if r["kind"].startswith("document") and int(r["tables_detected"]) > 0)
        lic = sorted({r["licence"] for r in self.rows})
        readme = [f"{self.purpose} (owner: {self.owner})", f"Built {time.strftime('%Y-%m-%d')}.", "",
                  "Contents",
                  f"  documents/      {docs} documents (PDF, or HTML with an extracted .txt); {withtab} contain detected tables",
                  f"  datasets/       {sets} tabular datasets (CSV, plus original bulk files where useful)",
                  "  tables/         each table detected in a PDF, as CSV: <document>_p<page>_t<n>.csv",
                  "  manifest.csv    one row per file: category, title, publisher, URL, licence, size, sha256",
                  f"  corpus_{self.code}.jsonl  text for retrieval in the copilot corpus format {{\"id\",\"text\",\"source\",\"url\"}}",
                  "", readme_extra.strip(), "",
                  "Load into the copilot (no code change needed)",
                  f"  cat corpus_{self.code}.jsonl >> <repo>/data/processed/corpus.jsonl",
                  "  python scripts/build_corpus.py --embed <endpoint flags>   # splits new records into passages and embeds them",
                  "", "Licences in this pack"] + [f"  - {l}" for l in lic] + [
                  "  Do not commit WHO / ICH / EU documents or their extracted text to the public repo.", ""]
        if self.notes:
            readme += ["Notes"] + [f"  - {n}" for n in self.notes]
        readme += ["  - Table extraction from PDFs is layout-based: numeric tables are clean, wrapped cells may be split."]
        (self.dir / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
        zp = BASE / f"{self.dir.name}.zip"
        zp.unlink(missing_ok=True)
        subprocess.run(["zip", "-qr", "-X", zp.name, self.dir.name, "-x", "*.DS_Store"], cwd=BASE, check=True)
        print(f"== {zp.name}: {docs} documents, {sets} datasets, corpus records={len(self.corpus)}", flush=True)


# ---------- shared sources ----------
def govinfo_cfr(pack, title_no, vol, part, desc, year=2025):
    url = f"https://www.govinfo.gov/content/pkg/CFR-{year}-title{title_no}-vol{vol}/pdf/CFR-{year}-title{title_no}-vol{vol}-part{part}.pdf"
    pack.pdf(url, "regulation", f"{title_no} CFR Part {part} - {desc}", "US GPO govinfo (CFR annual edition)", PD)


def who_doc(pack, page_slug, title):
    page = get(f"https://www.who.int/publications/m/item/{page_slug}")
    links = re.findall(r"https://cdn\.who\.int/media/docs/[^\"'<> ]+?\.pdf", page.decode("utf-8", "replace")) if page else []
    if not links:
        pack.notes.append(f"WHO page had no PDF link: {title}")
        return
    pack.pdf(html.unescape(links[0]) + "?download=true", "guidance", title, "World Health Organization", WHO_LIC)


def ich(pack, file, title):
    pack.pdf(f"https://database.ich.org/sites/default/files/{file}", "guidance", title, "ICH", ICH_LIC)


def eurlex(pack, celex, title):
    pack.pdf(f"https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:{celex}", "guidance", title,
             "European Commission (EUR-Lex)", EU_LIC)


def openfda_bulk(kind):
    url = f"https://download.open.fda.gov/drug/{kind}/drug-{kind}-0001-of-0001.json.zip"
    data = get(url)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        recs = json.loads(z.read(z.namelist()[0]))["results"]
    return url, data, recs


def j(v):
    return "; ".join(map(str, v)) if isinstance(v, list) else ("" if v is None else str(v))


def drug_match(text: str) -> str:
    t = (text or "").lower()
    for d in sorted(DRUGS, key=len, reverse=True):  # longest first: pegfilgrastim before filgrastim
        if d.lower() in t:
            return d
    return ""


def drugsfda_products():
    url, data, recs = openfda_bulk("drugsfda")
    rows = []
    for a in recs:
        g = j(a.get("openfda", {}).get("generic_name"))
        for p in a.get("products", []):
            ai = "; ".join(f"{x.get('name')} {x.get('strength')}" for x in p.get("active_ingredients", []))
            d = drug_match(ai + " " + g)
            if not d:
                continue
            rows.append({"drug": d, "application_number": a.get("application_number"), "sponsor_name": a.get("sponsor_name"),
                         "brand_name": p.get("brand_name"), "active_ingredients": ai, "dosage_form": p.get("dosage_form"),
                         "route": p.get("route"), "marketing_status": p.get("marketing_status"),
                         "te_code": p.get("te_code"), "reference_drug": p.get("reference_drug"),
                         "reference_standard": p.get("reference_standard"), "product_number": p.get("product_number")})
    return url, data, rows


def purple_book():
    url = "https://www.accessdata.fda.gov/drugsatfda_docs/PurpleBook/2026/purplebook-search-October-data-download.csv"
    data = get(url)
    text = data.decode("utf-8-sig", "replace").splitlines()
    start = next((i for i, l in enumerate(text) if "Proper Name" in l or "BLA Number" in l), 0)  # title lines precede the header
    rows = list(csv.DictReader(text[start:]))
    return url, data, rows


def establishments():
    url = "https://www.accessdata.fda.gov/cder/drls_reg.zip"
    data = get(url)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        txt = z.read("drls_reg.txt").decode("latin-1")
    keep = ["FEI_NUMBER", "DUNS_NUMBER", "FIRM_NAME", "ADDRESS", "EXPIRATION_DATE", "OPERATIONS", "REGISTRANT_NAME", "REGISTRANT_DUNS"]
    rows = []
    for r in csv.DictReader(io.StringIO(txt), delimiter="\t"):
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        addr = r.get("ADDRESS", "")
        m = re.search(r",\s*([^,]+?)\s*\(([A-Z]{3})\)\s*$", addr)
        out = {k: redact_ids(r.get(k, "")) for k in keep}
        out["COUNTRY"], out["COUNTRY_CODE"] = (m.group(1), m.group(2)) if m else ("", "")
        rows.append(out)
    return url, rows


def ndc_products():
    url = "https://www.accessdata.fda.gov/cder/ndctext.zip"
    data = get(url)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        prod = z.read("product.txt").decode("latin-1")
        pkg = z.read("package.txt").decode("latin-1")
    products = [r for r in csv.DictReader(io.StringIO(prod), delimiter="\t")
                if drug_match((r.get("NONPROPRIETARYNAME") or "") + " " + (r.get("SUBSTANCENAME") or ""))]
    ids = {r["PRODUCTID"] for r in products}
    packages = [r for r in csv.DictReader(io.StringIO(pkg), delimiter="\t") if r.get("PRODUCTID") in ids]
    for r in products:
        r["drug"] = drug_match((r.get("NONPROPRIETARYNAME") or "") + " " + (r.get("SUBSTANCENAME") or ""))
    return url, data, products, packages


def row_doc(prefix, i, title, fields: dict, source, url):
    body = "; ".join(f"{k}: {v}" for k, v in fields.items() if v not in ("", None))
    return {"id": f"{prefix}-{i}", "text": f"{title}\n{body}", "source": source, "url": url}


# =====================================================================
def pack_a():
    p = Pack("A", "UseCaseA_ColdChain_GDP_Guidance_and_Data", "Siddhesh",
             "Use case A - Cold-chain excursion triage and GDP release: WHO / EU GDP guidance, stability guidelines, core US rules")
    who_doc(p, "trs961-annex9-modelguidanceforstoragetransport", "WHO TRS 961 Annex 9 - Model guidance for the storage and transport of time- and temperature-sensitive pharmaceutical products")
    titles = {"a": "2: Design and procurement of storage facilities", "b": "3: Estimating the capacity of storage facilities",
              "c": "4: Building security and fire protection", "d": "5: Maintenance of storage facilities",
              "e": "6: Temperature and humidity monitoring systems for fixed storage areas", "f": "7: Qualification of temperature-controlled storage areas",
              "g": "8: Temperature mapping of storage areas", "h": "9: Maintenance of refrigeration equipment",
              "i": "10: Checking the accuracy of temperature control and monitoring devices", "j": "11: Qualification of refrigerated road vehicles",
              "k": "12: Temperature-controlled transport operations by road and by air", "l": "13: Qualification of shipping containers",
              "m": "14: Transport route profiling qualification", "n": "15: Temperature and humidity monitoring systems for transport operations",
              "o": "16: Environmental management of refrigeration equipment"}
    for s, t in titles.items():
        who_doc(p, f"Annex-9-{s}-trs-961", f"WHO TRS 961 Annex 9 Supplement {t}")
    eurlex(p, "52013XC1123(01)", "EU Guidelines of 5 November 2013 on Good Distribution Practice of medicinal products for human use (2013/C 343/01)")
    ich(p, "Q1A%28R2%29%20Guideline.pdf", "ICH Q1A(R2) Stability testing of new drug substances and products")
    ich(p, "Q1E%20Guideline.pdf", "ICH Q1E Evaluation for stability data")
    ich(p, "Q5C%20Guideline.pdf", "ICH Q5C Stability testing of biotechnological/biological products")
    govinfo_cfr(p, 21, 4, 211, "cGMP for finished pharmaceuticals (211.142 warehousing, 211.150 distribution, 211.166 stability)")
    govinfo_cfr(p, 21, 7, 600, "Biological products: general (600.15 temperatures during shipment)")
    govinfo_cfr(p, 21, 7, 610, "General biological products standards (610.53 dating periods and storage)")
    govinfo_cfr(p, 21, 4, 203, "Prescription drug marketing (storage and handling of samples)")
    p.finish("""
What this adds to UseCaseA_ColdChain_GDP_PDFs.zip (labels, NIST studies, US rules)
  - WHO TRS 961 Annex 9 and its technical supplements 2-16 (the WHO references in SOP-LOG-014 and the Blueprint)
  - EU GDP guidelines 2013/C 343/01 (chapters on operations and transportation)
  - ICH stability guidelines (Q1A, Q1E, Q5C) used to judge excursion impact against stability data
  - 21 CFR 211, 600, 610, 203 (so this pack is complete without the B pack)
  Eval question A05 (WHO TRS 961 Annex 9, temperature mapping) needs the Annex 9 documents in the corpus.
""")


def pack_b():
    p = Pack("B", "UseCaseB_Supplier_Qualification_Guidance_and_Data", "Shrilesh",
             "Use case B - Alternate API / generic supplier qualification: ICH and EU guidance plus supplier/equivalence datasets")
    for f, t in [("Q7%20Guideline.pdf", "ICH Q7 Good manufacturing practice guide for active pharmaceutical ingredients"),
                 ("ICH_Q9%28R1%29_Guideline_Step4_2022_1219.pdf", "ICH Q9(R1) Quality risk management"),
                 ("Q10%20Guideline.pdf", "ICH Q10 Pharmaceutical quality system"),
                 ("Q11%20Guideline.pdf", "ICH Q11 Development and manufacture of drug substances"),
                 ("Q3A%28R2%29%20Guideline.pdf", "ICH Q3A(R2) Impurities in new drug substances"),
                 ("Q3B%28R2%29%20Guideline.pdf", "ICH Q3B(R2) Impurities in new drug products"),
                 ("Q5E%20Guideline.pdf", "ICH Q5E Comparability of biotechnological/biological products after manufacturing changes")]:
        ich(p, f, t)
    eurlex(p, "52015XC0321(01)", "EU Guidelines of 19 March 2015 on principles of Good Distribution Practice of active substances (2015/C 95/01)")
    url, data, rows = drugsfda_products()
    p.table("drugsfda_products_project_drugs", rows, "dataset", "Drugs@FDA products for the project drugs (sponsor, TE code, reference drug, marketing status)",
            url, "FDA via openFDA", PD)
    p.passages([row_doc("drugsfdaB", i, f"Approved product: {r['brand_name']} ({r['drug']}) - {r['sponsor_name']} {r['application_number']}",
                        r, "openFDA Drugs@FDA products", url) for i, r in enumerate(rows)])
    url, data, rows = purple_book()
    p.raw("purplebook_october_2026.csv", data, "dataset", "FDA Purple Book (licensed biologics, biosimilars, interchangeability)", url, "FDA", PD, f"{len(rows)} rows")
    hits = [r for r in rows if drug_match(" ".join(r.values()))]
    p.passages([row_doc("purplebookB", i, f"Purple Book: {r.get('Proprietary Name', '')} {r.get('Proper Name', '')}", r, "FDA Purple Book", url)
                for i, r in enumerate(hits)])
    url, rows = establishments()
    p.table("fda_drug_establishments_registered", rows, "dataset", "FDA registered drug establishments (operations incl. API MANUFACTURE, country)",
            url, "FDA (DECRS)", PD, note="contact names, e-mails and phone numbers removed")
    api = [r for r in rows if "API MANUFACTURE" in r["OPERATIONS"]]
    p.passages([row_doc("estB", i, f"Registered establishment: {r['FIRM_NAME']} ({r['COUNTRY']})", r, "FDA drug establishment registration", url)
                for i, r in enumerate(api)])
    p.finish("""
What this adds to UseCaseB_Supplier_Qualification_PDFs.zip (CFR, quality reviews, approval letters, labels)
  - ICH Q7 (API GMP), Q9(R1), Q10, Q11, Q3A/Q3B (impurities), Q5E (comparability) - the course list's B guidance
  - EU GDP for active substances (2015/C 95/01)
  - datasets: Drugs@FDA products with TE codes and reference drugs for the project drugs; the FDA Purple Book;
    FDA registered drug establishments (API manufacturers, country, operations; personal contact columns removed)
  corpus_B.jsonl holds document text, one record per product, per Purple Book entry for project drugs,
  and per registered API manufacturing site.
""")


def import_alerts():
    lst = get("https://www.accessdata.fda.gov/cms_ia/ialist.html").decode("utf-8", "replace")
    rows = re.findall(r'<td>(\d{2}-\d{2})</td>\s*<td>[^<]*</td>\s*<td>([^<]*)</td>\s*<td><a href="(importalert_\d+\.html)">(.*?)</a>', lst, re.S)
    want = [r for r in rows if r[0].startswith("66-") or r[0] in ("54-18", "55-03", "55-05", "57-22", "99-34")]
    return [(n, d, u, html.unescape(re.sub(r"\s+", " ", t)).strip(' "*\u201c\u201d')) for n, d, u, t in want]


ENTRY = re.compile(
    r'<h4>(?P<country>[^<]+)</h4>'
    r'|<div class="div-name floatleft">(?P<firm>.*?)</div>\s*<div class="div-name floatright">\s*Date Published\s*:\s*(?P<fdate>[^<]*)</div>\s*<div class="clear">(?P<addr>.*?)</div>'
    r'|<div class="floatleft"><em>(?P<code>.*?)</em>(?P<prod>.*?)(?:<br>)?\s*</div>\s*<div class="floatright marginsmall">\s*Date Published:\s*(?P<pdate>[^<]*)</div>', re.S)


def clean(s):
    return redact_ids(re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip())


def pack_c():
    p = Pack("C", "UseCaseC_Customs_Import_PreClearance", "Swapnesh",
             "Use case C - Customs and import pre-clearance: FDA import alerts, import/customs rules, tariff and establishment data")
    alerts = import_alerts()
    firm_rows = []
    for num, date, page, title in alerts:
        url = f"https://www.accessdata.fda.gov/cms_ia/{page}"
        raw = get(url)
        if not raw:
            p.notes.append(f"Import alert not downloadable: {num}")
            continue
        h = redact_ids(raw.decode("utf-8", "replace"))
        raw = h.encode("utf-8")
        cut = re.search(r"List of firms", h, re.I)
        header = html_text(h[: cut.start()] if cut else h)
        start = header.find("Import Alert")
        header = header[start:] if start >= 0 else header
        country = firm = addr = ""
        n_alert = 0
        for m in ENTRY.finditer(h[cut.start():] if cut else ""):
            if m.group("country"):
                country = clean(m.group("country"))
            elif m.group("firm"):
                firm, addr = clean(m.group("firm")), clean(m.group("addr"))
                fdate = clean(m.group("fdate"))
            elif m.group("code") is not None and firm:
                n_alert += 1
                firm_rows.append({"import_alert": num, "alert_title": title, "country": country, "firm": firm, "address": addr,
                                  "firm_date_published": fdate, "product_code": clean(m.group("code")).replace(" ", ""),
                                  "product": clean(m.group("prod")), "product_date_published": clean(m.group("pdate")), "source_url": url})
        p.html_doc(url, "import_alert", f"FDA Import Alert {num} (revised {date}) - {title}", "FDA (ORA import alerts)", PD,
                   f"{header}\n\nFirms/products listed on this alert: {n_alert} (see datasets/import_alert_firms.csv)", raw)
    p.table("import_alert_firms", firm_rows, "dataset", "Firms and products on FDA drug import alerts (country, firm, product code, dates)",
            "https://www.accessdata.fda.gov/cms_ia/ialist.html", "FDA (ORA import alerts)", PD)
    groups = {}
    for r in firm_rows:
        groups.setdefault((r["import_alert"], r["country"]), []).append(r)
    p.passages([{"id": f"iafirmsC-{a}-{slug(c, 30)}", "url": rows[0]["source_url"], "source": "FDA import alert firm list",
                 "text": f"Import alert {a} ({rows[0]['alert_title']}) - firms in {c}\n" + "\n".join(
                     f"{r['firm']}, {r['address']}: {r['product_code']} {r['product']} (published {r['product_date_published']})" for r in rows)}
                for (a, c), rows in groups.items()])
    govinfo_cfr(p, 21, 1, 1, "General enforcement regulations (Subpart E: imports and exports)")
    govinfo_cfr(p, 21, 4, 207, "Establishment registration and drug listing")
    govinfo_cfr(p, 21, 4, 251, "Section 804 importation program")
    govinfo_cfr(p, 21, 5, 314, "Applications for FDA approval (314.410 imports and exports of new drugs)")
    govinfo_cfr(p, 19, 1, 12, "Customs: special classes of merchandise")
    govinfo_cfr(p, 19, 2, 141, "Customs: entry of merchandise")
    govinfo_cfr(p, 19, 2, 151, "Customs: examination, sampling and testing of merchandise")
    who_doc(p, "Annex-9-k-trs-961", "WHO TRS 961 Annex 9 Supplement 12: Temperature-controlled transport operations by road and by air")
    hts = []
    for lo, hi in (("2901", "2942"), ("3001", "3006")):
        u = f"https://hts.usitc.gov/reststop/exportList?from={lo}&to={hi}&format=CSV&styles=false"
        data = get(u)
        if data and len(data) > 200:
            hts += list(csv.DictReader(io.StringIO(data.decode("utf-8-sig", "replace"))))
    p.table("hts_tariff_chapters_29_30", hts, "dataset", "US Harmonized Tariff Schedule chapters 29 (organic chemicals/APIs) and 30 (pharmaceutical products)",
            "https://hts.usitc.gov/", "US International Trade Commission", PD)
    p.passages([row_doc("htsC", i, f"HTS {r.get('HTS Number')}: {r.get('Description', '')[:120]}", r, "USITC Harmonized Tariff Schedule", "https://hts.usitc.gov/")
                for i, r in enumerate(hts) if r.get("General Rate of Duty") or r.get("Unit of Quantity")])
    url, rows = establishments()
    p.table("fda_drug_establishments_registered", rows, "dataset", "FDA registered drug establishments (foreign site registration check before import)",
            url, "FDA (DECRS)", PD, note="contact names, e-mails and phone numbers removed")
    p.finish("""
  - FDA drug import alerts (all chapter 66 drug alerts, plus 54-18, 55-03 heparin, 55-05, 57-22, 99-34): alert text
    (reason, guidance, charge) as HTML + .txt, and every listed firm/product as rows in datasets/import_alert_firms.csv
    (national ID numbers of individuals listed on alert 99-34 are masked as [ID REDACTED]; saved pages are the masked copies)
  - 21 CFR 1 (imports), 207, 251, 314; 19 CFR 12, 141, 151 (customs entry and examination)
  - WHO TRS 961 Annex 9 Supplement 12 (temperature-controlled transport by road and air)
  - US Harmonized Tariff Schedule chapters 29 and 30; FDA registered drug establishments (personal contacts removed)
  corpus_C.jsonl: document text, firm lists grouped per alert and country, tariff lines.
  Alerts 66-72 and 66-80 list no firms on their pages, so they have alert text only.
""")


def pack_d():
    p = Pack("D", "UseCaseD_Shortage_Therapeutic_Substitution", "Sandeep",
             "Use case D - Drug shortage and therapeutic substitution: shortages, prices, equivalence and product data")
    url, data, recs = openfda_bulk("shortages")
    p.raw("openfda_drug_shortages.json.zip", data, "dataset", "openFDA drug shortages (bulk, all records)", url, "FDA via openFDA", PD, f"{len(recs)} records")
    rows = []
    for r in recs:
        o = r.get("openfda", {})
        rows.append({"generic_name": r.get("generic_name"), "proprietary_name": j(r.get("proprietary_name") or o.get("brand_name")),
                     "company_name": r.get("company_name"), "status": r.get("status"), "availability": r.get("availability"),
                     "shortage_reason": r.get("shortage_reason"), "dosage_form": r.get("dosage_form"), "presentation": r.get("presentation"),
                     "package_ndc": r.get("package_ndc"), "therapeutic_category": j(r.get("therapeutic_category")),
                     "related_info": r.get("related_info"), "resolved_note": r.get("resolved_note"),
                     "initial_posting_date": r.get("initial_posting_date"), "update_type": r.get("update_type"),
                     "update_date": r.get("update_date"), "discontinued_date": r.get("discontinued_date")})
    p.table("drug_shortages_all", rows, "dataset", "FDA drug shortages, one row per presentation", url, "FDA via openFDA", PD)
    p.passages([row_doc("shortageD", i, f"Drug shortage: {r['generic_name']} - {r['company_name']} ({r['status']})", r, "openFDA drug shortages", url)
                for i, r in enumerate(rows)])
    # NADAC: the full 2026 file is 1.4M rows; keep rows for the project drugs
    nurl = "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost-11-25-2026.csv"
    keys = sorted(((d, d.upper() if d.startswith("insulin") else d.upper()[:12]) for d in DRUGS), key=lambda x: -len(x[1]))
    nrows = []
    with S.get(nurl, stream=True, timeout=600) as r:
        lines = (l.decode("utf-8", "replace") for l in r.iter_lines())
        for row in csv.DictReader(lines):
            desc = (row.get("NDC Description") or "").upper()
            hit = next((d for d, k in keys if k in desc), "")
            if hit:
                row["drug"] = hit
                nrows.append(row)
    p.table("nadac_2026_project_drugs", nrows, "dataset", "CMS NADAC 2026 weekly prices for the project drugs (per unit)", nurl, "CMS (data.medicaid.gov)", PD,
            note="filtered from the full 2026 file (1.4M rows) by drug name")
    iso = lambda s: f"{s[6:10]}-{s[0:2]}-{s[3:5]}" if len(s or "") == 10 else (s or "")
    latest = {}
    for r in nrows:
        k = r["NDC"]
        if k not in latest or iso(r["Effective Date"]) > iso(latest[k]["Effective Date"]):
            latest[k] = r
    p.passages([row_doc("nadacD", i, f"NADAC price: {r['NDC Description']} (NDC {r['NDC']})", r, "CMS NADAC", nurl) for i, r in enumerate(latest.values())])
    curl = "https://download.medicaid.gov/data/nadac-comparison-11-25-2026.csv"
    comp = [r for r in csv.DictReader(io.StringIO(get(curl).decode("utf-8-sig", "replace"))) if drug_match(r.get("NDC Description", ""))]
    p.table("nadac_price_changes_project_drugs", comp, "dataset", "CMS NADAC price changes (old vs new rate) for the project drugs", curl, "CMS (data.medicaid.gov)", PD)
    url, data, rows = drugsfda_products()
    p.table("drugsfda_products_te_codes", rows, "dataset", "Drugs@FDA products with therapeutic-equivalence (TE) codes for substitution", url, "FDA via openFDA", PD)
    p.passages([row_doc("teD", i, f"Equivalence: {r['brand_name']} ({r['drug']}) TE {r['te_code'] or '-'}", r, "openFDA Drugs@FDA products", url)
                for i, r in enumerate(rows)])
    url, data, rows = purple_book()
    p.raw("purplebook_october_2026.csv", data, "dataset", "FDA Purple Book (biosimilars and interchangeable products)", url, "FDA", PD, f"{len(rows)} rows")
    url, data, prods, pkgs = ndc_products()
    p.table("ndc_products_project_drugs", prods, "dataset", "FDA NDC directory products for the project drugs (labeler, form, strength)", url, "FDA", PD)
    p.table("ndc_packages_project_drugs", pkgs, "dataset", "FDA NDC directory packages for the project drugs", url, "FDA", PD)
    govinfo_cfr(p, 21, 5, 314, "Applications for FDA approval (314.81 discontinuance / interruption reporting; ANDA equivalence)")
    govinfo_cfr(p, 21, 5, 320, "Bioavailability and bioequivalence requirements")
    p.finish("""
  - openFDA drug shortages (bulk) and a flat CSV, one row per presentation (status, reason, resupply dates)
  - CMS NADAC 2026 prices and price changes for the project drugs (for cost-delta tools)
  - Drugs@FDA products with TE codes and reference drugs; FDA Purple Book (biosimilar interchangeability)
  - FDA NDC directory products and packages for the project drugs
  - 21 CFR 314 (shortage notification, ANDA) and 320 (bioequivalence)
  corpus_D.jsonl: one record per shortage, per latest NADAC price per NDC, per approved product.
  NADAC prices retail-pharmacy products, so it covers 10 of the 20 project drugs (no oncology infusions or
  most hospital biologics); use Drugs@FDA / NDC data for those.
""")


def faers(drug):
    q = f'patient.drug.openfda.generic_name:"{drug}"'
    out = {}
    for name, field, lim in (("reactions", "patient.reaction.reactionmeddrapt.exact", 100),
                             ("lots", "patient.drug.drugbatchnumb.exact", 100),
                             ("serious", "serious", 10)):
        r = S.get("https://api.fda.gov/drug/event.json", params={"search": q, "count": field, "limit": lim}, timeout=120)
        out[name] = r.json().get("results", []) if r.status_code == 200 else []
        time.sleep(0.4)
    return out


def pack_e():
    p = Pack("E", "UseCaseE_Pharmacovigilance_Recall_Triage", "Sandeep",
             "Use case E - Post-market pharmacovigilance and lot-specific recall triage: recalls, adverse-event summaries, rules")
    url, data, recs = openfda_bulk("enforcement")
    p.raw("openfda_drug_enforcement.json.zip", data, "dataset", "openFDA drug enforcement reports (all recalls, bulk)", url, "FDA via openFDA", PD, f"{len(recs)} records")
    rows = [{"recall_number": r.get("recall_number"), "classification": r.get("classification"), "status": r.get("status"),
             "recall_initiation_date": r.get("recall_initiation_date"), "report_date": r.get("report_date"),
             "termination_date": r.get("termination_date"), "recalling_firm": r.get("recalling_firm"), "country": r.get("country"),
             "voluntary_mandated": r.get("voluntary_mandated"), "product_type": r.get("product_type"),
             "product_description": r.get("product_description"), "code_info": r.get("code_info"),
             "product_quantity": r.get("product_quantity"), "reason_for_recall": r.get("reason_for_recall"),
             "distribution_pattern": r.get("distribution_pattern"), "generic_name": j(r.get("openfda", {}).get("generic_name"))} for r in recs]
    p.table("drug_recalls_all", rows, "dataset", "FDA drug recalls, one row per recall (class, firm, lots in code_info, reason)", url, "FDA via openFDA", PD)
    sel = [r for r in rows if r["classification"] == "Class I" or drug_match(f"{r['product_description']} {r['generic_name']}")]
    p.passages([row_doc("recallE", r["recall_number"], f"Recall {r['recall_number']} ({r['classification']}): {r['recalling_firm']}", r,
                        "openFDA drug enforcement", url) for r in sel])
    reac, lots, summ = [], [], []
    for d in DRUGS:
        f = faers(d)
        reac += [{"drug": d, "reaction": x["term"], "reports": x["count"]} for x in f["reactions"]]
        lots += [{"drug": d, "lot_or_batch": x["term"], "reports": x["count"]} for x in f["lots"]]
        serious = {("serious" if str(x["term"]) == "1" else "non-serious"): x["count"] for x in f["serious"]}
        summ.append({"drug": d, **serious})
        top = ", ".join(f"{x['term']} ({x['count']})" for x in f["reactions"][:20])
        toplots = ", ".join(f"{x['term']} ({x['count']})" for x in f["lots"][:20])
        p.passages([{"id": f"faersE-{slug(d)}", "url": "https://open.fda.gov/apis/drug/event/", "source": "openFDA FAERS (count summaries)",
                     "text": f"FAERS adverse event summary: {d}\nTop reported reactions: {top}\nMost reported lot/batch numbers (any drug on the report): {toplots}\n"
                             f"Seriousness: {serious}"}])
        print(f"  [E] FAERS {d}: reactions={len(f['reactions'])} lots={len(f['lots'])}", flush=True)
    api = "https://api.fda.gov/drug/event.json"
    p.table("faers_top_reactions_project_drugs", reac, "dataset", "FAERS top 100 reported reactions per project drug (report counts)", api, "FDA via openFDA", PD)
    p.table("faers_lot_numbers_project_drugs", lots, "dataset", "FAERS most reported lot/batch numbers per project drug", api, "FDA via openFDA", PD,
            note="lot counts cover every drug on the report, not only the named drug")
    p.table("faers_summary_project_drugs", summ, "dataset", "FAERS serious vs non-serious report counts per project drug", api, "FDA via openFDA", PD)
    url, data, prods, pkgs = ndc_products()
    p.table("ndc_products_project_drugs", prods, "dataset", "FDA NDC directory products (NDC to labeler mapping) for the project drugs", url, "FDA", PD)
    p.table("ndc_packages_project_drugs", pkgs, "dataset", "FDA NDC directory packages for the project drugs", url, "FDA", PD)
    govinfo_cfr(p, 21, 1, 7, "Enforcement policy (Subpart C: recalls - classification, strategy, effectiveness checks)")
    govinfo_cfr(p, 21, 5, 310, "New drugs (310.305 postmarketing reporting of adverse drug experiences)")
    govinfo_cfr(p, 21, 5, 314, "Applications for FDA approval (314.80 postmarketing adverse drug experience reporting, 314.81 field alerts)")
    govinfo_cfr(p, 21, 7, 600, "Biological products: general (600.14 deviations, 600.80 adverse experience reporting)")
    govinfo_cfr(p, 21, 4, 211, "cGMP for finished pharmaceuticals (211.192 investigations, 211.198 complaint files)")
    ich(p, "E2A_Guideline.pdf", "ICH E2A Clinical safety data management: definitions and standards for expedited reporting")
    ich(p, "E2D_Guideline.pdf", "ICH E2D Post-approval safety data management")
    ich(p, "E2E_Guideline.pdf", "ICH E2E Pharmacovigilance planning")
    ich(p, "ICH_Q9%28R1%29_Guideline_Step4_2022_1219.pdf", "ICH Q9(R1) Quality risk management")
    p.finish("""
  - openFDA drug enforcement (all drug recalls, bulk) and a flat CSV with lot numbers (code_info), class, firm, reason
  - FAERS summaries for the project drugs: top reactions, most reported lot/batch numbers, serious vs non-serious counts
    (the full FAERS bulk download is 116 GB, so count queries are used instead)
  - FDA NDC directory products and packages (NDC to labeler mapping)
  - 21 CFR 7 (recalls), 310, 314, 600 (adverse event reporting), 211 (investigations, complaints)
  - ICH E2A, E2D, E2E (pharmacovigilance) and Q9(R1) (risk management)
  corpus_E.jsonl: document text, one record per Class I recall and per recall of a project drug, one FAERS summary per drug.
""")


if __name__ == "__main__":
    for f in sys.argv[2:] or ["a", "b", "c", "d", "e"]:
        print(f"===== pack {f.upper()}", flush=True)
        {"a": pack_a, "b": pack_b, "c": pack_c, "d": pack_d, "e": pack_e}[f]()
