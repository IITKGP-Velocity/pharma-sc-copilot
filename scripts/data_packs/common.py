"""Shared helpers for the data-pack builders: paths, and the Drugs@FDA document list per drug."""
import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("PACKS_WORK", ROOT / "data" / "raw" / "packs_work"))  # git-ignored download cache
DEFAULT_OUT = ROOT / "data" / "raw" / "packs"
DRUG_FILE = ROOT / "configs" / "drug_names.txt"


def project_drugs() -> list[str]:
    return [n.strip() for n in DRUG_FILE.read_text(encoding="utf-8").splitlines() if n.strip() and not n.startswith("#")]


def drugsfda_docs(names: list[str], cache_name: str) -> list[list]:
    """Rows [drug, prefix, application, sponsor, doc type, submission type, date, url] for every document
    openFDA Drugs@FDA links for these generic names (ANDA, NDA, BLA). Cached in WORK/<cache_name>.json."""
    cache = WORK / f"{cache_name}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rows = []
    for n in names:
        for prefix in ("ANDA", "NDA", "BLA"):
            r = requests.get("https://api.fda.gov/drug/drugsfda.json", timeout=60,
                             params={"search": f'openfda.generic_name:"{n}" AND application_number:{prefix}*', "limit": 100})
            for app in (r.json().get("results", []) if r.status_code == 200 else []):
                for s in app.get("submissions", []):
                    for a in s.get("application_docs") or []:
                        rows.append([n, prefix, app["application_number"], app.get("sponsor_name"), a.get("type"),
                                     s.get("submission_type"), a.get("date"), a.get("url")])
            time.sleep(0.25)
    uniq = list({r[-1]: r for r in rows}.values())
    WORK.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(uniq))
    return uniq
