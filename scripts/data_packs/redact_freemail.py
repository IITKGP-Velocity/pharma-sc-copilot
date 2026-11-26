"""Mask e-mail addresses at free-mail providers in pack files, refresh manifest sha256, re-zip. Usage: redact_freemail.py <base> <folder>..."""
import csv, hashlib, re, subprocess, sys
from pathlib import Path

FREE = re.compile(r"\b[A-Za-z0-9._%+-]+@(?:gmail|googlemail|yahoo|ymail|hotmail|outlook|live|msn|aol|gmx|web|mail|icloud|me|"
                  r"qq|163|126|sina|yandex|rediffmail|protonmail|zoho)\.[A-Za-z.]{2,6}\b", re.I)
base = Path(sys.argv[1])
for folder in sys.argv[2:]:
    d = base / folder
    changed = 0
    for f in list(d.glob("datasets/*.csv")) + list(d.glob("corpus_*.jsonl")) + list(d.glob("documents/*.txt")) + list(d.glob("documents/*.html")):
        t = f.read_text(encoding="utf-8", errors="replace")
        n = FREE.sub("[EMAIL REDACTED]", t)
        if n != t:
            f.write_text(n, encoding="utf-8")
            changed += len(FREE.findall(t))
    rows = list(csv.DictReader(open(d / "manifest.csv", encoding="utf-8")))
    for r in rows:
        r["sha256"] = hashlib.sha256((d / r["file"]).read_bytes()).hexdigest()
    with open(d / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    readme = d / "README.txt"
    note = "  - Personal free-mail addresses (gmail, yahoo, gmx, ...) in FDA records are masked as [EMAIL REDACTED]; official FDA and company contacts are kept as published.\n"
    if "EMAIL REDACTED" not in readme.read_text():
        readme.write_text(readme.read_text() + note)
    z = base / f"{folder}.zip"
    z.unlink(missing_ok=True)
    subprocess.run(["zip", "-qr", "-X", z.name, folder, "-x", "*.DS_Store"], cwd=base, check=True)
    print(folder, "masked", changed)
