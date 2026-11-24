#!/usr/bin/env python3
"""
build_eval_set.py - Turn the Deliverable 1 Word file into a machine-readable eval set.

Why: the 64 gold queries live in a .docx table. The harness needs them as JSONL so every
run reads the same file, and so the file can be hashed and frozen (Week 1 gate).

Usage (from the repo root):
  python EVAL/build_eval_set.py "path/to/Track_B_D1_Task_Decomposition_and_Eval_Set.docx"

Output: EVAL/eval_set_v1.jsonl (one query per line) + its SHA-256 printed to the console.
Uses only the standard library (a .docx is just a zip of XML files).
"""
import hashlib
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
OUT_FILE = Path(__file__).resolve().parent / "eval_set_v1.jsonl"

# Type labels used in the Word table -> full names used in the rest of the project.
TYPE_NAMES = {"ANS": "ANSWER", "CALC": "CALC", "ESC": "ESCALATE", "REF": "REFUSE",
              "ABS": "ABSTAIN", "PI": "PI", "PII": "PII", "OOS": "OOS"}


def cell_text(cell) -> str:
    """Join all text runs in a table cell, one line per paragraph."""
    paragraphs = []
    for p in cell.iter(f"{W}p"):
        paragraphs.append("".join(t.text or "" for t in p.iter(f"{W}t")))
    return " ".join(x.strip() for x in paragraphs if x.strip())


def read_rows(docx_path: str) -> list[list[str]]:
    """Return every table row in the document as a list of cell strings."""
    with zipfile.ZipFile(docx_path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    rows = []
    for tr in root.iter(f"{W}tr"):
        rows.append([cell_text(tc) for tc in tr.findall(f"{W}tc")])
    return rows


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)

    items = []
    for row in read_rows(sys.argv[1]):
        # Eval rows have exactly 5 cells: ID, Case, Query, Type, Gold answer.
        # The ID looks like "A01" (letter A-E + two digits); this skips every other table.
        if len(row) == 5 and len(row[0]) == 3 and row[0][0] in "ABCDE" and row[0][1:].isdigit():
            qid, case, query, qtype, gold = row
            items.append({"id": qid, "use_case": case, "query": query,
                          "type": TYPE_NAMES.get(qtype, qtype), "gold": gold})

    # Sanity checks: the document says 64 items and IDs must be unique.
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "duplicate IDs found"
    print(f"Parsed {len(items)} items (document states 64)")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("By type    :", dict(Counter(i["type"] for i in items)))
    print("By use case:", dict(Counter(i["use_case"] for i in items)))
    # This hash goes into the freeze record (section 9.2 of Deliverable 1).
    print("SHA-256    :", hashlib.sha256(OUT_FILE.read_bytes()).hexdigest())
    print("Wrote      :", OUT_FILE)


if __name__ == "__main__":
    main()
