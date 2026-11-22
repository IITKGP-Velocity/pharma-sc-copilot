#!/usr/bin/env python3
"""
parse_and_chunk.py  -  PDF parsing + structure-aware chunking for RAG.

Reads manifest.csv (columns: doc_id, bucket, use_case, published_date, optional:
title, authority, jurisdiction, doc_type, version, source_url) and docs/<doc_id>.pdf,
writes index/chunks.jsonl, index/chunk_report.json and (optionally)
index/debug/<doc_id>.txt.

Pipeline
  1. Extract lines with font info (size / bold) in reading order (2-column aware).
  2. Drop learned headers/footers, page numbers, TOC pages.
  3. Detect headings (regex + font cues), keep a heading stack -> section_path.
  4. Build paragraphs from layout (vertical gaps, list markers), de-hyphenate.
  5. Split into sentences, pack into ~target-token chunks that never cross a
     section boundary unless the section is tiny; sentence-level overlap.
  6. Tables extracted separately (markdown-like rows, header repeated per chunk).
  7. Validate / filter / de-duplicate and write a QA report.

Usage
  python parse_and_chunk.py --manifest index/manifest.csv --docs docs \
         --out index/chunks.jsonl --report index/chunk_report.json \
         --debug-dir index/debug [--strict]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import statistics
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf as fitz
from transformers import AutoTokenizer

log = logging.getLogger("chunker")

DEFAULT_SECTION = "General / Introduction"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    model_name: str = "BAAI/bge-small-en-v1.5"

    # chunk sizing (token counts are for the text that gets embedded, incl. prefix)
    target_tokens: int = 350          # soft target
    max_tokens: int = 400             # hard cap (bge max is 512 incl. [CLS]/[SEP])
    min_chunk_tokens: int = 80        # smaller chunks are merged forward/backward
    drop_below_tokens: int = 25       # chunks smaller than this are dropped
    overlap_sentences: int = 1        # sentences repeated at the start of next chunk
    overlap_max_tokens: int = 60      # ...but only if they are this small
    fallback_overlap_tokens: int = 40 # overlap when a single sentence must be cut
    prefix_max_tokens: int = 40       # cap for the "[doc | section path]" prefix

    # page layout
    header_margin: float = 0.07
    footer_margin: float = 0.93
    boilerplate_page_ratio: float = 0.30
    boilerplate_min_pages: int = 3

    # headings
    heading_font_delta: float = 0.8   # size above body size counts as "styled"
    style_heading_delta: float = 2.0  # size above body size => heading even w/o numbering
    max_heading_chars: int = 110
    min_headings_expected: int = 3    # if fewer found, retry without font requirement

    # paragraphs
    para_gap_factor: float = 0.6      # vertical gap (x font size) that starts a new paragraph

    # quality guardrails
    min_alpha_ratio: float = 0.45     # drop text chunks that are mostly digits/symbols
    min_chars_per_page: int = 100     # below this -> "probably scanned" warning
    keep_tables: bool = True

    skip_sections: tuple = (
        "table of contents", "contents", "references", "bibliography",
        "revision history", "document history", "change history", "index",
        "distribution list",
    )
    extra_meta_fields: tuple = (
        "title", "authority", "jurisdiction", "doc_type", "version", "source_url",
    )


# --------------------------------------------------------------------------- #
# Tokenizer helper
# --------------------------------------------------------------------------- #
class Tokens:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name)
        self.tok.model_max_length = 10**9  # silence "sequence too long" warnings

    def len(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def lens(self, texts: list[str]) -> list[int]:
        out: list[int] = []
        for i in range(0, len(texts), 512):
            batch = texts[i:i + 512]
            ids = self.tok(batch, add_special_tokens=False)["input_ids"]
            out.extend(len(x) for x in ids)
        return out

    def split(self, text: str, budget: int) -> list[str]:
        """Cut text into pieces of <= budget tokens by slicing the ORIGINAL string
        with character offsets (decode() would lowercase and mangle the text)."""
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if not ids:
            return []
        budget = max(20, budget)
        step = max(1, budget - self.cfg.fallback_overlap_tokens)
        pieces = []
        for s in range(0, len(ids), step):
            e = min(s + budget, len(ids))
            piece = text[offs[s][0]:offs[e - 1][1]].strip()
            if piece:
                pieces.append(piece)
            if e >= len(ids):
                break
        return pieces


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
_CTRL = re.compile("[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u200b-\u200f\u2060\ufeff\u00ad]")
_LEADERS = re.compile(r"(?:\s?\.){4,}")
_PAGE_NUM = re.compile(r"[-–\s]*(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?[-–\s]*", re.I)


def clean_text(text: str) -> str:
    """NFKC (ligatures, nbsp, full-width chars), drop control/zero-width chars."""
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL.sub("", text)
    return re.sub(r"[ \t\u00a0]+", " ", text).strip()


def normalize_for_freq(text: str) -> str:
    """'Page 1 of 29' and 'Page 2 of 29' -> same signature."""
    return re.sub(r"\d+", "#", " ".join(text.split()).lower())


def join_lines(texts: list[str]) -> str:
    """Join wrapped lines; repair end-of-line hyphenation (manufac- + turing)."""
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if not out:
            out = t
        elif re.search(r"[A-Za-z]{2}-$", out) and t[:1].islower():
            out = out[:-1] + t
        else:
            out += " " + t
    out = _LEADERS.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()


_ABBR = {"e.g", "i.e", "etc", "no", "fig", "sec", "approx", "vs", "cf", "al", "inc",
         "ltd", "co", "dr", "mr", "ms", "st", "min", "max", "ca", "ref", "vol",
         "para", "art", "ch", "annex"}


def split_sentences(text: str) -> list[str]:
    sents, start = [], 0
    for m in re.finditer(r"[.!?][\"'”’)\]]*\s+", text):
        end = m.end()
        nxt = text[end:end + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in "(§[\"“‘"):
            continue
        before = text[start:m.start() + 1].split()
        last = before[-1].lower().rstrip(".") if before else ""
        if m.group()[0] == "." and (last in _ABBR or (len(last) == 1 and last.isalpha())):
            continue
        sents.append(text[start:end].strip())
        start = end
    tail = text[start:].strip()
    if tail:
        sents.append(tail)
    return sents


# --------------------------------------------------------------------------- #
# Page extraction
# --------------------------------------------------------------------------- #
@dataclass
class Line:
    page: int
    text: str
    size: float
    bold: bool
    y0: float
    y1: float
    x0: float
    margin: bool = False
    kind: str = "text"          # "text" | "table"
    rows: list | None = None


def order_blocks(blocks: list[dict], page_w: float) -> list[dict]:
    """Reading order. Detects 2-column pages and reads left column then right,
    flushing at full-width blocks (titles, wide tables)."""
    blocks = sorted(blocks, key=lambda b: (round(b["bbox"][1], 0), b["bbox"][0]))
    mid, wide = page_w / 2, page_w * 0.6
    narrow = [b for b in blocks if (b["bbox"][2] - b["bbox"][0]) < wide]
    left = sum(1 for b in narrow if b["bbox"][2] <= mid + 10)
    right = sum(1 for b in narrow if b["bbox"][0] >= mid - 10)
    if left < 3 or right < 3:
        return blocks

    ordered: list[dict] = []
    L: list[dict] = []
    R: list[dict] = []

    def flush():
        ordered.extend(sorted(L, key=lambda b: b["bbox"][1]))
        ordered.extend(sorted(R, key=lambda b: b["bbox"][1]))
        L.clear()
        R.clear()

    for b in blocks:
        x0, _, x1, _ = b["bbox"]
        if x1 - x0 >= wide:
            flush()
            ordered.append(b)
        elif (x0 + x1) / 2 < mid:
            L.append(b)
        else:
            R.append(b)
    flush()
    return ordered


def read_page(page, pno: int, cfg: Config) -> list[Line]:
    h, w = page.rect.height, page.rect.width
    tables: list[Line] = []
    boxes = []
    if cfg.keep_tables:
        try:
            for t in page.find_tables().tables:
                rows = t.extract()
                if rows and len(rows) >= 2 and len(rows[0]) >= 2:
                    boxes.append(fitz.Rect(t.bbox))
                    tables.append(Line(pno, "", 0.0, False, t.bbox[1], t.bbox[3],
                                       t.bbox[0], kind="table", rows=rows))
        except Exception as e:  # old PyMuPDF or odd page: fall back to plain text
            log.debug("find_tables failed on page %d: %s", pno, e)

    blocks = [b for b in page.get_text("dict")["blocks"]
              if b.get("type") == 0 and b.get("lines")]
    lines: list[Line] = []
    for b in order_blocks(blocks, w):
        for ln in b["lines"]:
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = clean_text("".join(s["text"] for s in ln["spans"]))
            if not text:
                continue
            x0, y0, x1, y1 = ln["bbox"]
            pt = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
            if any(r.contains(pt) for r in boxes):
                continue  # text inside a table is emitted via the table
            size = max(s["size"] for s in spans)
            bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
            margin = y0 <= h * cfg.header_margin or y1 >= h * cfg.footer_margin
            lines.append(Line(pno, text, round(size, 1), bool(bold), y0, y1, x0, margin))

    for t in sorted(tables, key=lambda t: t.y0):
        idx = next((i for i, l in enumerate(lines) if l.y0 >= t.y0), len(lines))
        lines.insert(idx, t)
    return lines


def learn_boilerplate(pages: list[list[Line]], cfg: Config) -> set[str]:
    """Short margin lines that recur on >= ratio of pages (min N pages)."""
    n = len(pages)
    if n < 3:
        return set()
    counts: Counter = Counter()
    for plines in pages:
        seen = set()
        for ln in plines:
            if ln.kind == "text" and ln.margin and len(ln.text) <= 150:
                sig = normalize_for_freq(ln.text)
                if len(sig) >= 3:
                    seen.add(sig)
        counts.update(seen)
    threshold = max(cfg.boilerplate_min_pages, int(n * cfg.boilerplate_page_ratio))
    return {s for s, c in counts.items() if c >= threshold}


_TOC_LEADER = re.compile(r"\.{3,}\s*\d+\s*$|(?:\s\.){3,}\s*\d+\s*$")
_TOC_NUMBERED = re.compile(
    r"^(?:\d+(?:\.\d+)*|chapter\s+\d+|annex\s+\d+|appendix\s+[A-Z]|§\s*\d+\.\d+)\s+.+\s\d{1,3}$", re.I)


def is_toc_page(texts: list[str]) -> bool:
    if len(texts) < 6:
        return False
    n = sum(1 for t in texts if _TOC_LEADER.search(t) or _TOC_NUMBERED.match(t))
    ratio = n / len(texts)
    head = " ".join(texts[:4]).lower()
    return ratio > 0.5 or (ratio > 0.3 and ("table of contents" in head or head.startswith("contents")))


def extract_lines(doc, cfg: Config) -> tuple[list[Line], dict]:
    pages = [read_page(p, i, cfg) for i, p in enumerate(doc, start=1)]
    boiler = learn_boilerplate(pages, cfg)
    lines: list[Line] = []
    toc_pages = 0
    for plines in pages:
        keep = []
        for ln in plines:
            if ln.kind == "text" and ln.margin:
                if len(ln.text) <= 150 and normalize_for_freq(ln.text) in boiler:
                    continue
                if _PAGE_NUM.fullmatch(ln.text):
                    continue
            keep.append(ln)
        if is_toc_page([l.text for l in keep if l.kind == "text"]):
            toc_pages += 1
            continue
        lines.extend(keep)
    chars = sum(len(l.text) for l in lines if l.kind == "text")
    return lines, {"pages": len(pages), "chars": chars, "toc_pages_skipped": toc_pages,
                   "boilerplate_patterns": sorted(boiler)[:20]}


def body_font_size(lines: list[Line]) -> float:
    c: Counter = Counter()
    for l in lines:
        if l.kind == "text" and not l.margin:
            c[l.size] += len(l.text)
    return c.most_common(1)[0][0] if c else 10.0


# --------------------------------------------------------------------------- #
# Heading detection
# --------------------------------------------------------------------------- #
_SEP = r"(?:\s*[-–—:.]\s*|\s+)?"
STRONG_HEADINGS = [
    (re.compile(r"^((?i:subpart)\s+[A-Z]\b" + _SEP + r"(?:[A-Z].*)?)$"), 1),
    (re.compile(r"^(§\s*\d+\.\d+[a-z]?\s+[A-Z].*)$"), 2),
    (re.compile(r"^((?i:chapter)\s+(?:\d+|[IVXL]+)\b" + _SEP + r"(?:[A-Z].*)?)$"), 1),
    (re.compile(r"^((?i:annex|appendix)\s+(?:\d+|[A-Z]|[IVXL]+)\b" + _SEP + r"(?:[A-Z].*)?)$"), 1),
]
NUMBERED = re.compile(r"^((\d+(?:\.\d+){0,4})\.?\s+[A-Z].*)$")
_MODAL = re.compile(r"\b(shall|should|must|will|may not|can be)\b", re.I)


def sentence_like(t: str) -> bool:
    return bool(_MODAL.search(t)) or t.endswith((",", ";")) or len(t.split()) > 18


def classify_heading(ln: Line, body: float, cfg: Config, require_style: bool):
    """Return (level, title) or None."""
    t = ln.text.strip()
    if not (3 <= len(t) <= cfg.max_heading_chars):
        return None
    styled = ln.size >= body + cfg.heading_font_delta or ln.bold

    for rx, level in STRONG_HEADINGS:
        m = rx.match(t)
        if m:
            return None if sentence_like(t) else (level, m.group(1).strip())

    m = NUMBERED.match(t)
    if m and not t.endswith(".") and not sentence_like(t) and len(t.split()) <= 14:
        if require_style and not styled:
            return None
        return min(4, m.group(2).count(".") + 1), t

    if (ln.size >= body + cfg.style_heading_delta and len(t) <= 80
            and not t.endswith((".", ":")) and t[0].isupper() and len(t.split()) <= 12):
        return 1, t
    return None


def section_number(title: str) -> str:
    m = re.match(r"^§\s*(\d+\.\d+[a-z]?)", title)
    if m:
        return m.group(1)
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s", title)
    if m:
        return m.group(1)
    m = re.match(r"^((?i:chapter|annex|appendix|subpart))\s+([A-Za-z0-9]+)", title)
    if m:
        return f"{m.group(1).title()} {m.group(2)}"
    return ""


# --------------------------------------------------------------------------- #
# Paragraph building
# --------------------------------------------------------------------------- #
@dataclass
class Para:
    kind: str                  # "text" | "table"
    section_path: list
    heading: str
    level: int
    page_start: int
    page_end: int
    sec_id: int
    text: str = ""
    rows: list | None = None


_LIST_START = re.compile(r"^(?:[•▪●■◦·\-–—]\s+|\(\w{1,4}\)\s+|\d{1,2}[.)]\s+|[a-z]\)\s+)")


def is_para_break(prev: Line, ln: Line, cfg: Config) -> bool:
    if _LIST_START.match(ln.text):
        return True
    if ln.page == prev.page:
        if ln.y0 - prev.y1 > cfg.para_gap_factor * max(ln.size, 6.0):
            return True
        if ln.y0 < prev.y0 - 2:   # jumped back up: new column
            return True
    return False


class Builder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        alt = "|".join(re.escape(s) for s in cfg.skip_sections)
        self.skip_re = re.compile(rf"^(?:\d+(?:\.\d+)*\.?\s*)?(?:{alt})\s*$", re.I)
        self.stack: list[tuple[int, str]] = []
        self.cur: list[Line] = []
        self.paras: list[Para] = []
        self.skip_level: int | None = None
        self.sec_id = 0
        self.headings = 0

    def _section(self):
        if self.stack:
            return [t for _, t in self.stack], self.stack[-1][1], self.stack[-1][0]
        return [DEFAULT_SECTION], DEFAULT_SECTION, 0

    def flush(self):
        if not self.cur:
            return
        text = join_lines([l.text for l in self.cur])
        if text and self.skip_level is None:
            path, head, level = self._section()
            self.paras.append(Para("text", path, head, level, self.cur[0].page,
                                   self.cur[-1].page, self.sec_id, text))
        self.cur = []

    def add_table(self, ln: Line):
        self.flush()
        if self.skip_level is None:
            path, head, level = self._section()
            self.paras.append(Para("table", path, head, level, ln.page, ln.page,
                                   self.sec_id, rows=ln.rows))

    def heading(self, level: int, title: str):
        self.flush()
        is_skip = bool(self.skip_re.match(title))
        if self.skip_level is not None and level <= self.skip_level and not is_skip:
            self.skip_level = None
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title))
        self.sec_id += 1
        self.headings += 1
        if self.skip_level is None and is_skip:
            self.skip_level = level


def build_paras(lines: list[Line], body: float, cfg: Config, require_style: bool):
    b = Builder(cfg)
    prev: Line | None = None
    for ln in lines:
        if ln.kind == "table":
            b.add_table(ln)
            prev = None
            continue
        h = classify_heading(ln, body, cfg, require_style)
        if h:
            b.heading(*h)
            prev = None
            continue
        if prev is not None and is_para_break(prev, ln, cfg):
            b.flush()
        b.cur.append(ln)
        prev = ln
    b.flush()
    return b.paras, b.headings


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
@dataclass
class Unit:
    text: str
    toks: int
    page_start: int
    page_end: int
    section_path: tuple
    heading: str
    level: int
    sec_id: int
    first: bool
    head_toks: int


@dataclass
class Draft:
    units: list
    n_ov: int = 0
    forced: bool = False       # produced by cutting an oversized sentence


def render(units: list[Unit]) -> str:
    """Join units; if a chunk spans a (small) section boundary, keep the heading inline."""
    parts = []
    for i, u in enumerate(units):
        if i > 0 and u.sec_id != units[i - 1].sec_id:
            parts.append(f"[{u.heading}]")
        parts.append(u.text)
    return " ".join(parts)


def est_tokens(units: list[Unit]) -> int:
    n = sum(u.toks for u in units)
    n += sum(u.head_toks for i, u in enumerate(units) if i and u.sec_id != units[i - 1].sec_id)
    return n


class Chunker:
    def __init__(self, tk: Tokens, cfg: Config, doc_id: str, meta: dict):
        self.tk, self.cfg, self.doc_id, self.meta = tk, cfg, doc_id, meta
        self._prefix: dict = {}
        self.counters = {"c": 0, "t": 0}
        self.records: list[dict] = []

    # ---- prefix / budget ------------------------------------------------- #
    def prefix(self, path) -> tuple[str, int]:
        key = tuple(path) or (DEFAULT_SECTION,)
        if key not in self._prefix:
            label = " > ".join(key)
            p = f"[{self.doc_id} | {label}] "
            if self.tk.len(p) > self.cfg.prefix_max_tokens:
                p = f"[{self.doc_id} | {key[-1][:120]}] "
            self._prefix[key] = (p, self.tk.len(p))
        return self._prefix[key]

    def budget(self, path) -> int:
        return max(60, self.cfg.target_tokens - self.prefix(path)[1])

    # ---- units ------------------------------------------------------------ #
    def make_units(self, paras: list[Para]) -> list[Unit]:
        raw = [(p, s) for p in paras if p.kind == "text" for s in split_sentences(p.text)]
        lens = self.tk.lens([s for _, s in raw])
        head_cache: dict = {}
        units, prev_sec = [], None
        for (p, s), n in zip(raw, lens):
            if p.heading not in head_cache:
                head_cache[p.heading] = self.tk.len(f"[{p.heading}] ")
            units.append(Unit(s, n, p.page_start, p.page_end, tuple(p.section_path),
                              p.heading, p.level, p.sec_id, p.sec_id != prev_sec,
                              head_cache[p.heading]))
            prev_sec = p.sec_id
        return units

    # ---- packing ---------------------------------------------------------- #
    def pack(self, units: list[Unit]) -> list[Draft]:
        cfg = self.cfg
        drafts: list[Draft] = []
        cur: list[Unit] = []
        cur_toks = 0
        n_ov = 0

        def cost(u: Unit) -> int:
            if cur and u.sec_id != cur[-1].sec_id:
                return u.toks + u.head_toks
            return u.toks

        def flush(next_u: Unit | None):
            nonlocal cur, cur_toks, n_ov
            if not cur:
                return
            drafts.append(Draft(cur, n_ov))
            carry: list[Unit] = []
            k = cfg.overlap_sentences
            if k and next_u is not None and len(cur) - n_ov > k:
                tail = cur[-k:]
                if (all(t.section_path == next_u.section_path for t in tail)
                        and sum(t.toks for t in tail) <= cfg.overlap_max_tokens):
                    carry = list(tail)
            cur, cur_toks, n_ov = carry, sum(t.toks for t in carry), len(carry)

        for u in units:
            if u.toks > self.budget(u.section_path):       # single huge sentence
                flush(None)
                cur, cur_toks, n_ov = [], 0, 0
                for piece in self.tk.split(u.text, self.budget(u.section_path) - 2):
                    pu = Unit(piece, self.tk.len(piece), u.page_start, u.page_end,
                              u.section_path, u.heading, u.level, u.sec_id, True, u.head_toks)
                    drafts.append(Draft([pu], 0, forced=True))
                continue

            if cur:
                limit = self.budget(cur[0].section_path)
                new_section = u.sec_id != cur[-1].sec_id
                body_toks = cur_toks - sum(t.toks for t in cur[:n_ov])
                if cur_toks + cost(u) > limit or (new_section and body_toks >= cfg.min_chunk_tokens):
                    flush(u)
                    if cur and cur_toks + u.toks > self.budget(cur[0].section_path):
                        cur, cur_toks, n_ov = [], 0, 0   # drop overlap if it doesn't fit
            c = cost(u)
            cur.append(u)
            cur_toks += c
        flush(None)

        # merge short drafts into the previous one when they fit
        merged: list[Draft] = []
        for d in drafts:
            new = d.units[d.n_ov:]
            if (merged and not d.forced and not merged[-1].forced
                    and est_tokens(new) < cfg.min_chunk_tokens):
                cand = merged[-1].units + new
                if est_tokens(cand) <= self.budget(merged[-1].units[0].section_path):
                    merged[-1].units = cand
                    continue
            merged.append(d)
        return merged

    # ---- record emission -------------------------------------------------- #
    def emit(self, text: str, path, heading: str, level: int, ps: int, pe: int,
             sections: list[str], chunk_type: str = "text"):
        path = tuple(path) or (DEFAULT_SECTION,)
        pre, _ = self.prefix(path)
        embed = pre + text
        pieces = [text]
        if self.tk.len(embed) + 2 > self.cfg.max_tokens:      # hard safety net
            pieces = self.tk.split(text, self.cfg.max_tokens - self.prefix(path)[1] - 4)
        letter = "c" if chunk_type == "text" else "t"
        for piece in pieces:
            embed = pre + piece
            rec = {
                "chunk_id": f"{self.doc_id}_{letter}{self.counters[letter]:04d}",
                "doc_id": self.doc_id,
                "chunk_type": chunk_type,
                "section": heading or path[-1],
                "section_path": " > ".join(path),
                "section_number": section_number(heading or path[-1]),
                "heading_level": level,
                "sections_covered": sections,
                "page_start": ps,
                "page_end": pe,
                "use_case": self.meta.get("use_case", ""),
                "published_date": self.meta.get("published_date", ""),
            }
            for f in self.cfg.extra_meta_fields:
                if self.meta.get(f):
                    rec[f] = self.meta[f]
            rec.update({
                "token_count": self.tk.len(embed),
                "content_hash": hashlib.sha1(" ".join(piece.lower().split()).encode()).hexdigest()[:16],
                "text": piece,
                "embed_text": embed,
            })
            self.counters[letter] += 1
            self.records.append(rec)

    def emit_drafts(self, drafts: list[Draft]):
        for d in drafts:
            u0 = d.units[0]
            sections, last = [], None
            for u in d.units:
                if u.sec_id != last:
                    sections.append(" > ".join(u.section_path))
                    last = u.sec_id
            self.emit(render(d.units), u0.section_path, u0.heading, u0.level,
                      min(u.page_start for u in d.units), max(u.page_end for u in d.units),
                      sections)

    def emit_table(self, p: Para):
        rows = [[" ".join(clean_text(c or "").split()) for c in r] for r in p.rows]
        lines = [" | ".join(r) for r in rows if any(r)]
        if len(lines) < 2:
            return
        header, body = lines[0], lines[1:]
        path = tuple(p.section_path)
        budget = self.budget(path) - self.tk.len(header) - 6
        group: list[str] = []
        total = 0
        for ln, n in zip(body, self.tk.lens(body)):
            if group and total + n > budget:
                self.emit(header + "\n" + "\n".join(group), path, p.heading, p.level,
                          p.page_start, p.page_end, [" > ".join(path)], "table")
                group, total = [], 0
            group.append(ln)
            total += n
        if group:
            self.emit(header + "\n" + "\n".join(group), path, p.heading, p.level,
                      p.page_start, p.page_end, [" > ".join(path)], "table")


# --------------------------------------------------------------------------- #
# Document-level processing
# --------------------------------------------------------------------------- #
def pct(vals: list[int], q: float) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    return s[int(q * (len(s) - 1))]


def debug_dump(paras: list[Para], info: dict, body: float) -> str:
    out = [f"# body_font_size={body}  {json.dumps(info, ensure_ascii=False)}", ""]
    last = None
    for p in paras:
        if p.sec_id != last:
            out.append(f"\n## [L{p.level}] {' > '.join(p.section_path)}")
            last = p.sec_id
        if p.kind == "table":
            out.append(f"[TABLE p.{p.page_start} rows={len(p.rows)}]")
        else:
            out.append(f"(p.{p.page_start}-{p.page_end}) {p.text}")
    return "\n".join(out)


def chunk_document(doc_id: str, pdf_path: Path, meta: dict, cfg: Config, tk: Tokens):
    report: dict = {"doc_id": doc_id, "warnings": []}
    with fitz.open(pdf_path) as doc:
        if doc.needs_pass:
            raise RuntimeError("PDF is encrypted")
        lines, info = extract_lines(doc, cfg)
    report.update(info)

    pages = max(1, info["pages"])
    report["chars_per_page"] = round(info["chars"] / pages, 1)
    if info["chars"] == 0:
        report["warnings"].append("no extractable text (scanned PDF? run OCR first)")
        return [], report, ""
    if report["chars_per_page"] < cfg.min_chars_per_page:
        report["warnings"].append(f"only {report['chars_per_page']} chars/page - likely scanned or image-heavy")

    body = body_font_size(lines)
    paras, nheads = build_paras(lines, body, cfg, require_style=True)
    if nheads < cfg.min_headings_expected:
        p2, n2 = build_paras(lines, body, cfg, require_style=False)
        if n2 > nheads:
            paras, nheads = p2, n2
            report["warnings"].append("headings found without font cues (styles may be unreliable)")
    report["headings"] = nheads
    if nheads == 0:
        report["warnings"].append("no headings detected - all chunks fall under the default section")
    elif nheads / pages > 3:
        report["warnings"].append(f"{nheads / pages:.1f} headings/page - heading detection may be too loose")

    ck = Chunker(tk, cfg, doc_id, meta)
    ck.emit_drafts(ck.pack(ck.make_units(paras)))
    if cfg.keep_tables:
        for p in paras:
            if p.kind == "table":
                ck.emit_table(p)
    report["tables"] = sum(1 for p in paras if p.kind == "table")

    # ---- quality filters --------------------------------------------------- #
    kept, seen = [], set()
    dropped = Counter()
    for r in ck.records:
        if r["chunk_type"] == "text":
            body_txt = r["text"]
            alpha = sum(c.isalpha() for c in body_txt) / max(1, len(body_txt.replace(" ", "")))
            if r["token_count"] < cfg.drop_below_tokens:
                dropped["too_short"] += 1
                continue
            if alpha < cfg.min_alpha_ratio:
                dropped["low_alpha"] += 1
                continue
        if r["content_hash"] in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(r["content_hash"])
        kept.append(r)

    toks = [r["token_count"] for r in kept]
    text_chunks = [r for r in kept if r["chunk_type"] == "text"]
    report.update({
        "chunks": len(kept),
        "table_chunks": len(kept) - len(text_chunks),
        "tokens": {"min": min(toks, default=0), "median": int(statistics.median(toks)) if toks else 0,
                   "p95": pct(toks, 0.95), "max": max(toks, default=0)},
        "under_min_tokens": sum(1 for t in toks if t < cfg.min_chunk_tokens),
        "over_hard_cap": sum(1 for t in toks if t + 2 > cfg.max_tokens),
        "fallback_section_pct": round(100 * sum(r["section"] == DEFAULT_SECTION for r in kept) / max(1, len(kept)), 1),
        "no_terminal_punct_pct": round(100 * sum(not r["text"].rstrip().endswith((".", ";", ":", "?", "!", ")"))
                                                 for r in text_chunks) / max(1, len(text_chunks)), 1),
        "dropped": dict(dropped),
    })
    if report["fallback_section_pct"] > 30:
        report["warnings"].append(f"{report['fallback_section_pct']}% of chunks use the default section")
    if kept and report["under_min_tokens"] / len(kept) > 0.15:
        report["warnings"].append("more than 15% of chunks are below min_chunk_tokens")
    if report["over_hard_cap"]:
        report["warnings"].append(f"{report['over_hard_cap']} chunks exceed the hard token cap")

    return kept, report, debug_dump(paras, info, body)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def read_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = {"doc_id", "bucket"} - cols
        if missing:
            raise SystemExit(f"manifest is missing required columns: {sorted(missing)}")
        for c in ("use_case", "published_date"):
            if c not in cols:
                log.warning("manifest has no '%s' column - field will be empty", c)
        rows = list(reader)
    ids = [r["doc_id"] for r in rows]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        raise SystemExit(f"duplicate doc_id in manifest: {dup}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parent.parent
    index_dir = project_root / "index"
    ap.add_argument("--manifest", default=str(index_dir / "manifest.csv"))
    ap.add_argument("--docs", default=str(project_root / "docs"))
    ap.add_argument("--out", default=str(index_dir / "chunks.jsonl"))
    ap.add_argument("--report", default=str(index_dir / "chunk_report.json"))
    ap.add_argument("--debug-dir", default=str(index_dir / "debug"))
    ap.add_argument("--target-tokens", type=int)
    ap.add_argument("--min-chunk-tokens", type=int)
    ap.add_argument("--overlap-sentences", type=int)
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any error or warning")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = Config()
    if args.target_tokens:
        cfg.target_tokens = args.target_tokens
    if args.min_chunk_tokens:
        cfg.min_chunk_tokens = args.min_chunk_tokens
    if args.overlap_sentences is not None:
        cfg.overlap_sentences = args.overlap_sentences
    if args.no_tables:
        cfg.keep_tables = False

    tk = Tokens(cfg)
    docs_dir = Path(args.docs)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    reports: list[dict] = []
    problems = 0
    for row in read_manifest(args.manifest):
        if row["bucket"] != "text":
            continue
        doc_id = row["doc_id"]
        pdf_path = docs_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            log.error("MISSING PDF: %s", pdf_path)
            reports.append({"doc_id": doc_id, "error": "missing pdf"})
            problems += 1
            continue
        try:
            chunks, rep, dbg = chunk_document(doc_id, pdf_path, row, cfg, tk)
        except Exception as e:  # one bad PDF must not kill the whole run
            log.exception("FAILED %s", doc_id)
            reports.append({"doc_id": doc_id, "error": str(e)})
            problems += 1
            continue
        reports.append(rep)
        all_chunks.extend(chunks)
        if debug_dir and dbg:
            (debug_dir / f"{doc_id}.txt").write_text(dbg, encoding="utf-8")
        t = rep.get("tokens", {})
        log.info("%s: %d chunks (tokens min/med/p95/max: %s/%s/%s/%s)", doc_id, rep.get("chunks", 0),
                 t.get("min"), t.get("median"), t.get("p95"), t.get("max"))
        for w in rep["warnings"]:
            log.warning("  %s: %s", doc_id, w)
            problems += 1

    ids = [c["chunk_id"] for c in all_chunks]
    if len(ids) != len(set(ids)):
        log.error("duplicate chunk_ids found")
        problems += 1

    out_path = Path(args.out)
    report_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as out:
        for c in all_chunks:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "docs": reports}, f, indent=2, ensure_ascii=False)

    log.info("Saved %d chunks to %s (report: %s)", len(all_chunks), args.out, args.report)
    return 1 if (args.strict and problems) else 0


if __name__ == "__main__":
    sys.exit(main())