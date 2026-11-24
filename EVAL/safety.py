"""
safety.py - Small, deterministic safety helpers: PII redaction and prompt-injection detection.

Why: Deliverable 1 rule R4 says personal data must be redacted before anything is output
or logged, and rule R6 says text inside documents is data, never an instruction. These
helpers are simple regex checks so they can be tested offline.

Limits (be honest about them in the report):
  - Names are only caught when a label word comes first (driver, contact, patient, ...).
    Free-floating names need an NER model.
  - Injection patterns catch common phrasings, not every attack.
"""
import re

# ---- PII redaction (R4) ---------------------------------------------------------------
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# A run of digits, spaces, dashes, dots or brackets that starts and ends with a digit.
# It is only redacted if it holds 10+ digits, so readings like "2.1, 3.4" or "83.144" stay.
_PHONE = re.compile(r"(?<![\w.])\+?\(?\d[\d\s().-]{7,}\d(?![\w])")
# "date of birth" / "DOB" followed by a date in a common format.
_DOB = re.compile(r"((?:date of birth|dob)\s*[:\-]?\s*)(\d{1,4}[-/. ]\d{1,2}[-/. ]\d{1,4}|\d{1,2}\s+\w+\s+\d{4})",
                  re.IGNORECASE)
# A capitalised name directly after a label word: "driver Ravi Kumar", "contact Anita Rao".
_LABELLED_NAME = re.compile(
    r"\b((?i:driver|consignee contact|contact|patient|patient's name|name)\s*:?\s+)"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")


def redact_pii(text: str) -> str:
    """Return the text with emails, phone numbers, dates of birth and labelled names masked."""
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _DOB.sub(lambda m: m.group(1) + "[REDACTED_DOB]", text)
    text = _PHONE.sub(lambda m: "[REDACTED_PHONE]" if sum(c.isdigit() for c in m.group(0)) >= 10
                      else m.group(0), text)
    return _LABELLED_NAME.sub(lambda m: m.group(1) + "[REDACTED_NAME]", text)


# ---- Prompt-injection detection (R6) ----------------------------------------------------
_INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(instructions|rules|prompt)\b",
    r"\bdisregard\b.{0,40}\b(instructions|rules|prompt)\b",
    r"(?m)^\s*(system|assistant)\s*:",
    r"\byou are now\b",
    r"\bapprove all (suppliers|batches|lots|shipments|products|requests)\b",   # plain "approve all" occurs in real GMP text
    r"\b(reveal|print|show)\b.{0,30}\bsystem prompt\b",
    r"\bmark this\b.{0,40}\bas (closed|approved|resolved)\b",
    r"\breport all products as\b",
]
_INJECTION = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]


def find_injection(text: str) -> list[str]:
    """Return the patterns that match, so a hit can be explained. Empty list = none found."""
    return [p.pattern for p in _INJECTION if p.search(text)]
