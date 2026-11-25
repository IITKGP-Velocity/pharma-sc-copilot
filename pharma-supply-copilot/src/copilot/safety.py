"""Safety layer: runs before and after any model call. Rules mirror configs/refusal_rules.yaml."""
import re

from copilot.schemas import CopilotResponse

PHONE_PATTERNS = [
    re.compile(r"\+\d{1,3}[\s-]\d{4,5}[\s-]\d{5,6}"),
    re.compile(r"(?<!\d)(\+?\d{1,3}[\s-]?)?\d{10}(?!\d)"),
]
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
# "date of birth" / "DOB" followed by a date; other dates (shipping, expiry) are kept.
DOB = re.compile(r"((?:date of birth|dob)\s*[:\-]?\s*)(\d{1,4}[-/. ]\d{1,2}[-/. ]\d{1,4}|\d{1,2}\s+\w+\s+\d{4})",
                 re.IGNORECASE)
# A capitalised name right after a label word: "driver Ravi Kumar", "contact Anita Rao", "patient Asha Nair".
# Free-standing names are not caught; that needs a named-entity model.
LABELLED_NAME = re.compile(r"\b((?i:driver|consignee contact|contact|patient|name)\s*:?\s+)"
                           r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
INJECTION = re.compile(
    r"ignore (all |your |prior |previous )+(rules|instructions)|SYSTEM:|approve all|mark .* as closed",
    re.IGNORECASE,
)


def redact_pii(text: str) -> str:
    """R4: personal data is replaced before text reaches the model or the output:
    phone numbers, emails, dates of birth, and names that follow a label word."""
    for pattern in PHONE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = EMAIL.sub("[REDACTED]", text)
    text = DOB.sub(lambda m: m.group(1) + "[REDACTED]", text)
    return LABELLED_NAME.sub(lambda m: m.group(1) + "[REDACTED]", text)


def looks_like_injection(text: str) -> bool:
    """R6: instructions inside the input are data, never commands."""
    return bool(INJECTION.search(text))


def pre_check(query: str) -> CopilotResponse | None:
    """Decide clear rule cases without retrieval or a model. None means continue."""
    q = query.lower()
    if INJECTION.search(query):
        return CopilotResponse(decision="REFUSE", reason_code="R6")
    if re.search(r"change the lot|alter the|backdate|fabricat|fake|write a certificate|(delete|remove) the .*reports", q):
        return CopilotResponse(decision="REFUSE", reason_code="R2")
    if re.search(r"\bicu\b|patient.*dose|dose.*patient|switch patients", q):
        return CopilotResponse(decision="REFUSE", reason_code="R3")
    if re.search(r"will .*detain|predict", q):
        return CopilotResponse(decision="ABSTAIN", reason_code="R7",
                               missing_evidence="Outcome predictions are out of scope; only the current alert list can be checked.")
    if re.search(r"release (it|this|the batch)|ship (this|it)|sign (it |this )?off|disposition|i'll sign|approve .*supplier|sign the qualification", q):
        authority = "QA" if ("supplier" in q or "qualif" in q) else "QP"
        return CopilotResponse(decision="ESCALATE", reason_code="R1", escalate_to=authority)
    if re.search(r"\bpoem\b|write a (song|story)", q):
        return CopilotResponse(decision="REFUSE")
    return None


def guard_output(resp: CopilotResponse, retrieved_ids: set[str]) -> CopilotResponse:
    """Post-check: redact personal data, drop citations to documents that were not retrieved,
    and downgrade an ANSWER with no support to ABSTAIN (R8)."""
    resp.answer = redact_pii(resp.answer)
    resp.citations = [c for c in resp.citations if c.doc_id in retrieved_ids]
    if resp.decision == "ANSWER" and not resp.citations and not resp.computed:
        return CopilotResponse(decision="ABSTAIN", missing_evidence="No cited source supports this answer.")
    return resp
