"""Safety layer: runs before and after any model call. Rules mirror configs/refusal_rules.yaml."""
import re

from copilot.config import ROOT
from copilot.schemas import CopilotResponse

OUT_OF_SCOPE = ("Out of scope. This copilot only answers pharmaceutical supply-chain questions: cold-chain excursions "
                "and GDP release, alternate API and generic supplier qualification, customs and import pre-clearance, "
                "drug shortages and therapeutic substitution, and pharmacovigilance and recall triage. For example, "
                "ask about storage conditions, an import alert, a recall, a shortage or supplier qualification.")
CLINICAL = ("Out of scope. Patient-level dosing is a clinical decision; please ask the prescriber or pharmacist. "
            "This copilot can check product labels, shortages and recalls.")

# Words that mark a supply-chain question. STEMS match as word prefixes ("ship" matches shipping);
# WORDS (short acronyms and common words) must match whole, with an optional plural "s".
STEMS = """
    drug medicin medicat pharma batch shipment shipping shipped logistic excursion storage refrigerat freez frozen
    thaw labell recall adverse faers shortage supplier impurit certificate qualif customs import export manifest
    invoice demurrage detention nadac vial tablet capsule injection syringe generic biosimilar equivalen excipient
    diluent stabilit kinetic logger quarantine distribut warehouse courier vaccine insulin biologic antibod monograph
    sulfite benzyl formulation substitut pharmacist prescriber labeler wholesale manufactur inspection pharmacovigil
    deviation complaint humidity transport container temperature
""".split()
WORDS = """
    lot product label gdp gmp cfr ich ema fda ndc anda nda bla api coa mkt capa qp qa dmf cmc release port unit price
    cost brand alert trs orange purple cold chain box mapping specification facility warning patient
""".split()
SCOPE = re.compile(r"\b(" + "|".join(re.escape(t) for t in STEMS) + r")"
                   r"|\b(" + "|".join(re.escape(t) for t in WORDS) + r")s?\b"
                   r"|-?\d+(\.\d+)?\s*°\s*[cf]\b|\d\s*(to|-|–)\s*8\s*°", re.IGNORECASE)
DRUG_NAMES = [n.strip().lower() for n in (ROOT / "configs" / "drug_names.txt").read_text(encoding="utf-8").splitlines()
              if n.strip() and not n.startswith("#")]
# Requests that are off-topic even when they mention a supply-chain word ("a poem about shipping containers").
OFF_TOPIC = re.compile(
    r"\b(poem|poetry|song|lyrics|story|joke|riddle|recipe|horoscope|weather|movie|film|novel|sports?|football|cricket"
    r"|stock price|share price|crypto|bitcoin|homework|essay about|translate|capital of|who won|dating)\b"
    r"|write (a|me a) (song|story|poem)|\b(python|javascript|java|sql) (code|script|program)",
    re.IGNORECASE,
)
DOSING = re.compile(r"\bdosing\b|\bdos(age|e) (for|in|during)\b|during pregnancy|breastfeeding|how much .*(should|can) .*(take|give|inject)"
                    r"|\bdos(e|age|es)\b.*\b(child|children|pediatric|paediatric|infant|neonat\w*|pregnan\w*|elderly|adult|kg)\b",
                    re.IGNORECASE)

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


def in_scope(query: str) -> bool:
    """True when the question is about the pharmaceutical supply chain and is not an off-topic request."""
    if OFF_TOPIC.search(query):
        return False
    q = query.lower()
    return bool(SCOPE.search(query)) or any(name in q for name in DRUG_NAMES)


def pre_check(query: str) -> CopilotResponse | None:
    """Decide clear rule cases without retrieval or a model. None means continue."""
    q = query.lower()
    if INJECTION.search(query):
        return CopilotResponse(decision="REFUSE", reason_code="R6")
    if re.search(r"change the lot|alter the|backdate|fabricat|fake|write a certificate|(delete|remove) the .*reports", q):
        return CopilotResponse(decision="REFUSE", reason_code="R2")
    if re.search(r"\bicu\b|patient.*dose|dose.*patient|switch patients", q):
        return CopilotResponse(decision="REFUSE", reason_code="R3")
    if DOSING.search(query):
        return CopilotResponse(decision="REFUSE", reason_code="R3", answer=CLINICAL, escalate_to="Prescriber")
    if re.search(r"will .*detain|predict", q):
        return CopilotResponse(decision="ABSTAIN", reason_code="R7",
                               missing_evidence="Outcome predictions are out of scope; only the current alert list can be checked.")
    if re.search(r"release (it|this|the batch)|ship (this|it)|sign (it |this )?off|disposition|i'll sign|approve .*supplier|sign the qualification", q):
        authority = "QA" if ("supplier" in q or "qualif" in q) else "QP"
        return CopilotResponse(decision="ESCALATE", reason_code="R1", escalate_to=authority)
    if not in_scope(query):
        return CopilotResponse(decision="REFUSE", reason_code="OOS", answer=OUT_OF_SCOPE)
    return None


def guard_output(resp: CopilotResponse, retrieved_ids: set[str]) -> CopilotResponse:
    """Post-check: redact personal data, drop citations to documents that were not retrieved,
    and downgrade an ANSWER with no support to ABSTAIN (R8). Scope is decided by the rules before the model
    runs, so a model "out of scope" verdict on an in-scope question becomes ABSTAIN, not a refusal."""
    if resp.reason_code == "OOS":
        return CopilotResponse(decision="ABSTAIN", missing_evidence="Not enough evidence to answer this question.")
    resp.answer = redact_pii(resp.answer)
    resp.citations = [c for c in resp.citations if c.doc_id in retrieved_ids]
    if resp.decision == "ANSWER" and not resp.citations and not resp.computed:
        return CopilotResponse(decision="ABSTAIN", missing_evidence="No cited source supports this answer.")
    return resp
