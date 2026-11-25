"""Output schema (Deliverable 1, Section 5.1). Every response is validated against it."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Decision = Literal["ANSWER", "ESCALATE", "REFUSE", "ABSTAIN"]
Authority = Literal["QP", "QA", "Regulatory", "Pharmacist", "Prescriber"]


class Citation(BaseModel):
    """A cited passage: its id and the quoted text that supports the claim."""
    doc_id: str
    span: str


class ComputedValue(BaseModel):
    """A number produced by a tool: the tool name, its inputs and the result."""
    tool: str
    inputs: dict
    result: float | str | list | dict | bool


class CopilotResponse(BaseModel):
    """Every reply has this shape (Deliverable 1, section 5.1); the model's JSON is validated against it."""
    decision: Decision
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    computed: list[ComputedValue] = Field(default_factory=list)
    escalate_to: Optional[Authority] = None
    missing_evidence: Optional[str] = None
    reason_code: Optional[str] = None

    @field_validator("escalate_to", "missing_evidence", "reason_code", mode="before")
    @classmethod
    def null_string_is_none(cls, v):
        # Small models often write the string "null" or "" instead of JSON null.
        """Treat the strings 'null', 'none' and '' as a missing value."""
        return None if isinstance(v, str) and v.strip().lower() in ("", "null", "none") else v

    @field_validator("answer")
    @classmethod
    def answer_max_200_words(cls, v: str) -> str:
        """Reject answers over 200 words (the Deliverable 1 limit)."""
        if len(v.split()) > 200:
            raise ValueError("answer exceeds 200 words")
        return v
