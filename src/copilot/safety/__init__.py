"""Safety component: rules R1-R8 before any model call, personal-data redaction, and the output guard.

rules.py  pre_check (decides clear cases), redact_pii, looks_like_injection, guard_output
"""
from copilot.safety.rules import guard_output, looks_like_injection, pre_check, redact_pii

__all__ = ["guard_output", "looks_like_injection", "pre_check", "redact_pii"]
