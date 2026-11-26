"""Safety component: rules R1-R8 before any model call, personal-data redaction, and the output guard.

rules.py  pre_check (decides clear cases, incl. out-of-scope), redact_pii, looks_like_injection, in_scope, guard_output
"""
from copilot.safety.rules import OUT_OF_SCOPE, guard_output, in_scope, looks_like_injection, pre_check, redact_pii

__all__ = ["OUT_OF_SCOPE", "guard_output", "in_scope", "looks_like_injection", "pre_check", "redact_pii"]
