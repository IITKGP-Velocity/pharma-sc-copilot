"""Record the SHA-256 of the frozen evaluation file. Commit the output with tag v1.0-spec."""
import hashlib

import _paths

ROOT = _paths.ROOT
EVAL = ROOT / "eval" / "gold" / "eval_set_v1.0.jsonl"
OUT = ROOT / "eval" / "gold" / "eval_set_v1.0.sha256"

digest = hashlib.sha256(EVAL.read_bytes()).hexdigest()
OUT.write_text(f"{digest}  {EVAL.name}\n", encoding="utf-8")
print(f"{digest}  {EVAL.name}")
print(f"Written to {OUT.relative_to(ROOT)}")
