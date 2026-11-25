"""Ask one question from the command line.

  python scripts/ask.py "Logger shows 12 h at 9 C. Can we release the batch?"
  python scripts/ask.py --chat-endpoint https://HOST:PORT/v1 --model MODEL --chat-ca-cert PATH "question"
      (the API key is asked for with a hidden prompt)
"""
import argparse
import json

import _paths  # noqa: F401
from copilot.runtime import add_endpoint_args, apply_endpoint_args

ap = argparse.ArgumentParser(description="Ask the copilot one question.")
ap.add_argument("question", nargs="*")
add_endpoint_args(ap)
args = ap.parse_args()
apply_endpoint_args(args)

# Imported after the endpoint settings are in place, because the engine reads them when built.
from copilot.service import build_engine  # noqa: E402

question = " ".join(args.question) or "What is the storage temperature for unopened adalimumab pens?"
engine = build_engine()
print(f"Mode: {engine.mode}")
print(json.dumps(engine.answer(question).model_dump(), indent=2, ensure_ascii=False))
