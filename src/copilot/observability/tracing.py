"""Run log: one JSON line per step, written under eval/results/<run>/trace.jsonl."""
import json
import time
from pathlib import Path


def log_event(run_dir: Path, event: dict) -> None:
    """Append one event, with a timestamp, to run_dir/trace.jsonl."""
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), **event}
    with (run_dir / "trace.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
