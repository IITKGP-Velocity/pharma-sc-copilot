"""Rebuild the dashboard and failure log of a finished run (no model calls).

Use after editing failure_log.csv by hand: the labels people typed are kept, and their counts are printed.

  python scripts/rebuild_report.py eval/results/run_20261125_194250
"""
import json
import sys
from pathlib import Path

import _paths  # noqa: F401
from copilot.observability import (build_dashboard, label_counts, read_jsonl, unknown_labels,
                                   write_failure_log)


def main() -> int:
    """Rebuild failure_log.csv and dashboard.html for an existing run folder."""
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = Path(sys.argv[1])
    rows = read_jsonl(run_dir / "results.jsonl")
    if not rows:
        sys.exit(f"No results.jsonl in {run_dir} (runs made before the observability update only have trace.jsonl)")
    log = write_failure_log(run_dir, rows)
    page = build_dashboard(run_dir)
    print(f"Failure log: {log}\nDashboard:   {page}")
    print("Reviewed failure labels:", json.dumps(label_counts(log)) or "{}")
    bad = unknown_labels(log)
    if bad:
        print("Labels not in the taxonomy:", ", ".join(bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
