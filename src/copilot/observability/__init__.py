"""Observability component: what happened on each query, and how good a whole run was.

tracing.py      trace.jsonl: one line per agent stage, tagged with query id and time
metrics.py      results.jsonl rows, the run summary, retrieval recall@k and MRR
failure_log.py  failure_log.csv: wrong decisions for a person to label with the failure taxonomy
dashboard.py    dashboard.html: everything above on one page
"""
from copilot.observability.dashboard import build_dashboard
from copilot.observability.failure_log import TAXONOMY, label_counts, unknown_labels, write_failure_log
from copilot.observability.metrics import (EXPECTED, auto_label, percentile, read_jsonl, result_row,
                                           retrieval_metrics, summarise, write_jsonl)
from copilot.observability.tracing import log_event

__all__ = ["build_dashboard", "TAXONOMY", "label_counts", "unknown_labels", "write_failure_log", "EXPECTED",
           "auto_label", "percentile", "read_jsonl", "result_row", "retrieval_metrics", "summarise",
           "write_jsonl", "log_event"]
