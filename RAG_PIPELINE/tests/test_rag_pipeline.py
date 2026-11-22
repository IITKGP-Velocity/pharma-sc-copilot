#!/usr/bin/env python3
"""
test_rag_pipeline.py - tests for RAG_PIPELINE/scripts/rag_pipeline.py

Put this file in  RAG_PIPELINE/tests/  (it also works from RAG_PIPELINE/scripts/).

1) OFFLINE unit tests - no keys, no network, no index needed (fake retriever + fake LLM):
       python RAG_PIPELINE/tests/test_rag_pipeline.py
       python -m pytest RAG_PIPELINE/tests -v            # also works

2) LIVE retrieval check - real index + embeddings API, NO LLM call. Use it to calibrate
   the abstention thresholds (shows the cosine / BM25 scores for in- and out-of-scope questions):
       python RAG_PIPELINE/tests/test_rag_pipeline.py --retrieval-only

3) LIVE end-to-end smoke test - real index, embeddings API and Qwen API (~10-15 LLM calls):
       python RAG_PIPELINE/tests/test_rag_pipeline.py --live --show-answers

Edit CASES below to match your documents and questions.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
for cand in (HERE / "scripts", HERE.parent / "scripts", HERE):
    if (cand / "rag_pipeline.py").exists():
        sys.path.insert(0, str(cand))
        break

import rag_pipeline as rp  # noqa: E402  (loads the repo-root .env at import)

# --------------------------------------------------------------------------- #
# Fakes (offline tests)
# --------------------------------------------------------------------------- #
def make_hit(i: int, doc: str, sec: str, text: str, cos: float = 0.70, bm25: float = 5.0,
             ref: int | None = None) -> dict:
    return {
        "chunk_id": f"{doc}_c{i:04d}", "doc_id": doc, "section_number": sec, "section_path": sec,
        "citation": f"{doc} | {sec} | p.{i}", "page_start": i, "page_end": i, "text": text,
        "retrieval": {"mode": "hybrid", "rrf_score": 0.03, "dense_rank": i, "dense_cosine": cos,
                      "bm25_rank": i, "bm25_score": bm25, "ref_rank": ref,
                      "ref_match_type": "exact" if ref else None, "matched_ref": sec if ref else None},
    }


HITS = [
    make_hit(1, "CFR21_Part211", "211.142",
             "Warehousing procedures shall describe quarantine of drug products. "
             "Store at 2 to 8 degrees. See 211.142(a).", ref=1),
    make_hit(2, "WHO_TRS961_Annex9", "4.5.1", "Temperature mapping must be repeated every 12 months."),
]


class FakeRetriever:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.hits)


class FakeLLM:
    model = "fake-model"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, temperature=0.0, max_tokens=700):
        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def reply(text: str, finish: str = "stop") -> rp.LLMReply:
    return rp.LLMReply(text, finish, 300, 40)


def make_pipeline(hits, replies, **overrides) -> rp.RegulatoryRAGPipeline:
    """Build the pipeline without keys, certificates or an index."""
    p = rp.RegulatoryRAGPipeline.__new__(rp.RegulatoryRAGPipeline)
    p.retriever, p.llm = FakeRetriever(hits), FakeLLM(replies)
    p.min_cosine, p.min_bm25 = 0.48, 2.0
    p.enable_self_repair, p.strict_block_on_unverified = True, True
    p.max_tokens, p.system_prompt = 700, rp.SYSTEM_PROMPT
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


# --------------------------------------------------------------------------- #
# Offline unit tests
# --------------------------------------------------------------------------- #
class TestVerifier(unittest.TestCase):
    def check(self, answer: str):
        return rp.verify_grounded_response(answer, HITS, "what about 211.142?")

    def test_grounded_answer_passes(self):
        v = self.check("Quarantine is required [1]. Mapping every 12 months [2]. Store at 2 to 8 degrees [1].")
        self.assertTrue(v.passed, v.issues)
        self.assertEqual(v.citations_used, [1, 2])

    def test_parent_section_is_accepted(self):
        self.assertTrue(self.check("Mapping per 4.5 is repeated every 12 months [2].").passed)

    def test_multi_citation_forms(self):
        self.assertEqual(rp.parse_citations("a [1] b [2, 3] c [4-6]"), [1, 2, 3, 4, 5, 6])

    def test_ungrounded_number_fails(self):
        v = self.check("Store at 15 degrees [1].")
        self.assertFalse(v.passed)
        self.assertTrue(any("numeric" in i for i in v.issues))

    def test_ungrounded_section_fails(self):
        v = self.check("See section 211.999 for storage [1].")
        self.assertFalse(v.passed)
        self.assertTrue(any("section identifier" in i for i in v.issues))

    def test_out_of_range_citation_fails(self):
        v = self.check("Quarantine is required [7].")
        self.assertFalse(v.passed)
        self.assertTrue(any("Out-of-range" in i for i in v.issues))

    def test_missing_citation_fails(self):
        self.assertFalse(self.check("Quarantine is required.").passed)

    def test_abstention_variants_pass(self):
        for text in ("INSUFFICIENT_EVIDENCE: nothing found", "**INSUFFICIENT_EVIDENCE**: nothing"):
            v = self.check(text)
            self.assertTrue(v.passed and v.abstained, text)


class TestHelpers(unittest.TestCase):
    def test_strip_closed_think_block(self):
        self.assertEqual(rp.strip_reasoning_tags("<think>plan</think>Answer [1]"), "Answer [1]")

    def test_strip_unclosed_think_block(self):
        self.assertEqual(rp.strip_reasoning_tags("<think>cut off while thinking"), "")

    def test_fit_to_context_trims_lowest_ranked_first(self):
        big = [make_hit(i, "DOC", f"{i}.1", "x" * 20000) for i in range(1, 5)]
        p = make_pipeline(big, [])
        fitted = p._fit_to_context(big, "question")
        self.assertLess(len(fitted), len(big))
        self.assertEqual(fitted[0]["chunk_id"], big[0]["chunk_id"])


class TestEnvAndEndpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cert = Path(self.tmp.name) / "ca.crt"
        self.cert.write_text("dummy")

    def test_base_url_gets_chat_path_once(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            url, cert = rp.resolve_llm_endpoint("https://h:1/v1", self.cert)
            self.assertEqual(url, "https://h:1/v1/chat/completions")
            url, _ = rp.resolve_llm_endpoint("https://h:1/v1/chat/completions/", self.cert)
            self.assertEqual(url, "https://h:1/v1/chat/completions")
            self.assertEqual(cert, self.cert.resolve())

    def test_missing_values_raise_clear_errors(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "LLM_URL"):
                rp.resolve_llm_endpoint(None, self.cert)
            with self.assertRaisesRegex(ValueError, "LLM_CERT"):
                rp.resolve_llm_endpoint("https://h:1/v1", None)
            with self.assertRaises(FileNotFoundError):
                rp.resolve_llm_endpoint("https://h:1/v1", Path(self.tmp.name) / "missing.crt")

    def test_values_come_from_environment(self):
        env = {"LLM_URL": "https://h:2/v1", "LLM_CERT": str(self.cert)}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(rp.resolve_llm_endpoint()[0], "https://h:2/v1/chat/completions")

    def test_load_env_parses_and_never_overrides(self):
        env_file = Path(self.tmp.name) / ".env"
        env_file.write_text('# comment\nA_KEY="quoted value"\nB_KEY=plain # trailing\n'
                            "export C_KEY=exported\nEXISTING=from_file\n")
        with mock.patch.dict(os.environ, {"EXISTING": "from_shell"}, clear=True):
            rp.load_env(env_file)
            self.assertEqual(os.environ["A_KEY"], "quoted value")
            self.assertEqual(os.environ["B_KEY"], "plain")
            self.assertEqual(os.environ["C_KEY"], "exported")
            self.assertEqual(os.environ["EXISTING"], "from_shell")


class TestPipelineFlow(unittest.TestCase):
    def test_grounded_answer(self):
        p = make_pipeline(HITS, [reply("<think></think>Quarantine is required [1].")])
        r = p.ask("what about 211.142?")
        self.assertEqual(r.status, rp.STATUS_ANSWERED)
        self.assertEqual(r.llm_calls, 1)
        self.assertEqual([c["index"] for c in r.citations], [1])
        self.assertIn("Sources:", r.formatted_output)

    def test_repair_fixes_ungrounded_number(self):
        p = make_pipeline(HITS, [reply("Store at 15 degrees [1]."), reply("Store at 2 to 8 degrees [1].")])
        r = p.ask("storage?")
        self.assertEqual(r.status, rp.STATUS_ANSWERED)
        self.assertEqual(r.llm_calls, 2)
        self.assertTrue(r.verification.repair_attempted)

    def test_blocked_when_repair_fails(self):
        p = make_pipeline(HITS, [reply("Store at 15 degrees [1]."), reply("Still 15 degrees [1].")])
        r = p.ask("storage?")
        self.assertEqual(r.status, rp.STATUS_BLOCKED)
        self.assertTrue(r.abstained)
        self.assertIn("Closest retrieved passages", r.formatted_output)

    def test_allow_unverified_flags_answer(self):
        p = make_pipeline(HITS, [reply("Store at 15 degrees [1]."), reply("Still 15 degrees [1].")],
                          strict_block_on_unverified=False)
        r = p.ask("storage?")
        self.assertEqual(r.status, rp.STATUS_UNVERIFIED)
        self.assertTrue(r.answer.startswith("[UNVERIFIED]"))

    def test_truncated_answer_triggers_repair(self):
        p = make_pipeline(HITS, [reply("Quarantine is required [1]. Also", "length"),
                                 reply("Quarantine is required [1].")])
        r = p.ask("q")
        self.assertEqual(r.status, rp.STATUS_ANSWERED)
        self.assertEqual(r.llm_calls, 2)

    def test_weak_retrieval_abstains_without_llm_call(self):
        weak = [make_hit(1, "DOC", "1.1", "unrelated text", cos=0.20, bm25=0.5)]
        p = make_pipeline(weak, [])
        r = p.ask("how to bake bread?")
        self.assertEqual(r.status, rp.STATUS_ABSTAINED)
        self.assertEqual(r.llm_calls, 0)
        self.assertEqual(p.llm.calls, [])

    def test_reference_match_bypasses_gate(self):
        weak_but_ref = [make_hit(1, "DOC", "211.142", "Quarantine rules apply.", cos=0.20, bm25=0.5, ref=1)]
        p = make_pipeline(weak_but_ref, [reply("Quarantine rules apply [1].")])
        self.assertEqual(p.ask("211.142").status, rp.STATUS_ANSWERED)

    def test_no_hits_abstains(self):
        r = make_pipeline([], []).ask("anything")
        self.assertEqual((r.status, r.llm_calls), (rp.STATUS_ABSTAINED, 0))

    def test_model_abstention_is_respected(self):
        p = make_pipeline(HITS, [reply("INSUFFICIENT_EVIDENCE: not in the corpus")])
        r = p.ask("something obscure")
        self.assertEqual(r.status, rp.STATUS_ABSTAINED)
        self.assertEqual(r.llm_calls, 1)

    def test_llm_failure_returns_error_status(self):
        p = make_pipeline(HITS, [RuntimeError("HTTP 401")])
        r = p.ask("q")
        self.assertEqual(r.status, rp.STATUS_ERROR)
        self.assertIn("HTTP 401", r.formatted_output)

    def test_filters_are_passed_to_retriever(self):
        p = make_pipeline(HITS, [reply("Quarantine is required [1].")])
        p.ask("q", use_case="A", doc_ids=["CFR21_Part211"], top_k=3)
        call = p.retriever.calls[0]
        self.assertEqual((call["use_case"], call["doc_ids"], call["top_k"]), ("A", ["CFR21_Part211"], 3))

    def test_to_dict_is_json_serialisable(self):
        import json
        r = make_pipeline(HITS, [reply("Quarantine is required [1].")]).ask("q")
        d = r.to_dict()
        self.assertNotIn("retrieved_chunks", d)
        json.dumps(d)


# --------------------------------------------------------------------------- #
# Live checks (real index / APIs) - edit these to match your corpus
# --------------------------------------------------------------------------- #
CASES = [
    dict(name="CFR section lookup",
         query="What does 21 CFR 211.142 require for warehousing procedures?",
         status={"answered"}, docs={"CFR21_Part211"}, expect_ref=True),
    dict(name="WHO temperature mapping",
         query="How should temperature mapping be carried out in a storage area?",
         status={"answered"}, docs={"WHO_TRS961_Annex9"}),
    dict(name="ICH Q9 concept",
         query="What is quality risk management and what are its two primary principles?",
         status={"answered"}, docs={"ICH_Q9"}),
    dict(name="EU GDP transport",
         query="What does EU GDP require when transporting medicinal products?",
         status={"answered"}, docs={"EU_GDP_2013_C343_01"}),
    dict(name="Filter to one document",
         query="What are the risk assessment tools?", doc_ids=["ICH_Q9"],
         status={"answered", "abstained"}, only_docs={"ICH_Q9"}),
    dict(name="Out of scope (cooking)",
         query="What is the best way to bake sourdough bread?",
         status={"abstained"}, no_llm=True),
    dict(name="Out of scope (pharma-sounding)",
         query="What is the current FDA fee for submitting a new drug application?",
         status={"abstained", "blocked_unverified"}),
]


def evaluate_case(case: dict, res: rp.RAGResponse) -> list[str]:
    """Return a list of failure notes (empty list = pass)."""
    fails = []
    if res.status not in case["status"]:
        fails.append(f"status={res.status}, expected one of {sorted(case['status'])}")
    if res.status == rp.STATUS_ANSWERED:
        if not res.citations:
            fails.append("answered without citations")
        if case.get("docs") and not ({c["doc_id"] for c in res.citations} & case["docs"]):
            fails.append(f"no citation from {sorted(case['docs'])}")
    if case.get("only_docs"):
        stray = {h["doc_id"] for h in res.retrieved_chunks} - case["only_docs"]
        if stray:
            fails.append(f"filter leaked documents: {sorted(stray)}")
    if case.get("no_llm") and res.llm_calls != 0:
        fails.append(f"expected no LLM call, got {res.llm_calls}")
    if case.get("expect_ref") and not res.confidence.get("has_ref_match"):
        fails.append("section reference in the query was not matched in the index")
    return fails


def _top(hits: list[dict]) -> str:
    return hits[0]["citation"] if hits else "-"


def run_live(args) -> int:
    results, passed = [], 0
    if args.retrieval_only:
        from retriever import HybridRetriever
        embed_cert = rp.resolve_path(os.environ.get("EMBED_CERT") or rp.RAG_DIR / "secrets" / "vm02-ca.crt")
        index_dir = rp.resolve_path(args.index_dir or os.environ.get("RAG_INDEX_DIR") or rp.RAG_DIR / "index")
        retriever = HybridRetriever(index_dir=index_dir, embed_cert=embed_cert)
        gate = make_pipeline([], [], min_cosine=float(os.environ.get("RAG_MIN_COSINE", 0.48)),
                             min_bm25=float(os.environ.get("RAG_MIN_BM25", 2.0)))
        print(f"\nGate thresholds: cosine<{gate.min_cosine} AND bm25<{gate.min_bm25} -> abstain\n")
        for case in CASES:
            hits = retriever.search(case["query"], top_k=5, doc_ids=case.get("doc_ids"))
            sig = rp.confidence_signal(hits)
            abstain, _ = gate._should_abstain_early(sig)
            if case.get("no_llm"):                 # must be stopped by the gate itself
                gate_ok = abstain
            elif "answered" in case["status"]:     # valid question: the gate should let it through
                gate_ok = not abstain
            else:                                  # either the gate or the LLM may abstain
                gate_ok = True
            flag = "ok   " if gate_ok else "CHECK"
            print(f"[{flag}] {case['name']:<32} gate={'ABSTAIN' if abstain else 'answer ':<7} "
                  f"cos={sig['top_dense_cosine']} bm25={sig['top_bm25_score']} "
                  f"ref={sig['has_ref_match']} top={_top(hits)}")
        print("\nUse the cos/bm25 values of the out-of-scope rows vs the in-scope rows to set "
              "RAG_MIN_COSINE and RAG_MIN_BM25.")
        return 0

    pipeline = rp.RegulatoryRAGPipeline(index_dir=args.index_dir)
    print(f"\nModel: {pipeline.llm.model}\n")
    for case in CASES:
        t0 = time.monotonic()
        res = pipeline.ask(case["query"], doc_ids=case.get("doc_ids"))
        fails = evaluate_case(case, res)
        passed += not fails
        top = res.citations[0]["citation"] if res.citations else _top(res.retrieved_chunks)
        print(f"[{'PASS' if not fails else 'FAIL'}] {case['name']:<32} {res.status:<19} "
              f"calls={res.llm_calls} {int((time.monotonic() - t0) * 1000):>6} ms  top={top}")
        print(f"       cos={res.confidence.get('top_dense_cosine')} bm25={res.confidence.get('top_bm25_score')} "
              f"ref={res.confidence.get('has_ref_match')} verification={'ok' if res.verification.passed else res.verification.issues}")
        for f in fails:
            print(f"       -> {f}")
        if args.show_answers:
            print("       " + res.formatted_output.replace("\n", "\n       ")[:900])
        results.append(not fails)
    print(f"\n{passed}/{len(CASES)} live checks passed")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Offline unit tests (default) or live checks.")
    ap.add_argument("--live", action="store_true", help="end-to-end test with the real index, embeddings and LLM APIs")
    ap.add_argument("--retrieval-only", action="store_true", help="live retrieval + gate check, no LLM call")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--show-answers", action="store_true")
    known, rest = ap.parse_known_args()
    if known.live or known.retrieval_only:
        sys.exit(run_live(known))
    unittest.main(argv=[sys.argv[0]] + rest, verbosity=2)