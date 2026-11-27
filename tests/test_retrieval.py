"""BM25, dense search and reciprocal rank fusion on a tiny corpus."""
from copilot.rag.retrieval import BM25Index, DenseIndex, hybrid_search, reciprocal_rank_fusion, rerank, tokenize

DOCS = [
    {"id": "d1", "text": "Store unopened pens refrigerated at 2 to 8 C. Do not freeze."},
    {"id": "d2", "text": "Demurrage fees apply when containers stay at the port."},
    {"id": "d3", "text": "Haynes mean kinetic temperature for excursions."},
]


def test_tokenize_drops_stopwords():
    assert tokenize("Store at 2-8 C!") == ["store", "2", "8", "c"]


def test_bm25_ranks_matching_doc_first():
    assert BM25Index(DOCS).search("unopened pens storage", k=3)[0][0] == "d1"


def test_bm25_no_match_returns_empty():
    assert BM25Index(DOCS).search("submarine engine", k=3) == []


def test_dense_index_ranks_by_cosine():
    idx = DenseIndex(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    assert idx.search([0.9, 0.1], k=1)[0][0] == "a"


def test_rrf_prefers_documents_high_in_both_lists():
    assert set(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])[:2]) == {"a", "b"}


def test_hybrid_without_dense_equals_bm25_order():
    hits = hybrid_search("unopened pens", DOCS, BM25Index(DOCS), None, None, k=2)
    assert hits[0][0] == "d1"


class FakeChat:
    """Stands in for the course chat model: returns a fixed ranking or raises."""
    def __init__(self, reply=None, error=False):
        self.reply, self.error = reply, error

    def complete_json(self, messages, **kwargs):
        if self.error:
            raise RuntimeError("endpoint down")
        return self.reply


HITS = [("d1", 1.0), ("d2", 0.5), ("d3", 0.33)]
TEXTS = {d["id"]: d["text"] for d in DOCS}


def test_rerank_follows_model_order_and_keeps_left_out_passages():
    out = rerank("excursion", HITS, TEXTS, FakeChat({"ranking": [3, 1]}), k=3)
    assert [d for d, _ in out] == ["d3", "d1", "d2"]


def test_rerank_ignores_bad_numbers():
    out = rerank("q", HITS, TEXTS, FakeChat({"ranking": [9, "x", 2]}), k=2)
    assert [d for d, _ in out] == ["d2", "d1"]


def test_rerank_keeps_fused_order_without_model_or_on_error():
    assert rerank("q", HITS, TEXTS, None, k=2) == HITS[:2]
    assert rerank("q", HITS, TEXTS, FakeChat(error=True), k=2) == HITS[:2]
