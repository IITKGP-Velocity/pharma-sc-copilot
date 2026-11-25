from copilot.rag.retrieval import BM25Index, DenseIndex, hybrid_search, reciprocal_rank_fusion, tokenize

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
