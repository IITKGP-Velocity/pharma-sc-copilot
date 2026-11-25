"""RAG component: build the openFDA corpus and find the passages that answer a question.

corpus.py     download openFDA labels, shortages and recalls; split them into passages
retrieval.py  BM25 keyword search, dense (embedding) search, reciprocal rank fusion, optional reranker
"""
from copilot.rag.corpus import chunk_document, load_corpus, write_corpus
from copilot.rag.retrieval import BM25Index, DenseIndex, hybrid_search, reciprocal_rank_fusion, rerank, tokenize

__all__ = ["chunk_document", "load_corpus", "write_corpus", "BM25Index", "DenseIndex", "hybrid_search",
           "reciprocal_rank_fusion", "rerank", "tokenize"]
