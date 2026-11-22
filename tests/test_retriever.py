"""Tests for hybrid retrieval fusion, ranking, deduplication, and edge cases."""

import numpy as np

from rag_pipeline.core.models import DocumentChunk, RetrievalResult
from rag_pipeline.core.retriever import HybridRetriever


class FakeEmbeddingEngine:
    def __init__(self, dim: int = 4):
        self._dim = dim

    @property
    def dimension(self):
        return self._dim

    def embed_query(self, query: str, normalize: bool = True):
        return np.ones(self._dim, dtype=np.float32)

    def embed_texts(self, texts, batch_size: int = 32, normalize: bool = True):
        return np.ones((len(texts), self._dim), dtype=np.float32)


class FakeVectorStore:
    def __init__(self, chunks):
        self._chunks = {c.chunk_id: c for c in chunks}

    def count(self):
        return len(self._chunks)

    def query(self, query_embedding, top_k=5, where=None):
        # Return first chunks as dense hits with increasing distances.
        chunks = list(self._chunks.values())[:top_k]
        return {
            "ids": [[c.chunk_id for c in chunks]],
            "documents": [[c.text for c in chunks]],
            "metadatas": [[c.to_storage_metadata() for c in chunks]],
            "distances": [[0.2 + i * 0.1 for i in range(len(chunks))]],
        }

    def get_all(self, include_embeddings=False):
        chunks = list(self._chunks.values())
        data = {
            "ids": [c.chunk_id for c in chunks],
            "documents": [c.text for c in chunks],
            "metadatas": [c.to_storage_metadata() for c in chunks],
        }
        if include_embeddings:
            data["embeddings"] = []
        return data


def _chunk(cid, text, page=1, doc_id="doc1"):
    return DocumentChunk(
        chunk_id=cid,
        document_id=doc_id,
        filename="doc.pdf",
        source_path="/tmp/doc.pdf",
        text=text,
        page_number=page,
        chunk_index=0,
        doc_type="general",
        metadata={},
    )


def test_hybrid_search_fuses_and_deduplicates_results():
    chunks = [
        _chunk("c1", "Python embeddings and vector search for retrieval.", page=1),
        _chunk("c2", "Invoice total amount and payment due date details.", page=2),
        _chunk("c3", "Hybrid retrieval combines dense and sparse ranking.", page=3),
    ]
    store = FakeVectorStore(chunks)
    retriever = HybridRetriever(vector_store=store, embedding_engine=FakeEmbeddingEngine(), dense_weight=0.6, sparse_weight=0.4)

    # Force overlap: make dense and sparse return same top chunk.
    dense_chunk = chunks[0]
    sparse_chunk = chunks[0]
    dense = [RetrievalResult(chunk=dense_chunk, relevance_score=0.9, retrieval_method="vector", rank=1, dense_score=0.9)]
    sparse = [RetrievalResult(chunk=sparse_chunk, relevance_score=0.8, retrieval_method="knn", rank=1, sparse_score=0.8)]
    retriever.dense_search = lambda query, top_k=5: dense  # type: ignore[assignment]
    retriever.sparse_search = lambda query, top_k=3: sparse + [RetrievalResult(chunk=chunks[1], relevance_score=0.7, retrieval_method="knn", rank=2, sparse_score=0.7)]  # type: ignore[assignment]

    results = retriever.hybrid_search("retrieval query", top_k_total=5)
    ids = [r.chunk.chunk_id for r in results]
    assert len(ids) == len(set(ids)), "Results should be deduplicated by chunk ID"
    assert ids[0] == "c1", "Chunk present in both rankings should be ranked highly"
    assert all(r.rank == i + 1 for i, r in enumerate(results))


def test_hybrid_search_respects_configurable_top_k_and_empty_results():
    store = FakeVectorStore([])
    retriever = HybridRetriever(vector_store=store, embedding_engine=FakeEmbeddingEngine())
    assert retriever.hybrid_search("anything", top_k_total=5) == []

    chunks = [_chunk(f"c{i}", f"text {i} content keyword{i}", page=i + 1) for i in range(5)]
    store2 = FakeVectorStore(chunks)
    retriever2 = HybridRetriever(vector_store=store2, embedding_engine=FakeEmbeddingEngine())
    # Stub methods to produce deterministic sizes.
    retriever2.dense_search = lambda query, top_k=5: [RetrievalResult(chunk=chunks[i], relevance_score=1.0/(i+1), retrieval_method="vector", rank=i+1) for i in range(min(top_k, len(chunks)))]  # type: ignore[assignment]
    retriever2.sparse_search = lambda query, top_k=3: []  # type: ignore[assignment]
    results = retriever2.hybrid_search("text", top_k_total=3)
    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]


def test_sparse_search_empty_or_oov_query_returns_empty():
    chunks = [_chunk("c1", "alpha beta gamma", page=1)]
    store = FakeVectorStore(chunks)
    retriever = HybridRetriever(vector_store=store, embedding_engine=FakeEmbeddingEngine())
    assert retriever.sparse_search("", top_k=3) == []
    # Completely out-of-vocabulary query against English stopwords/terms often yields zero similarities.
    assert retriever.sparse_search("zzzzqqqq", top_k=3) == []
