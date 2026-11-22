"""Hybrid retrieval engine: dense vector search + sparse TF-IDF + weighted RRF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rag_pipeline.core.embeddings import EmbeddingEngine
from rag_pipeline.core.models import DocumentChunk, RetrievalResult
from rag_pipeline.services.exceptions import RAGApplicationError
from rag_pipeline.storage.vector_store import ChromaVectorStore
from rag_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _SparseIndex:
    signature: Tuple[int, Tuple[str, ...]]
    vectorizer: Any
    matrix: Any
    chunks: List[DocumentChunk]


class HybridRetriever:
    """Retrieve relevant chunks using dense, sparse, and hybrid strategies."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_engine: EmbeddingEngine,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.dense_weight = float(dense_weight)
        self.sparse_weight = float(sparse_weight)
        self.rrf_k = int(rrf_k)
        self._sparse_index: Optional[_SparseIndex] = None

    @staticmethod
    def _chunk_from_storage(chunk_id: str, text: str, metadata: Dict[str, Any]) -> DocumentChunk:
        metadata = metadata or {}
        return DocumentChunk(
            chunk_id=str(chunk_id),
            document_id=str(metadata.get("document_id", "")),
            filename=str(metadata.get("filename", metadata.get("source", ""))),
            source_path=str(metadata.get("source_path", "")),
            text=text or "",
            page_number=int(metadata.get("page_number", 0) or 0),
            chunk_index=int(metadata.get("chunk_index", 0) or 0),
            doc_type=str(metadata.get("doc_type", "general") or "general"),
            metadata={k: v for k, v in metadata.items()},
        )

    def dense_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Semantic vector search using embeddings and ChromaDB."""
        if not query or not str(query).strip():
            return []
        if top_k <= 0 or self.vector_store.count() == 0:
            return []

        query_embedding = self.embedding_engine.embed_query(query)
        results = self.vector_store.query(query_embedding=query_embedding.tolist(), top_k=top_k)

        ids = (results.get("ids") or [[]])[0] or []
        documents = (results.get("documents") or [[]])[0] or []
        metadatas = (results.get("metadatas") or [[]])[0] or []
        distances = (results.get("distances") or [[]])[0] or []

        retrieved: List[RetrievalResult] = []
        for idx, (cid, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances), start=1):
            # Chroma cosine distance is typically in [0, 2]; convert to a similarity-like score in [0, 1].
            try:
                distance = float(dist)
            except Exception:
                distance = 1.0
            similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            chunk = self._chunk_from_storage(cid, doc, meta or {})
            retrieved.append(
                RetrievalResult(
                    chunk=chunk,
                    relevance_score=similarity,
                    retrieval_method="vector",
                    rank=idx,
                    dense_score=similarity,
                )
            )
        return retrieved

    def _build_sparse_index(self) -> Optional[_SparseIndex]:
        data = self.vector_store.get_all(include_embeddings=False)
        ids = list(data.get("ids") or [])
        docs = list(data.get("documents") or [])
        metas = list(data.get("metadatas") or [])
        if not ids or not docs:
            self._sparse_index = None
            return None

        signature = (len(ids), tuple(str(i) for i in ids))
        if self._sparse_index and self._sparse_index.signature == signature:
            return self._sparse_index

        chunks: List[DocumentChunk] = []
        texts: List[str] = []
        for cid, text, meta in zip(ids, docs, metas):
            text_str = str(text or "")
            if not text_str.strip():
                continue
            chunk = self._chunk_from_storage(str(cid), text_str, meta or {})
            chunks.append(chunk)
            texts.append(text_str)

        if not texts:
            self._sparse_index = None
            return None

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception as exc:  # pragma: no cover - depends on runtime deps
            raise RAGApplicationError(f"scikit-learn is required for sparse retrieval: {exc}") from exc

        try:
            vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            # e.g. empty vocabulary after stop-word removal; keep a minimal fallback.
            vectorizer = TfidfVectorizer(max_features=10000)
            matrix = vectorizer.fit_transform(texts)
        except Exception as exc:
            raise RAGApplicationError(f"Failed to build TF-IDF index: {exc}") from exc

        self._sparse_index = _SparseIndex(signature=signature, vectorizer=vectorizer, matrix=matrix, chunks=chunks)
        logger.info("Rebuilt sparse TF-IDF index for %d chunk(s).", len(chunks))
        return self._sparse_index

    def sparse_search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Keyword relevance search using TF-IDF cosine similarity."""
        if not query or not str(query).strip() or top_k <= 0:
            return []
        if self.vector_store.count() == 0:
            return []

        index = self._build_sparse_index()
        if index is None or not index.chunks:
            return []

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            query_vec = index.vectorizer.transform([str(query)])
            similarities = cosine_similarity(query_vec, index.matrix).flatten()
        except Exception as exc:
            raise RAGApplicationError(f"Sparse retrieval failed: {exc}") from exc

        if similarities.size == 0:
            return []

        # Rank by similarity (desc), then chunk_id for stable ordering.
        ranked_indices = sorted(
            range(len(similarities)),
            key=lambda i: (-float(similarities[i]), index.chunks[i].chunk_id),
        )

        results: List[RetrievalResult] = []
        rank = 1
        for i in ranked_indices:
            score = float(similarities[i])
            if score <= 0.0 and len(results) >= top_k:
                break
            if score <= 0.0 and not results:
                # Keep searching; if all are zero we'll return empty.
                continue
            chunk = index.chunks[i]
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    relevance_score=max(0.0, min(1.0, score)),
                    retrieval_method="knn",
                    rank=rank,
                    sparse_score=max(0.0, min(1.0, score)),
                )
            )
            rank += 1
            if len(results) >= top_k:
                break

        # If query terms are entirely OOV, similarities can be all zeros -> empty result is correct.
        if all(float(similarities[i]) <= 0.0 for i in range(len(similarities))):
            return []
        return results

    def hybrid_search(self, query: str, top_k_total: int = 8) -> List[RetrievalResult]:
        """Combine dense and sparse rankings using weighted Reciprocal Rank Fusion (RRF)."""
        if not query or not str(query).strip() or top_k_total <= 0:
            return []
        if self.vector_store.count() == 0:
            return []

        dense_results = self.dense_search(query, top_k=max(top_k_total, 1))
        sparse_results = self.sparse_search(query, top_k=max(top_k_total, 1))

        if not dense_results and not sparse_results:
            return []

        # RRF(d) = Σ weight_i / (k + rank_i(d))
        k_rrf = max(1, int(self.rrf_k))
        fused: Dict[str, Dict[str, Any]] = {}

        def add_results(results: List[RetrievalResult], method_weight: float, score_field: str):
            for res in results:
                cid = res.chunk.chunk_id
                entry = fused.setdefault(
                    cid,
                    {
                        "chunk": res.chunk,
                        "rrf": 0.0,
                        "dense_score": None,
                        "sparse_score": None,
                        "methods": set(),
                    },
                )
                entry["rrf"] += method_weight / (k_rrf + int(res.rank))
                if score_field == "dense_score":
                    entry["dense_score"] = res.relevance_score if entry["dense_score"] is None else max(entry["dense_score"], res.relevance_score)
                if score_field == "sparse_score":
                    entry["sparse_score"] = res.relevance_score if entry["sparse_score"] is None else max(entry["sparse_score"], res.relevance_score)
                entry["methods"].add(res.retrieval_method)

        add_results(dense_results, self.dense_weight, "dense_score")
        add_results(sparse_results, self.sparse_weight, "sparse_score")

        ranked = sorted(
            fused.values(),
            key=lambda item: (-float(item["rrf"]), str(item["chunk"].chunk_id)),
        )

        # Normalize RRF scores to [0,1] for a user-facing relevance score while preserving order.
        max_rrf = max((float(item["rrf"]) for item in ranked), default=0.0)
        results: List[RetrievalResult] = []
        for rank, item in enumerate(ranked[:top_k_total], start=1):
            methods = sorted(m for m in item["methods"] if m)
            if "vector" in methods and "knn" in methods:
                method_label = "hybrid"
            elif len(methods) == 1:
                method_label = methods[0]
            else:
                method_label = "hybrid"
            normalized = (float(item["rrf"]) / max_rrf) if max_rrf > 0 else 0.0
            results.append(
                RetrievalResult(
                    chunk=item["chunk"],
                    relevance_score=max(0.0, min(1.0, normalized)),
                    retrieval_method=method_label,
                    rank=rank,
                    dense_score=item["dense_score"],
                    sparse_score=item["sparse_score"],
                )
            )
        return results

    def invalidate_sparse_index(self) -> None:
        """Invalidate cached sparse index so it will be rebuilt on next use."""
        self._sparse_index = None
