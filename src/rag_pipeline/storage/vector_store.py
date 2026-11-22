"""Persistent ChromaDB-backed vector storage."""

from __future__ import annotations

import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from rag_pipeline.core.models import DocumentChunk
from rag_pipeline.services.exceptions import StorageError
from rag_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStore:
    """Manage persistent storage and retrieval of document chunk embeddings."""

    def __init__(self, persist_directory: str, collection_name: str = "documents_collection"):
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._lock = Lock()

    @property
    def client(self):
        if self._client is None:
            try:
                import chromadb  # Lazy import to avoid hard dependency at module import time.

                Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            except Exception as exc:  # pragma: no cover - depends on chromadb runtime
                raise StorageError(f"Failed to initialize ChromaDB client: {exc}") from exc
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            try:
                # Never delete collections automatically on startup.
                self._collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:  # pragma: no cover
                raise StorageError(f"Failed to access ChromaDB collection: {exc}") from exc
        return self._collection

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
        """Insert chunk texts, embeddings, and metadata into the collection."""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise StorageError("Chunk count and embedding count must match for storage.")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [self._sanitize_metadata(chunk.to_storage_metadata()) for chunk in chunks]

        with self._lock:
            try:
                existing_ids: set[str] = set()
                get_fn = getattr(self.collection, "get", None)
                if callable(get_fn):
                    try:
                        existing = self.collection.get(ids=ids)
                        existing_ids = set(existing.get("ids", []) or [])
                    except Exception:
                        existing_ids = set()

                filtered = [
                    (cid, doc, emb, meta)
                    for cid, doc, emb, meta in zip(ids, documents, embeddings, metadatas)
                    if cid not in existing_ids
                ]
                if not filtered:
                    logger.info("All %d chunk(s) already exist in collection; skipping insert.", len(ids))
                    return 0

                f_ids, f_docs, f_embs, f_metas = zip(*filtered)

                # Prefer upsert when available for idempotent behavior.
                upsert_fn = getattr(self.collection, "upsert", None)
                if callable(upsert_fn):
                    upsert_fn(
                        ids=list(f_ids),
                        documents=list(f_docs),
                        embeddings=[list(map(float, e)) for e in f_embs],
                        metadatas=list(f_metas),
                    )
                else:
                    self.collection.add(
                        ids=list(f_ids),
                        documents=list(f_docs),
                        embeddings=[list(map(float, e)) for e in f_embs],
                        metadatas=list(f_metas),
                    )
                logger.info("Stored %d chunk(s) in ChromaDB collection '%s'.", len(f_ids), self.collection_name)
                return len(f_ids)
            except StorageError:
                raise
            except Exception as exc:  # pragma: no cover
                raise StorageError(f"Failed to store chunks in ChromaDB: {exc}") from exc

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Any]]:
        """Run a vector similarity search and return Chroma-style result payload."""
        if top_k <= 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        try:
            total = self.count()
            if total == 0:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            n_results = min(top_k, total)
            kwargs: Dict[str, Any] = {
                "query_embeddings": [list(map(float, query_embedding))],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where
            results = self.collection.query(**kwargs)
            # Normalize missing keys for callers.
            for key in ("ids", "documents", "metadatas", "distances"):
                results.setdefault(key, [[]])
            return results
        except Exception as exc:  # pragma: no cover
            raise StorageError(f"Vector search failed: {exc}") from exc

    def get_all(self, include_embeddings: bool = False) -> Dict[str, List[Any]]:
        """Return all stored records from the collection."""
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")
        try:
            data = self.collection.get(include=include)
            for key in ("ids", "documents", "metadatas", "embeddings"):
                data.setdefault(key, [] if key != "embeddings" else [])
            return data
        except Exception as exc:  # pragma: no cover
            raise StorageError(f"Failed to fetch stored chunks: {exc}") from exc

    def count(self) -> int:
        """Return the number of stored chunks."""
        try:
            return int(self.collection.count())
        except Exception as exc:  # pragma: no cover
            raise StorageError(f"Failed to count stored chunks: {exc}") from exc

    @staticmethod
    def _matches_document_id(chunk_id: str, metadata: Dict[str, Any], document_id: str) -> bool:
        if str(metadata.get("document_id", "")) == str(document_id):
            return True
        # Backward-compatible prefix match for legacy IDs.
        return str(chunk_id).startswith(f"{re.escape(str(document_id))}_") or str(chunk_id).startswith(f"{document_id}_")

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks associated with a document identifier."""
        if not document_id:
            raise StorageError("document_id is required for deletion.")
        with self._lock:
            try:
                # Prefer metadata-based deletion when supported.
                try:
                    self.collection.delete(where={"document_id": document_id})
                    # Chroma delete may not return count; compute best-effort by follow-up scan not necessary.
                    # Return -1 style avoided; caller can refresh stats. Here we return 0/unknown as 1+ best effort.
                    return 0
                except Exception:
                    pass

                data = self.collection.get(include=["metadatas"])
                ids_to_delete = []
                for cid, meta in zip(data.get("ids", []) or [], data.get("metadatas", []) or []):
                    if self._matches_document_id(cid, meta or {}, document_id):
                        ids_to_delete.append(cid)
                if ids_to_delete:
                    self.collection.delete(ids=ids_to_delete)
                return len(ids_to_delete)
            except Exception as exc:  # pragma: no cover
                raise StorageError(f"Failed to delete document '{document_id}': {exc}") from exc

    def clear(self) -> None:
        """Explicitly clear all stored chunks by deleting and recreating the collection."""
        with self._lock:
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                # Collection may not exist yet; proceed to recreate.
                pass
            try:
                self._collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("Cleared ChromaDB collection '%s'.", self.collection_name)
            except Exception as exc:  # pragma: no cover
                raise StorageError(f"Failed to clear ChromaDB collection: {exc}") from exc

    def get_stats(self) -> Dict[str, Any]:
        """Return basic database statistics for UI/reporting."""
        try:
            data = self.collection.get(include=["metadatas"])
            metadatas = data.get("metadatas", []) or []
            ids = data.get("ids", []) or []
            doc_ids = set()
            doc_types: Dict[str, int] = {}
            pages = set()
            for meta in metadatas:
                meta = meta or {}
                did = meta.get("document_id")
                if did:
                    doc_ids.add(str(did))
                dtype = str(meta.get("doc_type", "unknown") or "unknown")
                doc_types[dtype] = doc_types.get(dtype, 0) + 1
                page = meta.get("page_number")
                if page is not None:
                    pages.add((str(meta.get("document_id", "")), int(page)))
            return {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "total_chunks": len(ids),
                "total_documents": len(doc_ids) if doc_ids else 0,
                "unique_pages": len(pages),
                "document_type_counts": doc_types,
            }
        except Exception as exc:  # pragma: no cover
            raise StorageError(f"Failed to compute storage statistics: {exc}") from exc
