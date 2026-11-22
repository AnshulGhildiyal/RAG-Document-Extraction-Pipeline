"""Embedding engine using SentenceTransformers with lazy initialization."""

from __future__ import annotations

from threading import Lock
from typing import List, Optional

import numpy as np

from rag_pipeline.services.exceptions import RAGApplicationError
from rag_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingEngine:
    """Generate and normalize text embeddings for retrieval."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension: Optional[int] = None
        self._lock = Lock()

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model: %s", self.model_name)
                    model = SentenceTransformer(self.model_name, device=self.device)
                    self._model = model
                    self._dimension = int(model.get_sentence_embedding_dimension())
                    logger.info("Embedding model loaded (dimension=%s)", self._dimension)
                except Exception as exc:  # pragma: no cover - depends on runtime deps
                    raise RAGApplicationError(
                        f"Failed to load embedding model '{self.model_name}': {exc}"
                    ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension, loading the model if needed."""
        if self._dimension is None:
            self._load_model()
        return int(self._dimension or 0)

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode a list of texts into a 2D numpy array of embeddings."""
        if texts is None:
            raise RAGApplicationError("No texts provided for embedding.")
        cleaned = [str(text) for text in texts if text is not None and str(text).strip()]
        if not cleaned:
            raise RAGApplicationError("No non-empty texts available for embedding.")

        model = self._load_model()
        try:
            embeddings = model.encode(
                cleaned,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )
        except TypeError:
            # Older sentence-transformers versions may not support normalize_embeddings.
            embeddings = model.encode(
                cleaned,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms
        except Exception as exc:  # pragma: no cover
            raise RAGApplicationError(f"Embedding generation failed: {exc}") from exc

        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        if embeddings.shape[0] != len(cleaned):
            raise RAGApplicationError("Embedding count does not match input text count.")
        return embeddings.astype(np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Encode a single query string and return a 1D embedding vector."""
        if query is None or not str(query).strip():
            raise RAGApplicationError("Query text for embedding cannot be empty.")
        vector = self.embed_texts([str(query)], batch_size=1, normalize=normalize)[0]
        return vector
