"""Centralized, validated configuration for the RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from rag_pipeline.services.exceptions import ValidationError


@dataclass(frozen=True)
class RAGConfig:
    """Application configuration.

    Expensive resources (models, clients) must not be initialized here.
    """

    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_vector: int = 5
    top_k_knn: int = 3
    top_k_hybrid: int = 8
    similarity_threshold: float = 0.3
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rrf_k: int = 60
    gemini_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048
    chroma_path: str = "./data/chroma_db"
    collection_name: str = "documents_collection"
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    max_upload_size_mb: int = 20
    max_pdf_pages: int = 100

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "RAGConfig":
        """Load configuration from environment variables with defaults."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        def _get_int(name: str, default: int) -> int:
            value = os.getenv(name)
            if value is None or value == "":
                return default
            try:
                return int(value)
            except ValueError as exc:
                raise ValidationError(f"Invalid integer for {name}: {value}") from exc

        def _get_float(name: str, default: float) -> float:
            value = os.getenv(name)
            if value is None or value == "":
                return default
            try:
                return float(value)
            except ValueError as exc:
                raise ValidationError(f"Invalid float for {name}: {value}") from exc

        cfg = cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            chunk_size=_get_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_get_int("CHUNK_OVERLAP", cls.chunk_overlap),
            top_k_vector=_get_int("TOP_K_VECTOR", cls.top_k_vector),
            top_k_knn=_get_int("TOP_K_KNN", cls.top_k_knn),
            top_k_hybrid=_get_int("TOP_K_HYBRID", cls.top_k_hybrid),
            similarity_threshold=_get_float("SIMILARITY_THRESHOLD", cls.similarity_threshold),
            dense_weight=_get_float("DENSE_WEIGHT", cls.dense_weight),
            sparse_weight=_get_float("SPARSE_WEIGHT", cls.sparse_weight),
            rrf_k=_get_int("RRF_K", cls.rrf_k),
            gemini_model=os.getenv("GEMINI_MODEL", cls.gemini_model),
            llm_temperature=_get_float("LLM_TEMPERATURE", cls.llm_temperature),
            llm_max_tokens=_get_int("LLM_MAX_TOKENS", cls.llm_max_tokens),
            chroma_path=os.getenv("CHROMA_DB_PATH", cls.chroma_path),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", cls.collection_name),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
            log_file=os.getenv("LOG_FILE", cls.log_file),
            max_upload_size_mb=_get_int("MAX_UPLOAD_SIZE_MB", cls.max_upload_size_mb),
            max_pdf_pages=_get_int("MAX_PDF_PAGES", cls.max_pdf_pages),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Validate configuration constraints."""
        if self.chunk_size <= 0:
            raise ValidationError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValidationError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValidationError("chunk_overlap must be less than chunk_size")
        if self.top_k_vector <= 0 or self.top_k_knn <= 0 or self.top_k_hybrid <= 0:
            raise ValidationError("retrieval top-k values must be > 0")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValidationError("similarity_threshold must be between 0 and 1")
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValidationError("retrieval weights must be >= 0")
        if (self.dense_weight + self.sparse_weight) <= 0:
            raise ValidationError("sum of retrieval weights must be > 0")
        if self.rrf_k <= 0:
            raise ValidationError("rrf_k must be > 0")
        if not (0.0 <= self.llm_temperature <= 2.0):
            raise ValidationError("llm_temperature must be between 0 and 2")
        if self.llm_max_tokens <= 0:
            raise ValidationError("llm_max_tokens must be > 0")
        if self.max_upload_size_mb <= 0:
            raise ValidationError("max_upload_size_mb must be > 0")
        if self.max_pdf_pages <= 0:
            raise ValidationError("max_pdf_pages must be > 0")
        if not self.embedding_model.strip():
            raise ValidationError("embedding_model must not be empty")
        if not self.gemini_model.strip():
            raise ValidationError("gemini_model must not be empty")
        if not self.collection_name.strip():
            raise ValidationError("collection_name must not be empty")

    @property
    def chroma_dir(self) -> Path:
        """Return the ChromaDB persistence directory as a Path."""
        return Path(self.chroma_path)

    @property
    def gemini_api_key(self) -> Optional[str]:
        """Return the Gemini API key from the environment, if configured."""
        key = os.getenv("GEMINI_API_KEY")
        return key.strip() if key else None
