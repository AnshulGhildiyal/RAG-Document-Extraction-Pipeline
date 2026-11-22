"""Typed data models used across the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedPage:
    """Represents text extracted from a single PDF page."""

    page_number: int
    text: str
    char_count: int
    doc_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Represents an ingested source document."""

    document_id: str
    filename: str
    source_path: str
    pages: List[ExtractedPage] = field(default_factory=list)
    doc_type: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_characters(self) -> int:
        return sum(p.char_count for p in self.pages)

    @classmethod
    def from_path(
        cls,
        document_id: str,
        source_path: str,
        doc_type: str = "general",
        pages: Optional[List[ExtractedPage]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Document":
        path = Path(source_path)
        return cls(
            document_id=document_id,
            filename=path.name,
            source_path=str(path),
            pages=pages or [],
            doc_type=doc_type,
            metadata=metadata or {},
        )


@dataclass
class DocumentChunk:
    """A chunk of text prepared for embedding and retrieval."""

    chunk_id: str
    document_id: str
    filename: str
    source_path: str
    text: str
    page_number: int
    chunk_index: int
    doc_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_storage_metadata(self) -> Dict[str, Any]:
        """Return metadata suitable for vector-store persistence."""
        base = {
            "document_id": self.document_id,
            "filename": self.filename,
            "source_path": self.source_path,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "doc_type": self.doc_type,
            "char_count": len(self.text),
        }
        # Merge extra metadata without overriding core identifiers.
        for key, value in self.metadata.items():
            base.setdefault(key, value)
        return base


@dataclass
class RetrievalResult:
    """A ranked retrieval result with preserved source metadata."""

    chunk: DocumentChunk
    relevance_score: float
    retrieval_method: str
    rank: int
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None

    @property
    def source_label(self) -> str:
        return f"{self.chunk.filename} (page {self.chunk.page_number})"


@dataclass
class IngestionResult:
    """Summary of a document ingestion operation."""

    document_id: str
    filename: str
    source_path: str
    doc_type: str
    page_count: int
    chunk_count: int
    total_characters: int
    average_chunk_size: int
    message: str = ""
    success: bool = True


@dataclass
class QuestionAnswerResult:
    """Answer generated from retrieved document context."""

    question: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    context_used: bool = False
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None or self.context_used
