"""Document processing services: extraction-to-chunk preparation helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

from rag_pipeline.core.chunker import TextChunker
from rag_pipeline.core.classifier import DocumentClassifier
from rag_pipeline.core.models import Document, DocumentChunk, ExtractedPage
from rag_pipeline.services.exceptions import DocumentExtractionError
from rag_pipeline.services.extraction_service import PDFExtractionService
from rag_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentService:
    """High-level document processing service for ingestion preparation."""

    def __init__(
        self,
        extraction_service: PDFExtractionService | None = None,
        classifier: DocumentClassifier | None = None,
        chunker: TextChunker | None = None,
    ):
        self.classifier = classifier or DocumentClassifier()
        self.extraction_service = extraction_service or PDFExtractionService(classifier=self.classifier)
        self.chunker = chunker or TextChunker()

    @staticmethod
    def build_document_id(source_path: str) -> str:
        """Create a stable document identifier from the source path/name."""
        path = Path(source_path)
        digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
        stem = "".join(ch for ch in path.stem.lower() if ch.isalnum() or ch in ("-", "_"))[:40] or "document"
        return f"{stem}_{digest}"

    def process_pdf(
        self,
        pdf_path: str,
        max_pages: int = 100,
        max_size_mb: int = 20,
    ) -> Tuple[Document, List[DocumentChunk]]:
        """Validate, extract, classify, and chunk a PDF into pipeline-ready objects."""
        full_text, pages, overall_type = self.extraction_service.extract(
            pdf_path=pdf_path,
            max_pages=max_pages,
            max_size_mb=max_size_mb,
        )

        if not pages:
            raise DocumentExtractionError("No extractable pages were found in the document.")

        source_path = str(Path(pdf_path).resolve())
        filename = Path(source_path).name
        document_id = self.build_document_id(source_path)

        # Normalize page metadata and ensure per-page doc types exist.
        normalized_pages: List[ExtractedPage] = []
        for page in pages:
            page_text = (page.text or "").strip()
            if len(page_text) < 20:
                continue
            doc_type = page.doc_type or self.classifier.classify(page_text, filename=filename)
            normalized_pages.append(
                ExtractedPage(
                    page_number=page.page_number,
                    text=page_text,
                    char_count=len(page_text),
                    doc_type=doc_type,
                    metadata={
                        **(page.metadata or {}),
                        "document_id": document_id,
                        "filename": filename,
                        "source_path": source_path,
                    },
                )
            )

        if not normalized_pages:
            raise DocumentExtractionError("The PDF did not contain usable text after processing.")

        document = Document.from_path(
            document_id=document_id,
            source_path=source_path,
            doc_type=overall_type or self.classifier.classify_pages([p.text for p in normalized_pages], filename),
            pages=normalized_pages,
            metadata={
                "total_characters": sum(p.char_count for p in normalized_pages),
                "page_count": len(normalized_pages),
                "full_text_length": len(full_text or ""),
            },
        )

        chunks = self.chunker.chunk_document_pages(
            pages=normalized_pages,
            document_id=document.document_id,
            filename=document.filename,
            source_path=document.source_path,
        )

        if not chunks:
            raise DocumentExtractionError("No text chunks could be generated from the extracted document.")

        logger.info(
            "Document processed: %s (id=%s, pages=%d, chunks=%d, type=%s)",
            document.filename,
            document.document_id,
            document.page_count,
            len(chunks),
            document.doc_type,
        )
        return document, chunks
