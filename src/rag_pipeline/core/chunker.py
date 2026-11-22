"""Configurable, metadata-preserving text chunking."""

from __future__ import annotations

import hashlib
import re
from typing import List

from rag_pipeline.core.models import DocumentChunk, ExtractedPage
from rag_pipeline.services.exceptions import ValidationError


class TextChunker:
    """Split extracted page text into overlapping chunks with preserved metadata."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150, min_chunk_chars: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars
        self._validate_config()

    def _validate_config(self) -> None:
        if self.chunk_size <= 0:
            raise ValidationError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValidationError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValidationError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_chars <= 0:
            raise ValidationError("min_chunk_chars must be > 0")

    @staticmethod
    def _clean_text(text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"\s+", " ", line).strip() for line in normalized.split("\n")]
        # Preserve paragraph breaks while removing noisy blank lines.
        cleaned_lines: List[str] = []
        blank_pending = False
        for line in lines:
            if not line:
                blank_pending = bool(cleaned_lines)
                continue
            if blank_pending:
                cleaned_lines.append("")
            cleaned_lines.append(line)
            blank_pending = False
        return "\n".join(cleaned_lines).strip()

    def _find_boundary(self, text: str, start: int, proposed_end: int) -> int:
        if proposed_end >= len(text):
            return len(text)
        window_start = max(start, proposed_end - 120)
        window = text[window_start:proposed_end]
        # Prefer paragraph, then sentence, then whitespace boundaries.
        for sep in ("\n\n", "\n"):
            idx = window.rfind(sep)
            if idx != -1 and window_start + idx > start:
                return window_start + idx + len(sep)
        for sep in (". ", "? ", "! "):
            idx = window.rfind(sep)
            if idx != -1 and window_start + idx > start:
                return window_start + idx + 1
        idx = window.rfind(" ")
        if idx != -1 and window_start + idx > start:
            return window_start + idx + 1
        return proposed_end

    def _make_chunk_id(
        self,
        document_id: str,
        page_number: int,
        chunk_index: int,
        chunk_text: str,
    ) -> str:
        base = f"{document_id}|{page_number}|{chunk_index}|{chunk_text}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]

    def chunk_page(
        self,
        page: ExtractedPage,
        document_id: str,
        filename: str,
        source_path: str,
    ) -> List[DocumentChunk]:
        """Chunk a single extracted page into DocumentChunk objects."""
        text = self._clean_text(page.text or "")
        if len(text) < self.min_chunk_chars:
            return []

        chunks: List[DocumentChunk] = []

        if len(text) <= self.chunk_size:
            chunk_text = text.strip()
            if len(chunk_text) >= self.min_chunk_chars:
                chunks.append(
                    DocumentChunk(
                        chunk_id=self._make_chunk_id(document_id, page.page_number, 0, chunk_text),
                        document_id=document_id,
                        filename=filename,
                        source_path=source_path,
                        text=chunk_text,
                        page_number=page.page_number,
                        chunk_index=0,
                        doc_type=page.doc_type,
                        metadata={
                            **(page.metadata or {}),
                            "char_count": len(chunk_text),
                        },
                    )
                )
            return chunks

        start = 0
        chunk_index = 0
        text_length = len(text)
        max_iterations = (text_length // max(1, self.chunk_size - self.chunk_overlap)) + 10

        for _ in range(max_iterations):
            if start >= text_length:
                break
            proposed_end = min(start + self.chunk_size, text_length)
            end = self._find_boundary(text, start, proposed_end)
            if end <= start:
                end = proposed_end
            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.min_chunk_chars:
                chunks.append(
                    DocumentChunk(
                        chunk_id=self._make_chunk_id(document_id, page.page_number, chunk_index, chunk_text),
                        document_id=document_id,
                        filename=filename,
                        source_path=source_path,
                        text=chunk_text,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        doc_type=page.doc_type,
                        metadata={
                            **(page.metadata or {}),
                            "char_count": len(chunk_text),
                        },
                    )
                )
                chunk_index += 1

            if end >= text_length:
                break

            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = end  # Safety: guarantee forward progress.
            start = next_start

        # Deduplicate in the unlikely event of identical chunk IDs within a page.
        unique_chunks: List[DocumentChunk] = []
        seen_ids = set()
        for chunk in chunks:
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            unique_chunks.append(chunk)
        return unique_chunks

    def chunk_document_pages(
        self,
        pages: List[ExtractedPage],
        document_id: str,
        filename: str,
        source_path: str,
    ) -> List[DocumentChunk]:
        """Chunk all pages in a document and return a flat chunk list."""
        all_chunks: List[DocumentChunk] = []
        for page in pages:
            all_chunks.extend(
                self.chunk_page(
                    page=page,
                    document_id=document_id,
                    filename=filename,
                    source_path=source_path,
                )
            )
        return all_chunks
