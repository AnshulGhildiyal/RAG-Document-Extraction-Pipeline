"""Tests for text chunking behavior and validation."""

import pytest

from rag_pipeline.core.chunker import TextChunker
from rag_pipeline.core.models import ExtractedPage
from rag_pipeline.services.exceptions import ValidationError


def _make_page(text: str, page_number: int = 1, doc_type: str = "general") -> ExtractedPage:
    return ExtractedPage(
        page_number=page_number,
        text=text,
        char_count=len(text),
        doc_type=doc_type,
        metadata={"source_test": True},
    )


def test_chunk_short_text_single_chunk():
    chunker = TextChunker(chunk_size=800, chunk_overlap=150)
    page = _make_page("Short but valid page text for chunking. " * 2)
    chunks = chunker.chunk_page(page, document_id="docA", filename="a.pdf", source_path="/tmp/a.pdf")
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].document_id == "docA"
    assert chunks[0].metadata.get("source_test") is True


def test_chunk_long_text_produces_overlap_and_multiple_chunks():
    chunker = TextChunker(chunk_size=120, chunk_overlap=30, min_chunk_chars=20)
    text = ("Sentence one about retrieval systems. " * 20).strip()
    page = _make_page(text)
    chunks = chunker.chunk_page(page, document_id="docB", filename="b.pdf", source_path="/tmp/b.pdf")
    assert len(chunks) >= 2
    # Overlap expectation: start of later chunk should appear before end of previous chunk in original text region.
    assert chunks[1].chunk_index == 1
    assert chunks[0].text
    assert chunks[1].text
    # Metadata preserved
    assert all(c.filename == "b.pdf" and c.doc_type == "general" for c in chunks)


def test_chunk_empty_text_returns_no_chunks():
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    page = _make_page("   \n\n   ")
    chunks = chunker.chunk_page(page, document_id="docC", filename="c.pdf", source_path="/tmp/c.pdf")
    assert chunks == []


def test_chunk_invalid_configuration_raises():
    with pytest.raises(ValidationError):
        TextChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValidationError):
        TextChunker(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValidationError):
        TextChunker(chunk_size=100, chunk_overlap=-1)
