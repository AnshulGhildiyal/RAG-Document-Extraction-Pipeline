"""Shared pytest fixtures for the RAG pipeline tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src-layout package is importable when running tests without installation.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_pipeline.core.config import RAGConfig
from rag_pipeline.core.models import DocumentChunk, ExtractedPage


@pytest.fixture
def sample_config(tmp_path: Path) -> RAGConfig:
    return RAGConfig(
        chroma_path=str(tmp_path / "chroma_db"),
        log_file=str(tmp_path / "logs" / "test.log"),
    )


@pytest.fixture
def sample_pages() -> list[ExtractedPage]:
    return [
        ExtractedPage(
            page_number=1,
            text="John Doe is a software engineer with experience in Python, SQL, and machine learning. "
                 "Education: B.Tech in Computer Science. Skills include APIs, testing, and data processing.",
            char_count=170,
            doc_type="resume",
            metadata={},
        ),
        ExtractedPage(
            page_number=2,
            text="Invoice Number: INV-1001. Total amount due: $250.00. Payment due date: 2026-01-31. "
                 "Subtotal $230.00 and tax $20.00.",
            char_count=120,
            doc_type="invoice",
            metadata={},
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="doc1_p1_c0",
            document_id="doc1",
            filename="sample.pdf",
            source_path="/tmp/sample.pdf",
            text="Python developer with experience in APIs and testing.",
            page_number=1,
            chunk_index=0,
            doc_type="resume",
            metadata={"char_count": 52},
        ),
        DocumentChunk(
            chunk_id="doc1_p2_c0",
            document_id="doc1",
            filename="sample.pdf",
            source_path="/tmp/sample.pdf",
            text="Invoice total amount is $250.00 with payment due date 2026-01-31.",
            page_number=2,
            chunk_index=0,
            doc_type="invoice",
            metadata={"char_count": 66},
        ),
    ]
