"""Tests for RAGPipeline orchestration with mocked dependencies."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_pipeline.core.models import Document, DocumentChunk, ExtractedPage, QuestionAnswerResult, RetrievalResult
from rag_pipeline.core.pipeline import RAGPipeline
from rag_pipeline.services.exceptions import DocumentExtractionError, GenerationError, StorageError


def _write_minimal_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    return path


class FakeEmbeddingEngine:
    def embed_texts(self, texts, batch_size: int = 32, normalize: bool = True):
        # Return deterministic vectors with correct shape.
        return [[float((i + 1) % 5) / 10.0 for i in range(8)] for _ in texts]

    def embed_query(self, query: str, normalize: bool = True):
        return [0.1] * 8


class FakeVectorStore:
    def __init__(self):
        self._chunks = []
        self._count = 0
        self.cleared = False
        self.fail_add = False

    def add_chunks(self, chunks, embeddings):
        if self.fail_add:
            raise StorageError("storage failure")
        self._chunks.extend(chunks)
        self._count += len(chunks)
        return len(chunks)

    def delete_document(self, document_id):
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if getattr(c, "document_id", None) != document_id]
        self._count = len(self._chunks)
        return before - len(self._chunks)

    def count(self):
        return self._count

    def get_stats(self):
        return {
            "collection_name": "test",
            "persist_directory": "./data/chroma_db",
            "total_chunks": self._count,
            "total_documents": len({c.document_id for c in self._chunks}) if self._chunks else 0,
            "unique_pages": len({(c.document_id, c.page_number) for c in self._chunks}),
            "document_type_counts": {},
        }

    def query(self, query_embedding, top_k=5, where=None):
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def get_all(self, include_embeddings=False):
        return {"ids": [], "documents": [], "metadatas": []}

    def clear(self):
        self._chunks = []
        self._count = 0
        self.cleared = True


class FakeRetriever:
    def __init__(self, results=None):
        self._results = results or []
        self.invalidated = False

    def hybrid_search(self, query, top_k_total=8):
        return list(self._results)[:top_k_total]

    def dense_search(self, query, top_k=5):
        return list(self._results)[:top_k]

    def invalidate_sparse_index(self):
        self.invalidated = True


class FakeGenerationService:
    def __init__(self, result=None, fail=False):
        self.result = result
        self.fail = fail

    def generate_answer(self, question, contexts):
        if self.fail:
            raise GenerationError("generation failed")
        if self.result is not None:
            return self.result
        return QuestionAnswerResult(
            question=question,
            answer="This is a grounded answer [Source 1].",
            sources=[{"source_num": 1, "filename": "doc.pdf", "page_number": 1, "relevance_score": 0.9, "doc_type": "general"}],
            retrieval_results=contexts,
            context_used=True,
        )


class FakeDocumentService:
    def __init__(self, fail=False):
        self.fail = fail

    def process_pdf(self, pdf_path, max_pages=100, max_size_mb=20):
        if self.fail:
            raise DocumentExtractionError("extraction failed")
        page = ExtractedPage(page_number=1, text="Invoice total amount is $250.00", char_count=31, doc_type="invoice")
        doc = Document.from_path(document_id="doc_test", source_path=pdf_path, doc_type="invoice", pages=[page])
        chunk = DocumentChunk(
            chunk_id="chunk1",
            document_id=doc.document_id,
            filename=doc.filename,
            source_path=doc.source_path,
            text=page.text,
            page_number=1,
            chunk_index=0,
            doc_type="invoice",
        )
        return doc, [chunk]


def _make_pipeline(tmp_path: Path, retriever=None, generation_service=None, document_service=None, vector_store=None) -> RAGPipeline:
    pdf_placeholder = _write_minimal_pdf(tmp_path / "placeholder.pdf")  # ensures path helpers work if needed
    _ = pdf_placeholder
    pipeline = RAGPipeline.__new__(RAGPipeline)
    from rag_pipeline.core.config import RAGConfig
    pipeline.config = RAGConfig(chroma_path=str(tmp_path / "chroma"), log_file=str(tmp_path / "app.log"))
    pipeline.embedding_engine = FakeEmbeddingEngine()
    pipeline.vector_store = vector_store or FakeVectorStore()
    pipeline.document_service = document_service or FakeDocumentService()
    pipeline.retriever = retriever or FakeRetriever()
    pipeline.generation_service = generation_service or FakeGenerationService()
    return pipeline


def test_ingest_document_success(tmp_path: Path):
    pdf = _write_minimal_pdf(tmp_path / "sample.pdf")
    store = FakeVectorStore()
    retriever = FakeRetriever()
    pipeline = _make_pipeline(tmp_path, retriever=retriever, vector_store=store)

    result = pipeline.ingest_document(str(pdf))

    assert result.success is True
    assert result.chunk_count == 1
    assert result.doc_type == "invoice"
    assert store.count() == 1
    assert retriever.invalidated is True


def test_ingest_document_extraction_failure(tmp_path: Path):
    pdf = _write_minimal_pdf(tmp_path / "bad.pdf")
    pipeline = _make_pipeline(tmp_path, document_service=FakeDocumentService(fail=True))
    with pytest.raises(DocumentExtractionError):
        pipeline.ingest_document(str(pdf))


def test_ingest_document_storage_failure(tmp_path: Path):
    pdf = _write_minimal_pdf(tmp_path / "sample.pdf")
    store = FakeVectorStore()
    store.fail_add = True
    pipeline = _make_pipeline(tmp_path, vector_store=store)
    with pytest.raises(StorageError):
        pipeline.ingest_document(str(pdf))


def test_ask_question_success(tmp_path: Path):
    store = FakeVectorStore()
    store._count = 1
    chunk = DocumentChunk(
        chunk_id="c1", document_id="doc1", filename="doc.pdf", source_path="/tmp/doc.pdf",
        text="Invoice total amount is $250.00", page_number=1, chunk_index=0, doc_type="invoice"
    )
    retrieved = [RetrievalResult(chunk=chunk, relevance_score=0.9, retrieval_method="hybrid", rank=1)]
    pipeline = _make_pipeline(tmp_path, vector_store=store, retriever=FakeRetriever(results=retrieved))

    result = pipeline.ask_question("What is the total amount?", use_hybrid=True)

    assert result.context_used is True
    assert "grounded answer" in result.answer.lower()
    assert result.sources


def test_ask_question_missing_context_returns_insufficient_info(tmp_path: Path):
    store = FakeVectorStore()
    store._count = 1
    pipeline = _make_pipeline(tmp_path, vector_store=store, retriever=FakeRetriever(results=[]))

    result = pipeline.ask_question("What is unknown?", use_hybrid=True)

    assert result.context_used is False
    assert "cannot find this information" in result.answer.lower()


def test_ask_question_generation_failure(tmp_path: Path):
    store = FakeVectorStore()
    store._count = 1
    chunk = DocumentChunk(
        chunk_id="c1", document_id="doc1", filename="doc.pdf", source_path="/tmp/doc.pdf",
        text="Some content", page_number=1, chunk_index=0, doc_type="general"
    )
    retrieved = [RetrievalResult(chunk=chunk, relevance_score=0.9, retrieval_method="vector", rank=1)]
    pipeline = _make_pipeline(
        tmp_path,
        vector_store=store,
        retriever=FakeRetriever(results=retrieved),
        generation_service=FakeGenerationService(fail=True),
    )

    with pytest.raises(GenerationError):
        pipeline.ask_question("Question?", use_hybrid=False)
