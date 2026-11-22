"""Main RAG pipeline controller coordinating ingestion and question answering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from rag_pipeline.core.chunker import TextChunker
from rag_pipeline.core.classifier import DocumentClassifier
from rag_pipeline.core.config import RAGConfig
from rag_pipeline.core.embeddings import EmbeddingEngine
from rag_pipeline.core.models import IngestionResult, QuestionAnswerResult
from rag_pipeline.core.retriever import HybridRetriever
from rag_pipeline.services.document_service import DocumentService
from rag_pipeline.services.exceptions import GenerationError, RAGApplicationError, StorageError, ValidationError
from rag_pipeline.services.extraction_service import PDFExtractionService
from rag_pipeline.services.generation_service import GenerationService
from rag_pipeline.services.validation import validate_pdf_path, validate_question
from rag_pipeline.storage.vector_store import ChromaVectorStore
from rag_pipeline.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Coordinate document ingestion, retrieval, and grounded answer generation."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        extraction_service: Optional[PDFExtractionService] = None,
        document_service: Optional[DocumentService] = None,
        generation_service: Optional[GenerationService] = None,
        retriever: Optional[HybridRetriever] = None,
    ):
        self.config = config or RAGConfig.from_env()
        configure_logging(self.config.log_level, self.config.log_file)

        self.classifier = DocumentClassifier()
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.embedding_engine = embedding_engine or EmbeddingEngine(model_name=self.config.embedding_model)
        self.vector_store = vector_store or ChromaVectorStore(
            persist_directory=self.config.chroma_path,
            collection_name=self.config.collection_name,
        )
        self.extraction_service = extraction_service or PDFExtractionService(classifier=self.classifier)
        self.document_service = document_service or DocumentService(
            extraction_service=self.extraction_service,
            classifier=self.classifier,
            chunker=self.chunker,
        )
        self.retriever = retriever or HybridRetriever(
            vector_store=self.vector_store,
            embedding_engine=self.embedding_engine,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
            rrf_k=self.config.rrf_k,
        )
        self.generation_service = generation_service or GenerationService(
            model_name=self.config.gemini_model,
            temperature=self.config.llm_temperature,
            max_output_tokens=self.config.llm_max_tokens,
            api_key=self.config.gemini_api_key,
        )

        logger.info("RAG pipeline initialized.")

    def ingest_document(self, file_path: str) -> IngestionResult:
        """Ingest a PDF: validate, extract, classify, chunk, embed, and store."""
        validated_path = validate_pdf_path(file_path, max_size_mb=self.config.max_upload_size_mb)
        logger.info("Ingesting document: %s", validated_path.name)

        try:
            document, chunks = self.document_service.process_pdf(
                pdf_path=str(validated_path),
                max_pages=self.config.max_pdf_pages,
                max_size_mb=self.config.max_upload_size_mb,
            )
        except Exception as exc:
            logger.error("Document processing failed for %s: %s", validated_path.name, exc)
            raise

        texts = [chunk.text for chunk in chunks]
        try:
            embeddings = self.embedding_engine.embed_texts(texts)
        except Exception as exc:
            logger.error("Embedding generation failed for %s: %s", validated_path.name, exc)
            raise

        # Replace existing chunks for this document to keep ingestion consistent on re-upload.
        try:
            self.vector_store.delete_document(document.document_id)
        except Exception as exc:
            logger.warning("Failed to remove existing chunks for document %s before re-ingest: %s", document.document_id, exc)

        try:
            if hasattr(embeddings, "tolist"):
                embeddings_payload = embeddings.tolist()
            else:
                embeddings_payload = [list(vec) for vec in embeddings]
            inserted = self.vector_store.add_chunks(chunks, embeddings_payload)
        except Exception as exc:
            logger.error("Storage failed during ingestion for %s: %s", validated_path.name, exc)
            raise StorageError(f"Failed to store document '{document.filename}': {exc}") from exc

        self.retriever.invalidate_sparse_index()

        avg_chunk_size = int(sum(len(c.text) for c in chunks) / len(chunks)) if chunks else 0
        result = IngestionResult(
            document_id=document.document_id,
            filename=document.filename,
            source_path=document.source_path,
            doc_type=document.doc_type,
            page_count=document.page_count,
            chunk_count=len(chunks),
            total_characters=document.total_characters,
            average_chunk_size=avg_chunk_size,
            message=f"Document ingested successfully ({inserted} chunk(s) stored).",
            success=True,
        )
        logger.info(
            "Ingestion complete: %s (pages=%d chunks=%d stored=%d type=%s)",
            document.filename,
            document.page_count,
            len(chunks),
            inserted,
            document.doc_type,
        )
        return result

    def ask_question(self, question: str, use_hybrid: bool = True) -> QuestionAnswerResult:
        """Answer a question using retrieved document context and Gemini generation."""
        normalized_question = validate_question(question)
        logger.info("Received question: %s", normalized_question)

        if self.vector_store.count() == 0:
            return QuestionAnswerResult(
                question=normalized_question,
                answer="No documents have been uploaded yet. Please upload a PDF document first.",
                sources=[],
                retrieval_results=[],
                context_used=False,
                error="No documents in database.",
            )

        try:
            if use_hybrid:
                retrieved = self.retriever.hybrid_search(normalized_question, top_k_total=self.config.top_k_hybrid)
            else:
                retrieved = self.retriever.dense_search(normalized_question, top_k=self.config.top_k_vector)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            raise RAGApplicationError(f"Failed to retrieve relevant document context: {exc}") from exc

        if not retrieved:
            return QuestionAnswerResult(
                question=normalized_question,
                answer="I cannot find this information in the provided document context.",
                sources=[],
                retrieval_results=[],
                context_used=False,
                error="No relevant chunks found.",
            )

        filtered = [r for r in retrieved if r.relevance_score >= self.config.similarity_threshold]
        if not filtered:
            # If strict threshold filters everything out, still try the top retrieved chunks
            # to avoid missing relevant context due to score scaling differences.
            filtered = retrieved[: min(3, len(retrieved))]

        try:
            result = self.generation_service.generate_answer(normalized_question, filtered)
            # Ensure retrieval results returned even if generation service omits details.
            if not result.retrieval_results:
                result.retrieval_results = filtered
            result.question = normalized_question
            return result
        except GenerationError:
            raise
        except Exception as exc:
            logger.error("Answer generation failed: %s", exc)
            raise GenerationError(f"Failed to generate answer: {exc}") from exc

    def get_database_stats(self) -> Dict[str, Any]:
        """Return vector database statistics for UI and diagnostics."""
        stats = self.vector_store.get_stats()
        stats.update(
            {
                "embedding_model": self.config.embedding_model,
                "gemini_model": self.config.gemini_model,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
            }
        )
        return stats

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document's chunks from the vector store."""
        if not document_id or not str(document_id).strip():
            raise ValidationError("document_id is required.")
        deleted = self.vector_store.delete_document(str(document_id).strip())
        self.retriever.invalidate_sparse_index()
        logger.info("Delete requested for document_id=%s", document_id)
        return {"document_id": document_id, "deleted": deleted, "success": True}

    def clear_database(self) -> Dict[str, Any]:
        """Explicitly clear the vector database collection."""
        self.vector_store.clear()
        self.retriever.invalidate_sparse_index()
        logger.info("Vector database cleared.")
        return {"cleared": True, "collection_name": self.config.collection_name}
