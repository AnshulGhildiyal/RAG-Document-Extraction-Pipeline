"""Gradio interface for the RAG Document Extraction Pipeline."""

from __future__ import annotations

from typing import Any, Tuple

import gradio as gr

from rag_pipeline.core.pipeline import RAGPipeline
from rag_pipeline.interface.formatters import (
    format_answer_markdown,
    format_database_stats,
    format_error,
    format_ingestion_success,
    format_retrieval_details,
    format_sources_markdown,
)
from rag_pipeline.services.exceptions import RAGApplicationError, ValidationError
from rag_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class GradioApp:
    """Build and serve the Gradio UI backed by the RAG pipeline."""

    def __init__(self, pipeline: RAGPipeline | None = None):
        self.pipeline = pipeline or RAGPipeline()

    # --- Backend callbacks ---

    def upload_document(self, file_obj: Any) -> Tuple[str, dict, str]:
        if file_obj is None:
            return (
                format_error("Upload failed", "No file was selected. Please choose a PDF document."),
                {},
                self.get_database_stats(),
            )
        file_path = getattr(file_obj, "name", None) or str(file_obj)
        try:
            result = self.pipeline.ingest_document(file_path)
            stats = self.pipeline.get_database_stats()
            status = format_ingestion_success(result, total_chunks_in_db=int(stats.get("total_chunks", 0) or 0))
            details = {
                "document_id": result.document_id,
                "filename": result.filename,
                "doc_type": result.doc_type,
                "pages": result.page_count,
                "chunks": result.chunk_count,
                "total_characters": result.total_characters,
                "average_chunk_size": result.average_chunk_size,
            }
            return status, details, format_database_stats(stats)
        except ValidationError as exc:
            logger.warning("Upload validation failed: %s", exc)
            return (format_error("Upload failed", str(exc)), {}, self.get_database_stats())
        except RAGApplicationError as exc:
            logger.error("Upload processing failed: %s", exc)
            return (format_error("Upload failed", str(exc)), {}, self.get_database_stats())
        except Exception as exc:  # pragma: no cover - defensive UI guard
            logger.exception("Unexpected upload error")
            return (
                format_error("Upload failed", "An unexpected error occurred while processing the document."),
                {},
                self.get_database_stats(),
            )

    def ask_question(self, question: str, use_hybrid: bool) -> Tuple[str, str, str]:
        if not question or not str(question).strip():
            msg = format_error("Invalid question", "Please enter a question before submitting.")
            return msg, "### Sources\n\nNo sources.", "No retrieval details available."
        try:
            result = self.pipeline.ask_question(question, use_hybrid=bool(use_hybrid))
            answer_md = format_answer_markdown(result)
            sources_md = format_sources_markdown(result)
            details_md = format_retrieval_details(result.retrieval_results or [])
            return answer_md, sources_md, details_md
        except ValidationError as exc:
            msg = format_error("Invalid question", str(exc))
            return msg, "### Sources\n\nNo sources.", "No retrieval details available."
        except RAGApplicationError as exc:
            logger.error("Question answering failed: %s", exc)
            msg = format_error("Unable to answer question", str(exc))
            return msg, "### Sources\n\nNo sources.", "No retrieval details available."
        except Exception:  # pragma: no cover
            logger.exception("Unexpected QA error")
            msg = format_error(
                "Unable to answer question",
                "An unexpected error occurred while generating the answer. Please try again.",
            )
            return msg, "### Sources\n\nNo sources.", "No retrieval details available."

    def get_database_stats(self) -> str:
        try:
            stats = self.pipeline.get_database_stats()
            return format_database_stats(stats)
        except RAGApplicationError as exc:
            logger.error("Failed to fetch database stats: %s", exc)
            return format_error("Database status unavailable", str(exc))
        except Exception:  # pragma: no cover
            logger.exception("Unexpected stats error")
            return format_error("Database status unavailable", "Unable to retrieve database statistics.")

    def clear_database(self, confirm: bool) -> Tuple[str, str]:
        if not confirm:
            return (
                format_error("Clear cancelled", "Please confirm database clearing before proceeding."),
                self.get_database_stats(),
            )
        try:
            self.pipeline.clear_database()
            return ("### Database cleared\n\nAll stored document chunks were removed.", self.get_database_stats())
        except RAGApplicationError as exc:
            logger.error("Failed to clear database: %s", exc)
            return (format_error("Clear failed", str(exc)), self.get_database_stats())
        except Exception:  # pragma: no cover
            logger.exception("Unexpected clear error")
            return (
                format_error("Clear failed", "An unexpected error occurred while clearing the database."),
                self.get_database_stats(),
            )

    # --- UI construction ---

    def build(self) -> gr.Blocks:
        cfg = self.pipeline.config
        with gr.Blocks(title="RAG Document Extraction Pipeline") as demo:
            gr.Markdown(
                "# Advanced RAG Document Extraction Pipeline\n"
                "Upload PDF documents, ask questions, and review grounded answers with source citations."
            )

            with gr.Tabs():
                with gr.TabItem("Document Upload"):
                    gr.Markdown(
                        "Upload a PDF to extract text, classify pages, chunk content, generate embeddings, and store vectors in ChromaDB."
                    )
                    with gr.Row():
                        with gr.Column(scale=2):
                            file_input = gr.File(
                                label="PDF file",
                                file_types=[".pdf"],
                                type="filepath",
                            )
                            upload_btn = gr.Button("Process document", variant="primary")
                        with gr.Column(scale=1):
                            upload_details = gr.JSON(label="Ingestion details", value={})
                    upload_status = gr.Markdown(value="*No document processed yet.*")

                with gr.TabItem("Question Answering"):
                    gr.Markdown("Ask questions about the uploaded documents. Answers are generated from retrieved context.")
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="e.g., What is the total amount on the invoice?",
                        lines=3,
                    )
                    with gr.Row():
                        use_hybrid = gr.Checkbox(
                            label="Use hybrid retrieval (vector + TF-IDF with RRF)",
                            value=True,
                        )
                        ask_btn = gr.Button("Ask", variant="primary")
                    answer_output = gr.Markdown(label="Answer", value="### Answer\n\nAsk a question to get started.")
                    with gr.Row():
                        sources_output = gr.Markdown(label="Sources", value="### Sources\n\nNo sources yet.")
                    with gr.Accordion("Retrieved context details", open=False):
                        retrieval_output = gr.Markdown(value="No retrieval details available.")

                with gr.TabItem("Database Management"):
                    gr.Markdown("View and manage the persistent vector database.")
                    stats_output = gr.Markdown(value=self.get_database_stats())
                    with gr.Row():
                        refresh_btn = gr.Button("Refresh statistics")
                    gr.Markdown("#### Clear database")
                    gr.Markdown(
                        "Clearing removes all stored document chunks from the collection. This action cannot be undone."
                    )
                    confirm_clear = gr.Checkbox(label="I understand and want to clear the database", value=False)
                    clear_btn = gr.Button("Clear database", variant="stop")
                    clear_status = gr.Markdown(value="")

                with gr.TabItem("About"):
                    gr.Markdown(
                        f"""
### System configuration
- **Embedding model:** `{cfg.embedding_model}`
- **Chunk size / overlap:** {cfg.chunk_size} / {cfg.chunk_overlap}
- **Vector top-k:** {cfg.top_k_vector}
- **Keyword (TF-IDF) top-k:** {cfg.top_k_knn}
- **Hybrid top-k:** {cfg.top_k_hybrid}
- **Gemini model:** `{cfg.gemini_model}`
- **Temperature:** {cfg.llm_temperature}
- **ChromaDB path:** `{cfg.chroma_path}`

### Notes
- PDF extraction uses Docling with a pypdf fallback.
- Retrieval combines dense vector search and sparse TF-IDF search using weighted Reciprocal Rank Fusion (RRF).
- Answers are generated with Google Gemini using retrieved context and include source references.
- Uploaded document content is treated as untrusted data.
"""
                    )

            upload_btn.click(
                fn=self.upload_document,
                inputs=[file_input],
                outputs=[upload_status, upload_details, stats_output],
            )

            ask_btn.click(
                fn=self.ask_question,
                inputs=[question_input, use_hybrid],
                outputs=[answer_output, sources_output, retrieval_output],
            )

            refresh_btn.click(
                fn=self.get_database_stats,
                inputs=[],
                outputs=[stats_output],
            )

            clear_btn.click(
                fn=self.clear_database,
                inputs=[confirm_clear],
                outputs=[clear_status, stats_output],
            )

        demo.queue()
        return demo

    def launch(self, **kwargs):
        demo = self.build()
        launch_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": False,
            "show_error": False,
        }
        launch_kwargs.update(kwargs)
        return demo.launch(**launch_kwargs)


def create_app(pipeline: RAGPipeline | None = None) -> gr.Blocks:
    """Convenience factory for creating the Gradio Blocks app."""
    return GradioApp(pipeline=pipeline).build()
