"""Package entry point for launching the RAG pipeline application."""

from __future__ import annotations

from rag_pipeline.core.config import RAGConfig
from rag_pipeline.core.pipeline import RAGPipeline
from rag_pipeline.interface.gradio_app import GradioApp
from rag_pipeline.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    """Initialize the pipeline and launch the Gradio interface."""
    config = RAGConfig.from_env()
    configure_logging(config.log_level, config.log_file)
    logger.info("Starting RAG Document Extraction Pipeline application")
    pipeline = RAGPipeline(config=config)
    app = GradioApp(pipeline=pipeline)
    app.launch()


if __name__ == "__main__":
    main()
