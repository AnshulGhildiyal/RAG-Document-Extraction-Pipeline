"""Custom exception hierarchy for the RAG pipeline."""


class RAGApplicationError(Exception):
    """Base exception for application-level failures."""


class DocumentExtractionError(RAGApplicationError):
    """Raised when PDF extraction or parsing fails."""


class StorageError(RAGApplicationError):
    """Raised when vector storage operations fail."""


class GenerationError(RAGApplicationError):
    """Raised when answer generation fails."""


class ValidationError(RAGApplicationError):
    """Raised when user input or configuration is invalid."""
