# Viva Preparation

This document provides five concise technical questions and answers for each team member, aligned with the implemented project.

## Anshul (Core Architecture and Algorithms)

### 1. What is RAG and how is it implemented in this project?
**Answer:** Retrieval-Augmented Generation (RAG) combines information retrieval with LLM generation. In this project, uploaded PDFs are extracted, chunked, embedded, and stored in ChromaDB. For a user question, the system retrieves relevant chunks (dense and/or sparse), builds a context prompt with source metadata, and uses Gemini to generate a grounded answer with source references.

### 2. What embedding model is used and why?
**Answer:** The project uses `sentence-transformers/all-MiniLM-L6-v2` (default) via `EmbeddingEngine`. It provides compact semantic embeddings suitable for similarity search and is efficient for local embedding generation. The engine lazily loads the model, reuses it, and supports batch encoding and normalization.

### 3. How is ChromaDB used in the pipeline?
**Answer:** `ChromaVectorStore` uses a persistent ChromaDB client and `get_or_create_collection` (no automatic deletion on startup). It stores chunk texts, embeddings, and metadata (document/page/source/type info), supports vector queries, statistics, document deletion, and explicit collection clearing. Persistence is configured via `CHROMA_DB_PATH` (default `./data/chroma_db`).

### 4. What is hybrid retrieval in this implementation?
**Answer:** Hybrid retrieval combines dense semantic search (vector similarity over embeddings in ChromaDB) and sparse keyword search (TF-IDF cosine similarity over stored chunk texts). The retriever merges both result sets, deduplicates chunks by ID, and ranks them using weighted Reciprocal Rank Fusion (RRF) to balance semantic relevance and lexical matches.

### 5. How is Reciprocal Rank Fusion (RRF) applied here?
**Answer:** The retriever computes fused scores using \(RRF(d)=\sum_i \frac{w_i}{k + rank_i(d)}\), where ranks come from dense and sparse result lists. Defaults are `dense_weight=0.6`, `sparse_weight=0.4`, and `rrf_k=60`. Chunks are deduplicated, sorted by fused score (with stable tie-breaking), and the top-k results are returned with preserved metadata.

## Ravi (Backend Services and Data Processing)

### 1. Why is Docling used for PDF extraction, and what is the fallback?
**Answer:** Docling is used as the primary parser to extract structured text/markdown content from PDFs. If Docling fails or returns unusable content, `PDFExtractionService` falls back to `pypdf` (with compatibility handling) to extract page text. This improves robustness across different PDF formats.

### 2. How does the extraction service handle invalid or problematic PDFs?
**Answer:** Before extraction, `validate_pdf_path` checks file existence, extension, size, emptiness, and PDF signature. The extraction service logs failures, attempts fallback parsing, and raises `DocumentExtractionError` when no usable text/pages can be produced. It avoids inventing page references and only returns structured pages with extracted content.

### 3. How is Gemini integrated for answer generation?
**Answer:** `GenerationService` uses the maintained `google-genai` SDK (`google.genai.Client`) and reads `GEMINI_API_KEY` from configuration/environment. It builds a grounded prompt containing retrieved source snippets and metadata, instructs the model to answer only from evidence, supports `[Source N]` citations, and retries on transient/rate-limit failures with bounded backoff.

### 4. What exception handling strategy is used in backend services?
**Answer:** The project defines `RAGApplicationError` as a base with specific subclasses: `ValidationError`, `DocumentExtractionError`, `StorageError`, and `GenerationError`. Services validate inputs early, catch low-level/runtime/API errors, log them, and raise meaningful domain-specific exceptions so the pipeline and UI can respond appropriately.

### 5. How is logging implemented and used?
**Answer:** `utils/logger.py` provides centralized logging configuration (`configure_logging`) and module loggers (`get_logger`). Logging records application startup, ingestion and extraction events, embedding/retrieval operations, and generation/storage/database errors. Log level and file path are configurable through `RAGConfig`/environment variables.

## Sarthak (User Interface and Application Integration)

### 1. How is the Gradio interface structured?
**Answer:** `GradioApp` builds a tabbed interface with Document Upload, Question Answering, Database Management, and About sections. It exposes callbacks for document processing, question submission, stats refresh, and database clearing, and uses `formatters.py` to present consistent Markdown/structured outputs.

### 2. How do UI callbacks integrate with the backend pipeline?
**Answer:** The UI holds a `RAGPipeline` instance and calls its methods directly: `ingest_document()` for uploads, `ask_question()` for QA, `get_database_stats()` for status display, and `clear_database()` for explicit deletion. Callback outputs update status messages, answer/source panels, retrieval details, and database statistics.

### 3. How is UI state and feedback managed during operations?
**Answer:** The interface provides user-visible status areas (upload status, ingestion details JSON, answer/sources/retrieval panels, database stats, clear status). Buttons trigger backend actions and refresh dependent views (e.g., stats after upload/clear). The app enables Gradio queueing to improve responsiveness during longer operations.

### 4. How does the interface handle errors safely?
**Answer:** Callbacks catch validation and application-specific exceptions and render friendly Markdown error messages via `format_error(...)`. Raw Python tracebacks are not shown to users. Unexpected exceptions are logged and replaced with generic user-facing failure messages.

### 5. How are citations and database information presented?
**Answer:** Answers are displayed separately from source references. `format_sources_markdown()` lists retrieved sources with filename, page number, document type, relevance, retrieval method, and previews, distinguishing cited/retrieved references. Database management shows real collection statistics (documents, chunks, pages, type counts) and requires explicit confirmation before clearing.

## Jivika (Quality Assurance and Documentation)

### 1. What is the role of pytest in this project?
**Answer:** Pytest is used to implement automated tests for core components (classification, chunking, retrieval, validation, storage, pipeline orchestration). Tests verify expected behavior, edge cases, and failure handling. The project configures pytest in `pyproject.toml` (`testpaths`, `pythonpath`, default options) for consistent execution.

### 2. What is the difference between unit and integration-style testing here?
**Answer:** Unit tests focus on isolated components (e.g., classifier rules, chunker behavior, validators, retriever fusion logic with fakes). Integration-style tests exercise orchestration across components, such as `RAGPipeline` ingestion/QA flows using mocked services/stores/generation to validate component interactions and error propagation.

### 3. How are fixtures used in the test suite?
**Answer:** `tests/conftest.py` provides shared fixtures (e.g., temporary config paths, sample extracted pages/chunks) to standardize test setup and avoid duplication. Fixtures help ensure isolated, repeatable tests and simplify working with temporary directories and sample data structures.

### 4. How do tests avoid external dependencies and expensive operations?
**Answer:** Tests mock external APIs and expensive ML components (e.g., Gemini generation, embedding/retrieval dependencies in pipeline tests) and use temporary directories/isolated storage paths. Storage tests use `pytest.importorskip("chromadb")` so they can be skipped if the dependency is unavailable, and no real API key or internet access is required.

### 5. How are dependencies and environment setup documented for reproducibility?
**Answer:** The project includes `requirements.txt`, `pyproject.toml`, and `.env.example` to define dependencies and configurable environment variables. The README documents virtual environment setup for Windows/Linux/macOS, dependency installation, API key configuration, application launch (`python run.py`), and test execution commands.
