# Team Contributions

## Workload Allocation (Proposed)
The following distribution reflects the required project ownership and workload allocation for this MCA mini project. It represents proposed responsibilities, **not measured Git statistics**.

| Member | Role | Contribution |
|---|---|---:|
| Anshul | Project Lead, Core Architecture and Algorithms | 35% |
| Ravi | Backend Services and Data Processing | 30% |
| Sarthak | User Interface and Application Integration | 30% |
| Jivika | Quality Assurance and Documentation | 5% |

**Hierarchy:** Anshul (35%) > Ravi (30%) = Sarthak (30%) > Jivika (5%)

> Note: Actual commits in this repository reflect the account/person who performed the implementation in this environment. Do not interpret commit history as fabricated individual contributions.

## Detailed Ownership

### Anshul — 35% (Project Lead and Core Architecture Engineer)
**Owned files (primary):**
- `src/rag_pipeline/core/config.py`
- `src/rag_pipeline/core/models.py`
- `src/rag_pipeline/core/classifier.py`
- `src/rag_pipeline/core/chunker.py`
- `src/rag_pipeline/core/embeddings.py`
- `src/rag_pipeline/core/retriever.py`
- `src/rag_pipeline/core/pipeline.py`
- `src/rag_pipeline/storage/vector_store.py`
- `src/rag_pipeline/main.py`
- `run.py` (architecture/entry integration)

**Main functions/classes:**
- `RAGConfig`, config validation and env loading
- `Document`, `ExtractedPage`, `DocumentChunk`, `RetrievalResult`, `IngestionResult`, `QuestionAnswerResult`
- `DocumentClassifier`
- `TextChunker`
- `EmbeddingEngine`
- `HybridRetriever` (dense/sparse retrieval and weighted RRF fusion)
- `ChromaVectorStore`
- `RAGPipeline` (ingestion and QA orchestration)

**Technical responsibilities:**
- Core architecture and module boundaries
- Configuration and typed data models
- Classification, chunking, embedding, retrieval, and hybrid ranking algorithms
- Persistent storage architecture and ingestion consistency
- Main pipeline controller and final integration decisions

**Suggested Git branch:** `feature/anshul-core`

**Suggested commit messages:**
- `feat(core): add config, models, and classifier modules`
- `feat(core): implement chunking and embedding engine`
- `feat(retrieval): add hybrid retriever with weighted RRF`
- `feat(storage): implement persistent ChromaDB vector store`
- `feat(pipeline): integrate ingestion and QA orchestration`

### Ravi — 30% (Backend Services and Data Processing Engineer)
**Owned files (primary):**
- `src/rag_pipeline/services/document_service.py`
- `src/rag_pipeline/services/extraction_service.py`
- `src/rag_pipeline/services/generation_service.py`
- `src/rag_pipeline/services/validation.py`
- `src/rag_pipeline/services/exceptions.py`
- `src/rag_pipeline/utils/logger.py`

**Main functions/classes:**
- `PDFExtractionService` (Docling primary extraction, pypdf fallback)
- `DocumentService` (document processing/chunk preparation)
- `GenerationService` (Google GenAI/Gemini prompt construction, retries, citations)
- Validation helpers (`validate_pdf_path`, `validate_question`, etc.)
- Custom exception hierarchy (`RAGApplicationError`, `DocumentExtractionError`, `StorageError`, `GenerationError`, `ValidationError`)
- Logging utilities (`configure_logging`, `get_logger`)

**Technical responsibilities:**
- PDF extraction and fallback integration
- Backend document processing services
- Gemini answer generation integration and prompt grounding
- Input validation and service-level error handling
- Logging and error reporting
- Service-to-core integration support

**Suggested Git branch:** `feature/ravi-backend`

**Suggested commit messages:**
- `feat(services): add validation, logging, and exception hierarchy`
- `feat(extraction): implement Docling extraction with pypdf fallback`
- `feat(services): add document processing service`
- `feat(generation): integrate Google GenAI grounded answer generation`
- `fix(services): improve error handling and input safety`

### Sarthak — 30% (Frontend, UX and Application Integration Engineer)
**Owned files (primary):**
- `src/rag_pipeline/interface/gradio_app.py`
- `src/rag_pipeline/interface/formatters.py`
- `src/rag_pipeline/interface/__init__.py` (UI package integration)
- `run.py` (UI launch integration)

**Main functions/classes:**
- `GradioApp` (`build`, `launch`, upload/ask/stats/clear callbacks)
- `create_app` factory
- Formatter functions for ingestion status, answers, sources, retrieval details, and database stats

**Technical responsibilities:**
- Gradio interface architecture and layout (upload, QA, database, about)
- Backend integration through pipeline callbacks
- Result presentation (answers, citations, retrieval details, statistics)
- User input validation feedback and error presentation
- Database clear confirmation and responsive UI interaction
- Interface-level usability and integration behavior

**Suggested Git branch:** `feature/sarthak-interface`

**Suggested commit messages:**
- `feat(ui): build Gradio tabs for upload, QA, and database management`
- `feat(ui): add formatters for answers, sources, and stats`
- `feat(ui): integrate backend pipeline callbacks and clear confirmation`
- `fix(ui): improve user-facing error handling and status messages`

### Jivika — 5% (QA and Academic Documentation Support)
**Owned files (primary):**
- `tests/conftest.py`
- `tests/test_classifier.py`
- `tests/test_chunker.py`
- `tests/test_retriever.py`
- `tests/test_validation.py`
- `tests/test_storage.py`
- `tests/test_pipeline.py`
- `sample_data/README.md`
- `requirements.txt`
- `.env.example`
- `README.md`
- `docs/PROJECT_REPORT.md`
- `docs/ARCHITECTURE.md`
- `docs/TEAM_CONTRIBUTIONS.md`
- `docs/VIVA_PREPARATION.md`

**Main contributions:**
- Basic unit/integration test organization and fixtures
- Dependency and environment documentation
- README and academic documentation formatting/support
- Architecture/DFD documentation support and viva preparation materials

**Technical responsibilities:**
- Test scaffolding and execution support
- Documentation consistency and completeness checks
- Requirements/environment template maintenance
- Academic deliverables preparation (report, diagrams documentation, viva material)

**Suggested Git branch:** `feature/jivika-qa-docs`

**Suggested commit messages:**
- `test: add pytest suite and fixtures for core pipeline components`
- `docs: add README and environment/dependency documentation`
- `docs: prepare project report, architecture, and viva materials`
- `chore: update sample data notes and QA documentation`

## Contribution Summary
- Total proposed allocation: **100%** (35% + 30% + 30% + 5%)
- All members are responsible for validating their owned components; testing is shared, with Jivika's QA/documentation ownership scoped proportionately.
- Ownership proposals are documented for academic evaluation and do not imply fabricated commit authorship.
