"""PDF extraction service with Docling primary parsing and pypdf fallback."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rag_pipeline.core.classifier import DocumentClassifier
from rag_pipeline.core.models import ExtractedPage
from rag_pipeline.services.exceptions import DocumentExtractionError
from rag_pipeline.services.validation import validate_pdf_path
from rag_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class PDFExtractionService:
    """Extract structured page content from PDF files."""

    def __init__(self, classifier: DocumentClassifier | None = None):
        self.classifier = classifier or DocumentClassifier()

    def extract(self, pdf_path: str, max_pages: int = 100, max_size_mb: int = 20) -> Tuple[str, List[ExtractedPage], str]:
        """Extract full text, per-page content, and overall document type."""
        validated_path = validate_pdf_path(pdf_path, max_size_mb=max_size_mb)
        logger.info("Starting PDF extraction: %s", validated_path.name)

        try:
            full_text, pages = self._extract_with_docling(validated_path, max_pages=max_pages)
            parser_used = "docling"
        except Exception as docling_exc:
            logger.warning("Docling extraction failed for %s: %s", validated_path.name, docling_exc)
            try:
                full_text, pages = self._extract_with_pypdf(validated_path, max_pages=max_pages)
                parser_used = "pypdf"
            except Exception as fallback_exc:
                logger.error("PDF extraction failed for %s", validated_path.name)
                raise DocumentExtractionError(
                    f"Unable to extract text from PDF '{validated_path.name}'. "
                    "The file may be unreadable or contain no extractable text."
                ) from fallback_exc

        if not pages:
            raise DocumentExtractionError("No pages could be extracted from the PDF.")

        # Ensure page types and overall type are set consistently.
        for page in pages:
            if not page.doc_type:
                page.doc_type = self.classifier.classify(page.text, filename=validated_path.name)
            page.metadata.setdefault("parser", parser_used)

        overall_type = self.classifier.classify_pages([p.text for p in pages], filename=validated_path.name)
        for page in pages:
            page.metadata.setdefault("overall_doc_type", overall_type)

        full_text = (full_text or "\n\n".join(p.text for p in pages)).strip()
        if len(full_text) < 20:
            raise DocumentExtractionError("Extracted PDF text is empty or too short to process.")

        logger.info(
            "PDF extraction complete (%s): pages=%d chars=%d type=%s",
            parser_used,
            len(pages),
            len(full_text),
            overall_type,
        )
        return full_text, pages, overall_type

    def _extract_with_docling(self, path: Path, max_pages: int) -> Tuple[str, List[ExtractedPage]]:
        try:
            from docling.document_converter import DocumentConverter
        except Exception as exc:  # pragma: no cover - optional dependency/runtime
            raise DocumentExtractionError(f"Docling is not available: {exc}") from exc

        try:
            converter = DocumentConverter()
            result = converter.convert(str(path))
        except Exception as exc:
            raise DocumentExtractionError(f"Docling conversion failed: {exc}") from exc

        document = getattr(result, "document", None)
        if document is None:
            raise DocumentExtractionError("Docling returned no document content.")

        full_text = self._export_docling_full_text(document)
        pages = self._extract_docling_pages(document, path.name, max_pages=max_pages)

        # Fallback within Docling path: split full text by available page count if per-page extraction is weak.
        if (not pages or all(len(p.text.strip()) < 20 for p in pages)) and full_text.strip():
            page_count = self._docling_page_count(document)
            pages = self._split_text_into_pages(full_text, path.name, page_count=page_count, max_pages=max_pages)

        full_text = (full_text or "\n\n".join(p.text for p in pages)).strip()
        if not pages or len(full_text) < 20:
            raise DocumentExtractionError("Docling could not extract usable text.")
        return full_text, pages

    @staticmethod
    def _docling_page_count(document: Any) -> int:
        pages = getattr(document, "pages", None)
        try:
            if pages is None:
                return 1
            if isinstance(pages, dict):
                return max(len(pages), 1)
            return max(len(pages), 1)
        except Exception:
            return 1

    @staticmethod
    def _export_docling_full_text(document: Any) -> str:
        for method_name in ("export_to_markdown", "export_to_text"):
            method = getattr(document, method_name, None)
            if callable(method):
                try:
                    text = method() or ""
                    if str(text).strip():
                        return str(text)
                except Exception:
                    continue
        for attr in ("text", "content"):
            value = getattr(document, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        # Last resort: concatenate page-level exports if available.
        pages = getattr(document, "pages", None)
        page_texts: List[str] = []
        if pages is not None:
            iterable = pages.values() if isinstance(pages, dict) else pages
            for page in iterable:
                text = PDFExtractionService._export_docling_page_text(page)
                if text.strip():
                    page_texts.append(text)
        return "\n\n".join(page_texts).strip()

    @staticmethod
    def _export_docling_page_text(page: Any) -> str:
        for method_name in ("export_to_markdown", "export_to_text"):
            method = getattr(page, method_name, None)
            if callable(method):
                try:
                    text = method() or ""
                    if str(text).strip():
                        return str(text)
                except Exception:
                    continue
        for attr in ("text", "content"):
            value = getattr(page, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _extract_docling_pages(self, document: Any, filename: str, max_pages: int) -> List[ExtractedPage]:
        pages_obj = getattr(document, "pages", None)
        if pages_obj is None:
            return []

        extracted: List[ExtractedPage] = []
        if isinstance(pages_obj, dict):
            items = sorted(pages_obj.items(), key=lambda kv: self._safe_page_number(kv[0], default=len(extracted) + 1))
            iterable = [(self._safe_page_number(k, default=i + 1), v) for i, (k, v) in enumerate(items)]
        else:
            iterable = list(enumerate(list(pages_obj), start=1))

        for idx, page in iterable:
            page_num = self._safe_page_number(idx, default=len(extracted) + 1)
            if page_num > max_pages:
                logger.warning("Skipping pages beyond max_pages=%d", max_pages)
                break
            text = self._export_docling_page_text(page).strip()
            if len(text) < 20:
                continue
            doc_type = self.classifier.classify(text, filename=filename)
            extracted.append(
                ExtractedPage(
                    page_number=page_num,
                    text=text,
                    char_count=len(text),
                    doc_type=doc_type,
                    metadata={"parser": "docling"},
                )
            )
        return extracted

    @staticmethod
    def _safe_page_number(value: Any, default: int) -> int:
        try:
            num = int(value)
            return num if num > 0 else default
        except Exception:
            digits = "".join(ch for ch in str(value) if ch.isdigit())
            if digits:
                try:
                    num = int(digits)
                    return num if num > 0 else default
                except Exception:
                    return default
            return default

    def _extract_with_pypdf(self, path: Path, max_pages: int) -> Tuple[str, List[ExtractedPage]]:
        reader = None
        last_exc: Exception | None = None
        for module_name in ("pypdf", "PyPDF2"):
            try:
                module = __import__(module_name, fromlist=["PdfReader"])
                reader_cls = getattr(module, "PdfReader")
                with path.open("rb") as handle:
                    reader = reader_cls(handle)
                    pages_obj = list(getattr(reader, "pages", []))
                # Re-open properly for extraction to avoid closed-stream issues in some implementations.
                with path.open("rb") as handle:
                    reader = reader_cls(handle)
                    pages_obj = list(getattr(reader, "pages", []))
                    break
            except Exception as exc:  # pragma: no cover - depends on installed lib/version
                last_exc = exc
                reader = None
                continue

        if reader is None:
            raise DocumentExtractionError(f"pypdf/PyPDF2 fallback unavailable: {last_exc}")

        extracted_pages: List[ExtractedPage] = []
        page_texts: List[str] = []
        try:
            with path.open("rb") as handle:
                module = None
                reader_obj = None
                for module_name in ("pypdf", "PyPDF2"):
                    try:
                        module = __import__(module_name, fromlist=["PdfReader"])
                        reader_obj = module.PdfReader(handle)
                        break
                    except Exception:
                        handle.seek(0)
                        continue
                if reader_obj is None:
                    raise DocumentExtractionError("Unable to initialize PDF reader for fallback extraction.")

                total_pages = min(len(reader_obj.pages), max_pages)
                for idx in range(total_pages):
                    page = reader_obj.pages[idx]
                    text = ""
                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        text = ""
                    text = (text or "").strip()
                    page_num = idx + 1
                    if not text:
                        continue
                    page_texts.append(text)
                    doc_type = self.classifier.classify(text, filename=path.name)
                    extracted_pages.append(
                        ExtractedPage(
                            page_number=page_num,
                            text=text,
                            char_count=len(text),
                            doc_type=doc_type,
                            metadata={"parser": "pypdf"},
                        )
                    )
        except DocumentExtractionError:
            raise
        except Exception as exc:
            raise DocumentExtractionError(f"Fallback PDF extraction failed: {exc}") from exc

        full_text = "\n\n".join(page_texts).strip()
        if not extracted_pages or len(full_text) < 20:
            raise DocumentExtractionError("Fallback parser could not extract usable text from the PDF.")
        return full_text, extracted_pages

    def _split_text_into_pages(
        self,
        full_text: str,
        filename: str,
        page_count: int,
        max_pages: int,
    ) -> List[ExtractedPage]:
        text = (full_text or "").strip()
        if not text:
            return []
        count = max(1, min(page_count or 1, max_pages))
        approx = max(1, len(text) // count)
        pages: List[ExtractedPage] = []
        start = 0
        for page_num in range(1, count + 1):
            if start >= len(text):
                break
            if page_num == count:
                end = len(text)
            else:
                target = min(len(text), start + approx)
                window_start = max(start, target - 120)
                window_end = min(len(text), target + 120)
                split_at = text.rfind("\n\n", window_start, window_end)
                if split_at <= start:
                    split_at = text.rfind("\n", window_start, window_end)
                if split_at <= start:
                    split_at = target
                end = split_at
            page_text = text[start:end].strip()
            start = end
            if len(page_text) < 20:
                continue
            pages.append(
                ExtractedPage(
                    page_number=page_num,
                    text=page_text,
                    char_count=len(page_text),
                    doc_type=self.classifier.classify(page_text, filename=filename),
                    metadata={"parser": "docling", "reconstructed_page": True},
                )
            )
        return pages
