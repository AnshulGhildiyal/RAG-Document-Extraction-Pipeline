"""Deterministic, rule-based document classification."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional


class DocumentClassifier:
    """Classify document or page text into predefined categories.

    This is a heuristic, rule-based classifier (not a trained ML model).
    """

    SUPPORTED_TYPES = (
        "resume",
        "invoice",
        "payslip",
        "contract",
        "financial_statement",
        "report",
        "general",
    )

    _RULES = [
        (
            "resume",
            ("resume", "cv", "curriculum"),
            ("experience", "education", "skills", "employment", "objective", "career summary", "projects"),
        ),
        (
            "invoice",
            ("invoice", "bill", "receipt", "fee", "worksheet"),
            ("invoice", "total", "amount", "payment", "due date", "subtotal", "purchase price", "closing", "tax"),
        ),
        (
            "payslip",
            ("payslip", "salary", "paystub"),
            ("gross pay", "net pay", "deductions", "salary", "earnings", "employee id", "working days"),
        ),
        (
            "contract",
            ("contract", "agreement", "terms"),
            ("party", "agreement", "terms and conditions", "hereby", "whereas", "obligations", "effective date"),
        ),
        (
            "financial_statement",
            ("balance", "statement", "financial"),
            ("assets", "liabilities", "equity", "balance sheet", "fiscal", "income", "cash flow"),
        ),
        (
            "report",
            ("report", "analysis", "study"),
            ("executive summary", "findings", "conclusion", "methodology", "recommendations"),
        ),
    ]

    def classify(self, text: str, filename: Optional[str] = None) -> str:
        """Classify text (and optional filename hint) into a document type."""
        text_lower = (text or "").lower()[:4000]
        filename_lower = (filename or "").lower()

        if not text_lower.strip() and not filename_lower.strip():
            return "general"

        best_type = "general"
        best_score = 0

        for doc_type, filename_keywords, content_keywords in self._RULES:
            filename_match = any(keyword in filename_lower for keyword in filename_keywords)
            content_matches = sum(1 for keyword in content_keywords if keyword in text_lower)

            # Preserve notebook behavior: filename match or >=2 content keyword matches.
            matched = filename_match or content_matches >= 2
            if not matched:
                continue

            score = content_matches + (2 if filename_match else 0)
            if score > best_score:
                best_score = score
                best_type = doc_type

        return best_type

    def classify_pages(self, page_texts: Iterable[str], filename: Optional[str] = None) -> str:
        """Determine an overall document type from per-page classifications."""
        page_types = [self.classify(text, filename=filename) for text in page_texts]
        page_types = [doc_type for doc_type in page_types if doc_type]
        if not page_types:
            return "general"

        counts = Counter(page_types)
        # Prefer the most common specific type over "general" when counts tie.
        most_common = sorted(
            counts.items(),
            key=lambda item: (item[1], 0 if item[0] == "general" else 1),
            reverse=True,
        )
        return most_common[0][0]
