"""Tests for input and file validation."""

from pathlib import Path

import pytest

from rag_pipeline.services.exceptions import ValidationError
from rag_pipeline.services.validation import validate_pdf_path, validate_question


def _write_minimal_pdf(path: Path, content: bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n") -> Path:
    path.write_bytes(content)
    return path


def test_validate_question_rejects_empty_and_accepts_trimmed_text():
    with pytest.raises(ValidationError):
        validate_question("")
    with pytest.raises(ValidationError):
        validate_question("   ")
    assert validate_question("  What is   the total?  ") == "What is the total?"


def test_validate_pdf_path_rejects_missing_and_non_pdf(tmp_path: Path):
    with pytest.raises(ValidationError):
        validate_pdf_path(tmp_path / "missing.pdf")

    txt = tmp_path / "notes.txt"
    txt.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(ValidationError):
        validate_pdf_path(txt)


def test_validate_pdf_path_rejects_empty_and_invalid_signature(tmp_path: Path):
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    with pytest.raises(ValidationError):
        validate_pdf_path(empty_pdf)

    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"NOTPDF content")
    with pytest.raises(ValidationError):
        validate_pdf_path(fake_pdf)


def test_validate_pdf_path_accepts_valid_pdf_and_enforces_size(tmp_path: Path):
    valid = _write_minimal_pdf(tmp_path / "valid.pdf")
    resolved = validate_pdf_path(valid, max_size_mb=5)
    assert resolved.exists() and resolved.suffix == ".pdf"

    # Create a file just over 1 MB limit when interpreted as MB threshold.
    large = tmp_path / "large.pdf"
    large.write_bytes(b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10))
    with pytest.raises(ValidationError):
        validate_pdf_path(large, max_size_mb=1)
