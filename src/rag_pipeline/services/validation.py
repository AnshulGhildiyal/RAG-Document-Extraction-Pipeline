"""Input and file validation helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from rag_pipeline.services.exceptions import ValidationError


PDF_SIGNATURE = b"%PDF"


def _is_pdf_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header = handle.read(5)
        return header.startswith(PDF_SIGNATURE)
    except OSError:
        return False


def validate_pdf_path(
    file_path: str | os.PathLike[str],
    max_size_mb: int = 20,
    allowed_root: Optional[str | os.PathLike[str]] = None,
) -> Path:
    """Validate that a path points to a readable, non-empty PDF file.

    Returns a resolved Path on success.
    """
    if file_path is None:
        raise ValidationError("No file path provided.")

    path = Path(file_path)
    if not str(path).strip():
        raise ValidationError("File path is empty.")

    # Resolve to an absolute path and optionally constrain to an allowed root.
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValidationError(f"File not found: {path}") from exc
    except OSError as exc:
        raise ValidationError(f"Unable to access file: {path}") from exc

    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValidationError("File path is outside the allowed upload directory.") from exc

    if not resolved.is_file():
        raise ValidationError("Uploaded path is not a file.")

    if resolved.suffix.lower() != ".pdf":
        raise ValidationError("Only PDF files are supported.")

    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise ValidationError(f"Unable to read file metadata: {resolved.name}") from exc

    if size <= 0:
        raise ValidationError("The uploaded PDF file is empty.")

    max_bytes = max_size_mb * 1024 * 1024
    if size > max_bytes:
        raise ValidationError(
            f"File size ({size / (1024 * 1024):.2f} MB) exceeds the limit of {max_size_mb} MB."
        )

    if not _is_pdf_file(resolved):
        raise ValidationError("The file does not appear to be a valid PDF document.")

    return resolved


def validate_question(question: str, min_length: int = 1, max_length: int = 2000) -> str:
    """Validate and normalize a user question."""
    if question is None:
        raise ValidationError("Question cannot be empty.")
    normalized = " ".join(str(question).strip().split())
    if len(normalized) < min_length:
        raise ValidationError("Please enter a non-empty question.")
    if len(normalized) > max_length:
        raise ValidationError(f"Question is too long (maximum {max_length} characters).")
    return normalized


def validate_non_empty_text(text: Optional[str], context: str = "text") -> str:
    """Ensure extracted or provided text is not empty."""
    if text is None or not str(text).strip():
        raise ValidationError(f"No usable {context} was provided.")
    return str(text)
