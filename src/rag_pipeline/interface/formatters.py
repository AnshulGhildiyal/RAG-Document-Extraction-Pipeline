"""Formatting helpers for presenting pipeline results in the UI."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from rag_pipeline.core.models import IngestionResult, QuestionAnswerResult, RetrievalResult


def format_ingestion_success(result: IngestionResult, total_chunks_in_db: int | None = None) -> str:
    """Format a successful ingestion summary as Markdown."""
    lines = [
        "### Document processed successfully",
        "",
        f"- **File:** {result.filename}",
        f"- **Document ID:** `{result.document_id}`",
        f"- **Detected type:** {result.doc_type}",
        f"- **Pages processed:** {result.page_count}",
        f"- **Chunks created:** {result.chunk_count}",
        f"- **Total characters:** {result.total_characters:,}",
        f"- **Average chunk size:** {result.average_chunk_size} chars",
    ]
    if total_chunks_in_db is not None:
        lines.append(f"- **Total chunks in database:** {total_chunks_in_db}")
    if result.message:
        lines.extend(["", result.message])
    return "\n".join(lines)


def format_error(title: str, message: str, details: Iterable[str] | None = None) -> str:
    """Format a user-facing error message without exposing raw tracebacks."""
    lines = [f"### {title}", "", str(message).strip() or "An unexpected error occurred."]
    detail_items = [d for d in (details or []) if str(d).strip()]
    if detail_items:
        lines.extend(["", "**Details:**"])
        for item in detail_items:
            lines.append(f"- {item}")
    return "\n".join(lines)


def format_answer_markdown(result: QuestionAnswerResult) -> str:
    """Format the generated answer for display."""
    answer = (result.answer or "").strip() or "No answer was generated."
    header = "### Answer"
    if result.error and not result.context_used:
        header = "### Answer (insufficient context)"
    return f"{header}\n\n{answer}"


def format_sources_markdown(result: QuestionAnswerResult) -> str:
    """Format cited/retrieved source references for display."""
    sources = result.sources or []
    if not sources:
        if result.retrieval_results:
            return "### Sources\n\nRetrieved context is available in the details section, but no explicit source citations were returned."
        return "### Sources\n\nNo source references available."

    lines = ["### Sources"]
    for src in sources:
        source_num = src.get("source_num", "?")
        filename = src.get("filename", "Unknown document")
        page = src.get("page_number", "?")
        doc_type = src.get("doc_type", "unknown")
        score = src.get("relevance_score")
        method = src.get("retrieval_method", "retrieval")
        cited = src.get("cited_in_answer")
        cited_label = "Yes" if cited else ("No" if cited is not None else "Retrieved")
        lines.extend(
            [
                "",
                f"**[Source {source_num}] {filename} — page {page}**",
                f"- Document type: {doc_type}",
                f"- Retrieval: {method}",
                f"- Relevance: {float(score):.3f}" if score is not None else "- Relevance: n/a",
                f"- Cited in answer: {cited_label}",
            ]
        )
        preview = str(src.get("text_preview", "")).strip()
        if preview:
            lines.append(f"- Preview: {preview}")
    return "\n".join(lines)


def format_retrieval_details(results: List[RetrievalResult]) -> str:
    """Format retrieval details for an expandable UI section."""
    if not results:
        return "No retrieval details available."
    lines = [f"Retrieved {len(results)} chunk(s):", ""]
    for res in results:
        chunk = res.chunk
        lines.append(
            f"- **Rank {res.rank}** | score={res.relevance_score:.3f} | method={res.retrieval_method} | "
            f"doc=`{chunk.document_id or 'n/a'}` | file={chunk.filename or 'n/a'} | page={chunk.page_number} | type={chunk.doc_type}"
        )
        snippet = (chunk.text or "").replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + "..."
        if snippet:
            lines.append(f"  - Snippet: {snippet}")
    return "\n".join(lines)


def format_database_stats(stats: Dict[str, Any]) -> str:
    """Format database/system statistics as Markdown."""
    if not stats:
        return "### Database status\n\nNo statistics available."
    total_chunks = int(stats.get("total_chunks", 0) or 0)
    total_docs = int(stats.get("total_documents", 0) or 0)
    unique_pages = int(stats.get("unique_pages", 0) or 0)
    lines = [
        "### Database status",
        "",
        f"- **Collection:** `{stats.get('collection_name', 'n/a')}`",
        f"- **Persist directory:** `{stats.get('persist_directory', 'n/a')}`",
        f"- **Stored documents:** {total_docs}",
        f"- **Stored chunks:** {total_chunks}",
        f"- **Unique pages:** {unique_pages}",
    ]
    type_counts = stats.get("document_type_counts") or {}
    if isinstance(type_counts, dict) and type_counts:
        lines.extend(["", "**Chunks by document type:**"])
        for doc_type, count in sorted(type_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            lines.append(f"- {doc_type}: {int(count)}")
    if total_chunks == 0:
        lines.extend(["", "_Database is empty. Upload a PDF to begin._"])
    return "\n".join(lines)
