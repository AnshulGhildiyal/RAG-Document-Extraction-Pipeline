"""Tests for ChromaDB vector storage behavior."""

from pathlib import Path

import pytest

from rag_pipeline.core.models import DocumentChunk
from rag_pipeline.storage.vector_store import ChromaVectorStore

chromadb = pytest.importorskip("chromadb")


def _chunk(cid: str, doc_id: str = "doc1", text: str = "sample text", page: int = 1) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid,
        document_id=doc_id,
        filename="sample.pdf",
        source_path="/tmp/sample.pdf",
        text=text,
        page_number=page,
        chunk_index=0,
        doc_type="general",
        metadata={"char_count": len(text)},
    )


def _embedding(seed: float = 0.1, dim: int = 8):
    return [seed + (i * 0.01) for i in range(dim)]


def test_insert_and_retrieve_and_stats(tmp_path: Path):
    store = ChromaVectorStore(persist_directory=str(tmp_path / "chroma"), collection_name="test_collection")
    chunks = [_chunk("c1", text="alpha beta"), _chunk("c2", text="gamma delta", page=2)]
    inserted = store.add_chunks(chunks, [_embedding(0.1), _embedding(0.2)])
    assert inserted == 2
    assert store.count() == 2

    results = store.query(query_embedding=_embedding(0.1), top_k=2)
    assert results["ids"] and len(results["ids"][0]) >= 1

    stats = store.get_stats()
    assert stats["total_chunks"] == 2
    assert stats["collection_name"] == "test_collection"


def test_persistence_across_instances(tmp_path: Path):
    persist_dir = str(tmp_path / "chroma_persist")
    store1 = ChromaVectorStore(persist_directory=persist_dir, collection_name="persist_collection")
    store1.add_chunks([_chunk("p1", text="persistent text")], [_embedding(0.3)])
    assert store1.count() == 1

    # New instance should see persisted data.
    store2 = ChromaVectorStore(persist_directory=persist_dir, collection_name="persist_collection")
    assert store2.count() == 1
    all_data = store2.get_all()
    assert "p1" in (all_data.get("ids") or [])


def test_duplicate_handling_and_delete_and_clear(tmp_path: Path):
    store = ChromaVectorStore(persist_directory=str(tmp_path / "chroma_dup"), collection_name="dup_collection")
    chunk = _chunk("dup1", doc_id="docX", text="duplicate text")
    first = store.add_chunks([chunk], [_embedding(0.4)])
    second = store.add_chunks([chunk], [_embedding(0.4)])
    assert first == 1
    assert second == 0
    assert store.count() == 1

    # Add another doc and delete by document id.
    store.add_chunks([_chunk("other1", doc_id="docY", text="other")], [_embedding(0.5)])
    assert store.count() == 2
    store.delete_document("docX")
    # Depending on Chroma delete semantics, verify by remaining IDs/metadata.
    remaining = store.get_all()
    remaining_ids = set(remaining.get("ids") or [])
    assert "dup1" not in remaining_ids
    assert "other1" in remaining_ids

    store.clear()
    assert store.count() == 0


def test_empty_database_queries_and_stats(tmp_path: Path):
    store = ChromaVectorStore(persist_directory=str(tmp_path / "chroma_empty"), collection_name="empty_collection")
    assert store.count() == 0
    results = store.query(query_embedding=_embedding(0.1), top_k=5)
    assert results["ids"] == [[]]
    assert results["documents"] == [[]]
    stats = store.get_stats()
    assert stats["total_chunks"] == 0
    assert stats["total_documents"] == 0
