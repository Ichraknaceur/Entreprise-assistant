"""Tests for markdown document chunking."""

import pytest

from enterprise_knowledge_assistant.rag.chunking import chunk_documents
from enterprise_knowledge_assistant.rag.loaders import KnowledgeDocument


def test_chunk_documents_returns_single_chunk_for_small_document() -> None:
    """Short documents should remain in one chunk."""
    document = KnowledgeDocument(
        document="remote_work_policy.md",
        category="hr",
        path="hr/remote_work_policy.md",
        title="Remote Work Policy",
        content="# Remote Work Policy\n\nEmployees may work remotely up to three days.",
    )

    chunks = chunk_documents([document], target_size=500, overlap_size=50)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "hr/remote_work_policy.md::chunk-000"
    assert chunks[0].document == "remote_work_policy.md"
    assert chunks[0].category == "hr"
    assert "Employees may work remotely" in chunks[0].text


def test_chunk_documents_preserves_overlap_between_chunks() -> None:
    """Chunks should share trailing block context when overlap is enabled."""
    paragraph_alpha = ("Alpha " * 10).strip()
    paragraph_beta = ("Beta " * 10).strip()
    paragraph_gamma = ("Gamma " * 10).strip()
    document = KnowledgeDocument(
        document="access_control_policy.md",
        category="compliance",
        path="compliance/access_control_policy.md",
        title="Access Control Policy",
        content=(
            "# Access Control Policy\n\n"
            f"{paragraph_alpha}\n\n"
            f"{paragraph_beta}\n\n"
            f"{paragraph_gamma}"
        ),
    )

    chunks = chunk_documents([document], target_size=140, overlap_size=40)

    assert len(chunks) >= 2
    assert paragraph_beta in chunks[0].text
    assert paragraph_beta in chunks[1].text


def test_chunk_documents_splits_large_paragraphs_into_bounded_chunks() -> None:
    """Oversized paragraphs should be split so every chunk respects the limit."""
    long_paragraph = " ".join(f"word{i}" for i in range(150))
    document = KnowledgeDocument(
        document="vpn_access_guide.md",
        category="it",
        path="it/vpn_access_guide.md",
        title="VPN Access Guide",
        content=long_paragraph,
    )

    chunks = chunk_documents([document], target_size=120, overlap_size=20)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 120 for chunk in chunks)


@pytest.mark.parametrize(
    ("target_size", "overlap_size", "match"),
    [
        (0, 10, "target_size"),
        (100, -1, "overlap_size"),
        (100, 100, "overlap_size"),
    ],
)
def test_chunk_documents_rejects_invalid_sizes(
    target_size: int,
    overlap_size: int,
    match: str,
) -> None:
    """Chunker parameters should be validated early."""
    document = KnowledgeDocument(
        document="doc.md",
        category="hr",
        path="hr/doc.md",
        title="Doc",
        content="content",
    )

    with pytest.raises(ValueError, match=match):
        chunk_documents(
            [document],
            target_size=target_size,
            overlap_size=overlap_size,
        )
