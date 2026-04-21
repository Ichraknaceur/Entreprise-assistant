"""Tests for markdown document loading."""

from pathlib import Path

import pytest

from enterprise_knowledge_assistant.rag.loaders import load_markdown_documents


def test_load_markdown_documents_extracts_metadata(tmp_path: Path) -> None:
    """The loader should parse markdown files into normalized document objects."""
    docs_dir = tmp_path / "sample_docs"
    hr_dir = docs_dir / "hr"
    hr_dir.mkdir(parents=True)
    remote_policy = hr_dir / "remote_work_policy.md"
    remote_policy.write_text(
        "# Remote Work Policy\n\nEmployees may work remotely up to three days.",
        encoding="utf-8",
    )

    documents = load_markdown_documents(docs_dir)

    assert len(documents) == 1
    document = documents[0]
    assert document.document == "remote_work_policy.md"
    assert document.category == "hr"
    assert document.path == "hr/remote_work_policy.md"
    assert document.title == "Remote Work Policy"
    assert "Employees may work remotely" in document.content


def test_load_markdown_documents_uses_filename_when_title_is_missing(
    tmp_path: Path,
) -> None:
    """The loader should derive a readable title when no H1 heading exists."""
    docs_dir = tmp_path / "sample_docs"
    it_dir = docs_dir / "it"
    it_dir.mkdir(parents=True)
    guide = it_dir / "vpn_access_guide.md"
    guide.write_text(
        "Access to the VPN requires MFA and an approved device.",
        encoding="utf-8",
    )

    documents = load_markdown_documents(docs_dir)

    assert documents[0].title == "Vpn Access Guide"


def test_load_markdown_documents_raises_for_missing_directory() -> None:
    """The loader should fail loudly when the configured directory is absent."""
    with pytest.raises(FileNotFoundError):
        load_markdown_documents(Path("does-not-exist"))
