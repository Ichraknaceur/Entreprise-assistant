"""Document loading utilities for markdown files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from enterprise_knowledge_assistant.core.config import get_settings

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True, frozen=True)
class KnowledgeDocument:
    """A markdown knowledge-base document with normalized metadata."""

    document: str
    category: str
    path: str
    title: str
    content: str


def load_markdown_documents(data_dir: Path | None = None) -> list[KnowledgeDocument]:
    """Load markdown documents from the configured knowledge base directory."""
    base_dir = data_dir or get_settings().data_dir
    if not base_dir.exists():
        msg = f"Knowledge base directory does not exist: {base_dir}"
        raise FileNotFoundError(msg)

    return [
        _build_document(markdown_path=markdown_path, base_dir=base_dir)
        for markdown_path in sorted(base_dir.rglob("*.md"))
    ]


def _build_document(markdown_path: Path, base_dir: Path) -> KnowledgeDocument:
    """Build a normalized document object from a markdown file."""
    content = markdown_path.read_text(encoding="utf-8").strip()
    relative_path = markdown_path.relative_to(base_dir)
    return KnowledgeDocument(
        document=markdown_path.name,
        category=(
            relative_path.parts[0]
            if len(relative_path.parts) > 1
            else "uncategorized"
        ),
        path=relative_path.as_posix(),
        title=_extract_title(content=content, fallback=markdown_path.stem),
        content=content,
    )


def _extract_title(content: str, fallback: str) -> str:
    """Extract a title from the first level-one markdown heading."""
    for line in content.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith("# "):
            return stripped_line.removeprefix("# ").strip()
    return fallback.replace("_", " ").strip().title()
