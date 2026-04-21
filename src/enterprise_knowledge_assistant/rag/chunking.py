"""Document chunking utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from enterprise_knowledge_assistant.rag.loaders import KnowledgeDocument


@dataclass(slots=True, frozen=True)
class KnowledgeChunk:
    """A retrieval-ready chunk linked back to its source document."""

    chunk_id: str
    document: str
    category: str
    path: str
    title: str
    text: str


def chunk_documents(
    documents: Sequence[KnowledgeDocument],
    *,
    target_size: int = 500,
    overlap_size: int = 100,
) -> list[KnowledgeChunk]:
    """Split loaded documents into retrieval-friendly chunks."""
    if target_size <= 0:
        msg = "target_size must be greater than 0"
        raise ValueError(msg)
    if overlap_size < 0:
        msg = "overlap_size must be greater than or equal to 0"
        raise ValueError(msg)
    if overlap_size >= target_size:
        msg = "overlap_size must be smaller than target_size"
        raise ValueError(msg)

    chunks: list[KnowledgeChunk] = []
    for document in documents:
        document_units = _split_document_units(
            document.content,
            target_size=target_size,
        )
        chunks.extend(
            _build_document_chunks(
                document=document,
                units=document_units,
                target_size=target_size,
                overlap_size=overlap_size,
            ),
        )
    return chunks


def _build_document_chunks(
    *,
    document: KnowledgeDocument,
    units: list[str],
    target_size: int,
    overlap_size: int,
) -> list[KnowledgeChunk]:
    """Pack content units into bounded chunks while preserving source metadata."""
    chunks: list[KnowledgeChunk] = []
    buffer: list[str] = []

    for unit in units:
        if buffer and _joined_length(buffer) + len(unit) + 2 > target_size:
            chunks.append(
                _create_chunk(
                    document=document,
                    units=buffer,
                    index=len(chunks),
                ),
            )
            buffer = _select_overlap_units(units=buffer, overlap_size=overlap_size)
            while buffer and _joined_length(buffer) + len(unit) + 2 > target_size:
                buffer.pop(0)

        if not buffer or _joined_length(buffer) + len(unit) + 2 <= target_size:
            buffer.append(unit)

    if buffer:
        chunks.append(
            _create_chunk(
                document=document,
                units=buffer,
                index=len(chunks),
            ),
        )
    return chunks


def _create_chunk(
    *,
    document: KnowledgeDocument,
    units: list[str],
    index: int,
) -> KnowledgeChunk:
    """Create a chunk object with stable source metadata."""
    return KnowledgeChunk(
        chunk_id=f"{document.path}::chunk-{index:03d}",
        document=document.document,
        category=document.category,
        path=document.path,
        title=document.title,
        text="\n\n".join(units).strip(),
    )


def _split_document_units(content: str, *, target_size: int) -> list[str]:
    """Split markdown content into block-level units and sub-split large blocks."""
    units = [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]
    bounded_units: list[str] = []
    for unit in units:
        if len(unit) <= target_size:
            bounded_units.append(unit)
            continue
        bounded_units.extend(_split_large_unit(unit=unit, target_size=target_size))
    return bounded_units


def _split_large_unit(unit: str, *, target_size: int) -> list[str]:
    """Split an oversized block into smaller sentence-aware segments."""
    sentences = _split_sentences(unit)
    if len(sentences) == 1:
        return _split_long_sentence(sentence=sentences[0], target_size=target_size)

    segments: list[str] = []
    current_segment: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*current_segment, sentence]).strip()
        if current_segment and len(candidate) > target_size:
            segments.append(" ".join(current_segment).strip())
            current_segment = [sentence]
            continue
        current_segment.append(sentence)

    if current_segment:
        segments.append(" ".join(current_segment).strip())

    final_segments: list[str] = []
    for segment in segments:
        if len(segment) <= target_size:
            final_segments.append(segment)
        else:
            final_segments.extend(
                _split_long_sentence(sentence=segment, target_size=target_size),
            )
    return final_segments


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like segments."""
    return [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+", text)
        if segment.strip()
    ]


def _split_long_sentence(sentence: str, *, target_size: int) -> list[str]:
    """Split a very long sentence into word-bounded segments."""
    words = sentence.split()
    segments: list[str] = []
    current_words: list[str] = []
    for word in words:
        candidate = " ".join([*current_words, word]).strip()
        if current_words and len(candidate) > target_size:
            segments.append(" ".join(current_words))
            current_words = [word]
            continue
        current_words.append(word)

    if current_words:
        segments.append(" ".join(current_words))
    return segments


def _select_overlap_units(*, units: list[str], overlap_size: int) -> list[str]:
    """Keep a suffix of the previous chunk to provide retrieval overlap."""
    if overlap_size == 0:
        return []

    overlap_units: list[str] = []
    current_size = 0
    for unit in reversed(units):
        unit_size = len(unit) + (2 if overlap_units else 0)
        overlap_units.insert(0, unit)
        current_size += unit_size
        if current_size >= overlap_size:
            break
    return overlap_units


def _joined_length(units: list[str]) -> int:
    """Return the length of units when joined as chunk text."""
    return len("\n\n".join(units))
