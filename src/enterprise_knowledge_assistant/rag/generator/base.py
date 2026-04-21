"""Base protocol for answer generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class BaseGenerator(ABC):
    """Interface for pluggable answer generators."""

    @abstractmethod
    def generate(self, question: str, contexts: Sequence[str]) -> str:
        """Generate an answer from the provided contexts."""
