"""Retriever interface for the knowledge base."""


def retrieve_context() -> None:
    """Retrieve the most relevant chunks for a user query."""
    raise NotImplementedError
