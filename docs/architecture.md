# Architecture

## Design principles

The current codebase follows a simple layered structure:

- API layer for HTTP routes and schemas
- core layer for configuration and dependencies
- service layer for orchestration
- RAG layer for ingestion, chunking, retrieval, and generation concerns

The goal is to keep modules small and composable without introducing premature
abstractions.

## Package layout

```text
src/enterprise_knowledge_assistant/
├── api/
│   ├── routes/
│   └── schemas/
├── core/
├── rag/
│   └── generator/
└── services/
```

## Responsibilities

### `api/`

- exposes HTTP routes
- validates request and response schemas
- stays thin and delegates work to services

### `core/`

- centralizes settings
- wires shared dependencies
- configures logging

### `services/`

- orchestrates application use cases
- keeps route handlers lightweight

### `rag/`

- loads markdown documents
- chunks documents into retrieval-ready units
- will later handle embeddings, vector storage, retrieval, and prompts

## Current flow

```text
Markdown files -> Loader -> Chunker -> Future embeddings/vector store
                                       -> Future retriever
                                       -> Generator abstraction
                                       -> FastAPI /query
```

## Why Milvus next

The project is intentionally moving toward an open-source vector database
instead of a proprietary managed service. Milvus fits that direction while
still giving the project a serious production-oriented feel.
