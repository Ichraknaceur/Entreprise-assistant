# Architecture

## Design principles

The current codebase follows a simple layered structure:

- API layer for HTTP routes and schemas
- core layer for configuration, observability, and dependencies
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
- hosts Langfuse observability integration

### `services/`

- orchestrates application use cases
- keeps route handlers lightweight

### `rag/`

- loads markdown documents
- chunks documents into retrieval-ready units
- handles embeddings, vector storage, retrieval, prompts, and generators

## Current flow

```text
Markdown files
  -> Loader
  -> Chunker
  -> Embeddings
  -> Milvus

User question
  -> Query embedding
  -> Milvus retriever
  -> Generator abstraction (mock/openai)
  -> FastAPI /query
  -> Langfuse tracing
```

## Why this structure works

- Milvus keeps the vector layer open-source and production-oriented
- provider abstraction keeps generation extensible
- Langfuse adds observability without tightly coupling business logic to one UI
- service-layer orchestration keeps API routes thin and testable
