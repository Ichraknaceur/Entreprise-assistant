# Current Status

This page documents the current implementation state of the project.

## Implemented

### Backend foundation

- FastAPI app entrypoint
- route registration
- configuration and dependency wiring
- query service orchestration

### API contracts

- `GET /health`
- `GET /health/database`
- `POST /admin/index`
- `POST /query`

### RAG pipeline

- markdown corpus under `data/sample_docs/`
- `KnowledgeDocument` loader
- `KnowledgeChunk` chunker with overlap support
- embeddings with `sentence-transformers`
- Milvus ingestion and collection creation
- Milvus retrieval
- generator abstraction with `mock` and `openai`

### Observability

- Langfuse integration for the `/query` flow
- trace root observation
- retrieval span
- generation span

### Quality and developer tooling

- `uv` dependency management
- `Ruff` linting
- `pytest` tests
- MkDocs documentation
- Make targets for API, tests, and docs

## Not implemented yet

- richer refusal behavior based on retrieval quality thresholds
- Langfuse prompt management
- Langfuse evaluations / datasets / experiments
- upload endpoint
- Streamlit UI
- Docker packaging

## Current design direction

The repository is now ready for the next product-hardening milestone:

1. improve refusal behavior
2. expand Langfuse usage beyond tracing
3. add deployment-facing polish
4. add a lightweight UI
