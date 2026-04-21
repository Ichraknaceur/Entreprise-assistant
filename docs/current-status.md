# Current Status

This page documents the current implementation state of the project.

## Implemented

### Backend foundation

- FastAPI app entrypoint
- route registration
- configuration and dependency wiring
- query service scaffold

### API contracts

- `GET /health`
- `POST /query`

The `/query` route already exposes the response contract that the project will
keep later when retrieval and grounded generation are connected.

### RAG preparation

- markdown corpus under `data/sample_docs/`
- `KnowledgeDocument` loader
- `KnowledgeChunk` chunker with overlap support
- generator abstraction and mock generator

### Quality and developer tooling

- `uv` dependency management
- `Ruff` linting
- `pytest` tests
- MkDocs documentation
- Make targets for API, tests, and docs

## Not implemented yet

- embeddings generation
- Milvus collection management
- chunk ingestion into a vector database
- retrieval over indexed chunks
- grounded answer generation from retrieved context
- upload endpoint
- Streamlit UI
- Docker packaging

## Current design direction

The repository is now ready for the first vector retrieval milestone:

1. add Milvus support
2. embed chunks
3. index chunks in Milvus
4. retrieve top matches for a question
5. wire retrieval results into `/query`
