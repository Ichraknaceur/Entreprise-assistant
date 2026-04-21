# Enterprise Knowledge Assistant

Enterprise Knowledge Assistant is a production-oriented RAG project that
simulates an internal company knowledge base. It is being built incrementally,
with a focus on clean backend architecture, retrieval quality, source-backed
answers, and a realistic engineering workflow.

## Current baseline

At this stage, the repository already includes:

- a working FastAPI application skeleton
- a stable API contract for `/health` and `/query`
- a simulated internal markdown knowledge base
- a tested document loader
- a tested chunking pipeline
- a generator abstraction with a mock implementation

The next major milestone is integrating **Milvus** as the open-source vector
database for ingestion and retrieval.

## Project goals

- ingest internal documentation from markdown files
- prepare retrieval-ready chunks with traceable metadata
- support vector retrieval over enterprise documents
- generate concise, source-backed answers
- keep the codebase structured like a serious backend project

## Read next

- [Current Status](current-status.md)
- [Architecture](architecture.md)
- [API Overview](api.md)
- [Development Workflow](development-workflow.md)
