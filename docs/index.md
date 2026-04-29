# Enterprise Knowledge Assistant

Enterprise Knowledge Assistant is a production-oriented RAG project that
simulates an internal company knowledge base. It is being built incrementally,
with a focus on clean backend architecture, retrieval quality, source-backed
answers, LLM provider flexibility, and a realistic engineering workflow.

## Current baseline

At this stage, the repository already includes:

- a working FastAPI backend
- a stable API contract for health, indexing, and query flows
- a simulated internal markdown knowledge base
- a tested document loader
- a tested chunking and embeddings pipeline
- Milvus indexing and retrieval
- `mock` and `openai` answer generation
- Langfuse tracing for `/query`

The next major milestone is expanding from tracing into Langfuse prompt
management and evaluation.

## Project goals

- ingest internal documentation from markdown files
- prepare retrieval-ready chunks with traceable metadata
- support vector retrieval over enterprise documents
- generate concise, source-backed answers
- trace the full RAG query loop
- keep the codebase structured like a serious backend project

## Read next

- [Current Status](current-status.md)
- [Architecture](architecture.md)
- [API Overview](api.md)
- [Langfuse Observability](langfuse.md)
- [Development Workflow](development-workflow.md)
