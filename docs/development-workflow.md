# Development Workflow

## Initial baseline

The repository is being prepared for an initial baseline push that captures the
current foundation:

- API skeleton
- markdown knowledge base
- loader
- chunker
- tests
- documentation

## Branching strategy

After the initial push, development should move feature-by-feature using
dedicated branches.

Recommended naming:

- `feature/milvus-integration`
- `feature/embeddings-pipeline`
- `feature/retriever`
- `feature/query-grounding`
- `feature/streamlit-ui`

## Suggested workflow

1. create a branch from `main`
2. implement one coherent feature
3. run lint and tests locally
4. update docs when behavior changes
5. merge only when the feature is stable

## Why this matters

This project is meant to look like a real engineering project. A disciplined
branch strategy makes the portfolio story stronger and keeps progress easier to
explain.
