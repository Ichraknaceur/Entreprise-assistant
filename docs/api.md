# API Overview

## Base endpoints

### `GET /health`

Returns a simple service health payload.

Example response:

```json
{
  "status": "ok"
}
```

### `POST /query`

Accepts a natural language question.

Example request:

```json
{
  "question": "What is the remote work policy?"
}
```

Current example response:

```json
{
  "answer": "Generation is scaffolded but not implemented yet. Received question: 'What is the remote work policy?'. Retrieved context chunks: 0.",
  "sources": [
    {
      "document": "not_implemented_yet",
      "snippet": "Retrieval pipeline will provide source-backed snippets here."
    }
  ]
}
```

## Important note

At this stage, `/query` is intentionally returning a placeholder answer. The
goal so far was to lock the API shape before plugging in the real retrieval and
generation pipeline.

This lets the project evolve feature-by-feature without redesigning the public
contract every time.
