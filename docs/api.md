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

Accepts a natural language question and an optional provider override.

Example request:

```json
{
  "question": "What is the remote work policy?",
  "provider": "openai"
}
```

Example response:

```json
{
  "answer": "Employees may work remotely up to three days per week with manager approval.",
  "sources": [
    {
      "document": "remote_work_policy.md",
      "snippet": "Employees may work remotely up to three days per week..."
    }
  ]
}
```

Supported providers today:

- `mock`
- `openai`

If no grounded context is retrieved, the API returns a refusal-style answer
instead of inventing a response.

### `GET /health/database`

Returns vector store health information for the configured Milvus backend.

Example response:

```json
{
  "status": "ok",
  "provider": "milvus",
  "collection_name": "knowledge_chunks",
  "collection_exists": true
}
```

### `POST /admin/index`

Triggers the end-to-end indexing flow:

1. load markdown documents
2. chunk them
3. create embeddings
4. ingest records into Milvus

Example response:

```json
{
  "status": "ok",
  "documents_count": 15,
  "chunks_count": 42,
  "inserted_count": 42
}
```
