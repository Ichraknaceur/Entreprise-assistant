# Langfuse Observability

The project now includes a first Langfuse integration focused on observability
for the `/query` flow.

## What is traced

For each query request, the application can emit:

- a root `query-request` observation
- a nested `retrieve-context` observation
- a nested `generate-answer` observation

These traces capture the core RAG loop:

- the user question
- the selected provider
- retrieval result counts and source documents
- the generated answer

![Langfuse Observability UI](https://langfuse.com/_next/static/media/observability-ui.7ec35254.png)

## Environment variables

Add the following values to `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=development
```

If your project is in the US region, set:

```env
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

## Current behavior

The Langfuse client is optional.

- if credentials are configured, `/query` emits traces
- if credentials are missing, the app falls back to a no-op client
- the API continues to work without Langfuse

To make traces visible quickly in local development, the app flushes Langfuse
events after each query request.

## Current scope

This integration currently covers observability only.

It does not yet include:

- prompt management through Langfuse prompts
- evaluation datasets
- automated scoring
- experiments or A/B prompt comparison

These are strong next steps for the project.
