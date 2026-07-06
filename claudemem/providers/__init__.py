"""Pluggable providers: embeddings, reranker, contextual-retrieval enrichment.
Get instances via the factory functions in each module (they read Config and
degrade gracefully when an optional backend is unavailable)."""
