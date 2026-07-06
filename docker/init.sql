-- Runs once on first cluster init (docker-entrypoint-initdb.d).
-- Only guaranteed-present extensions here; optional ones (vectorscale) are attempted
-- by the application migration with graceful fallback.
CREATE EXTENSION IF NOT EXISTS pg_search;
CREATE EXTENSION IF NOT EXISTS vector;
