# Security model

This is a single-user, local memory service. It is intentionally not a remotely deployable API.

## Boundaries

- The HTTP service refuses non-loopback binds and rejects non-loopback peers, hostile browser
  origins, and untrusted Host headers. It has no network authentication because it is not exposed.
- Lifecycle hooks execute only versioned local scripts. They never download or execute bootstrap
  code, and installation removes the retired remote-code carrier from existing Claude settings.
- Transcript recall is untrusted reference data. It is labeled `trust="data-only"`, its instructions
  must never be followed, and injected envelopes are stripped before transcripts are indexed.
- Curated notes are trusted only as the operator's authored files. Writes are secret-scanned,
  slugged, confined to approved roots, and performed atomically.
- Credentials belong in environment variables. Note parsing, recall output, MCP output, and backups
  redact credential-shaped material.

## Data and history

`~/.agent-memory/notes` is the canonical file store. The database is derived and may be rebuilt.
Lifecycle metadata preserves superseded facts for audit while excluding them from automatic recall.
Delivery receipts prove what reached a client; usefulness feedback records whether it helped or hurt.

## Deliberate non-goals

Do not bind the service to `0.0.0.0`, a LAN interface, a tunnel, or a public reverse proxy. A future
multi-user deployment would require authentication, authorization, CSRF protection, tenant-scoped
encryption, request limits, and a separate threat review.
