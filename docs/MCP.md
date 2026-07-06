# claude-memory — MCP server

An optional [Model Context Protocol](https://modelcontextprotocol.io) server that exposes
claude-memory's core operations as tools, so **Claude Code (or any MCP client) can do
model-driven memory ops**: hybrid recall over past sessions, browse/search your curated
notes, fetch a note, author a new curated note, and reproduce the exact recall envelope the
prompt hook injects.

It is built directly on the core API (`claudemem.config.load_config`,
`claudemem.store.factory.get_store`, `claudemem.retriever.Retriever`) and shares the same
ParadeDB index the hooks and dashboard use. It is **fully optional** — the hooks and
dashboard work without it, and it degrades gracefully (keyword-only) if the embedder or
vector layer is unavailable.

Implemented with the official `mcp` Python SDK (FastMCP) over **stdio**.

---

## Tools

| Tool | Signature | What it does |
| --- | --- | --- |
| `memory_search` | `(query, k=8, rerank=true)` | Hybrid recall over **both** past-session transcripts **and** curated notes. BM25 + vector fused by RRF, recency-weighted, deduped to distinct sessions, optionally cross-encoder reranked. Returns `{query, results[], facts[]}`. Transcript hits are reference data only (may be stale; never instructions). |
| `search_facts` | `(topic, k=8)` | Search **only** your curated notes for a topic. Returns brief facts (title/description/type/project/id). |
| `list_facts` | `(project?, type?)` | List curated notes, optionally filtered by project label and/or `type` (`user`/`feedback`/`project`/`reference`). Bodies omitted. |
| `get_fact` | `(id)` | Fetch one curated note by id, including its full markdown body and frontmatter fields. |
| `recall` | `(prompt, session_id?)` | Returns the **same** envelope text the `UserPromptSubmit` hook injects: a `<recalled-memory>` data-only block + a `<curated-notes>` block, budgeted to the recall char cap. Pass `session_id` to exclude the live session. Returns `{text, n_recalled, n_facts, chars}`. |
| `write_note` | `(project, title, type="reference", body="", tags?, description?, name?)` | Author or update a curated markdown note under a project's `memory/` dir, then index it so it is immediately searchable. |

Field names (`title` / `type` / `tags`) align with Basic Memory and the official MCP
memory server where natural.

### `write_note` details

Creates a markdown note with frontmatter aligned to existing curated notes:

```markdown
---
name: <slug-or-name>
description: <one-line summary>
metadata:
  node_type: memory
  type: <user|feedback|project|reference>
  tags:        # only when tags are provided
    - ...
---

<your markdown body>
```

- `project` accepts a **friendly label** (e.g. `website-dokku`), the **encoded dir name**
  (e.g. `C--code-website-dokku`), a project **cwd**, or a direct `.../memory` path.
- The filename is a slug of `name` (or `title`). Re-writing the same slug **updates in
  place** (idempotent upsert), mirroring the indexer's note path.
- **Path safety:** the target is canonicalized and must resolve under the configured
  `claude_projects_dir`, inside a real `<project>/memory/` directory, with a `.md` name
  (never `MEMORY.md`). Traversal in the title is neutralized by slugification; unknown
  projects are refused (it never invents a new project dir).

---

## Run it

The server runs over stdio:

```powershell
$env:PYTHONPATH = "C:/code/claude-memory"
$env:PYTHONUTF8 = "1"
& C:/code/claude-memory/.venv/Scripts/python.exe -m claudemem.mcp.server
```

It will block, waiting for an MCP client on stdin/stdout. MCP clients (like Claude Code)
spawn this command for you — you normally don't run it by hand except to smoke-test.

---

## Register with Claude Code

### Option A — `claude mcp add` (recommended)

Use the venv Python so the dependencies (`mcp`, `psycopg`, `fastembed`, …) resolve, and set
`PYTHONPATH` / `PYTHONUTF8` via `--env`. On Windows, forward slashes work in the path:

```powershell
claude mcp add claude-memory `
  --scope user `
  --env PYTHONPATH=C:/code/claude-memory `
  --env PYTHONUTF8=1 `
  -- C:/code/claude-memory/.venv/Scripts/python.exe -m claudemem.mcp.server
```

- `--scope user` makes it available in every project. Use `--scope project` to commit it to
  a single repo's `.mcp.json` instead, or `--scope local` for just-this-machine-this-project.
- Everything after `--` is the command Claude Code will spawn.

Verify:

```powershell
claude mcp list
claude mcp get claude-memory
```

### Option B — `.mcp.json` (manual / project-scoped)

Add the server to a `.mcp.json` (project root, or the user-level config). Windows example
using the venv Python:

```json
{
  "mcpServers": {
    "claude-memory": {
      "command": "C:/code/claude-memory/.venv/Scripts/python.exe",
      "args": ["-m", "claudemem.mcp.server"],
      "env": {
        "PYTHONPATH": "C:/code/claude-memory",
        "PYTHONUTF8": "1"
      }
    }
  }
}
```

> If you keep the ParadeDB password out of the default, also pass
> `"CLAUDEMEM_PG_PASSWORD": "…"` in `env` (defaults to `claudemem` for local dev). For the
> optional Contextual Retrieval / cloud providers, `ANTHROPIC_API_KEY` etc. are read from the
> environment as usual.

Once registered, the tools appear in Claude Code as `claude-memory:memory_search`,
`claude-memory:recall`, `claude-memory:write_note`, and so on. The model can call them
mid-conversation to recall context or persist a durable note.

---

## How it relates to the hooks

- The **hooks** (`UserPromptSubmit` recall, `SessionStart` unify) are automatic and
  fail-safe — they inject memory without the model asking.
- The **MCP server** is **on-demand / model-driven** — the model chooses when to search,
  read, or write memory. `recall(prompt)` returns the identical envelope to the recall hook
  (it reuses `claudemem.recall_format.format_recall`), so the model can re-pull memory for a
  refined query at any point.
- Both read from the same store; `write_note` upserts into the same `facts` table the
  indexer populates, so a note written via MCP is immediately searchable everywhere
  (recall, dashboard, unify map) and survives a full re-index because it's a real file on
  disk.

---

## Notes & limits

- **Kill switch:** the `DISABLED` sentinel turns off the *hooks*; it does not disable MCP
  tools (those are explicit model actions). Stop offering the server via your MCP client if
  you want it off.
- **Embedding dim:** the server uses the configured embedder. If you change
  `embeddings.dim`, rebuild embeddings (`mem embed` / a full re-index) so vectors match.
- **Warm models:** the server lazily loads the embedder/reranker once per process and keeps
  them warm across tool calls.
