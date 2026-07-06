export const meta = {
  name: 'claudemem-surfaces',
  description: 'Build the dashboard UI, MCP server, eval, selftest, promote-mining, scripts+docs against the frozen claude-memory core',
  phases: [{ title: 'Surfaces', detail: '6 parallel builders against the verified core + SPEC' }],
}

const SHARED = `
PROJECT: claude-memory — a local agent-memory layer for Claude Code on Windows.
Root: C:/code/claude-memory   Python package: claudemem   venv python: C:/code/claude-memory/.venv/Scripts/python.exe
The CORE IS BUILT, VERIFIED, and the database is populated with real data. DO NOT modify core files
unless your task explicitly owns them. Read these for the exact API/contract before writing:
  - C:/code/claude-memory/docs/SPEC.md   (authoritative spec — read the sections relevant to you)
  - claudemem/retriever.py (Retriever.search/search_facts; Result has .chunk and .score)
  - claudemem/store/base.py (Store DAO interface + Chunk/Fact/Candidate dataclasses)
  - claudemem/config.py (load_config() -> Config; sections scope/store/embeddings/reranker/recall/unify/server/index)
  - claudemem/dashboard/api.py (the live HTTP endpoints — the dashboard contract)
  - claudemem/recall_format.py, claudemem/facts.py, claudemem/indexer.py, claudemem/paths.py
ParadeDB (Postgres) is live at localhost:55432 (db/user/pass = claudemem), already indexed.
To run Python use the venv with env PYTHONPATH=C:/code/claude-memory and PYTHONUTF8=1, e.g. (PowerShell):
  $env:PYTHONPATH="C:/code/claude-memory"; $env:PYTHONUTF8="1"; & C:/code/claude-memory/.venv/Scripts/python.exe -c "import claudemem; print('ok')"
Use the PowerShell tool for commands (not Git Bash) so Windows paths work. Use the Read/Write/Edit tools for files.
STRICT RULES: only create/edit the files your task lists (disjoint from other builders). Do not edit other modules.
Match the existing code's style (stdlib-first, type hints, terse docstrings, fail-safe). Keep it genuinely high quality.
Return a concise report: the files you wrote and any verification you ran + its result.
YOUR TASK:
`

const TASKS = [
  {
    key: 'dashboard-ui',
    prompt: `Build the GORGEOUS web hub front-end (the user explicitly wants best-of-the-best aesthetics).
OWN ONLY: claudemem/dashboard/static/index.html, claudemem/dashboard/static/css/app.css,
claudemem/dashboard/static/js/app.js, and claudemem/dashboard/static/js/vendor/ (vendored libs you download).
The FastAPI server (claudemem/dashboard/server.py) serves static/index.html at "/" and mounts static/ at "/static".
Consume the EXISTING JSON API in claudemem/dashboard/api.py (do NOT modify api.py). Endpoints available:
  GET /api/search?q=&kind=all|transcripts|facts&k=&rerank=  -> {results:[{id,kind,project,session,role,block,ts,age,score,reranked,title,snippet}], facts:[{id,type,title,description,project,path,tags}], timing_ms}
  GET /api/facts?project=&type= ; GET /api/fact/{id} (has body_html) ; GET /ui/search?q= ; GET /ui/fact/{id}
  GET /api/graph -> {nodes:[{id,label,type,group}],edges:[{source,target,kind}]}
  GET /api/metrics ; GET /api/stats ; GET /api/injections?limit= ; GET /api/injections/stream (SSE)
  GET /api/promotions ; POST /api/promotions/{id} {action:accept|reject}
  GET/POST /api/killswitch {on} ; POST /api/index {full} ; GET /api/index/stream (SSE)
DESIGN: dark, refined, dense-but-calm "knowledge hub". The user loves the golden ratio (PHI 1.618) as a
spacing/scale motif and beautiful, maximal, polished UI. Use a tasteful dark palette, good typography,
smooth micro-interactions. Sections (sidebar nav, Alpine.js state): Search (instant typeahead hitting
/api/search with ~250ms debounce; show recalls + curated notes with trust badges, provenance chips, scores),
Notes (browse curated facts grouped by project/type; click -> render body_html with backlinks),
Graph (Cytoscape.js from /api/graph; filter by type; click node opens the note),
Metrics (ECharts line/area charts from /api/metrics: chunks/facts/embedded over time),
Injections (live SSE table from /api/injections/stream — the audit window of what memory fed the model),
Promotions (candidate cards with accept/reject), Settings (show /api/stats; kill-switch toggle; reindex button
with live /api/index/stream progress). VENDOR the JS libs offline: download htmx, alpine.js, cytoscape,
echarts, and markdown-it into static/js/vendor/ via PowerShell Invoke-WebRequest from jsdelivr, and reference
them locally (fall back gracefully / hide a panel if a lib is missing). Use Tailwind-via-CDN is NOT allowed
(vendor or hand-write CSS). Hand-write a beautiful app.css (no build step). After building, START the server
(C:/code/claude-memory/.venv/Scripts/python.exe -m claudemem serve --no-browser in the background) and curl
a few endpoints (e.g. /api/stats, /api/search?q=deploy) to confirm the page + API render; then stop the server.`,
  },
  {
    key: 'mcp-server',
    prompt: `Build the optional MCP server so Claude Code (and any MCP client) can do model-driven memory ops.
OWN ONLY: claudemem/mcp/server.py, claudemem/mcp/__init__.py, and docs/MCP.md.
Use the installed 'mcp' Python SDK (FastMCP/stdio). Expose tools, implemented via the core API
(claudemem.config.load_config, claudemem.store.factory.get_store, claudemem.retriever.Retriever):
  - memory_search(query, k=8, rerank=true) -> hybrid transcript+fact results (use Retriever.search + search_facts)
  - get_fact(id) and list_facts(project?, type?) and search_facts(topic, k)
  - write_note(project, title, type, body, tags?) -> creates a curated markdown note (frontmatter: name/description/metadata.type)
    in the matching C:/Users/zcobe/.claude/projects/<project>/memory/ dir and upserts it (align format to existing notes;
    see claudemem/facts.py + existing notes). Sanitize paths (only write under a real .../memory dir).
  - recall(prompt, session_id?) -> the same envelope text the hook injects (reuse claudemem.recall_format.format_recall).
Make it runnable as: python -m claudemem.mcp.server (stdio). Document in docs/MCP.md how to register it with
Claude Code (claude mcp add / .mcp.json example for Windows using the venv python). Verify the module imports
cleanly with the venv python (PYTHONPATH set). Keep tool schemas clear; align field names to Basic Memory / the
official MCP memory server where natural (title/type/tags).`,
  },
  {
    key: 'eval',
    prompt: `Build the recall@k evaluation harness. OWN ONLY: claudemem/eval.py and eval/golden.jsonl.
eval/golden.jsonl: one JSON object per line: {"q": "<natural query>", "expect_sessions": ["<session_id>", ...],
"expect_facts": ["<fact title or name substring>", ...], "note": "..."}. SEED it with ~12 realistic queries by
INSPECTING the live indexed data: connect via the core (load_config + get_store) and/or query ParadeDB
(localhost:55432) to find real session_ids and curated-fact titles, then craft queries whose expected hits you
verified actually exist (e.g. topics like dokku deploy, golden ratio design, plant genome, blob PvP netcode,
explore-before-building feedback). claudemem/eval.py: run_eval() loads golden.jsonl, runs Retriever.search +
search_facts for each q, computes recall@k (k from config.recall.top_k; also report @1/@3/@k), prints a table,
records a 'recall_at_k' metric via store.record_metric, and prints the delta vs the previous run (drift). Make
'python -m claudemem eval' work (cli.cmd_eval already calls claudemem.eval.run_eval). Actually RUN it with the
venv python and include the output in your report (it must produce non-zero recall on your seeded set).`,
  },
  {
    key: 'selftest',
    prompt: `Build the regression self-test. OWN ONLY: claudemem/selftest.py. Implement run_selftest(verbose=True)->bool
with >=18 independent checks, printing PASS/FAIL per check and returning False if any fail (cli.cmd_selftest exits
1 on failure). Cover (per SPEC section 9): config loads; store connects+migrates (health ok); embedder loads with
correct dim; index a SYNTHETIC tiny transcript .jsonl + a synthetic note into a temp/throwaway scope and confirm
they're retrievable (or, to avoid polluting real data, exercise chunking/transcripts/facts parsing + retriever
fusion on in-memory/synthetic inputs and a real query against the populated DB returning >0 hits); BM25 search
returns hits; vector search returns hits (skip-with-note if embedder unavailable); hybrid fuse returns hits;
recall hook on synthetic stdin emits a valid additionalContext JSON envelope (run hooks/recall.py via subprocess
with the venv python, feeding a JSON event on stdin where cwd is in-scope; assert valid JSON w/ hookSpecificOutput);
trivial-prompt -> no output; out-of-scope cwd -> no output; kill switch (create DISABLED) -> recall+unify emit
nothing, then remove DISABLED; unify hook emits a <memory-map>; char cap respected (<=10000); strip_injected_blocks
removes a <recalled-memory> block; meaningful_term_count gating; install-hooks writes valid settings.json then
restores (use a temp settings path or snapshot+restore C:/Users/zcobe/.claude/settings.json carefully — prefer
testing claudemem.hooks_install on a COPY, do not corrupt the real settings). RUN it with the venv python and
include the pass/fail summary in your report. It is OK for a check to be SKIPPED with a clear reason if a dep is
genuinely unavailable, but core checks must pass.`,
  },
  {
    key: 'promote',
    prompt: `Build auto-promotion mining. OWN ONLY: claudemem/promote.py. Implement mine_candidates()->int that scans
the indexed transcript chunks for recurring lessons/gotchas/decisions NOT already captured as curated facts, drafts
candidate notes, and stores them via store.add_promotion_candidate(title, body, type, support, score). Approach
(no external LLM required; keep it deterministic + offline): pull chunks (especially role=user and assistant 'text'
that look like corrections/lessons — heuristics: phrases like "always", "never", "don't", "make sure", "from now on",
"the issue was", "root cause", "remember to"), cluster near-duplicates by embedding cosine similarity (reuse the
embeddings already in the DB via the store / a fresh query embedding) or by keyword overlap, score by cluster size
(support) and novelty vs existing facts (low max-similarity to any existing fact = more novel), and draft a concise
title + 2-4 sentence body + a guessed type (feedback/project/reference) for the top N (cap ~15). 'mem promote' calls
this. Do NOT auto-write notes (acceptance happens in the dashboard). RUN it with the venv python against the live DB
and report how many candidates it produced + 2-3 example titles.`,
  },
  {
    key: 'scripts-docs',
    prompt: `Build the ops scripts + docs. OWN ONLY: files under C:/code/claude-memory/scripts/*.ps1 (NEW ones only;
do NOT edit existing db.sh/smoke.py/diag.py/*.sql), mem.ps1 and mem.cmd at the project root, and docs/README.md +
docs/ARCHITECTURE.md. Deliver:
  - mem.ps1 / mem.cmd: thin launchers that call "C:/code/claude-memory/.venv/Scripts/python.exe -m claudemem @args"
    with PYTHONUTF8=1 (so the user can run 'mem index', 'mem serve', etc).
  - scripts/serve.ps1: start the dashboard (mem serve).
  - scripts/index.ps1: run 'mem index'.
  - scripts/install_hooks.ps1 / uninstall_hooks.ps1: call 'mem install-hooks' / 'mem uninstall-hooks'.
  - scripts/fetch_vendor.ps1: download htmx, alpine, cytoscape, echarts, markdown-it into
    claudemem/dashboard/static/js/vendor/ via Invoke-WebRequest (idempotent).
  - scripts/install_persistence.ps1: create a Windows Scheduled Task (at logon, hidden) that keeps the stack alive:
    keeps the WSL2 VM pinned (wsl -d Ubuntu -- sleep infinity), ensures the ParadeDB container is up
    (wsl -d Ubuntu -- bash /mnt/c/code/claude-memory/scripts/db.sh up), and starts the dashboard server
    (mem serve --no-browser). Include an uninstall_persistence.ps1 that removes the task. Explain WHY (WSL idle
    terminates the VM, stopping the DB) in a comment + the README.
  - docs/README.md: what it is, the architecture at a glance, quickstart (db up -> pip install -e . -> mem index ->
    mem serve -> mem install-hooks -> install_persistence), config, the kill switch (DISABLED file / mem killswitch),
    how recall/unify work, the dashboard, MCP, eval/selftest, and troubleshooting (port 55432 vs native PG16; WSL
    keepalive). docs/ARCHITECTURE.md: deeper component diagram + data flow + design decisions (ParadeDB hybrid,
    warm-server hot path, files-as-truth, fail-safe hooks). Verify mem.cmd runs 'mem stats' (or '-m claudemem stats')
    successfully and include the output.`,
  },
]

phase('Surfaces')
const reports = await parallel(
  TASKS.map((t) => () => agent(SHARED + t.prompt, { label: t.key, phase: 'Surfaces' }))
)

return TASKS.map((t, i) => ({ task: t.key, report: reports[i] }))
