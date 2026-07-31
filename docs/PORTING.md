# Porting claude-memory to a new machine

This repo carries the **code and doctrine**. The **memory itself** — the curated note stores, the
derived database, and (optionally) the raw transcripts — is personal data that never goes to GitHub
and moves out-of-band. This document is the complete bring-up, in order. Do the steps in order;
step 4's ordering rule is load-bearing.

---

## 0. What lives where (the full system map)

The engine is more than this repo. On a configured machine the install consists of:

| piece | location | how it gets there |
|---|---|---|
| code + config | `<repo>` (this repo) | `git clone` |
| venv + package | `<repo>/.venv` | step 2 |
| derived DB (chunks, facts, injections, metrics, promotion queue, anti-memory) | `<repo>/data/claudemem.db` | copied from old machine, or rebuilt |
| embedding model cache | `<repo>/.fastembed_cache/` | auto-downloaded on first embed (~130 MB, needs internet once) |
| curated note stores (**the source of truth**) | `~/.claude/projects/<store>/memory/*.md` + `MEMORY.md` per store | copied from old machine |
| raw transcripts (optional reindex source) | `~/.claude/projects/<store>/*.jsonl` | copied only if you want transcript-derived recall rebuilt from source |
| hook wiring | `~/.claude/settings.json` (UserPromptSubmit → `hooks/recall.py`; SessionStart startup/resume/clear → `hooks/unify.py`; SessionEnd + PreCompact → `hooks/index_trigger.py`) | `mem install-hooks` (regenerates absolute paths from the repo's location — never copy these entries between machines) |
| supervisor | Scheduled Task `ClaudeMemoryPersistence` → `scripts/persistence_run.ps1` | `install_persistence.ps1` (elevated) |
| health beacon | `~/.claude/memory-health.json` (written by unify + recall) | appears on first hook fire |
| statusline segment | `~/.claude/statusline.js` (or your own) | graft the snippet in §7 |
| kill switch | `<repo>/DISABLED` sentinel | `mem killswitch on|off` |

## 1. Prerequisites

Windows 11 · Python 3.12 · git. Node is **not** required (dashboard JS is vendored, CSS pre-built).
WSL2 + Docker are needed **only** if you re-enable the optional ParadeDB backend — the pinned
default is sqlite and needs neither.

## 2. Clone + install

```powershell
git clone https://github.com/ZacTallberg/claude-memory.git C:\code\claude-memory
cd C:\code\claude-memory
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\scripts\fetch_vendor.ps1          # one-time: vendored dashboard JS
```

## 3. Point config at the new machine

Edit `config.toml`:

- `[scope] workspace_roots` — the roots under which recall/unify activate on the new machine.
- `[scope] claude_projects_dir` — the new machine's `~/.claude/projects` absolute path.
- Leave `[store] backend = "sqlite"` pinned. `"auto"` forked the store into two diverging copies
  whenever ParadeDB flapped — that is a repaired outage, not a preference.

## 4. Move the memory (ORDER MATTERS)

**Copy the curated note stores BEFORE the DB, and both BEFORE any `mem index` runs.** The indexer
prunes facts whose backing file is absent from disk — run it against a copied DB with the note
stores missing and it will wipe the facts index it was supposed to serve.

```powershell
# 4a. curated note stores (source of truth) — every per-project store you care about
robocopy \\old\c$\Users\<olduser>\.claude\projects $env:USERPROFILE\.claude\projects /E /XF *.jsonl

# 4b. the derived DB — carries transcript-derived memory the new machine cannot rebuild
#     (it has no old transcripts), plus injections history, metrics, promotion queue, anti-memory
robocopy \\old\C$\code\claude-memory\data C:\code\claude-memory\data claudemem.db

# 4c. (optional, heavy) raw transcripts, only if you want reindex-from-source ability
#     — drop the /XF *.jsonl filter in 4a instead
```

Fresh start on a machine with no history: skip 4a/4b and just run `.\mem.cmd index`.

> The old machine also keeps hook-event history in `~/.claude/settings.json` and the statusline in
> `~/.claude/statusline.js`. **Do not copy either file wholesale** — settings.json is full of
> old-machine absolute paths. Hooks are regenerated in step 5; the statusline segment is grafted
> in step 7.

## 5. Wire the hooks

```powershell
.\mem.cmd install-hooks      # idempotent; preserves every non-claude-memory hook entry
```

This writes absolute venv + hook paths derived from where the repo actually sits, so it is correct
on any machine with no editing.

## 6. Persistence (supervisor)

```powershell
# from an ELEVATED PowerShell — Register-ScheduledTask silently no-ops without it,
# and the installer verifies registration precisely because that failure was once silent
.\scripts\install_persistence.ps1
Get-ScheduledTask ClaudeMemoryPersistence      # expect State: Ready
```

The supervisor (`persistence_run.ps1`, named-mutex singleton) starts and revives the warm server
(`mem serve --no-browser` on 127.0.0.1:7777), probes `/healthz` for a real 200 with 90s warm-up
grace and a 3-strike rule, and clears the port holder before a restart. Its WSL/ParadeDB legs
re-arm automatically only when `backend != "sqlite"`. Every Claude Code session start is also a
watchdog: `hooks/unify.py` re-arms the supervisor if 7777 is closed.

## 7. Statusline beacon (delivery health must never be silent)

The hooks write `~/.claude/memory-health.json`:
`{ts, source: "server"|"local-fallback", backend, facts_shown?, facts_total?}`. Graft this into
whatever statusline the new machine uses (this is the exact segment from the old machine):

```js
function memSegment() {
  const p = path.join(os.homedir(), '.claude', 'memory-health.json');
  if (!fs.existsSync(p)) return red('mem ?');
  const h = JSON.parse(fs.readFileSync(p, 'utf8'));
  const ageH = (Date.now() / 1000 - (h.ts || 0)) / 3600;
  if (ageH > 48) return red('mem stale ' + Math.round(ageH) + 'h');
  if (h.source === 'server') return green('mem ok');
  const backend = h.backend === 'PostgresStore' ? 'pg' : 'sq';
  const counts = h.facts_total != null ? ' ' + h.facts_shown + '/' + h.facts_total : '';
  return yellow('mem ' + backend + counts);   // yellow = alive but on keyword-only fallback
}
```

Wire it via `~/.claude/settings.json` → `"statusLine": {"type": "command", "command": ...}`.

## 8. Verify (all of it, before trusting it)

```powershell
.\mem.cmd selftest                          # expect FAIL=0 — includes 5 wedge guards.
                                            # Before data lands (step 4 skipped, index not run),
                                            # corpus-dependent checks SKIP with "corpus empty";
                                            # after migration expect zero corpus skips.
.\mem.cmd stats                             # backend=sqlite, chunk/fact counts match the old box
curl.exe -s http://127.0.0.1:7777/healthz   # {"ok": true, "store": "ok"}
```

Then the end-to-end proof: open a Claude Code session under a workspace root, send a real prompt,
and confirm (a) a `<recalled-memory>` block appears, (b) the statusline shows `mem ok` (green —
i.e. `source: "server"`, not fallback), (c) a new row lands in the `injections` table:

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; print(sqlite3.connect('data/claudemem.db').execute('select ts, hook, latency_ms from injections order by rowid desc limit 3').fetchall())"
```

A port is **not done** until that end-to-end chain is seen working. Port-open is not healthy;
`is-active` is not `is-working` — ask for receipts (the injections rows are the receipts).

## Known gotchas (each of these cost real time once)

- **`.ps1` files must be ASCII-only** (no BOM, no smart quotes) or PowerShell 5.1 mangles them.
- **`PYTHONUTF8=1` is required** on Windows — always run via `mem.cmd` / `mem.ps1`, which set it.
- **Changing the embedding model or `dim` requires `mem index --full`** — mixed-dim vectors search
  wrong, silently. 44k chunks re-embed in hours on CPU; schedule it.
- **First embed needs internet** to populate `.fastembed_cache/`; after that it is fully offline.
- **The dashboard opens a browser by default** — the supervisor uses `--no-browser`; do the same in
  any script.
- Backup floor: once ported, the DB is the only copy of old-transcript memory — put
  `data/claudemem.db` + the note stores on a nightly copy with a rehearsed restore.
