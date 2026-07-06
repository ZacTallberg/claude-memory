"""claudemem — local best-of-breed agent-memory layer for Claude Code.

Files are the source of truth (Claude Code JSONL transcripts + curated markdown notes);
the database (ParadeDB primary, SQLite fallback) is a rebuildable derived index.

Public entry points:
    python -m claudemem <subcommand>     (see claudemem.cli)
    from claudemem.config import load_config
    from claudemem.store.factory import get_store
    from claudemem.retriever import Retriever
"""

__version__ = "0.1.0"
