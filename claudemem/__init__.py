"""claudemem — private machine-wide memory for local agent runtimes.

Files are the source of truth (adapter-normalized transcripts + curated Markdown notes);
the database (ParadeDB primary, SQLite fallback) is a rebuildable derived index.

Public entry points:
    python -m claudemem <subcommand>     (see claudemem.cli)
    from claudemem.config import load_config
    from claudemem.store.factory import get_store
    from claudemem.retriever import Retriever
"""

__version__ = "0.3.0"
