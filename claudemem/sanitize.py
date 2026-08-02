"""One-way credential purge for the legacy SQLite corpus.

This module intentionally reports only aggregate secret types and row identifiers. It never
returns, prints, or logs matched plaintext. Vector rows for changed text are removed because
an embedding of credential-bearing text is also stale and must be regenerated.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

from .config import Config
from .security import SecretFinding, redact_secrets


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name=? LIMIT 1", (name,)
    ).fetchone() is not None


def sanitize_sqlite(cfg: Config, *, apply: bool = False) -> dict:
    if cfg.store.backend != "sqlite":
        raise RuntimeError("legacy sanitation requires the pinned sqlite backend")
    path = Path(cfg.store.sqlite.path)
    if not path.is_absolute():
        path = cfg.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    vec_available = False
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        vec_available = True
    except Exception:
        vec_available = False

    counts: Counter[str] = Counter()
    fingerprints: dict[str, SecretFinding] = {}
    changed_rows: list[dict] = []
    scanned = 0

    def record(findings: list[SecretFinding]) -> None:
        for finding in findings:
            counts[finding.kind] += 1
            fingerprints.setdefault(finding.fingerprint, finding)

    for row in conn.execute("SELECT id,content,context_blurb FROM chunks"):
        scanned += 1
        content, found_content = redact_secrets(row["content"] or "")
        blurb, found_blurb = redact_secrets(row["context_blurb"] or "")
        found = found_content + found_blurb
        if not found:
            continue
        record(found)
        search_text = ((blurb + " ") if blurb else "") + content
        changed_rows.append({"table": "chunks", "id": row["id"], "content": content,
                             "context_blurb": blurb or None, "search_text": search_text})

    for row in conn.execute(
        "SELECT id,name,title,project,tags,description,body FROM facts"
    ):
        scanned += 1
        fields: dict[str, str] = {}
        found: list[SecretFinding] = []
        for field in ("name", "title", "project", "description", "body"):
            fields[field], hits = redact_secrets(row[field] or "")
            found.extend(hits)
        try:
            original_tags = json.loads(row["tags"] or "[]")
        except Exception:
            original_tags = []
        safe_tags = []
        for tag in original_tags if isinstance(original_tags, list) else []:
            safe, hits = redact_secrets(str(tag))
            safe_tags.append(safe)
            found.extend(hits)
        if not found:
            continue
        record(found)
        search_text = " ".join([
            fields["name"], fields["title"], fields["project"], " ".join(safe_tags),
            fields["description"], fields["body"],
        ])
        changed_rows.append({"table": "facts", "id": row["id"], **fields,
                             "tags": json.dumps(safe_tags), "search_text": search_text})

    auxiliary = {
        "injections": ("id", ("prompt_excerpt", "details")),
        "promotion_candidates": ("id", ("title", "body", "support")),
        "anti_memory": ("id", ("key", "reason")),
        "metrics": ("id", ("details",)),
        "kv": ("key", ("value",)),
        "graph_nodes": ("id", ("label",)),
    }
    for table, (key_col, columns) in auxiliary.items():
        if not _table_exists(conn, table):
            continue
        selected = ",".join((key_col, *columns))
        for row in conn.execute(f"SELECT {selected} FROM {table}"):
            scanned += 1
            updates: dict[str, str] = {}
            found: list[SecretFinding] = []
            for column in columns:
                safe, hits = redact_secrets(row[column] or "")
                updates[column] = safe
                found.extend(hits)
            if found:
                record(found)
                changed_rows.append({"table": table, "key_col": key_col,
                                     "id": row[key_col], **updates})

    if apply and changed_rows:
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""CREATE TABLE IF NOT EXISTS secret_tombstones(
                fingerprint TEXT PRIMARY KEY, kind TEXT NOT NULL,
                detected_at TEXT NOT NULL DEFAULT (datetime('now'))
            )""")
            for finding in fingerprints.values():
                conn.execute(
                    "INSERT OR IGNORE INTO secret_tombstones(fingerprint,kind) VALUES (?,?)",
                    (finding.fingerprint, finding.kind),
                )
            for item in changed_rows:
                table = item["table"]
                if table == "chunks":
                    conn.execute(
                        "UPDATE chunks SET content=?,context_blurb=?,search_text=? WHERE id=?",
                        (item["content"], item["context_blurb"], item["search_text"], item["id"]),
                    )
                    conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (item["id"],))
                    conn.execute("INSERT INTO chunks_fts(rowid,search_text) VALUES (?,?)",
                                 (item["id"], item["search_text"]))
                    if vec_available and _table_exists(conn, "chunks_vec"):
                        conn.execute("DELETE FROM chunks_vec WHERE rowid=?", (item["id"],))
                elif table == "facts":
                    conn.execute(
                        """UPDATE facts SET name=?,title=?,project=?,tags=?,description=?,body=?,
                           search_text=? WHERE id=?""",
                        (item["name"], item["title"], item["project"], item["tags"],
                         item["description"], item["body"], item["search_text"], item["id"]),
                    )
                    conn.execute("DELETE FROM facts_fts WHERE rowid=?", (item["id"],))
                    conn.execute("INSERT INTO facts_fts(rowid,search_text) VALUES (?,?)",
                                 (item["id"], item["search_text"]))
                    if vec_available and _table_exists(conn, "facts_vec"):
                        conn.execute("DELETE FROM facts_vec WHERE rowid=?", (item["id"],))
                else:
                    columns = [key for key in item if key not in ("table", "key_col", "id")]
                    assignments = ",".join(f"{column}=?" for column in columns)
                    conn.execute(
                        f"UPDATE {table} SET {assignments} WHERE {item['key_col']}=?",
                        [item[column] for column in columns] + [item["id"]],
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    conn.close()

    by_table = Counter(item["table"] for item in changed_rows)
    return {
        "ok": True,
        "applied": bool(apply),
        "database": str(path),
        "rows_scanned": scanned,
        "rows_changed": len(changed_rows),
        "rows_by_table": dict(sorted(by_table.items())),
        "detections_by_type": dict(sorted(counts.items())),
        "unique_fingerprints": len(fingerprints),
        "vectors_invalidated": sum(1 for item in changed_rows
                                   if item["table"] in ("chunks", "facts")) if apply else 0,
    }

