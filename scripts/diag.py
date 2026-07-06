import traceback
from pathlib import Path
from claudemem.config import load_config

cfg = load_config()
print("PG target:", cfg.store.postgres.host, cfg.store.postgres.port, cfg.store.postgres.dbname,
      "user=", cfg.store.postgres.user)
from claudemem.store.postgres_store import PostgresStore
s = PostgresStore(cfg)
try:
    s.connect(); print("connected OK")
    s.migrate(); print("migrated OK")
    print("health:", s.health())
except Exception:
    print("---- POSTGRES ERROR ----")
    traceback.print_exc()

sq = cfg.root / "data" / "claudemem.db"
print("sqlite fallback exists:", sq.exists(), (sq.stat().st_size if sq.exists() else 0), "bytes")
