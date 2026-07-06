"""Core smoke test: config -> store -> index real data -> hybrid query. Run with the venv python."""
import sys
import time

from claudemem.config import load_config
from claudemem.store.factory import get_store
from claudemem.indexer import index
from claudemem.retriever import Retriever
from claudemem.providers.embeddings import get_embedding_provider

cfg = load_config()
print("config: backend=", cfg.store.backend, "embed=", cfg.embeddings.model, "dim=", cfg.embeddings.dim)

emb = get_embedding_provider(cfg)
print("embedder available:", emb.available(), "name:", getattr(emb, "name", "?"), "dim:", emb.dim)

store = get_store(cfg)
print("store health:", store.health())

t0 = time.time()
stats = index(cfg, store, full=False, progress=lambda m: print("   ", m))
print("index stats:", stats.as_dict(), "in %.1fs" % (time.time() - t0))
print("counts:", store.counts())

r = Retriever(cfg, store, embedder=emb)
for q in (sys.argv[1:] or ["explore before building", "deploy to prod dokku", "procedural plant genome"]):
    t0 = time.time()
    res = r.search(q, tier="hot", k=4)
    print(f"\n=== HOT query: {q!r}  ({(time.time()-t0)*1000:.0f} ms, {len(res)} hits) ===")
    for x in res:
        c = x.chunk
        print(f"  [{x.score:.4f}] {c.project} | {c.role}/{c.kind} | {c.session_id} :: {c.content[:110]!r}")
    facts = r.search_facts(q, k=3)
    print(f"  facts ({len(facts)}):", [f"{f.type}:{f.title}" for f in facts])

print("\nSMOKE_OK")
