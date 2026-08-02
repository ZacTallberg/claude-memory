"""`mem` command-line interface. Subcommand imports are lazy to keep startup fast."""
from __future__ import annotations

import argparse
import json
import sys


def _store_and_cfg():
    from .config import load_config
    from .store.factory import get_store
    cfg = load_config()
    return cfg, get_store(cfg)


def cmd_index(args):
    from .indexer import index
    cfg, store = _store_and_cfg()
    stats = index(cfg, store, full=args.full, progress=lambda m: print(m))
    print(json.dumps(stats.as_dict(), indent=2))
    if getattr(args, "promote", False):
        from .promote import mine_candidates
        n = mine_candidates()
        print(f"{n} promotion candidates drafted")


def cmd_embed(args):
    from .indexer import embed_pending
    cfg, store = _store_and_cfg()
    n = embed_pending(cfg, store, progress=lambda m: print(m))
    print(f"embedded {n} chunks")


def cmd_query(args):
    from .retriever import Retriever
    cfg, store = _store_and_cfg()
    r = Retriever(cfg, store)
    res = r.search(args.query, tier=("full" if args.rerank else "hot"),
                   k=args.k, do_rerank=args.rerank or None)
    for x in res:
        c = x.chunk
        print(f"[{x.score:.4f}] {c.project} | {c.role}/{c.kind} | {c.session_id}")
        print(f"    {c.content[:200]}")
    print(f"\n{len(res)} results")


def cmd_facts(args):
    from .retriever import Retriever
    cfg, store = _store_and_cfg()
    r = Retriever(cfg, store)
    for f in r.search_facts(args.topic, k=args.k):
        print(f"[{f.type}] {f.title}  ({f.project})")
        print(f"    {f.description}")
        if args.full:
            print("    " + f.body.replace("\n", "\n    "))


def cmd_stats(args):
    cfg, store = _store_and_cfg()
    print(json.dumps(store.health(), indent=2, default=str))


def cmd_serve(args):
    from .config import load_config
    cfg = load_config()
    from .dashboard.server import run
    run(host=cfg.server.host, port=(args.port or cfg.server.port),
        open_browser=cfg.server.open_browser and not args.no_browser)


def cmd_selftest(args):
    from .selftest import run_selftest
    sys.exit(0 if run_selftest(verbose=True) else 1)


def cmd_eval(args):
    from .eval import run_eval
    run_eval()


def cmd_promote(args):
    from .promote import mine_candidates
    n = mine_candidates()
    print(f"{n} promotion candidates drafted")


def cmd_install_hooks(args):
    from .hooks_install import install
    print(install())


def cmd_uninstall_hooks(args):
    from .hooks_install import uninstall
    print(uninstall())


def cmd_install_codex_hooks(args):
    from .codex_hooks_install import install
    print(install())


def cmd_uninstall_codex_hooks(args):
    from .codex_hooks_install import uninstall
    print(uninstall())


def cmd_killswitch(args):
    from .paths import killed, set_killed
    if args.action == "status":
        print("DISABLED" if killed() else "enabled")
    else:
        set_killed(args.action == "on")
        print("DISABLED" if killed() else "enabled")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mem", description="shared local agent-memory CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("index"); s.add_argument("--full", action="store_true")
    s.add_argument("--promote", action="store_true"); s.set_defaults(fn=cmd_index)
    s = sub.add_parser("embed"); s.set_defaults(fn=cmd_embed)
    s = sub.add_parser("query"); s.add_argument("query"); s.add_argument("--k", type=int, default=6)
    s.add_argument("--rerank", action="store_true"); s.set_defaults(fn=cmd_query)
    s = sub.add_parser("facts"); s.add_argument("topic"); s.add_argument("--k", type=int, default=6)
    s.add_argument("--full", action="store_true"); s.set_defaults(fn=cmd_facts)
    s = sub.add_parser("stats"); s.set_defaults(fn=cmd_stats)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int); s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("selftest"); s.set_defaults(fn=cmd_selftest)
    s = sub.add_parser("eval"); s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("promote"); s.set_defaults(fn=cmd_promote)
    s = sub.add_parser("install-hooks"); s.set_defaults(fn=cmd_install_hooks)
    s = sub.add_parser("uninstall-hooks"); s.set_defaults(fn=cmd_uninstall_hooks)
    s = sub.add_parser("install-codex-hooks"); s.set_defaults(fn=cmd_install_codex_hooks)
    s = sub.add_parser("uninstall-codex-hooks"); s.set_defaults(fn=cmd_uninstall_codex_hooks)
    s = sub.add_parser("killswitch"); s.add_argument("action", choices=["on", "off", "status"])
    s.set_defaults(fn=cmd_killswitch)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
