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
    stats = index(cfg, store, full=args.full, progress=lambda m: print(m),
                  only_provider=args.provider)
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


def cmd_integrations(args):
    from .diagnostics import integration_status
    status = integration_status()
    print(json.dumps(status, indent=2, default=str))
    sys.exit(0 if status["ok"] else 1)


def cmd_delivery_check(args):
    from .diagnostics import delivery_check
    result = delivery_check(concurrency=args.concurrency, under_index_load=args.load,
                            cwd=args.cwd)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["ok"] else 1)


def cmd_backup(args):
    from pathlib import Path
    from .backup import create_backup, verify_snapshot
    from .config import load_config
    if args.verify:
        result = verify_snapshot(Path(args.verify))
    else:
        result = create_backup(load_config(), if_due=args.if_due, retention=args.retention)
    print(json.dumps(result, indent=2))


def cmd_sanitize(args):
    from .config import load_config
    from .sanitize import sanitize_sqlite
    result = sanitize_sqlite(load_config(), apply=args.apply)
    print(json.dumps(result, indent=2))


def cmd_promote(args):
    from .promote import mine_candidates
    n = mine_candidates()
    print(f"{n} promotion candidates drafted")


def cmd_conflicts(args):
    from .config import load_config
    from .facts import load_notes
    from .memory_types import find_claim_conflicts
    conflicts = find_claim_conflicts(load_notes(load_config()))
    print(json.dumps({"count": len(conflicts), "conflicts": conflicts}, indent=2))
    sys.exit(1 if conflicts and args.strict else 0)


def cmd_adapters(args):
    from .config import load_config
    from .transcript_adapters import adapter_status
    print(json.dumps({"adapters": adapter_status(load_config())}, indent=2))


def cmd_models(args):
    from .model_bakeoff import supported_model_specs
    rows = [vars(spec) for spec in supported_model_specs().values()]
    print(json.dumps({"models": rows}, indent=2))


def cmd_model_bakeoff(args):
    from .model_bakeoff import run_bakeoff
    report = run_bakeoff(args.models, pool_k=args.pool_k, k=args.k)
    print(json.dumps(report, indent=2))


def cmd_feedback_experiment(args):
    from .feedback_learning import run_experiment
    report = run_experiment(min_feedback=args.min_feedback)
    print(json.dumps(report, indent=2))


def cmd_tune_ranking(args):
    from .ranking_experiment import run_sweep
    report = run_sweep(rrf_values=args.rrf, half_life_values=args.half_life)
    print(json.dumps(report, indent=2))


def cmd_longmemeval(args):
    from pathlib import Path
    from .benchmarks.longmemeval import run
    report = run(Path(args.path), mode=args.mode, k=args.k, limit=args.limit)
    print(json.dumps(report, indent=2))


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
    s.add_argument("--provider")
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
    s = sub.add_parser("integrations"); s.set_defaults(fn=cmd_integrations)
    s = sub.add_parser("delivery-check")
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--load", action="store_true", help="run while a live index pass is active")
    s.add_argument("--cwd")
    s.set_defaults(fn=cmd_delivery_check)
    s = sub.add_parser("backup"); s.add_argument("--if-due", action="store_true")
    s.add_argument("--retention", type=int, default=14); s.add_argument("--verify")
    s.set_defaults(fn=cmd_backup)
    s = sub.add_parser("sanitize"); s.add_argument("--apply", action="store_true")
    s.set_defaults(fn=cmd_sanitize)
    s = sub.add_parser("promote"); s.set_defaults(fn=cmd_promote)
    s = sub.add_parser("consolidate", help="draft typed memory candidates for human review")
    s.set_defaults(fn=cmd_promote)
    s = sub.add_parser("conflicts", help="inspect unresolved structured temporal claims")
    s.add_argument("--strict", action="store_true", help="exit non-zero when conflicts exist")
    s.set_defaults(fn=cmd_conflicts)
    s = sub.add_parser("adapters", help="list built-in and installed transcript adapters")
    s.set_defaults(fn=cmd_adapters)
    s = sub.add_parser("models", help="list embedding models supported by installed FastEmbed")
    s.set_defaults(fn=cmd_models)
    s = sub.add_parser("model-bakeoff", help="compare embedding models without mutating the live index")
    s.add_argument("models", nargs="+", help="FastEmbed model names; use 'current' for the live model")
    s.add_argument("--pool-k", type=int, default=50)
    s.add_argument("--k", type=int)
    s.set_defaults(fn=cmd_model_bakeoff)
    s = sub.add_parser("feedback-experiment", help="offline usefulness-feedback ranking gate")
    s.add_argument("--min-feedback", type=int, default=5)
    s.set_defaults(fn=cmd_feedback_experiment)
    s = sub.add_parser("tune-ranking", help="read-only golden sweep of RRF and recency constants")
    s.add_argument("--rrf", type=int, nargs="+")
    s.add_argument("--half-life", type=float, nargs="+")
    s.set_defaults(fn=cmd_tune_ranking)
    s = sub.add_parser("longmemeval", help="official-format LongMemEval session retrieval benchmark")
    s.add_argument("path"); s.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid")
    s.add_argument("--k", type=int, default=10); s.add_argument("--limit", type=int)
    s.set_defaults(fn=cmd_longmemeval)
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
