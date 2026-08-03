"""Path discovery, project-name decoding, scoping predicate, and the kill switch.

Claude Code encodes a project's cwd into its transcript dir name by replacing ':' and
path separators with '-'. Decoding is ambiguous (real '-' vs separator), so we never
rely on decoding for correctness — transcript records carry an explicit `cwd` field.
We only derive a *friendly display label* from the encoded dir name.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import Config, ROOT

DISABLED_SENTINEL = ROOT / "DISABLED"


def killed() -> bool:
    """True iff the single-file kill switch is present (turns both hooks off)."""
    return DISABLED_SENTINEL.exists()


def set_killed(on: bool) -> None:
    if on:
        DISABLED_SENTINEL.write_text("memory layer disabled\n", encoding="utf-8")
    elif DISABLED_SENTINEL.exists():
        DISABLED_SENTINEL.unlink()


def _norm(p: str | os.PathLike) -> str:
    try:
        return os.path.normcase(str(Path(p).resolve()))
    except Exception:
        return os.path.normcase(str(p))


def in_scope(cwd: str | None, cfg: Config) -> bool:
    """Whether an installed lifecycle hook should activate for this context.

    ``installed_clients`` is machine-wide by design: the user-level hook registration is the
    trust boundary, not an arbitrary filesystem prefix. ``workspace_roots`` remains available
    for deliberately constrained installations and for path-policy regression tests.
    """
    if cfg.scope.activation == "installed_clients":
        return True
    if cfg.scope.activation != "workspace_roots":
        return False
    if not cwd:
        return False
    c = _norm(cwd)
    for root in cfg.scope.workspace_roots:
        r = _norm(root)
        if c == r or c.startswith(r + os.sep):
            return True
    return False


def friendly_project(encoded_dir_name: str) -> str:
    """'C--code-website-dokku' -> 'website-dokku'; 'C--code' -> 'code';
    'C--Users-zcobe' -> 'Users-zcobe'. Strips a leading drive token and common
    'code-' workspace prefix for a compact, human label."""
    name = encoded_dir_name
    # Strip a leading drive token like 'C--' (drive letter + ':' + first sep -> 'C--').
    if len(name) >= 3 and name[1:3] == "--":
        name = name[3:]
    # Strip a leading 'code-' workspace prefix if present.
    if name.startswith("code-"):
        name = name[len("code-"):]
    return name or encoded_dir_name


def project_from_cwd(cwd: str | None) -> str | None:
    """Best-effort project label for visibility checks on live prompt delivery."""
    if not cwd:
        return None
    try:
        return Path(cwd).resolve().name or None
    except Exception:
        return Path(str(cwd)).name or None


@dataclass(frozen=True)
class TranscriptFile:
    path: Path
    encoded_dir: str
    project: str
    provider: str = "claude"


@dataclass(frozen=True)
class MemoryDir:
    path: Path           # the .../memory directory
    encoded_dir: str
    project: str


def projects_root(cfg: Config) -> Path:
    return Path(cfg.scope.claude_projects_dir)


def canonical_memory_root(cfg: Config) -> Path:
    """Agent-neutral source of truth for newly authored curated notes."""
    return Path(cfg.scope.memory_root)


def memory_write_roots(cfg: Config) -> list[Path]:
    """All roots in which an existing note may be updated safely."""
    roots = [canonical_memory_root(cfg)]
    if cfg.scope.include_legacy_claude_notes:
        roots.append(projects_root(cfg))
    return roots


def iter_transcript_files(cfg: Config) -> list[TranscriptFile]:
    """All configured agent transcripts, tagged for provider-specific parsing."""
    out: list[TranscriptFile] = []
    providers = set(cfg.index.transcript_providers)
    if "claude" in providers:
        base = projects_root(cfg)
        if base.exists():
            for proj_dir in sorted(base.iterdir()):
                if not proj_dir.is_dir():
                    continue
                label = friendly_project(proj_dir.name)
                for jf in sorted(proj_dir.glob("*.jsonl")):
                    out.append(TranscriptFile(path=jf, encoded_dir=proj_dir.name,
                                              project=label, provider="claude"))
    if "codex" in providers:
        home = Path(cfg.scope.codex_home)
        for folder in (home / "sessions", home / "archived_sessions"):
            if folder.exists():
                for jf in sorted(folder.rglob("*.jsonl")):
                    out.append(TranscriptFile(path=jf, encoded_dir="codex",
                                              project="codex", provider="codex"))
    return out


def iter_memory_dirs(cfg: Config) -> list[MemoryDir]:
    """Every curated-note directory, canonical first and legacy Claude dirs second."""
    out: list[MemoryDir] = []
    seen: set[str] = set()

    neutral = canonical_memory_root(cfg)
    if neutral.is_dir():
        if any(p.is_file() and p.suffix.lower() == ".md" for p in neutral.iterdir()):
            out.append(MemoryDir(path=neutral, encoded_dir="global", project="global"))
            seen.add(_norm(neutral))
        for project_dir in sorted(p for p in neutral.iterdir() if p.is_dir()):
            out.append(MemoryDir(path=project_dir, encoded_dir=project_dir.name,
                                 project=project_dir.name))
            seen.add(_norm(project_dir))

    if cfg.scope.include_legacy_claude_notes:
        base = projects_root(cfg)
        if base.exists():
            for proj_dir in sorted(base.iterdir()):
                mem = proj_dir / "memory"
                if mem.is_dir() and _norm(mem) not in seen:
                    out.append(MemoryDir(path=mem, encoded_dir=proj_dir.name,
                                         project=friendly_project(proj_dir.name)))
                    seen.add(_norm(mem))
    return out


def iter_note_files(cfg: Config) -> list[tuple[Path, str]]:
    """All curated note markdown files (excluding the MEMORY.md index) with project label."""
    out: list[tuple[Path, str]] = []
    for md in iter_memory_dirs(cfg):
        for nf in sorted(md.path.glob("*.md")):
            if nf.name.upper() == "MEMORY.MD":
                continue
            out.append((nf, md.project))
    return out


def safe_under(path: str | os.PathLike, roots: list[str | os.PathLike]) -> bool:
    """Path-traversal guard: True iff `path` resolves under one of `roots`."""
    p = _norm(path)
    for r in roots:
        rr = _norm(r)
        if p == rr or p.startswith(rr + os.sep):
            return True
    return False


def is_curated_note_path(path: str | os.PathLike, cfg: Config) -> bool:
    """True only for an allowed one-level canonical or legacy curated-note Markdown path."""
    try:
        p = Path(path).resolve()
        neutral = canonical_memory_root(cfg).resolve()
        legacy = projects_root(cfg).resolve()
    except Exception:
        return False
    if p.suffix.lower() != ".md" or p.name.upper() == "MEMORY.MD":
        return False
    if p.parent == neutral or p.parent.parent == neutral:
        return True
    return (cfg.scope.include_legacy_claude_notes
            and p.parent.name.lower() == "memory"
            and p.parent.parent.parent == legacy)
