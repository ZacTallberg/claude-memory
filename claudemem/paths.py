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
    """True iff cwd resolves under one of the configured workspace roots."""
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


@dataclass(frozen=True)
class TranscriptFile:
    path: Path
    encoded_dir: str
    project: str


@dataclass(frozen=True)
class MemoryDir:
    path: Path           # the .../memory directory
    encoded_dir: str
    project: str


def projects_root(cfg: Config) -> Path:
    return Path(cfg.scope.claude_projects_dir)


def iter_transcript_files(cfg: Config) -> list[TranscriptFile]:
    """All *.jsonl session transcripts across every per-project dir."""
    out: list[TranscriptFile] = []
    base = projects_root(cfg)
    if not base.exists():
        return out
    for proj_dir in sorted(base.iterdir()):
        if not proj_dir.is_dir():
            continue
        label = friendly_project(proj_dir.name)
        for jf in sorted(proj_dir.glob("*.jsonl")):
            out.append(TranscriptFile(path=jf, encoded_dir=proj_dir.name, project=label))
    return out


def iter_memory_dirs(cfg: Config) -> list[MemoryDir]:
    """Every per-project '.../memory' dir that holds curated notes."""
    out: list[MemoryDir] = []
    base = projects_root(cfg)
    if not base.exists():
        return out
    for proj_dir in sorted(base.iterdir()):
        mem = proj_dir / "memory"
        if mem.is_dir():
            out.append(MemoryDir(path=mem, encoded_dir=proj_dir.name, project=friendly_project(proj_dir.name)))
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
