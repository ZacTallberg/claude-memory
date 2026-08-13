"""Extensible transcript-adapter contract.

Built-ins cover Claude Code and Codex. Third-party adapters may register the Python entry-point
group ``claudemem.transcript_adapters``. Adapter loading occurs only in the background indexer;
prompt hooks never import or execute adapter code.
"""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Protocol, runtime_checkable

from . import codex_transcripts, transcripts
from .config import Config
from .log import get_logger
from .paths import TranscriptFile, friendly_project, projects_root

log = get_logger(__name__)
ENTRY_POINT_GROUP = "claudemem.transcript_adapters"


@runtime_checkable
class TranscriptAdapter(Protocol):
    """Minimal contract for a new agent-runtime transcript format."""

    name: str

    def discover(self, cfg: Config) -> list[TranscriptFile]: ...

    def parse_new(self, path: Path, start_byte: int, cfg: Config): ...


class ClaudeAdapter:
    name = "claude"

    def discover(self, cfg: Config) -> list[TranscriptFile]:
        out: list[TranscriptFile] = []
        base = projects_root(cfg)
        if not base.exists():
            return out
        for project_dir in sorted(base.iterdir()):
            if not project_dir.is_dir():
                continue
            label = friendly_project(project_dir.name)
            out.extend(TranscriptFile(path=path, encoded_dir=project_dir.name,
                                      project=label, provider=self.name)
                       for path in sorted(project_dir.glob("*.jsonl")))
        return out

    def parse_new(self, path: Path, start_byte: int, cfg: Config):
        return transcripts.parse_new(path, start_byte, cfg)


class CodexAdapter:
    name = "codex"

    def discover(self, cfg: Config) -> list[TranscriptFile]:
        out: list[TranscriptFile] = []
        home = Path(cfg.scope.codex_home)
        for folder in (home / "sessions", home / "archived_sessions"):
            if folder.exists():
                out.extend(TranscriptFile(path=path, encoded_dir="codex", project="codex",
                                          provider=self.name)
                           for path in sorted(folder.rglob("*.jsonl")))
        return out

    def parse_new(self, path: Path, start_byte: int, cfg: Config):
        return codex_transcripts.parse_new(path, start_byte, cfg)


_BUILTINS: dict[str, TranscriptAdapter] = {
    "claude": ClaudeAdapter(),
    "codex": CodexAdapter(),
}
_CACHE: dict[str, TranscriptAdapter] | None = None


def _coerce_adapter(value) -> TranscriptAdapter | None:
    try:
        candidate = value() if isinstance(value, type) else value
        if callable(candidate) and not hasattr(candidate, "discover"):
            candidate = candidate()
    except Exception as exc:
        log.warning("transcript adapter factory failed: %s", exc)
        return None
    if not isinstance(candidate, TranscriptAdapter):
        log.warning("transcript adapter rejected: missing name/discover/parse_new contract")
        return None
    return candidate


def adapters(*, refresh: bool = False) -> dict[str, TranscriptAdapter]:
    global _CACHE
    if _CACHE is not None and not refresh:
        return dict(_CACHE)
    found = dict(_BUILTINS)
    try:
        eps = metadata.entry_points().select(group=ENTRY_POINT_GROUP)
    except Exception:
        eps = []
    for ep in eps:
        try:
            adapter = _coerce_adapter(ep.load())
        except Exception as exc:
            log.warning("cannot load transcript adapter %s: %s", ep.name, exc)
            continue
        if adapter is None:
            continue
        name = str(adapter.name).strip().casefold()
        if not name or name in _BUILTINS:
            log.warning("transcript adapter %s cannot replace a built-in", ep.name)
            continue
        found[name] = adapter
    _CACHE = found
    return dict(found)


def get_adapter(name: str) -> TranscriptAdapter:
    key = str(name or "").strip().casefold()
    try:
        return adapters()[key]
    except KeyError as exc:
        raise ValueError(f"unknown transcript adapter {name!r}; available={sorted(adapters())}") from exc


def discover_transcripts(cfg: Config, *, only_provider: str | None = None) -> list[TranscriptFile]:
    enabled = [only_provider] if only_provider else list(cfg.index.transcript_providers)
    out: list[TranscriptFile] = []
    for name in enabled:
        adapter = get_adapter(name)
        try:
            files = adapter.discover(cfg)
        except Exception as exc:
            log.exception("transcript adapter %s discovery failed: %s", name, exc)
            continue
        for item in files:
            if not isinstance(item, TranscriptFile):
                log.warning("transcript adapter %s returned a non-TranscriptFile; skipping", name)
                continue
            # Enforce scope centrally rather than per adapter: every provider funnels through
            # here, so an excluded directory cannot re-enter the corpus via a different adapter.
            if _out_of_scope(item, cfg):
                continue
            out.append(item)
    return sorted(out, key=lambda item: (item.provider, str(item.path).casefold()))


def _out_of_scope(item: TranscriptFile, cfg: Config) -> bool:
    """Whether this transcript belongs to a project the config excludes from the corpus."""
    from .paths import is_excluded_project
    candidates = [item.encoded_dir or "", item.project or ""]
    try:
        candidates.append(Path(item.path).parent.name)
    except Exception:
        pass
    return any(is_excluded_project(c, cfg) for c in candidates if c)


def adapter_status(cfg: Config) -> list[dict]:
    registered = adapters()
    enabled = set(cfg.index.transcript_providers)
    return [{"name": name, "builtin": name in _BUILTINS, "enabled": name in enabled,
             "entry_point_group": ENTRY_POINT_GROUP}
            for name in sorted(registered)]
