from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SYSTEM_MEMORY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    root: Path = Field(default_factory=lambda: Path.cwd())
    database_path: Path = Path("data/system-memory.db")
    archive_path: Path = Path("data/archive")
    token_path: Path = Path("data/admin.token")
    busy_timeout_ms: int = Field(default=5_000, ge=100, le=120_000)
    host: str = "127.0.0.1"
    port: int = Field(default=7788, ge=1024, le=65_535)
    request_body_limit: int = Field(default=2_200_000, ge=1024, le=10_000_000)
    lexical_limit: int = Field(default=40, ge=1, le=500)
    abstention_min_score: float = Field(default=0.30, ge=0, le=10)
    vector_min_similarity: float = Field(default=0.72, ge=-1, le=1)
    query_inference_timeout_seconds: float = Field(default=3.0, ge=0.05, le=60)
    inference_capacity: int = Field(default=16, ge=1, le=128)
    embedding_threads: int = Field(default=2, ge=1, le=32)
    embedding_cache_path: Path = Path("var/models")
    live_embedding_poll_seconds: float = Field(default=1.0, ge=0.05, le=60)

    def resolve_path(self, path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    @property
    def resolved_database_path(self) -> Path:
        return self.resolve_path(self.database_path)

    @property
    def resolved_archive_path(self) -> Path:
        return self.resolve_path(self.archive_path)

    @property
    def resolved_token_path(self) -> Path:
        return self.resolve_path(self.token_path)

    @property
    def resolved_embedding_cache_path(self) -> Path:
        return self.resolve_path(self.embedding_cache_path)
