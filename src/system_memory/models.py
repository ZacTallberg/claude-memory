from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM_OBSERVATION = "system_observation"


class Authority(StrEnum):
    USER_AUTHORED = "user_authored"
    USER_DECLARATION = "user_declaration"
    USER_BEHAVIOR = "user_behavior"
    EXPLICIT_DECISION = "explicit_decision"
    TOOL_OUTCOME = "tool_outcome"
    ASSISTANT_SYNTHESIS = "assistant_synthesis"
    IMPORTED_UNKNOWN = "imported_unknown"


class EventKind(StrEnum):
    MESSAGE = "message"
    TOOL_OUTCOME = "tool_outcome"
    CHECKPOINT = "checkpoint"
    DECISION = "decision"
    HUB_EVENT = "hub_event"
    ARTIFACT = "artifact"
    LEGACY_RECOVERED = "legacy_recovered"


class IngestEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=64)
    source_kind: str = Field(min_length=1, max_length=64)
    source_locator: str = Field(min_length=1, max_length=4096)
    provider_event_id: str | None = Field(default=None, max_length=512)
    agent_id: str = Field(min_length=1, max_length=256)
    session_id: str = Field(min_length=1, max_length=512)
    parent_session_id: str | None = Field(default=None, max_length=512)
    episode_sequence: int = Field(default=0, ge=0)
    project_id: str | None = Field(default=None, max_length=256)
    task_id: str | None = Field(default=None, max_length=512)
    hub_instance_id: str | None = Field(default=None, max_length=512)
    worktree: str | None = Field(default=None, max_length=4096)
    commit_sha: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{7,64}$")
    role: Role
    authority: Authority
    kind: EventKind = EventKind.MESSAGE
    occurred_at: datetime
    content: str = Field(min_length=1, max_length=2_000_000)
    source_offset_start: int | None = Field(default=None, ge=0)
    source_offset_end: int | None = Field(default=None, ge=0)
    visibility: Literal["private", "shared", "public"] = "private"
    trust: Literal["authored", "observed", "derived", "legacy"] = "authored"
    loss_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", "source_kind", "agent_id")
    @classmethod
    def normalize_identifier(cls, value: str) -> str:
        return value.strip().lower().replace(" ", "-")


class IngestResult(BaseModel):
    event_id: str
    source_id: str
    episode_id: str
    inserted: bool
    redaction_count: int
    content_sha256: str


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    span_start: int = Field(default=0, ge=0)
    span_end: int | None = Field(default=None, ge=0)
    relation: Literal["supports", "contradicts", "context"] = "supports"


class ClaimOperation(StrEnum):
    ADD = "ADD"
    MERGE = "MERGE"
    SUPERSEDE = "SUPERSEDE"
    RETRACT = "RETRACT"
    DISPUTE = "DISPUTE"
    NOOP = "NOOP"


class ClaimProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series_id: str | None = None
    operation: ClaimOperation
    subject: str = Field(min_length=1, max_length=512)
    predicate: str = Field(min_length=1, max_length=256)
    value: Any
    rendering: str = Field(min_length=1, max_length=32_000)
    authority: Authority
    confidence: float = Field(ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    predecessor_revision_id: str | None = None
    created_by: str = Field(min_length=1, max_length=256)
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)


class RecallScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_ids: tuple[str, ...] = ()
    providers: tuple[str, ...] = ()
    session_ids: tuple[str, ...] = ()
    roles: tuple[Role, ...] = ()
    hard_filter: bool = False


class RecallQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=32_000)
    current_project_id: str | None = None
    current_task_id: str | None = None
    current_provider: str | None = None
    as_of: datetime | None = None
    scope: RecallScope = Field(default_factory=RecallScope)
    limit: int = Field(default=6, ge=1, le=50)
    max_chars: int = Field(default=8_000, ge=256, le=100_000)


class RecallEvidence(BaseModel):
    document_id: str
    memory_type: str
    ref_id: str
    title: str
    text: str
    provider: str | None
    project_id: str | None
    session_id: str | None
    role: str | None
    authority: str
    occurred_at: datetime | None
    score: float
    reasons: tuple[str, ...]


class RecallResult(BaseModel):
    request_id: str
    mode: Literal["hybrid", "keyword_only", "empty", "timeout", "shed", "error"]
    evidence: tuple[RecallEvidence, ...]
    elapsed_ms: float
    generation_id: str | None
    abstained: bool
    reason: str | None = None


class EmbeddingManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    revision: str
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dimension: int = Field(gt=0, le=16_384)
    native_dimension: int = Field(gt=0, le=16_384)
    normalized: bool = True
    query_prefix: str = ""
    document_prefix: str = ""
    matryoshka: bool = False

    @field_validator("dimension")
    @classmethod
    def dimension_must_be_supported(cls, value: int, info):
        native = info.data.get("native_dimension")
        # native_dimension may be parsed after dimension, so the model-level invariant
        # is completed in model_post_init below.
        if native is not None and value > native:
            raise ValueError("embedding dimension exceeds the model's native dimension")
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.dimension > self.native_dimension:
            raise ValueError("embedding dimension exceeds the model's native dimension")
        if self.dimension != self.native_dimension and not self.matryoshka:
            raise ValueError("dimension reduction requires an explicitly Matryoshka-trained model")
