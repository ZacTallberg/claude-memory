"""Anthropic Contextual Retrieval: generate a short situating context per chunk before
indexing (improves recall). Off the hot path, prompt-cached per document, fully optional.
"""
from __future__ import annotations

from ..config import Config
from ..log import get_logger

log = get_logger(__name__)

_SYSTEM = (
    "You situate a chunk of text within its source document for a search index. "
    "Reply with ONLY a terse 1-2 sentence context (no preamble), naming the document/topic, "
    "project, and what the chunk is about, so the chunk is findable by keyword and meaning."
)


class Contextualizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = None
        if cfg.contextual.enabled and cfg.anthropic_api_key:
            try:
                import anthropic  # lazy
                self._client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
            except Exception as e:
                log.warning("contextual retrieval unavailable: %s", e)
                self._client = None

    def available(self) -> bool:
        return self._client is not None

    def contextualize(self, document: str, chunk: str) -> str | None:
        """Return a short context blurb, or None if unavailable/failed. The `document` block
        is marked cache-able so all chunks of the same doc reuse the cached prefix."""
        if not self.available():
            return None
        doc = (document or "")[: self.cfg.contextual.max_doc_chars]
        try:
            msg = self._client.messages.create(
                model=self.cfg.contextual.model,
                max_tokens=120,
                system=_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"<document>\n{doc}\n</document>",
                         "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text":
                            f"<chunk>\n{chunk}\n</chunk>\n"
                            "Give the 1-2 sentence situating context for this chunk."},
                    ],
                }],
            )
            parts = [b.text for b in msg.content if getattr(b, "type", "") == "text"]
            out = " ".join(parts).strip()
            return out or None
        except Exception as e:
            log.warning("contextualize failed: %s", e)
            return None
