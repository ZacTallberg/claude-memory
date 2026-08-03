# Transcript adapters

Built-in Claude Code and Codex adapters normalize their transcripts into one client-neutral corpus.
Additional local runtimes can be added without editing the indexer by registering a Python entry
point in the `claudemem.transcript_adapters` group.

An adapter implements this deliberately small protocol:

```python
class TranscriptAdapter(Protocol):
    name: str

    def discover(self, cfg: Config) -> list[TranscriptFile]: ...
    def parse_new(self, path: Path, start_byte: int, cfg: Config): ...
```

The package exposing an adapter registers it in `pyproject.toml`:

```toml
[project.entry-points."claudemem.transcript_adapters"]
my_agent = "my_agent_memory:Adapter"
```

Then add the adapter name to `[index].transcript_providers` and run `mem adapters` followed by
`mem index`. Built-in names cannot be replaced. Malformed discoveries and parse failures are logged
and skipped instead of blocking other sources.

Extension code is discovered and executed only by the background indexer. Prompt hooks and the warm
retrieval path never import third-party adapter code, preserving the latency and trust boundaries.
The adapter must still return the core `TranscriptFile` and parsed-record shapes; that stable
normalization boundary keeps storage, retrieval, evaluation, and lifecycle logic independent of any
one agent vendor.
