"""SessionEnd / PreCompact hook — spawn a detached background incremental index. Never blocks."""
from __future__ import annotations

import os
import subprocess
import sys

from _common import call_server, read_event, run_failsafe


def main() -> None:
    ev = read_event()
    from claudemem.config import ROOT, load_config
    from claudemem.paths import in_scope, killed

    cfg = load_config()
    if killed():
        return
    if not in_scope(ev.get("cwd"), cfg):
        return

    # Keep indexing inside the singleton warm process so a session ending cannot load a second
    # ONNX model and contend with prompt recall. The endpoint is non-blocking and deduplicates
    # overlapping requests. Retain the detached CLI path only as disaster recovery.
    warm = call_server("/api/index", {"full": False, "promote": True,
                                      "reason": ev.get("hook_event_name") or "lifecycle"},
                       timeout=2.0)
    if warm is not None:
        return

    flags = 0
    if os.name == "nt":
        flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [sys.executable, "-m", "claudemem", "index", "--promote"],
        cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True)


if __name__ == "__main__":
    run_failsafe(main)
