"""Fleet auto-enrollment carrier (SessionStart) — isolated, stdlib-only, fail-safe.

claude-memory is the one thing every Vigor fleet machine installs (fleet ADR-0004), so its
harness is the machine-wide carrier for fleet-hub enrollment: if this machine has no fleet
config, fetch the enrollment brain from the hub's open bootstrap endpoint and run it inline
(its stdout lands in the session so the agent sees and reports what happened). The brain
itself is idempotent and allowlist-gated server-side; every mint is a visible board event.

Deliberately imports NOTHING from claudemem — a broken or absent memory venv can never
break enrollment, and vice versa. Exit 0 on every path (the claude-memory hook hard rule).
Kill switches honored: ~/.fleet-hub/DISABLED and the FLEET_AUTOENROLL=0 env.
"""
from __future__ import annotations

import os
import ssl
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HUB_URL = os.environ.get("FLEET_HUB_URL", "https://gitlab.vigor.local/apps/fleet-hub").rstrip("/")
FLEET_DIR = Path.home() / ".fleet-hub"


def main() -> int:
    try:
        if os.environ.get("FLEET_AUTOENROLL") == "0":
            return 0
        if (FLEET_DIR / "DISABLED").exists() or (FLEET_DIR / "config.json").exists():
            return 0
        req = urllib.request.Request(f"{HUB_URL}/enroll/bootstrap")
        with urllib.request.urlopen(req, timeout=8,
                                    context=ssl._create_unverified_context()) as r:
            brain = r.read()
        with tempfile.NamedTemporaryFile("wb", suffix="_bootstrap_enroll.py",
                                         delete=False) as f:
            f.write(brain)
            path = f.name
        try:
            proc = subprocess.run([sys.executable, path], capture_output=True,
                                  text=True, timeout=75)
            if proc.stdout:
                print(proc.stdout.strip())
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except Exception:
        pass  # never break a session; the next SessionStart retries
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
