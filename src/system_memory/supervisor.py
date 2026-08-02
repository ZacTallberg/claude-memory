from __future__ import annotations

import ctypes
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .clock import utc_iso


class SupervisorError(RuntimeError):
    pass


class AlreadySupervised(SupervisorError):
    pass


class PortConflict(SupervisorError):
    pass


class OwnershipLost(SupervisorError):
    pass


class SupervisorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path
    host: str = "127.0.0.1"
    port: int = Field(default=7788, ge=1024, le=65_535)
    probe_interval_seconds: float = Field(default=2.0, ge=0.1, le=60)
    probe_timeout_seconds: float = Field(default=0.75, ge=0.1, le=10)
    startup_timeout_seconds: float = Field(default=30.0, ge=1, le=300)
    shutdown_grace_seconds: float = Field(default=10.0, ge=0, le=120)
    failed_probe_limit: int = Field(default=3, ge=1, le=30)
    maximum_restart_delay_seconds: float = Field(default=60.0, ge=1, le=600)

    @property
    def run_root(self) -> Path:
        return self.root.resolve() / "run"

    @property
    def state_path(self) -> Path:
        return self.run_root / "supervisor.json"

    @property
    def lock_path(self) -> Path:
        return self.run_root / "supervisor.lock"

    @property
    def log_path(self) -> Path:
        return self.run_root / "system-memory.log"

    @property
    def supervisor_log_path(self) -> Path:
        return self.run_root / "supervisor.jsonl"

    @property
    def token_path(self) -> Path:
        return self.root.resolve() / "data" / "admin.token"


class SupervisorState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    status: Literal["starting", "running", "conflict", "stopped", "fault"]
    supervisor_pid: int
    supervisor_nonce: str
    child_pid: int | None = None
    child_nonce: str | None = None
    child_fingerprint: str | None = None
    build_id: str
    started_at: datetime
    updated_at: datetime
    restart_count: int = 0
    last_reason: str | None = None


class OwnedChild:
    def __init__(
        self,
        *,
        pid: int,
        nonce: str,
        fingerprint: str,
        process: subprocess.Popen[bytes] | None,
    ) -> None:
        self.pid = pid
        self.nonce = nonce
        self.fingerprint = fingerprint
        self.process = process


class SingleInstanceLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: BinaryIO | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise AlreadySupervised("another Supervisor V2 instance owns the lock") from error
        self.handle = handle
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self.handle:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


class Supervisor:
    def __init__(self, config: SupervisorConfig, *, build_id: str) -> None:
        self.config = config
        self.build_id = build_id
        self.supervisor_nonce = secrets.token_urlsafe(24)
        self.started_at = datetime.now().astimezone()
        self.stop_requested = threading.Event()
        self.child: OwnedChild | None = None
        self.restart_count = 0

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    def run(self) -> None:
        self.config.run_root.mkdir(parents=True, exist_ok=True)
        with SingleInstanceLock(self.config.lock_path):
            self._install_signal_handlers()
            try:
                self.child = self._adopt_or_start()
                self._monitor()
            except PortConflict as error:
                self._write_state("conflict", reason=str(error))
                self._log("port_conflict", detail=str(error))
                raise
            except Exception as error:
                self._write_state("fault", reason=type(error).__name__)
                self._log("supervisor_fault", error=type(error).__name__)
                raise
            finally:
                if self.stop_requested.is_set() and self.child:
                    self._terminate_owned(self.child)
                    self.child = None
                if self.stop_requested.is_set():
                    self._write_state("stopped", reason="supervisor stop requested")

    def request_stop(self, *_args) -> None:
        self.stop_requested.set()

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

    def _adopt_or_start(self) -> OwnedChild:
        live = self._live_identity()
        state = self._read_state()
        if live:
            pid, nonce, _ = live
            fingerprint = process_fingerprint(pid)
            if (
                state
                and state.child_pid == pid
                and state.child_nonce == nonce
                and state.child_fingerprint == fingerprint
                and fingerprint
            ):
                child = OwnedChild(pid=pid, nonce=nonce, fingerprint=fingerprint, process=None)
                self.restart_count = state.restart_count
                self._log("child_adopted", child_pid=pid)
                self.child = child
                if live[2] != self.build_id:
                    self._log(
                        "owned_build_replaced",
                        child_pid=pid,
                        previous_build=live[2],
                        next_build=self.build_id,
                    )
                    self._terminate_owned(child)
                    self.restart_count += 1
                    self.child = None
                    return self._start_child()
                self._write_state("running", reason="adopted verified child")
                return child
            raise PortConflict("a live but unowned service is already bound to the configured port")
        if self._port_accepts_connections():
            raise PortConflict("the configured port is occupied by an unidentified service")
        return self._start_child()

    def _start_child(self) -> OwnedChild:
        nonce = secrets.token_urlsafe(32)
        environment = os.environ.copy()
        environment["SYSTEM_MEMORY_INSTANCE_NONCE"] = nonce
        environment["SYSTEM_MEMORY_BUILD_ID"] = self.build_id
        self._rotate_log(self.config.log_path)
        log = self.config.log_path.open("ab", buffering=0)
        command = [
            sys.executable,
            "-m",
            "system_memory",
            "--root",
            str(self.config.root.resolve()),
            "serve",
            "--port",
            str(self.config.port),
        ]
        creation_flags = 0
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                cwd=self.config.root.resolve(),
                creationflags=creation_flags,
            )
        finally:
            log.close()
        fingerprint = process_fingerprint(process.pid)
        if not fingerprint:
            raise SupervisorError("child exited before process ownership could be recorded")
        child = OwnedChild(
            pid=process.pid,
            nonce=nonce,
            fingerprint=fingerprint,
            process=process,
        )
        self.child = child
        self._write_state("starting", reason="spawned exact child")
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while time.monotonic() < deadline and not self.stop_requested.is_set():
            if process.poll() is not None:
                raise SupervisorError(f"child exited during startup with code {process.returncode}")
            if self._identity_is_live(child):
                self._write_state("running", reason="child passed identity liveness")
                self._log("child_started", child_pid=child.pid)
                return child
            self.stop_requested.wait(0.2)
        self._terminate_owned(child)
        raise SupervisorError("child did not pass identity liveness before startup timeout")

    def _monitor(self) -> None:
        failed_probes = 0
        restart_delay = 1.0
        while not self.stop_requested.is_set():
            child = self.child
            if not child:
                child = self._start_child()
                self.child = child
            if process_fingerprint(child.pid) != child.fingerprint:
                if self._port_accepts_connections():
                    raise PortConflict(
                        "owned child exited and another service now occupies the configured port"
                    )
                self.restart_count += 1
                self._write_state("starting", reason="owned child exited")
                self.stop_requested.wait(restart_delay)
                if self.stop_requested.is_set():
                    break
                restart_delay = min(self.config.maximum_restart_delay_seconds, restart_delay * 2)
                self.child = self._start_child()
                failed_probes = 0
                continue
            if self._identity_is_live(child):
                failed_probes = 0
                restart_delay = 1.0
                self._write_state("running", reason="identity liveness verified")
            else:
                failed_probes += 1
                self._log(
                    "liveness_probe_failed",
                    child_pid=child.pid,
                    consecutive_failures=failed_probes,
                )
                if failed_probes >= self.config.failed_probe_limit:
                    self._terminate_owned(child)
                    self.restart_count += 1
                    self.child = None
                    failed_probes = 0
            self.stop_requested.wait(self.config.probe_interval_seconds)

    def _terminate_owned(self, child: OwnedChild) -> None:
        if process_fingerprint(child.pid) != child.fingerprint:
            raise OwnershipLost("refusing to terminate a PID whose creation identity changed")
        live = self._live_identity()
        if live and (live[0] != child.pid or live[1] != child.nonce):
            raise OwnershipLost("refusing to terminate a process with a different live nonce")
        token = self._admin_token()
        if live and token:
            with suppress(httpx.HTTPError):
                httpx.post(
                    f"{self.base_url}/v1/admin/shutdown",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=self.config.probe_timeout_seconds,
                )
        deadline = time.monotonic() + self.config.shutdown_grace_seconds
        while time.monotonic() < deadline:
            if process_fingerprint(child.pid) != child.fingerprint:
                return
            time.sleep(0.1)
        if process_fingerprint(child.pid) != child.fingerprint:
            return
        if child.process and child.process.pid == child.pid:
            child.process.terminate()
        else:
            os.kill(child.pid, signal.SIGTERM)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if process_fingerprint(child.pid) != child.fingerprint:
                return
            time.sleep(0.1)
        if process_fingerprint(child.pid) != child.fingerprint:
            return
        if child.process and child.process.pid == child.pid:
            child.process.kill()
        elif hasattr(signal, "SIGKILL"):
            os.kill(child.pid, signal.SIGKILL)

    def _identity_is_live(self, child: OwnedChild) -> bool:
        live = self._live_identity()
        return bool(live and live[0] == child.pid and live[1] == child.nonce)

    def _live_identity(self) -> tuple[int, str, str] | None:
        try:
            response = httpx.get(
                f"{self.base_url}/livez", timeout=self.config.probe_timeout_seconds
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("ok") is not True:
                return None
            pid = payload.get("pid")
            nonce = payload.get("nonce")
            build = payload.get("build")
            if not isinstance(pid, int) or not isinstance(nonce, str) or not nonce:
                return None
            return pid, nonce, str(build or "unknown")
        except (httpx.HTTPError, ValueError, TypeError):
            return None

    def _port_accepts_connections(self) -> bool:
        try:
            with socket.create_connection(
                (self.config.host, self.config.port),
                timeout=self.config.probe_timeout_seconds,
            ):
                return True
        except OSError:
            return False

    def _admin_token(self) -> str | None:
        try:
            token = self.config.token_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return token or None

    def _read_state(self) -> SupervisorState | None:
        try:
            return SupervisorState.model_validate_json(
                self.config.state_path.read_text(encoding="utf-8")
            )
        except Exception:
            return None

    def _write_state(self, status: str, *, reason: str) -> None:
        child = self.child
        state = SupervisorState(
            status=status,
            supervisor_pid=os.getpid(),
            supervisor_nonce=self.supervisor_nonce,
            child_pid=child.pid if child else None,
            child_nonce=child.nonce if child else None,
            child_fingerprint=child.fingerprint if child else None,
            build_id=self.build_id,
            started_at=self.started_at,
            updated_at=datetime.now().astimezone(),
            restart_count=self.restart_count,
            last_reason=reason,
        )
        self._atomic_write(self.config.state_path, state.model_dump_json(indent=2) + "\n")

    def _log(self, event: str, **fields) -> None:
        self._rotate_log(self.config.supervisor_log_path)
        record = {
            "at": utc_iso(),
            "event": event,
            "supervisor_pid": os.getpid(),
            **fields,
        }
        with self.config.supervisor_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
        try:
            with temp.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()

    @staticmethod
    def _rotate_log(path: Path, *, maximum_bytes: int = 5_000_000, retained: int = 4) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size < maximum_bytes:
            return
        oldest = path.with_suffix(path.suffix + f".{retained}")
        if oldest.exists():
            oldest.unlink()
        for index in range(retained - 1, 0, -1):
            source = path.with_suffix(path.suffix + f".{index}")
            if source.exists():
                os.replace(source, path.with_suffix(path.suffix + f".{index + 1}"))
        os.replace(path, path.with_suffix(path.suffix + ".1"))


def process_fingerprint(pid: int) -> str | None:
    if pid <= 0:
        return None
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return None
        try:
            creation = ctypes.c_ulonglong()
            exit_time = ctypes.c_ulonglong()
            kernel = ctypes.c_ulonglong()
            user = ctypes.c_ulonglong()
            ok = ctypes.windll.kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            )
            return f"windows-filetime:{creation.value}" if ok else None
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            fields = proc_stat.read_text(encoding="utf-8").split()
            return f"proc-start:{fields[21]}"
        except (OSError, IndexError):
            return None
    return None
