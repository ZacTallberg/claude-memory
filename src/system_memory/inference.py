from __future__ import annotations

import itertools
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, TypeVar

T = TypeVar("T")


class WorkKind(IntEnum):
    QUERY_EMBEDDING = 0
    INTERACTIVE_RERANK = 10
    CONTEXT_COMPILATION = 20
    DOCUMENT_EMBEDDING = 100


class InferenceShed(RuntimeError):
    pass


class InferenceTimeout(RuntimeError):
    pass


@dataclass(order=True)
class _Work[T]:
    priority: int
    sequence: int
    function: Callable[[], T] | None = field(compare=False)
    future: Future[T] | None = field(compare=False)


class InferenceScheduler:
    """One priority-owned inference lane with bounded outstanding work.

    A caller timing out never releases admission early. The running operation continues
    to occupy its slot until the model exits, preventing abandoned work from causing the
    exact oversubscription spiral observed in v1.
    """

    def __init__(self, *, capacity: int = 16, name: str = "system-memory-inference") -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._queue: queue.PriorityQueue[_Work[Any]] = queue.PriorityQueue()
        self._admission = threading.BoundedSemaphore(capacity)
        self._sequence = itertools.count()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, kind: WorkKind, function: Callable[[], T], *, timeout: float) -> T:
        if self._closed.is_set():
            raise InferenceShed("inference scheduler is closed")
        if not self._admission.acquire(blocking=False):
            raise InferenceShed("inference admission capacity is full")
        future: Future[T] = Future()
        work = _Work(int(kind), next(self._sequence), function, future)
        try:
            self._queue.put_nowait(work)
        except Exception:
            self._admission.release()
            raise
        try:
            return future.result(timeout=timeout)
        except TimeoutError as error:
            raise InferenceTimeout(f"inference exceeded {timeout:.3f}s") from error

    def _run(self) -> None:
        while True:
            work = self._queue.get()
            if work.function is None or work.future is None:
                self._queue.task_done()
                return
            try:
                if work.future.set_running_or_notify_cancel():
                    try:
                        work.future.set_result(work.function())
                    except BaseException as error:
                        work.future.set_exception(error)
            finally:
                self._admission.release()
                self._queue.task_done()

    def close(self, *, wait: bool = True) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(_Work(1_000_000, next(self._sequence), None, None))
        if wait:
            self._thread.join(timeout=30)

    @property
    def queued(self) -> int:
        return self._queue.qsize()
