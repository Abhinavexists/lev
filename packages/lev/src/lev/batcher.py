"""Cross-request batching and admission control for the HTTP server (ADR-030).

One worker thread owns the GPU; each forward takes every queued request that fits.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
import traceback
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Protocol

from .model import MAX_BATCH_TOKENS, Prepared
from .types import SystemOneResponse


class AnswerEngine(Protocol):
    def answer(self, requests: list[Prepared]) -> list[SystemOneResponse]: ...


class Overloaded(Exception):
    pass


class ClientGone(Exception):
    pass


class WorkerStopped(RuntimeError):
    pass


@dataclass
class _Job:
    prepared: Prepared
    future: Future
    deadline: float


class Batcher:
    def __init__(
        self,
        engine: AnswerEngine,
        max_pending: int = 64,
        max_batch_tokens: int = MAX_BATCH_TOKENS,
        timeout: float = 30.0,
    ):
        self.engine = engine
        self.max_pending = max_pending
        self.max_batch_tokens = max_batch_tokens  # Bounds activation memory per forward pass.
        self.timeout = timeout

        self._pending = 0  # Counts work until the worker finishes, even if the client leaves.
        self._pending_lock = threading.Lock()
        self._queue: queue.SimpleQueue[_Job] = queue.SimpleQueue()
        self._held: _Job | None = None
        self._batch: list[_Job] = []

        self._worker = threading.Thread(
            target=self._run,
            name="lev-gpu",
            daemon=True,
        )
        self._worker.start()

    @property
    def pending(self) -> int:
        with self._pending_lock:
            return self._pending

    @property
    def alive(self) -> bool:
        return self._worker.is_alive()

    async def submit(
        self,
        prepared: Prepared,
        disconnected: Callable[[], Awaitable] | None = None,
    ) -> SystemOneResponse:

        if not self.alive:
            raise WorkerStopped("the GPU worker has stopped; restart the server")

        with self._pending_lock:
            if self._pending >= self.max_pending:
                raise Overloaded(f"{self._pending} requests in flight; retry shortly")
            self._pending += 1

        job = _Job(prepared, Future(), time.monotonic() + self.timeout)
        job.future.add_done_callback(self._release)
        self._queue.put(job)

        result = asyncio.wrap_future(job.future)  # Cancelling may cancel the queued job.

        if disconnected is None:
            return await result

        watcher = asyncio.ensure_future(disconnected())

        try:
            await asyncio.wait(
                {result, watcher},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not result.done() and watcher.exception() is not None:
                return await result
        finally:
            if watcher.done() and not watcher.cancelled():
                watcher.exception()  # Mark the exception as retrieved.
            watcher.cancel()

        if not result.done():
            result.cancel()
            raise ClientGone

        return result.result()

    def _release(self, _future: Future) -> None:
        with self._pending_lock:
            self._pending -= 1

    def _run(self) -> None:
        try:
            while True:
                self._batch = []

                try:
                    self._fill(self._batch)
                    self._compute(self._batch)
                except Exception as error:  # noqa: BLE001 -- worker must outlive one batch
                    traceback.print_exc()
                    _fail([job.future for job in self._batch], error)
        finally:
            self._fail_queued()

    def _compute(self, batch: list[_Job]) -> None:
        try:
            responses = self.engine.answer([job.prepared for job in batch])
        except Exception as error:  # noqa: BLE001 -- failed forward should not kill worker
            if len(batch) == 1:
                batch[0].future.set_exception(error)
                return

            # A batch may fail where its requests alone would not.
            for job in batch:
                self._compute([job])
            return

        for job, response in zip(batch, responses, strict=True):
            job.future.set_result(response)

    def _fill(self, batch: list[_Job]) -> None:
        # Held jobs remain cancellable until they actually join a batch.
        rows = width = 0

        while (job := self._take(block=not batch)) is not None:
            grown_rows = rows + job.prepared.rows
            grown_width = max(width, job.prepared.width)

            if batch and grown_rows * grown_width > self.max_batch_tokens:
                self._held = job
                return

            if not job.future.set_running_or_notify_cancel():
                continue  # Client disconnected while queued.

            batch.append(job)
            rows, width = grown_rows, grown_width

    def _take(self, block: bool) -> _Job | None:
        """Return the next request worth computing."""

        while True:
            if self._held is not None:
                job, self._held = self._held, None
            else:
                try:
                    job = self._queue.get(block=block)
                except queue.Empty:
                    return None

            if job.future.cancelled():
                continue

            if time.monotonic() > job.deadline:
                if job.future.set_running_or_notify_cancel():
                    job.future.set_exception(
                        TimeoutError(f"queued longer than the {self.timeout:.0f}s deadline")
                    )
                continue

            return job

    def _fail_queued(self) -> None:

        stopped = WorkerStopped("the GPU worker has stopped; restart the server")
        _fail([job.future for job in self._batch], stopped)

        while (job := self._held or self._next_queued()) is not None:
            self._held = None

            if job.future.set_running_or_notify_cancel():
                job.future.set_exception(stopped)

    def _next_queued(self) -> _Job | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


def _fail(futures: list[Future], error: BaseException) -> None:
    for future in futures:
        if not future.done():
            future.set_exception(error)
