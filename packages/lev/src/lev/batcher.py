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

from .model import MAX_BATCH_TOKENS, DecisionEngine, Prepared
from .types import SystemOneResponse


class Overloaded(Exception):
    """`max_pending` requests are already in flight."""


class ClientGone(Exception):
    """The client disconnected before its answer was ready."""


class WorkerStopped(RuntimeError):
    """The GPU worker thread has exited; the server needs a restart."""


@dataclass
class _Job:
    prepared: Prepared
    future: Future
    deadline: float


class Batcher:
    def __init__(
        self,
        engine: DecisionEngine,
        max_pending: int = 64,
        max_batch_tokens: int = MAX_BATCH_TOKENS,
        timeout: float = 30.0,
    ):
        self.engine = engine
        self.max_pending = max_pending
        # Padded tokens per forward (rows x widest row); bounds activation memory.
        # A request larger than the budget still runs, alone.
        self.max_batch_tokens = max_batch_tokens
        self.timeout = timeout
        # A request counts until the worker finishes or skips it, not until its
        # client leaves: abandoned work still occupies the GPU.
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._queue: queue.SimpleQueue[_Job] = queue.SimpleQueue()
        self._held: _Job | None = None
        self._batch: list[_Job] = []
        self._worker = threading.Thread(target=self._run, name="lev-gpu", daemon=True)
        self._worker.start()

    @property
    def pending(self) -> int:
        with self._pending_lock:
            return self._pending

    @property
    def alive(self) -> bool:
        return self._worker.is_alive()

    async def submit(
        self, prepared: Prepared, disconnected: Callable[[], Awaitable] | None = None
    ) -> SystemOneResponse:
        """Queue one request and wait for its answer. `disconnected` makes an
        awaitable that completes if the client leaves; it is called only once
        the request is admitted. If it raises instead, disconnects go undetected
        and the answer is awaited alone."""
        if not self.alive:
            raise WorkerStopped("the GPU worker has stopped; restart the server")
        with self._pending_lock:
            if self._pending >= self.max_pending:
                raise Overloaded(f"{self._pending} requests in flight; retry shortly")
            self._pending += 1
        job = _Job(prepared, Future(), time.monotonic() + self.timeout)
        job.future.add_done_callback(self._release)
        self._queue.put(job)
        # Cancelling this wrapper cancels `job.future` unless the worker has
        # already started it.
        result = asyncio.wrap_future(job.future)
        if disconnected is None:
            return await result
        watcher = asyncio.ensure_future(disconnected())
        try:
            await asyncio.wait({result, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if not result.done() and watcher.exception() is not None:
                return await result
        finally:
            if watcher.done() and not watcher.cancelled():
                watcher.exception()  # retrieved, so asyncio does not log it as lost
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
                except Exception as error:  # noqa: BLE001 -- the worker must outlive any one batch
                    traceback.print_exc()
                    _fail([job.future for job in self._batch], error)
        finally:
            self._fail_queued()

    def _compute(self, batch: list[_Job]) -> None:
        try:
            responses = self.engine.answer([job.prepared for job in batch])
        except Exception as error:  # noqa: BLE001 -- a failed forward fails requests, not the worker
            if len(batch) == 1:
                batch[0].future.set_exception(error)
                return
            # A batch can fail where its requests alone would not (out of
            # memory), and one bad request must not fail the others.
            for job in batch:
                self._compute([job])
            return
        for job, response in zip(batch, responses, strict=True):
            job.future.set_result(response)

    def _fill(self, batch: list[_Job]) -> None:
        # Marked running only on joining, so a request held for the next batch
        # can still be cancelled or time out.
        rows = width = 0
        while (job := self._take(block=not batch)) is not None:
            grown_rows, grown_width = rows + job.prepared.rows, max(width, job.prepared.width)
            if batch and grown_rows * grown_width > self.max_batch_tokens:
                self._held = job
                return
            if not job.future.set_running_or_notify_cancel():
                continue  # the client left between `_take` and here
            batch.append(job)
            rows, width = grown_rows, grown_width

    def _take(self, block: bool) -> _Job | None:
        """The next request still worth computing, not yet marked running."""
        while True:
            if self._held is not None:
                job, self._held = self._held, None
            else:
                try:
                    job = self._queue.get(block=block)
                except queue.Empty:
                    return None
            if job.future.cancelled():
                continue  # the client left while it waited
            if time.monotonic() > job.deadline:
                if job.future.set_running_or_notify_cancel():
                    job.future.set_exception(
                        TimeoutError(f"queued longer than the {self.timeout:.0f}s deadline")
                    )
                continue
            return job

    def _fail_queued(self) -> None:
        """On worker exit, fail the running batch and everything still waiting,
        rather than leave them hanging."""
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
