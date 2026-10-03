"""Cross-request batching and admission control for the HTTP server.

One worker thread owns the GPU. Requests queue while a forward runs, and the
next forward takes every queued request that fits the token budget, so batches
grow with load and an idle server adds no wait. Past `max_pending` requests in
flight, a new one is refused at once instead of queueing behind work whose
clients have already given up; a client that disconnects, or a request that
waited past its deadline, is dropped before it reaches the GPU.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from dataclasses import dataclass

from .model import DecisionEngine, Prepared
from .types import SystemOneResponse


class Overloaded(Exception):
    """`max_pending` requests are already in flight."""


class ClientGone(Exception):
    """The client disconnected before its answer was ready."""


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
        max_batch_tokens: int = 16384,
        timeout: float = 30.0,
    ):
        self.engine = engine
        self.max_pending = max_pending
        # Padded tokens per forward (rows x widest row); bounds activation memory.
        # A request larger than the budget still runs, alone.
        self.max_batch_tokens = max_batch_tokens
        self.timeout = timeout
        # Read and written only on the event loop, so it needs no lock.
        self.pending = 0
        self._queue: queue.SimpleQueue[_Job] = queue.SimpleQueue()
        self._held: _Job | None = None
        threading.Thread(target=self._run, name="lev-gpu", daemon=True).start()

    async def submit(
        self, prepared: Prepared, disconnected: Callable[[], Awaitable] | None = None
    ) -> SystemOneResponse:
        """Queue one request and wait for its answer. `disconnected` makes an
        awaitable that completes if the client leaves; it is called only once
        the request is admitted."""
        if self.pending >= self.max_pending:
            raise Overloaded(f"{self.pending} requests in flight; retry shortly")
        job = _Job(prepared, Future(), time.monotonic() + self.timeout)
        self.pending += 1
        self._queue.put(job)
        # Cancelling this wrapper cancels `job.future`, which the worker skips.
        result = asyncio.wrap_future(job.future)
        try:
            if disconnected is None:
                return await result
            watcher = asyncio.ensure_future(disconnected())
            try:
                await asyncio.wait({result, watcher}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                watcher.cancel()
            if not result.done():
                result.cancel()
                raise ClientGone
            return result.result()
        finally:
            self.pending -= 1

    def _run(self) -> None:
        while True:
            self._compute(self._next_batch())

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

    def _next_batch(self) -> list[_Job]:
        """Block for one live request, then take every queued one that fits."""
        first = self._take(block=True)
        assert first is not None
        batch, rows, width = [first], first.prepared.rows, first.prepared.width
        while (job := self._take(block=False)) is not None:
            grown_rows, grown_width = rows + job.prepared.rows, max(width, job.prepared.width)
            if grown_rows * grown_width > self.max_batch_tokens:
                self._held = job
                break
            batch.append(job)
            rows, width = grown_rows, grown_width
        return batch

    def _take(self, block: bool) -> _Job | None:
        """The next request still worth computing, marked running."""
        while True:
            if self._held is not None:
                job, self._held = self._held, None
                return job
            try:
                job = self._queue.get(block=block)
            except queue.Empty:
                return None
            if not job.future.set_running_or_notify_cancel():
                continue  # the client left while it was queued
            if time.monotonic() > job.deadline:
                job.future.set_exception(
                    TimeoutError(f"queued longer than the {self.timeout:.0f}s deadline")
                )
                continue
            return job
