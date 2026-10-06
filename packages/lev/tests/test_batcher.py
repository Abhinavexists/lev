"""Cross-request batching and admission control, with a fake engine and no GPU."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from lev.batcher import Batcher, ClientGone, Overloaded, WorkerStopped


def request(name: str, rows: int = 1, width: int = 10):
    return SimpleNamespace(name=name, rows=rows, width=width)


class GatedEngine:
    """Hold the first forward so later requests queue behind it."""

    def __init__(self):
        self.forwards: list[list[str]] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def answer(self, requests):
        self.started.set()
        self.release.wait(timeout=5)
        names = [r.name for r in requests]
        self.forwards.append(names)
        if "boom" in names:
            raise RuntimeError("CUDA out of memory")
        return [f"answer to {n}" for n in names]


async def submit_while_busy(batcher, engine, first, queued):
    """Occupy the worker with `first` and queue `queued` behind it."""
    head = asyncio.ensure_future(batcher.submit(first))
    while not engine.started.is_set():
        await asyncio.sleep(0.001)
    tasks = [asyncio.ensure_future(batcher.submit(r)) for r in queued]
    await asyncio.sleep(0.01)
    return head, tasks


def test_requests_queued_behind_a_forward_share_the_next_one():
    async def run():
        engine = GatedEngine()
        batcher = Batcher(engine)
        head, tasks = await submit_while_busy(
            batcher, engine, request("a"), [request("b"), request("c"), request("d")]
        )
        engine.release.set()
        return engine, await head, await asyncio.gather(*tasks)

    engine, first, rest = asyncio.run(run())
    assert engine.forwards == [["a"], ["b", "c", "d"]]
    assert first == "answer to a"
    assert rest == ["answer to b", "answer to c", "answer to d"], "each caller gets its own row"


def test_the_token_budget_splits_a_batch():
    async def run():
        engine = GatedEngine()
        # 3 rows x width 10 fits; a fourth row would not.
        batcher = Batcher(engine, max_batch_tokens=30)
        head, tasks = await submit_while_busy(
            batcher, engine, request("a"), [request(n) for n in "bcde"]
        )
        engine.release.set()
        await head
        await asyncio.gather(*tasks)
        return engine

    assert asyncio.run(run()).forwards == [["a"], ["b", "c", "d"], ["e"]]


def test_past_max_pending_a_request_is_refused_at_once():
    async def run():
        engine = GatedEngine()
        batcher = Batcher(engine, max_pending=2)
        head, tasks = await submit_while_busy(batcher, engine, request("a"), [request("b")])
        with pytest.raises(Overloaded):
            await batcher.submit(request("c"))
        engine.release.set()
        await head
        await asyncio.gather(*tasks)
        return engine, batcher

    engine, batcher = asyncio.run(run())
    assert engine.forwards == [["a"], ["b"]], "the refused request never reached the GPU"
    assert batcher.pending == 0


def test_a_client_that_disconnects_while_queued_is_never_computed():
    async def run():
        engine = GatedEngine()
        batcher = Batcher(engine)
        head = asyncio.ensure_future(batcher.submit(request("a")))
        while not engine.started.is_set():
            await asyncio.sleep(0.001)
        gone = asyncio.Event()
        leaving = asyncio.ensure_future(batcher.submit(request("b"), gone.wait))
        staying = asyncio.ensure_future(batcher.submit(request("c")))
        await asyncio.sleep(0.01)
        gone.set()
        with pytest.raises(ClientGone):
            await leaving
        engine.release.set()
        await head
        await staying
        return engine

    assert asyncio.run(run()).forwards == [["a"], ["c"]]


def test_a_request_queued_past_its_deadline_times_out_without_a_forward():
    async def run():
        engine = GatedEngine()
        batcher = Batcher(engine, timeout=0.05)
        head, tasks = await submit_while_busy(batcher, engine, request("a"), [request("b")])
        await asyncio.sleep(0.1)  # "b" waits behind "a" past its deadline
        engine.release.set()
        await head
        with pytest.raises(TimeoutError):
            await tasks[0]
        return engine

    assert asyncio.run(run()).forwards == [["a"]]


def test_a_failed_batch_is_retried_one_request_at_a_time():
    async def run():
        engine = GatedEngine()
        batcher = Batcher(engine)
        head, tasks = await submit_while_busy(
            batcher, engine, request("a"), [request("boom"), request("b")]
        )
        engine.release.set()
        await head
        results = await asyncio.gather(*tasks, return_exceptions=True)
        after = await batcher.submit(request("c"))
        return engine, results, after

    engine, results, after = asyncio.run(run())
    assert engine.forwards[1:4] == [["boom", "b"], ["boom"], ["b"]]
    assert isinstance(results[0], RuntimeError), "only the failing request fails"
    assert results[1] == "answer to b"
    assert after == "answer to c", "the worker survives a failed forward"


class SteppedEngine:
    """Every forward waits for one `step()`, so a test controls when each runs."""

    def __init__(self, fail_with: BaseException | None = None, short_by: int = 0):
        self.forwards: list[list[str]] = []
        self.calls = 0
        self.permits = threading.Semaphore(0)
        self.fail_with = fail_with
        self.short_by = short_by

    def step(self):
        self.permits.release()

    def answer(self, requests):
        self.calls += 1
        self.permits.acquire(timeout=5)
        names = [r.name for r in requests]
        self.forwards.append(names)
        if self.fail_with is not None:
            error, self.fail_with = self.fail_with, None
            raise error
        answers = [f"answer to {n}" for n in names]
        if self.short_by:
            answers, self.short_by = answers[: -self.short_by], 0
        return answers


async def until(condition, timeout=2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        assert loop.time() < deadline, "condition never became true"
        await asyncio.sleep(0.001)


def test_a_held_request_can_still_be_cancelled_before_it_runs():

    async def run():
        engine = SteppedEngine()
        batcher = Batcher(engine, max_batch_tokens=10)
        a = asyncio.ensure_future(batcher.submit(request("a")))
        await until(lambda: engine.calls == 1)
        gone = asyncio.Event()
        b = asyncio.ensure_future(batcher.submit(request("b")))
        c = asyncio.ensure_future(batcher.submit(request("c"), gone.wait))
        await asyncio.sleep(0.01)
        engine.step()
        await until(lambda: engine.calls == 2)  # `b` running, `c` held
        gone.set()
        with pytest.raises(ClientGone):
            await c
        engine.step()
        engine.step()  # a permit for `c` too, so running it would show up
        await asyncio.gather(a, b)
        await asyncio.sleep(0.05)
        return engine

    assert asyncio.run(run()).forwards == [["a"], ["b"]]


def test_a_held_request_past_its_deadline_times_out_without_running():
    async def run():
        engine = SteppedEngine()
        batcher = Batcher(engine, max_batch_tokens=10, timeout=0.1)
        a = asyncio.ensure_future(batcher.submit(request("a")))
        await until(lambda: engine.calls == 1)
        b = asyncio.ensure_future(batcher.submit(request("b")))
        c = asyncio.ensure_future(batcher.submit(request("c")))
        engine.step()
        await until(lambda: engine.calls == 2)  # `b` running, `c` held
        await asyncio.sleep(0.15)  # past `c`'s deadline, while it is held
        engine.step()
        await asyncio.gather(a, b)
        with pytest.raises(TimeoutError):
            await c
        return engine

    assert asyncio.run(run()).forwards == [["a"], ["b"]]


def test_an_abandoned_running_request_still_counts_as_pending():
    """Disconnecting does not release GPU capacity already in use."""

    async def run():
        engine = SteppedEngine()
        batcher = Batcher(engine)
        gone = asyncio.Event()
        a = asyncio.ensure_future(batcher.submit(request("a"), gone.wait))
        await until(lambda: engine.calls == 1)
        gone.set()
        with pytest.raises(ClientGone):
            await a
        while_running = batcher.pending
        engine.step()
        await until(lambda: batcher.pending == 0)
        return while_running

    assert asyncio.run(run()) == 1


def test_a_failing_disconnect_watch_still_returns_the_answer():

    async def broken_watch():
        raise RuntimeError("receive() is not available")

    async def run():
        engine = SteppedEngine()
        batcher = Batcher(engine)
        answer = asyncio.ensure_future(batcher.submit(request("a"), broken_watch))
        await until(lambda: engine.calls == 1)
        engine.step()
        return await answer

    assert asyncio.run(run()) == "answer to a"


def test_an_error_outside_the_forward_fails_its_batch_and_the_worker_keeps_serving():
    """A short response list triggers zip(strict=True) outside forward error handling."""

    async def run():
        engine = SteppedEngine(short_by=1)
        batcher = Batcher(engine)
        first = asyncio.ensure_future(batcher.submit(request("a")))
        engine.step()
        with pytest.raises(ValueError):
            await first
        second = asyncio.ensure_future(batcher.submit(request("b")))
        engine.step()
        return batcher, await second

    batcher, after = asyncio.run(run())
    assert after == "answer to b"
    assert batcher.alive


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_stopped_worker_fails_its_requests_and_refuses_new_ones():
    async def run():
        engine = SteppedEngine(fail_with=SystemExit("worker killed"))
        batcher = Batcher(engine)
        running = asyncio.ensure_future(batcher.submit(request("a")))
        await until(lambda: engine.calls == 1)
        queued = asyncio.ensure_future(batcher.submit(request("b")))
        await asyncio.sleep(0.01)
        engine.step()
        results = await asyncio.gather(running, queued, return_exceptions=True)
        await until(lambda: not batcher.alive)
        with pytest.raises(WorkerStopped):
            await batcher.submit(request("c"))
        return results, batcher

    results, batcher = asyncio.run(run())
    assert all(isinstance(r, WorkerStopped) for r in results), results
    assert batcher.pending == 0
