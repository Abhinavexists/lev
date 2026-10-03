"""Cross-request batching and admission control, with a fake engine and no GPU."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from lev.batcher import Batcher, ClientGone, Overloaded


def request(name: str, rows: int = 1, width: int = 10):
    return SimpleNamespace(name=name, rows=rows, width=width)


class GatedEngine:
    """Records each forward's requests; the first forward waits for `release`,
    so requests submitted meanwhile queue up behind it."""

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
    """Occupy the worker with `first`, queue `queued`, then let it run."""
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
