"""Serve typed decisions and health; return 422, 529, 503, or 504 for request failures."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Literal

# Module level: FastAPI resolves the endpoint's string annotations here.
from fastapi import Request

from .batcher import Batcher, ClientGone, Overloaded, WorkerStopped
from .model import MAX_BATCH_TOKENS, load
from .types import SystemOneRequest, SystemOneResponse


def create_app(
    checkpoint_dir: str | None = None,
    model_cache: str | None = None,
    calibration: str | None = None,
    model_id: str = "Qwen/Qwen3.5-4B-Base",
    noul_readout: Literal["rating", "binary"] | None = None,
    compile: bool = False,
    max_label_options: int | None = None,
    prompt_style: Literal["plain", "chat"] | None = None,
    skip_multi_token_codes: bool = True,
    max_pending: int = 64,
    max_batch_tokens: int = MAX_BATCH_TOKENS,
    timeout: float = 30.0,
    score_order_average: Literal["off", "reversed", "cyclic"] = "off",
) -> Any:
    from fastapi import FastAPI, HTTPException, Response
    from starlette.concurrency import run_in_threadpool

    from . import __version__

    state: dict[str, Any] = {"engine": None, "batcher": None}

    @asynccontextmanager
    async def lifespan(app):
        engine = load(
            checkpoint_dir,
            model_id=model_id,
            cache_dir=model_cache,
            calibration=calibration,
            noul_readout=noul_readout,
            prompt_style=prompt_style,
            compile=compile,
            max_label_options=max_label_options,
            skip_multi_token_codes=skip_multi_token_codes,
            score_order_average=score_order_average,
            max_batch_tokens=max_batch_tokens,
        )
        state["batcher"] = Batcher(
            engine, max_pending=max_pending, max_batch_tokens=max_batch_tokens, timeout=timeout
        )
        state["engine"] = engine
        yield

    app = FastAPI(title="lev", version=__version__, lifespan=lifespan)

    # Async, so a saturated thread pool cannot starve it.
    @app.get("/health")
    async def health() -> dict:
        engine = state["engine"]
        if engine is None:
            return {"status": "loading", "model": model_id}
        config = engine.config
        return {
            "status": "ok" if state["batcher"].alive else "worker stopped",
            "model": config.model_id,
            "checkpoint": str(engine.checkpoint) if engine.checkpoint else None,
            "calibrated": bool(engine.calibration.temperatures),
            "mode_b": engine.mode_b_head is not None,
            "noul_readout": config.noul_readout,
            "max_label_options": config.max_label_options,
            "order_average": config.order_average,
            "score_order_average": config.score_order_average,
            "max_batch_tokens": config.max_batch_tokens,
            "prefix_mode": config.prefix_mode,
            "compiled": config.compile,
            "prompt_style": config.prompt_style,
            "skip_multi_token_codes": config.skip_multi_token_codes,
            "in_flight": state["batcher"].pending,
            "max_pending": max_pending,
        }

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def system_one(body: SystemOneRequest, request: Request) -> Any:
        if state["engine"] is None:
            raise HTTPException(status_code=529, detail="model still loading")
        if not body.questions:
            raise HTTPException(status_code=422, detail="at least one question required")
        try:
            prepared = await run_in_threadpool(state["engine"].prepare, body.state, body.questions)
            return await state["batcher"].submit(prepared, lambda: _disconnected(request))
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Overloaded as exc:
            raise HTTPException(
                status_code=503, detail=str(exc), headers={"Retry-After": "1"}
            ) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except WorkerStopped as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ClientGone:
            return Response(status_code=499)  # nobody is listening; for the access log

    return app


async def _disconnected(request: Request) -> None:
    """Wait for an ASGI disconnect; if none is reported, wait without polling."""
    if (await request.receive())["type"] == "http.disconnect":
        return
    await asyncio.Event().wait()
