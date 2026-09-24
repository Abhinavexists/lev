"""FastAPI server for `/v1/systemone`.

Wire-identical to TypeSafe's endpoint, so the benchmark measures us and Jev with
the same code path and one changed flag:

    levbench eval --backend jev --base-url http://localhost:8000

Error codes mirror the real API (422 validation, 429 rate limit, 529 overloaded)
so client retry logic behaves identically against either.

Serves the base backbone, or a trained checkpoint when `--checkpoint` names one:
the LoRA adapter, the Mode B head and any `calibration.json` beside them are all
picked up. Start it with `lev serve` and read `/health` before trusting a number
off it -- that is where you see which checkpoint resolved and whether a
calibration profile is actually in effect.
"""

from __future__ import annotations

from typing import Any

from .model import load
from .types import SystemOneRequest, SystemOneResponse


def create_app(
    checkpoint_dir: str | None = None,
    model_cache: str | None = None,
    calibration: str | None = None,
    model_id: str = "Qwen/Qwen3.5-4B-Base",
    noul_readout: str | None = None,
    compile: bool = False,
    max_label_options: int | None = None,
    prompt_style: str | None = None,
    skip_multi_token_codes: bool = True,
) -> Any:
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="lev", version="0.1.0")
    state: dict[str, Any] = {"engine": None}

    @app.on_event("startup")
    def _load() -> None:
        state["engine"] = load(
            checkpoint_dir,
            model_id=model_id,
            cache_dir=model_cache,
            calibration=calibration,
            noul_readout=noul_readout,
            prompt_style=prompt_style,
            compile=compile,
            max_label_options=max_label_options,
            skip_multi_token_codes=skip_multi_token_codes,
        )

    @app.get("/health")
    def health() -> dict:
        engine = state["engine"]
        if engine is None:
            return {"status": "loading", "model": model_id}
        config = engine.config
        return {
            "status": "ok",
            "model": config.model_id,
            "checkpoint": str(engine.checkpoint) if engine.checkpoint else None,
            "calibrated": bool(engine.calibration.temperatures),
            "mode_b": engine.mode_b_head is not None,
            "noul_readout": config.noul_readout,
            "max_label_options": config.max_label_options,
            "order_average": config.order_average,
            "prefix_mode": config.prefix_mode,
            "compiled": config.compile,
            "prompt_style": config.prompt_style,
            "skip_multi_token_codes": config.skip_multi_token_codes,
        }

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def system_one(request: SystemOneRequest) -> SystemOneResponse:
        if state["engine"] is None:
            raise HTTPException(status_code=529, detail="model still loading")
        if not request.questions:
            raise HTTPException(status_code=422, detail="at least one question required")
        try:
            return state["engine"].system_one(request.state, request.questions)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
