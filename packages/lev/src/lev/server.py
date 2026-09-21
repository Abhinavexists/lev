"""FastAPI server for `/v1/systemone`.

Wire-identical to TypeSafe's endpoint, so the benchmark measures us and Jev with
the same code path and one changed flag:

    levbench eval --backend jev --base-url http://localhost:8000

Error codes mirror the real API (422 validation, 429 rate limit, 529 overloaded)
so client retry logic behaves identically against either.

NOT YET RUN. Start it with `lev serve` and check `/health` first.
"""

from __future__ import annotations

from typing import Any

from .calibrate import CalibrationProfile
from .model import DecisionEngine, EngineConfig
from .types import SystemOneRequest, SystemOneResponse


def create_app(
    checkpoint_dir: str | None = None,
    model_cache: str | None = None,
    calibration: str | None = None,
    model_id: str = "Qwen/Qwen3.5-4B-Base",
) -> Any:
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="lev", version="0.1.0")
    state: dict[str, Any] = {"engine": None}

    @app.on_event("startup")
    def _load() -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = checkpoint_dir or model_id
        tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=model_cache)
        model = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.bfloat16, device_map="auto", cache_dir=model_cache
        ).eval()

        profile = CalibrationProfile()
        if calibration:
            try:
                profile = CalibrationProfile.load(calibration)
            except FileNotFoundError:
                # Serving uncalibrated is legitimate (it is build step 1), but it
                # must be loud: uncalibrated confidence is the failure mode the
                # benchmark measured at ECE 0.43.
                print(f"WARNING: no calibration at {calibration}; serving raw softmax")

        state["engine"] = DecisionEngine(model, tokenizer, EngineConfig(model_id=source), profile)
        state["calibrated"] = bool(profile.temperatures)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok" if state["engine"] else "loading",
            "model": model_id,
            "calibrated": state.get("calibrated", False),
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
