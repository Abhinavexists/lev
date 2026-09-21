"""FastAPI server for `/v1/systemone`.

Wire-identical to TypeSafe's endpoint, so the benchmark measures us and Jev with
the same code path and one changed flag:

    levbench eval --backend jev --base-url http://localhost:8000

Error codes mirror the real API (422 validation, 429 rate limit, 529 overloaded)
so client retry logic behaves identically against either.

Runs: serves Mode A requests against a local `Qwen/Qwen3.5-4B-Base`. Start it with
`lev serve` and check `/health` first.
"""

from __future__ import annotations

from typing import Any

from .calibrate import CalibrationProfile
from .model import DecisionEngine, EngineConfig
from .types import SystemOneRequest, SystemOneResponse


def _load_head(checkpoint, model):
    """Load the Mode B head saved beside the adapter, if there is one.

    Absent is fine and means Mode A only -- the router will refuse a question
    that needs Mode B rather than answer it wrongly.
    """
    import torch

    from .readout.mode_b import CandidatePathReadout

    path = checkpoint / "mode_b_head.pt"
    if not path.is_file():
        return None

    state_dict = torch.load(path, map_location="cpu")
    # Shapes come from the saved tensors, not from a config that might have
    # drifted since the run that produced them.
    proj_dim, hidden_size = state_dict["question_proj.weight"].shape
    head = CandidatePathReadout(hidden_size=hidden_size, proj_dim=proj_dim)
    head.load_state_dict(state_dict)
    return head.to(device=next(model.parameters()).device, dtype=torch.float32).eval()


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

        # A checkpoint directory holds a LoRA *adapter*, not a full model, so
        # the base has to be loaded first and the adapter applied on top.
        # `AutoModelForCausalLM.from_pretrained(checkpoint_dir)` -- which this
        # did before anything had been trained -- cannot work: there is no
        # model config there, only `adapter_config.json`.
        checkpoint = None
        if checkpoint_dir:
            from .train.loop import resolve_checkpoint

            checkpoint = resolve_checkpoint(checkpoint_dir)

        # Tokenizer from the checkpoint when it saved one: the label-token
        # readout depends on which ids a code encodes to, so a mismatch here
        # produces plausible-looking wrong answers rather than an error.
        tok_source = (
            checkpoint if checkpoint and (checkpoint / "tokenizer.json").is_file() else model_id
        )
        tokenizer = AutoTokenizer.from_pretrained(str(tok_source), cache_dir=model_cache)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.device_count() > 1 else None,
            cache_dir=model_cache,
        )
        if torch.cuda.is_available() and torch.cuda.device_count() == 1:
            model = model.cuda()

        head = None
        if checkpoint:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(checkpoint))
            head = _load_head(checkpoint, model)
        model = model.eval()

        profile = CalibrationProfile()
        # Default to the profile sitting beside the weights it was fitted for.
        resolved_calibration = calibration or (
            str(checkpoint.parent / "calibration.json") if checkpoint else None
        )
        if resolved_calibration:
            try:
                profile = CalibrationProfile.load(resolved_calibration)
            except FileNotFoundError:
                # Serving uncalibrated is legitimate (it is build step 1), but it
                # must be loud: uncalibrated confidence is the failure mode the
                # benchmark measured at ECE 0.43.
                print(f"WARNING: no calibration at {resolved_calibration}; serving raw softmax")

        state["engine"] = DecisionEngine(
            model, tokenizer, EngineConfig(model_id=model_id), profile, mode_b_head=head
        )
        state["calibrated"] = bool(profile.temperatures)
        state["checkpoint"] = str(checkpoint) if checkpoint else None
        state["mode_b"] = head is not None

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok" if state["engine"] else "loading",
            "model": model_id,
            "checkpoint": state.get("checkpoint"),
            "calibrated": state.get("calibrated", False),
            "mode_b": state.get("mode_b", False),
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
