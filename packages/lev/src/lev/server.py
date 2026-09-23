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

import json
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
    noul_readout: str | None = None,
    compile: bool = False,
    max_label_options: int | None = None,
    prompt_style: str | None = None,
) -> Any:
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="lev", version="0.1.0")
    state: dict[str, Any] = {"engine": None}

    @app.on_event("startup")
    def _load() -> None:
        nonlocal model_id, prompt_style
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # A checkpoint directory holds a LoRA *adapter*, not a full model: it
        # has `adapter_config.json` and no model config. So the base is loaded
        # from `model_id` and the adapter applied on top.
        checkpoint = None
        if checkpoint_dir:
            from .train.checkpoints import resolve_checkpoint

            checkpoint = resolve_checkpoint(checkpoint_dir, model_cache)
            # A packaged release names the base its adapter was trained on. The
            # weights are tied to it, so the manifest wins over the argument.
            manifest_file = checkpoint / "lev_release.json"
            if manifest_file.is_file():
                manifest = json.loads(manifest_file.read_text())
                # The prompt style is a property of the weights too.
                prompt_style = manifest.get("prompt_style", prompt_style)
                if manifest["base_model"] != model_id:
                    print(
                        f"release manifest names base {manifest['base_model']!r}; "
                        f"using it instead of {model_id!r}"
                    )
                    model_id = manifest["base_model"]

        # Tokenizer from the checkpoint when it saved one: the label-token
        # readout depends on which ids a code encodes to, so a mismatch produces
        # plausible wrong answers rather than an error.
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
        # Default to the profile sitting beside the weights it was fitted for:
        # inside a flat release directory, or one level up in a training tree.
        resolved_calibration = calibration
        if resolved_calibration is None and checkpoint:
            candidates = [checkpoint / "calibration.json", checkpoint.parent / "calibration.json"]
            resolved_calibration = str(next((c for c in candidates if c.is_file()), candidates[-1]))
        if resolved_calibration:
            try:
                profile = CalibrationProfile.load(resolved_calibration)
            except FileNotFoundError:
                # Serving uncalibrated is legitimate (it is build step 1), but it
                # must be loud: uncalibrated confidence is the failure mode the
                # benchmark measured at ECE 0.43.
                print(f"WARNING: no calibration at {resolved_calibration}; serving raw softmax")

        # A stock checkpoint pins the 0-8 rating scale at one end regardless of
        # content (ADR-007), so untrained serving reads Noul as two options.
        readout = noul_readout or ("rating" if checkpoint else "binary")
        config = EngineConfig(
            model_id=model_id,
            noul_readout=readout,
            compile=compile,
            prompt_style=prompt_style or "plain",
        )
        if max_label_options is not None:
            # A serving-side experiment knob (ADR-025): the trained policy is the
            # EngineConfig default, and /health reports whatever is in effect.
            config.max_label_options = max_label_options
        engine = DecisionEngine(model, tokenizer, config, profile, mode_b_head=head)
        if compile:
            # Pay compile and graph capture now, on every shape bucket the
            # benchmark and the demo send, so no request ever does.
            print(f"warmup: compiled forward in {engine.warmup():.0f}s", flush=True)
        state["engine"] = engine
        state["calibrated"] = bool(profile.temperatures)
        state["checkpoint"] = str(checkpoint) if checkpoint else None
        state["mode_b"] = head is not None
        state["noul_readout"] = readout
        state["max_label_options"] = config.max_label_options
        state["order_average"] = config.order_average
        state["prefix_mode"] = config.prefix_mode
        state["compiled"] = config.compile
        state["prompt_style"] = config.prompt_style

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok" if state["engine"] else "loading",
            "model": model_id,
            "checkpoint": state.get("checkpoint"),
            "calibrated": state.get("calibrated", False),
            "mode_b": state.get("mode_b", False),
            "noul_readout": state.get("noul_readout"),
            "max_label_options": state.get("max_label_options"),
            "order_average": state.get("order_average"),
            "prefix_mode": state.get("prefix_mode"),
            "compiled": state.get("compiled"),
            "prompt_style": state.get("prompt_style"),
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
