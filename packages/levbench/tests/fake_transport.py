"""Fake Jev transport with deterministic pseudo-random answers; tests plumbing, not accuracy."""

from __future__ import annotations

import hashlib
import json

import httpx2


def _rand(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _probabilities(key: str, options: list[str] | range, state_sig: str) -> dict:
    weights = [_rand(state_sig, key, str(option)) + 1e-6 for option in options]
    total = sum(weights)
    return {o: round(w / total, 4) for o, w in zip(options, weights, strict=True)}


def _answer(key: str, question: dict, state_sig: str) -> dict:
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": round(_rand(state_sig, key, "noul"), 4)}

    if kind == "choice":
        probs = _probabilities(key, list(question["criteria"]), state_sig)
        return {
            "type": "choice",
            "choice": max(probs, key=probs.__getitem__),
            "probabilities": probs,
            "confidence": max(probs.values()),
        }

    if kind == "score":
        levels = question["criteria"]
        probs = _probabilities(key, range(len(levels)), state_sig)
        return {
            "type": "score",
            "score": round(sum(i * p for i, p in probs.items()), 4),
            "probabilities": probs,
            "confidence": max(probs.values()),
            "legend": {i: lv for i, lv in enumerate(levels)},
        }

    raise ValueError(kind)


def handler(request: httpx2.Request) -> httpx2.Response:
    body = json.loads(request.content)
    state = body["state"]
    state_text = state if isinstance(state, str) else json.dumps(state, sort_keys=True)
    state_sig = hashlib.sha256(state_text.encode()).hexdigest()[:16]

    questions = body["questions"]
    answers = {k: _answer(k, q, state_sig) for k, q in questions.items()}

    # Bill state tokens once per request and output tokens per question.
    input_tokens = max(1, len(state_text) // 4)
    output_tokens = 4 * len(questions)

    return httpx2.Response(
        200,
        json={
            "model": body.get("model", "jev-latest"),
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    )


def transport() -> httpx2.MockTransport:
    return httpx2.MockTransport(handler)
