"""A fake Jev endpoint, so the harness can be exercised without an API key.

It parses the real request body and emits wire-shaped responses, which means it
catches serialisation and parsing bugs. It does NOT model Jev's judgement --
answers are deterministic pseudo-random, so accuracy numbers from this
transport are meaningless by construction. Only plumbing is under test here.
"""

from __future__ import annotations

import hashlib
import json

import httpx2


def _rand(*parts: str) -> float:
    """Deterministic pseudo-random float in [0,1) from the given strings."""
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _answer(key: str, question: dict, state_sig: str) -> dict:
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": round(_rand(state_sig, key, "noul"), 4)}

    if kind == "choice":
        options = list(question["criteria"])
        weights = [_rand(state_sig, key, o) + 1e-6 for o in options]
        total = sum(weights)
        probs = {o: round(w / total, 4) for o, w in zip(options, weights, strict=True)}
        return {
            "type": "choice",
            "choice": max(probs, key=probs.get),
            "probabilities": probs,
            "confidence": round(max(probs.values()), 4),
        }

    if kind == "score":
        levels = question["criteria"]
        weights = [_rand(state_sig, key, str(i)) + 1e-6 for i in range(len(levels))]
        total = sum(weights)
        probs = {i: round(w / total, 4) for i, w in enumerate(weights)}
        return {
            "type": "score",
            "score": round(sum(i * p for i, p in probs.items()), 4),
            "probabilities": probs,
            "confidence": round(max(probs.values()), 4),
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

    # Input scales with the state (billed once per request), output with the
    # question count: the shape the batching sweep must find from responses.
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
