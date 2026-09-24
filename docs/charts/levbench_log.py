"""Parse `levbench eval` output into per-subset numbers.

The charts are built from these, never from retyped values: a levbench log is
the measurement, and this reads it as written.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_HEADER = re.compile(r"^=== (\S+)  \((\d+) items\) ===$", re.M)
_SERVED = re.compile(r"^=== (\w+) / (.+?) ===$", re.M)
_FIELDS = {
    "p50_s": re.compile(r"latency p50\s+([0-9.]+)s"),
    "mean_s": re.compile(r"latency mean\s+([0-9.]+)s"),
    "p95_s": re.compile(r"latency p95\s+([0-9.]+)s"),
    "input_tokens": re.compile(r"input tokens\s+([0-9,]+)"),
    "cost_usd": re.compile(r"total cost\s+\$([0-9.]+)"),
    "wall_s": re.compile(r"wall time\s+([0-9.]+)s"),
}
_SCORES = re.compile(r"accuracy ([0-9.]+)\s+log-loss ([0-9.]+)\s+brier ([0-9.]+)\s+ECE ([0-9.]+)")


def parse(text: str) -> dict:
    """`{subset: {n, accuracy, log_loss, brier, ece, p50_s, ...}}` plus `served_by`."""
    out: dict = {"subsets": {}}
    marks = list(_HEADER.finditer(text))
    for i, mark in enumerate(marks):
        body = text[mark.end() : marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        scores = _SCORES.search(body)
        if not scores:
            continue
        row = {"n": int(mark.group(2))}
        row.update(
            zip(("accuracy", "log_loss", "brier", "ece"), map(float, scores.groups()), strict=True)
        )
        for key, pattern in _FIELDS.items():
            if m := pattern.search(body):
                row[key] = float(m.group(1).replace(",", ""))
        if m := _SERVED.search(body):
            out["served_by"] = m.group(2)
        out["subsets"][mark.group(1)] = row
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(parse(Path(sys.argv[1]).read_text()), indent=2))
