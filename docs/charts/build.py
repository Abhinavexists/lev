# ruff: noqa: E501 -- SVG and HTML markup, one element per line
"""Render docs/charts/lev-vs-jev.html from measured logs and recorded history.

    uv run python docs/charts/build.py

Inputs, all checked in:
  logs/jev-s1bench-*.txt            Jev on the 13 S1Bench task files, through levbench
  logs/lev-s1bench-*.txt            lev on the same files, same laptop
  logs/pre-nimble/                  both runs on the earlier six-subset definitions,
                                    the scale the iterations in history.json were on
  history.json                      lev iterations and speed measurements, each
                                    with the FINDINGS section it was recorded in
  ../../data/s1bench-snapshot.json  the public S1Bench board: every other model

lev and Jev keep fixed colours (slot 1 blue, slot 2 orange; validated for
colour-vision deficiency in both modes); every other model is context, drawn
in the de-emphasis gray. One scale per chart, and every value is also in the
table beneath it.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path

from lev.data.s1bench import EVAL_SUBSETS, definition
from levbench_log import parse

HERE = Path(__file__).parent
SNAPSHOT = HERE.parent.parent / "data" / "s1bench-snapshot.json"
# The six subsets every model on the board completed; the leaderboard is over these.
SUBSETS = ["aegis2", "boolq", "helpsteer2", "massive-en-US", "paws", "vitaminc-dev"]
ALL_SUBSETS = list(EVAL_SUBSETS)
W = 760

LEV, JEV, CTX = "--series-1", "--series-2", "--context"


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def bar_path(x0: float, x1: float, y: float, h: float, r: float = 4) -> str:
    """Square at the baseline, 4px rounded at the data end."""
    if x1 - x0 <= 0.5:
        return ""
    r = min(r, (x1 - x0) / 2, h / 2)
    return (
        f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
        f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z"
    )


def mark(d: str, var: str, tip: str) -> str:
    return f'<path d="{d}" fill="var({var})" class="mark" data-tip="{esc(tip)}" tabindex="0"/>'


def legend(items) -> str:
    keys = "".join(
        f'<span class="key"><i style="background:var({var})"></i>{esc(name)}</span>'
        for name, var in items
    )
    return f'<div class="legend">{keys}</div>'


def table(headers, rows) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (
        '<details class="tbl"><summary>Data</summary>'
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></details>"
    )


def leaderboard(rows, vmin=0.2, vmax=0.8) -> str:
    """Ranked horizontal bars. Rows: dicts with name, value, var, meta, bold."""
    label_w, meta_w, row_h, bar_h, top = 196, 150, 23, 13, 6
    plot_w = W - label_w - meta_w - 40
    x = lambda v: label_w + (v - vmin) / (vmax - vmin) * plot_w  # noqa: E731
    h = top + len(rows) * row_h + 24
    out = [f'<svg viewBox="0 0 {W} {h}" role="img" class="chart">']
    for k in range(round((vmax - vmin) / 0.1) + 1):
        t = vmin + k * 0.1
        out.append(
            f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{h - 22}" class="grid"/>'
            f'<text x="{x(t):.1f}" y="{h - 6}" class="tick" text-anchor="middle">{t * 100:.0f}%</text>'
        )
    for i, r in enumerate(rows):
        cy = top + i * row_h + row_h / 2
        b = " b" if r.get("bold") else ""
        out.append(
            f'<text x="{label_w - 12}" y="{cy + 4:.1f}" class="cat{b}" text-anchor="end">{esc(r["name"])}</text>'
        )
        out.append(
            mark(
                bar_path(label_w, x(r["value"]), cy - bar_h / 2, bar_h),
                r["var"],
                f"{r['name']}|{pct(r['value'])}",
            )
        )
        out.append(
            f'<text x="{x(r["value"]) + 7:.1f}" y="{cy + 4:.1f}" class="val{b}">{pct(r["value"])}</text>'
        )
        out.append(
            f'<text x="{W - 4}" y="{cy + 4:.1f}" class="meta" text-anchor="end">{esc(r["meta"])}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def grouped_bars(cats, series, vmax, fmt, ticks, label_w=118) -> str:
    """One group per category, one bar per series."""
    bar_h, gap, group_gap, top = 12, 2, 16, 4
    group_h = len(series) * bar_h + (len(series) - 1) * gap
    h = top + len(cats) * (group_h + group_gap) - group_gap + 24
    plot_w = W - label_w - 64
    x = lambda v: label_w + v / vmax * plot_w  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {h}" role="img" class="chart">']
    for t in ticks:
        out.append(
            f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{h - 22}" class="grid"/>'
            f'<text x="{x(t):.1f}" y="{h - 6}" class="tick" text-anchor="middle">{esc(fmt(t))}</text>'
        )
    for i, cat in enumerate(cats):
        gy = top + i * (group_h + group_gap)
        out.append(
            f'<text x="{label_w - 12}" y="{gy + group_h / 2 + 4:.1f}" class="cat" text-anchor="end">{esc(cat)}</text>'
        )
        for j, s in enumerate(series):
            v, y = s["values"][i], gy + j * (bar_h + gap)
            var = s["vars"][i] if "vars" in s else s["var"]
            out.append(
                mark(bar_path(label_w, x(v), y, bar_h), var, f"{s['name']} · {cat}|{fmt(v)}")
            )
            out.append(
                f'<text x="{x(v) + 6:.1f}" y="{y + bar_h - 2:.1f}" class="val">{esc(fmt(v))}</text>'
            )
    out.append("</svg>")
    return "".join(out)


def scatter(
    points, xr, yr, xfmt, yfmt, xticks, yticks, xlabel, ylabel, xlog=False, hline=None
) -> str:
    """Points: dicts with x, y, name, var, and label (direct-label it)."""
    h, left, right, top, bottom = 380, 58, 24, 14, 44
    pw, ph = W - left - right, h - top - bottom
    tx = math.log10 if xlog else (lambda v: v)
    px = lambda v: left + (tx(v) - tx(xr[0])) / (tx(xr[1]) - tx(xr[0])) * pw  # noqa: E731
    py = lambda v: top + (1 - (v - yr[0]) / (yr[1] - yr[0])) * ph  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {h}" role="img" class="chart">']
    for t in xticks:
        out.append(
            f'<line x1="{px(t):.1f}" x2="{px(t):.1f}" y1="{top}" y2="{top + ph}" class="grid"/>'
            f'<text x="{px(t):.1f}" y="{top + ph + 16}" class="tick" text-anchor="middle">{esc(xfmt(t))}</text>'
        )
    for t in yticks:
        out.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{py(t):.1f}" y2="{py(t):.1f}" class="grid"/>'
            f'<text x="{left - 8}" y="{py(t) + 4:.1f}" class="tick" text-anchor="end">{esc(yfmt(t))}</text>'
        )
    out.append(
        f'<text x="{left + pw / 2}" y="{h - 6}" class="axlab" text-anchor="middle">{esc(xlabel)}</text>'
        f'<text x="14" y="{top + ph / 2}" class="axlab" text-anchor="middle" transform="rotate(-90 14 {top + ph / 2})">{esc(ylabel)}</text>'
    )
    if hline:
        y = py(hline["value"])
        out.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{y:.1f}" y2="{y:.1f}" class="ref" stroke="var({hline["var"]})"/>'
            f'<text x="{left + 6}" y="{y - 6:.1f}" class="reflab">{esc(hline["name"])}</text>'
        )
    for p in sorted(points, key=lambda p: p["var"] != CTX):  # context underneath
        cx, cy = px(p["x"]), py(p["y"])
        r = 6 if p["var"] != CTX else 4.5
        tip = f"{p['name']}|{xfmt(p['x'])} · {yfmt(p['y'])}"
        out.append(
            f'<g class="pt" data-tip="{esc(tip)}" tabindex="0"><circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" fill="transparent"/>'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="var({p["var"]})" stroke="var(--surface)" stroke-width="2"/></g>'
        )
        if p.get("label"):
            b = " b" if p["var"] != CTX else ""
            out.append(
                f'<text x="{cx + p.get("dx", 10):.1f}" y="{cy + p.get("dy", 4):.1f}" class="plab{b}" text-anchor="{p.get("anchor", "start")}">{esc(p["name"])}</text>'
            )
    out.append("</svg>")
    return "".join(out)


def line(xs, values, refs, ymin, ymax) -> str:
    h, left, right, top, bottom = 280, 48, 132, 18, 42
    pw, ph = W - left - right, h - top - bottom
    px = lambda i: left + i / (len(xs) - 1) * pw  # noqa: E731
    py = lambda v: top + (1 - (v - ymin) / (ymax - ymin)) * ph  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {h}" role="img" class="chart">']
    for k in range(round((ymax - ymin) / 0.1) + 1):
        t = ymin + k * 0.1
        out.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{py(t):.1f}" y2="{py(t):.1f}" class="grid"/>'
            f'<text x="{left - 8}" y="{py(t) + 4:.1f}" class="tick" text-anchor="end">{t * 100:.0f}%</text>'
        )
    for i, label in enumerate(xs):
        for k, part in enumerate(label.split("\n")):
            out.append(
                f'<text x="{px(i):.1f}" y="{top + ph + 17 + 13 * k}" class="tick" text-anchor="middle">{esc(part)}</text>'
            )
    for ref in refs:
        y = py(ref["value"])
        out.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{y:.1f}" y2="{y:.1f}" class="ref" stroke="var({ref["var"]})"/>'
            f'<text x="{left + pw + 8}" y="{y + 4 + ref.get("dy", 0):.1f}" class="reflab">{esc(ref["name"])} {pct(ref["value"])}</text>'
        )
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
    out.append(
        f'<polyline points="{pts}" fill="none" stroke="var({LEV})" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    for i, v in enumerate(values):
        tip = xs[i].replace("\n", " ") + "|" + pct(v)
        out.append(
            f'<g class="pt" data-tip="{esc(tip)}" tabindex="0">'
            f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="12" fill="transparent"/>'
            f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4.5" fill="var({LEV})" stroke="var(--surface)" stroke-width="2"/></g>'
            f'<text x="{px(i) - (8 if i == len(values) - 1 else 0):.1f}" y="{py(v) + 20:.1f}" class="val" text-anchor="{"end" if i == len(values) - 1 else "middle"}">{pct(v)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def size_label(params_m) -> str:
    if params_m is None:
        return "size n/a"
    return f"{params_m / 1000:g}B" if params_m >= 1000 else f"{params_m}M"


@dataclass
class Inputs:
    lev: dict  # subset -> levbench scores
    jev: dict
    lev_served_by: str
    lev_log_name: str
    jev_log_name: str
    history: dict
    board: dict
    lev_acc: float  # macro over the board's six
    jev_acc: float
    lev_ece: float
    jev_ece: float
    lev_acc_all: float  # macro over all thirteen
    jev_acc_all: float
    lev_ece_all: float
    jev_ece_all: float
    lev_acc_pre_nimble: float
    jev_acc_pre_nimble: float


def load_inputs() -> Inputs:
    jev_log = sorted(HERE.glob("logs/jev-s1bench-*.txt"))[-1]
    jev = parse(jev_log.read_text())["subsets"]
    lev_log = sorted(HERE.glob("logs/lev-s1bench-*.txt"))[-1]
    lev_parsed = parse(lev_log.read_text())
    lev = lev_parsed["subsets"]
    pre_nimble = {
        name: parse((HERE / f"logs/pre-nimble/{name}").read_text())["subsets"]
        for name in ("lev-s1bench-final.txt", "jev-s1bench-2026-09-22.txt")
    }

    def mean(scores, key, subsets=SUBSETS):
        return sum(scores[s][key] for s in subsets) / len(subsets)

    return Inputs(
        lev=lev,
        jev=jev,
        lev_served_by=lev_parsed.get("served_by", "?"),
        lev_log_name=lev_log.name,
        jev_log_name=jev_log.name,
        history=json.loads((HERE / "history.json").read_text()),
        board=json.loads(SNAPSHOT.read_text()),
        lev_acc=mean(lev, "accuracy"),
        jev_acc=mean(jev, "accuracy"),
        lev_ece=mean(lev, "ece"),
        jev_ece=mean(jev, "ece"),
        lev_acc_all=mean(lev, "accuracy", ALL_SUBSETS),
        jev_acc_all=mean(jev, "accuracy", ALL_SUBSETS),
        lev_ece_all=mean(lev, "ece", ALL_SUBSETS),
        jev_ece_all=mean(jev, "ece", ALL_SUBSETS),
        lev_acc_pre_nimble=mean(pre_nimble["lev-s1bench-final.txt"], "accuracy"),
        jev_acc_pre_nimble=mean(pre_nimble["jev-s1bench-2026-09-22.txt"], "accuracy"),
    )


def majority_rate(subset: str) -> float:
    """Accuracy of always answering the most common label."""
    labels = definition(subset)["labels"]
    return max(labels.values()) / sum(labels.values())


def completed_targets(board: dict) -> list[dict]:
    """Board models that completed all six subsets; a model stopped after one
    subset reports a one-subset macro and is not comparable."""
    return [
        t
        for t in board["targets"]
        if sum(t["subsets"].get(s, {}).get("acc") is not None for s in SUBSETS) == 6
    ]


def leaderboard_rows(done: list[dict], data: Inputs) -> list[dict]:
    """Board models plus lev and Jev as measured here, best first."""
    rows = []
    for t in done:
        is_jev = t["target"] == "jev"
        rows.append(
            {
                "name": ("Jev (board)" if is_jev else t["target"])
                + (" †" if t.get("contamination") else ""),
                "value": t["macro_acc"],
                "var": JEV if is_jev else CTX,
                "meta": ("TypeSafe API" if is_jev else size_label(t.get("params_m")))
                + f"  ·  ECE {t['ece']:.3f}",
                "bold": is_jev,
            }
        )
    rows.append(
        {
            "name": "lev (measured here)",
            "value": data.lev_acc,
            "var": LEV,
            "meta": f"4B  ·  ECE {data.lev_ece:.3f}",
            "bold": True,
        }
    )
    rows.append(
        {
            "name": "Jev (measured here)",
            "value": data.jev_acc,
            "var": JEV,
            "meta": f"TypeSafe API  ·  ECE {data.jev_ece:.3f}",
            "bold": True,
        }
    )
    rows.sort(key=lambda r: -r["value"])
    return rows


# Board models that get a direct label in the scatter charts.
LABELLED = {
    "reflex-4b",
    "decider-2b",
    "simplejev-qwen38-27b",
    "djev-full",
    "laya-gpu",
    "open-jev-deberta",
    "qwen3-8b-full",
    "jeff-gpu-full",
}
# Direct-label positions (dx, dy, anchor) per chart, set by eye to avoid collisions.
LABEL_OFFSETS = {
    "cal": {
        "simplejev-qwen38-27b": (10, 4, "start"),
        "djev-full": (10, 0, "start"),
        "reflex-4b": (10, 14, "start"),
        "decider-2b": (-10, 4, "end"),
        "open-jev-deberta": (-10, 4, "end"),
    },
    "size": {
        "reflex-4b": (0, 18, "middle"),
        "decider-2b": (-10, 4, "end"),
        "simplejev-qwen38-27b": (-10, 14, "end"),
        "open-jev-deberta": (-10, 4, "end"),
    },
}


def place(point: dict, chart: str) -> dict:
    if point["name"] in LABEL_OFFSETS[chart]:
        point["dx"], point["dy"], point["anchor"] = LABEL_OFFSETS[chart][point["name"]]
    return point


def calibration_points(done: list[dict], data: Inputs) -> list[dict]:
    points = [
        place(
            {
                "x": t["macro_acc"],
                "y": t["ece"],
                "name": "Jev" if t["target"] == "jev" else t["target"],
                "var": JEV if t["target"] == "jev" else CTX,
                "label": t["target"] in LABELLED or t["target"] == "jev",
            },
            "cal",
        )
        for t in done
    ]
    points.append(
        {
            "x": data.lev_acc,
            "y": data.lev_ece,
            "name": "lev",
            "var": LEV,
            "label": True,
            "dx": -10,
            "dy": -8,
            "anchor": "end",
        }
    )
    return points


def size_points(done: list[dict], data: Inputs) -> list[dict]:
    points = [
        place(
            {
                "x": t["params_m"] / 1000,
                "y": t["macro_acc"],
                "name": t["target"],
                "var": CTX,
                "label": t["target"] in LABELLED and t["target"] != "djev-full",
            },
            "size",
        )
        for t in done
        if t.get("params_m") and t["target"] != "jev"
    ]
    points.append(
        {
            "x": 4.0,
            "y": data.lev_acc,
            "name": "lev",
            "var": LEV,
            "label": True,
            "dx": 0,
            "dy": -12,
            "anchor": "middle",
        }
    )
    return points


def section(kicker, title, sub, body, data, note="") -> str:
    return (
        f'<section><div class="kicker">{esc(kicker)}</div><h2>{esc(title)}</h2>'
        f'<p class="sub">{esc(sub)}</p>{body}'
        + (f'<p class="note">{esc(note)}</p>' if note else "")
        + f"{data}</section>"
    )


LEV_AND_JEV = [("lev", LEV), ("Jev", JEV)]


def leaderboard_section(rows: list[dict], jev_board: dict, data: Inputs) -> str:
    above = [r for r in rows if r["var"] == CTX and r["value"] > data.lev_acc]
    return section(
        "Where lev stands",
        f"Behind only Jev and {len(above)} open models of 26B and up",
        "Macro accuracy over the six S1Bench subsets that every listed model completed.",
        legend([("lev", LEV), ("Jev", JEV), ("open models on the S1Bench board", CTX)])
        + leaderboard(rows),
        table(
            ["model", "macro accuracy", "size", "ECE"],
            [
                [
                    r["name"],
                    pct(r["value"]),
                    r["meta"].split("  ·  ")[0],
                    r["meta"].split("ECE ")[1],
                ]
                for r in rows
            ],
        ),
        f"Board models were scored by S1Bench's own harness, lev by ours, on the same pinned items. On the same model our harness scores Jev "
        f"{(jev_board['macro_acc'] - data.jev_acc) * 100:.1f} points lower than the board, all of it on aegis2. "
        "† self-declared training contamination on an evaluation subset. Models stopped after one subset are omitted. "
        "Gemini, Claude and GPT have not been run on these subsets, so they are not shown.",
    )


def calibration_section(points: list[dict]) -> str:
    return section(
        "Accuracy and calibration",
        "Further right is more accurate; lower is better calibrated",
        "Macro accuracy against calibration error (ECE). The ideal model sits in the bottom-right corner.",
        scatter(
            points,
            (0.28, 0.86),
            (0.0, 0.46),
            pct,
            lambda v: f"{v:.2f}",
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0, 0.1, 0.2, 0.3, 0.4],
            "macro accuracy",
            "ECE",
        ),
        table(
            ["model", "macro accuracy", "ECE"],
            [
                [p["name"], pct(p["x"]), f"{p['y']:.3f}"]
                for p in sorted(points, key=lambda p: -p["x"])
            ],
        ),
        "lev's point uses our harness for both axes; the others use the board's.",
    )


def size_section(points: list[dict], jev_board: dict) -> str:
    return section(
        "Accuracy per parameter",
        "lev matches open models six times its size",
        "Macro accuracy against parameter count, log scale. Jev's size is not published, so it is drawn as a line.",
        scatter(
            points,
            (0.06, 40),
            (0.3, 0.8),
            lambda v: f"{v:g}B" if v >= 1 else f"{v * 1000:.0f}M",
            pct,
            [0.1, 0.3, 1, 3, 10, 30],
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "parameters",
            "macro accuracy",
            xlog=True,
            hline={
                "value": jev_board["macro_acc"],
                "var": JEV,
                "name": f"Jev (board) {pct(jev_board['macro_acc'])}",
            },
        ),
        table(
            ["model", "parameters", "macro accuracy"],
            [
                [p["name"], f"{p['x']:g}B", pct(p["y"])]
                for p in sorted(points, key=lambda p: p["x"])
            ],
        ),
    )


def per_subset_series(data: Inputs, value) -> list[dict]:
    """lev and Jev as grouped-bar series, `value(scores)` per subset."""
    return [
        {"name": "lev", "var": LEV, "values": [value(data.lev[s]) for s in ALL_SUBSETS]},
        {"name": "Jev", "var": JEV, "values": [value(data.jev[s]) for s in ALL_SUBSETS]},
    ]


def accuracy_by_subset_section(data: Inputs) -> str:
    lev, jev = data.lev, data.jev
    return section(
        "Head to head",
        "Accuracy by subset",
        f"lev and Jev on all 13 S1Bench subsets ({sum(lev[s]['n'] for s in ALL_SUBSETS):,} task items), same harness, same laptop.",
        legend(LEV_AND_JEV)
        + grouped_bars(
            ALL_SUBSETS,
            per_subset_series(data, lambda r: r["accuracy"]),
            1.0,
            pct,
            [0, 0.25, 0.5, 0.75, 1.0],
            label_w=150,
        ),
        table(
            ["subset", "n", "lev", "Jev", "lev − Jev", "majority label"],
            [
                [
                    s,
                    lev[s]["n"],
                    pct(lev[s]["accuracy"]),
                    pct(jev[s]["accuracy"]),
                    f"{(lev[s]['accuracy'] - jev[s]['accuracy']) * 100:+.1f} pts",
                    pct(majority_rate(s)),
                ]
                for s in ALL_SUBSETS
            ],
        ),
        "At these sizes a per-subset difference needs roughly 5–9 points to clear sampling noise. "
        "On civil_comments, helpsteer2 and both summeval subsets, always answering the most common label beats both models.",
    )


def ece_by_subset_section(data: Inputs) -> str:
    lev, jev = data.lev, data.jev
    return section(
        "Head to head",
        "Calibration error by subset",
        "The gap between stated confidence and observed accuracy. Lower is better; 0 is perfect.",
        legend(LEV_AND_JEV)
        + grouped_bars(
            ALL_SUBSETS,
            per_subset_series(data, lambda r: r["ece"]),
            0.32,
            lambda v: f"{v:.3f}",
            [0, 0.1, 0.2, 0.3],
            label_w=150,
        ),
        table(
            ["subset", "lev ECE", "Jev ECE", "lev log-loss", "Jev log-loss"],
            [
                [
                    s,
                    f"{lev[s]['ece']:.3f}",
                    f"{jev[s]['ece']:.3f}",
                    f"{lev[s]['log_loss']:.3f}",
                    f"{jev[s]['log_loss']:.3f}",
                ]
                for s in ALL_SUBSETS
            ],
        ),
    )


def progress_section(data: Inputs) -> str:
    runs = data.history["macro_accuracy"]
    xs = [r["label"] for r in runs] + ["final\nserving fixes"]
    vals = [r["value"] for r in runs] + [data.lev_acc_pre_nimble]
    return section(
        "Progress",
        f"From {pct(vals[0])} to {pct(vals[-1])} across four iterations",
        "lev's macro accuracy per model iteration, against Jev and the frozen backbone it was trained from. "
        "Measured on the six subsets as this repo defined them before adopting S1Bench's pinned items, "
        "so the scale differs from the charts above.",
        line(
            xs,
            vals,
            [
                {"name": "Jev", "var": JEV, "value": data.jev_acc_pre_nimble, "dy": -3},
                {
                    "name": "frozen 4B",
                    "var": CTX,
                    "value": data.history["frozen_backbone"],
                    "dy": 7,
                },
            ],
            0.4,
            0.8,
        ),
        table(
            ["iteration", "macro accuracy", "recorded in"],
            [[r["label"].replace("\n", " "), pct(r["value"]), r["source"]] for r in runs]
            + [["final, serving fixes", pct(vals[-1]), "logs/pre-nimble/lev-s1bench-final.txt"]],
        ),
    )


def compute_section(data: Inputs) -> str:
    compute = data.history["speed"]["container_compute_ms"]
    throughput = data.history["speed"]["benchmark_throughput"]
    return section(
        "Speed",
        f"lev compute per call: {compute[0]['value']:.0f} ms → {compute[-1]['value']:.0f} ms",
        "Inside the container on one H100, before any network. lev only; Jev's internals are not observable.",
        grouped_bars(
            [r["label"] for r in compute],
            [
                {
                    "name": "lev",
                    "values": [r["value"] for r in compute],
                    "vars": [CTX] * (len(compute) - 1) + [LEV],
                }
            ],
            200,
            lambda v: f"{v:.0f} ms",
            [0, 50, 100, 150, 200],
            label_w=196,
        ),
        table(["engine path", "ms per call"], [[r["label"], f"{r['value']:.0f}"] for r in compute]),
        f"Benchmark throughput rose from {throughput[0]['value']:.2f} to {throughput[-1]['value']:.2f} items/s "
        f"({throughput[-1]['label']}); a question costs the same whether a request asks one or eight.",
    )


def headline_tiles(data: Inputs) -> str:
    return "".join(
        f'<div class="tile"><div class="tl">{esc(label)}</div>'
        f'<div class="tv"><b style="color:var({LEV})">{esc(a)}</b><span>lev</span></div>'
        f'<div class="tv j"><b>{esc(b)}</b><span>Jev</span></div>'
        f'<div class="tn">{esc(note)}</div></div>'
        for label, a, b, note in (
            (
                "Accuracy",
                pct(data.lev_acc_all),
                pct(data.jev_acc_all),
                "macro, all 13 subsets · higher is better",
            ),
            (
                "Accuracy, board subsets",
                pct(data.lev_acc),
                pct(data.jev_acc),
                "macro, the 6 the board completed",
            ),
            (
                "Calibration error",
                f"{data.lev_ece_all:.3f}",
                f"{data.jev_ece_all:.3f}",
                "mean ECE, 13 subsets · lower is better",
            ),
        )
    )


def main() -> None:
    data = load_inputs()
    done = completed_targets(data.board)
    jev_board = next(t for t in done if t["target"] == "jev")
    rows = leaderboard_rows(done, data)

    sections = [
        leaderboard_section(rows, jev_board, data),
        calibration_section(calibration_points(done, data)),
        size_section(size_points(done, data), jev_board),
        accuracy_by_subset_section(data),
        ece_by_subset_section(data),
        progress_section(data),
        compute_section(data),
    ]
    provenance = (
        f"lev: logs/{data.lev_log_name} ({data.lev_served_by}) · "
        f"Jev: logs/{data.jev_log_name} · board: data/s1bench-snapshot.json (build {data.board.get('build', '?')}) · "
        "iterations and speed: history.json"
    )
    page = (
        TEMPLATE.replace("{{TILES}}", headline_tiles(data))
        .replace("{{SECTIONS}}", "".join(sections))
        .replace("{{PROVENANCE}}", esc(provenance))
    )
    (HERE / "lev-vs-jev.html").write_text(page)
    rank = next(i for i, r in enumerate(rows) if r["name"].startswith("lev")) + 1
    print(
        f"wrote lev-vs-jev.html  rank {rank}/{len(rows)}  board six: lev {pct(data.lev_acc)} Jev {pct(data.jev_acc)}"
        f"  all 13: lev {pct(data.lev_acc_all)} ECE {data.lev_ece_all:.3f}  Jev {pct(data.jev_acc_all)} ECE {data.jev_ece_all:.3f}"
    )


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>lev — S1Bench performance and speed</title>
<style>
.viz-root{color-scheme:light;--bg:#fcfcfb;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--ink-3:#8a8984;--rule:#ecebe7;
--grid:#efeee9;--series-1:#2a78d6;--series-2:#eb6834;--context:#c7c6c0}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--bg:#1a1a19;--surface:#1a1a19;
--ink:#ffffff;--ink-2:#c3c2b7;--ink-3:#8f8e86;--rule:#2b2b29;--grid:#262624;--series-1:#3987e5;--series-2:#d95926;--context:#55554f}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--bg:#1a1a19;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--ink-3:#8f8e86;
--rule:#2b2b29;--grid:#262624;--series-1:#3987e5;--series-2:#d95926;--context:#55554f}
*{box-sizing:border-box}html,body{margin:0}
.viz-root{background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Inter,sans-serif;
-webkit-font-smoothing:antialiased;padding:56px 24px 72px;min-height:100vh}
.wrap{max-width:820px;margin:0 auto}
header{margin-bottom:8px}.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);margin-bottom:10px}
h1{font-size:34px;line-height:1.15;font-weight:650;letter-spacing:-.02em;margin:0 0 12px}
.lede{color:var(--ink-2);font-size:16px;max-width:640px;margin:0}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);margin:36px 0 0}
.tile{padding:20px 0 18px}.tile+.tile{border-left:1px solid var(--rule);padding-left:22px}
.tl{font-size:12px;color:var(--ink-3);letter-spacing:.04em;text-transform:uppercase;margin-bottom:10px}
.tv{display:flex;align-items:baseline;gap:8px}.tv b{font-size:30px;font-weight:650;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.tv.j b{font-size:17px;font-weight:550;color:var(--ink-2)}.tv span{font-size:12.5px;color:var(--ink-3)}
.tn{font-size:12px;color:var(--ink-3);margin-top:6px}
section{padding:40px 0 8px;border-top:1px solid var(--rule);margin-top:32px}
.kicker{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);margin-bottom:6px}
h2{font-size:21px;font-weight:620;letter-spacing:-.01em;margin:0 0 6px}
.sub{color:var(--ink-2);margin:0 0 18px;font-size:14px;max-width:640px}
.legend{display:flex;flex-wrap:wrap;gap:18px;margin:0 0 12px;font-size:13px;color:var(--ink-2)}
.key{display:inline-flex;align-items:center;gap:7px}.key i{width:10px;height:10px;border-radius:3px;display:inline-block}
.chart{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}.ref{stroke-width:1.5}
.tick{fill:var(--ink-3);font-size:11px;font-variant-numeric:tabular-nums}.axlab{fill:var(--ink-3);font-size:11.5px}
.cat{fill:var(--ink-2);font-size:12.5px}.cat.b{fill:var(--ink);font-weight:620}
.val{fill:var(--ink-3);font-size:11.5px;font-variant-numeric:tabular-nums}.val.b{fill:var(--ink);font-weight:620}
.meta{fill:var(--ink-3);font-size:11.5px;font-variant-numeric:tabular-nums}
.plab{fill:var(--ink-3);font-size:11.5px}.plab.b{fill:var(--ink);font-weight:620;font-size:12.5px}.reflab{fill:var(--ink-2);font-size:11.5px}
.mark,.pt{outline:none}.mark{transition:opacity .12s}.mark:hover,.mark:focus{opacity:.72}
.pt:hover circle:last-child,.pt:focus circle:last-child{stroke:var(--ink);stroke-width:1.5}
.note{color:var(--ink-3);font-size:12.5px;margin:14px 0 0;max-width:680px}
.tbl{margin-top:14px;font-size:12.5px;color:var(--ink-2)}.tbl summary{cursor:pointer;color:var(--ink-3);list-style:none}
.tbl summary::-webkit-details-marker{display:none}.tbl summary::before{content:"+ "}.tbl[open] summary::before{content:"– "}
.tbl table{border-collapse:collapse;margin-top:10px;width:100%}.tbl th{font-weight:550;color:var(--ink-3)}
.tbl th,.tbl td{text-align:left;padding:5px 12px 5px 0;border-bottom:1px solid var(--rule);font-variant-numeric:tabular-nums}
footer{margin-top:48px;padding-top:18px;border-top:1px solid var(--rule);color:var(--ink-3);font-size:12px}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--rule);border-radius:8px;
padding:7px 11px;font-size:12px;box-shadow:0 6px 20px rgba(0,0,0,.10);display:none;z-index:10}
#tip b{display:block;font-size:14px;font-variant-numeric:tabular-nums}#tip span{color:var(--ink-2)}
@media (max-width:680px){h1{font-size:27px}.tiles{grid-template-columns:1fr}.tile+.tile{border-left:0;border-top:1px solid var(--rule);padding-left:0}}
</style></head>
<body><div class="viz-root"><div class="wrap">
<header><div class="eyebrow">S1Bench · performance and speed</div>
<h1>lev: a 4B open System One model, against Jev and the open field</h1>
<p class="lede">Qwen3.5-4B with a LoRA adapter, a candidate-path head and fitted temperatures. Scored on the same S1Bench task files as TypeSafe's Jev, and placed against every model on the public S1Bench board.</p>
<div class="tiles">{{TILES}}</div></header>
{{SECTIONS}}
<footer>Sources: {{PROVENANCE}}</footer>
</div></div>
<div id="tip" role="status"><span></span><b></b></div>
<script>
const tip=document.getElementById('tip'),tb=tip.querySelector('b'),ts=tip.querySelector('span');
function show(label,value,x,y){ts.textContent=label;tb.textContent=value;tip.style.display='block';
 const r=tip.getBoundingClientRect();tip.style.left=Math.min(x+14,innerWidth-r.width-8)+'px';tip.style.top=Math.max(8,y-r.height-12)+'px';}
function hide(){tip.style.display='none';}
document.querySelectorAll('[data-tip]').forEach(m=>{const [l,v]=m.dataset.tip.split('|');
 m.addEventListener('pointermove',e=>show(l,v,e.clientX,e.clientY));m.addEventListener('pointerleave',hide);
 m.addEventListener('focus',()=>{const b=m.getBoundingClientRect();show(l,v,b.right,b.top);});m.addEventListener('blur',hide);});
</script></body></html>
"""

if __name__ == "__main__":
    main()
