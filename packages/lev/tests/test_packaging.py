"""Guards against the repo's own configuration drifting from its documentation.

Every documented command is run from the repo root, so an extra that exists only
on a member package is not reachable by the command the docs give you. That is a
real bug shipped once already: `docs/TRAINING.md` said `uv sync --extra serve`
while the root forwarded only `train`.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").is_file() and "uv.workspace" in (p / "pyproject.toml").read_text()
)


def _extras(pyproject: Path) -> set[str]:
    data = tomllib.loads(pyproject.read_text())
    return set(data.get("project", {}).get("optional-dependencies", {}))


def _documented_extras() -> dict[str, set[str]]:
    """Every `--extra <name>` appearing in docs, the README or the Makefile."""
    found: dict[str, set[str]] = {}
    sources = [ROOT / "README.md", ROOT / "Makefile", ROOT / "CONTRIBUTING.md"]
    sources += sorted((ROOT / "docs").glob("*.md"))
    for src in sources:
        if not src.is_file():
            continue
        for name in re.findall(r"--extra[= ]([a-z][a-z0-9-]*)", src.read_text()):
            found.setdefault(name, set()).add(src.name)
    return found


def test_root_forwards_every_member_extra():
    """A member extra unreachable from the root is invisible to every doc command."""
    root = _extras(ROOT / "pyproject.toml")
    for member in sorted((ROOT / "packages").glob("*/pyproject.toml")):
        missing = _extras(member) - root
        assert not missing, (
            f"{member.parent.name} defines extras {sorted(missing)} that the root "
            f"workspace does not forward. `uv sync --extra {sorted(missing)[0]}` "
            "fails from the repo root, which is where the docs run it."
        )


def test_every_documented_extra_exists():
    root = _extras(ROOT / "pyproject.toml")
    for name, where in sorted(_documented_extras().items()):
        assert name in root, (
            f"{sorted(where)} document `--extra {name}`, but the root workspace "
            f"defines only {sorted(root)}."
        )


@pytest.mark.parametrize("expected", ["train", "serve", "modal"])
def test_known_extras_are_present(expected):
    assert expected in _extras(ROOT / "pyproject.toml")


def test_every_documented_backend_exists():
    """A `--backend x` in the docs must be a choice the CLI actually accepts.

    Same failure shape as the missing `serve` extra: the docs tell you to run
    something the tool rejects.
    """
    cli = (ROOT / "packages" / "levbench" / "src" / "levbench" / "cli.py").read_text()
    valid = set(re.findall(r"choices=\[([^\]]+)\]", cli))
    accepted = {v.strip().strip("\"'") for group in valid for v in group.split(",")}

    documented: dict[str, set[str]] = {}
    sources = [ROOT / "README.md", ROOT / "Makefile"] + sorted((ROOT / "docs").glob("*.md"))
    for src in sources:
        if src.is_file():
            for name in re.findall(r"--backend[= ]([a-z][a-z0-9-]*)", src.read_text()):
                documented.setdefault(name, set()).add(src.name)

    for name, where in sorted(documented.items()):
        assert name in accepted, (
            f"{sorted(where)} document `--backend {name}`, but the CLI accepts "
            f"only {sorted(accepted)}."
        )
