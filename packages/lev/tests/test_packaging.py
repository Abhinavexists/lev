"""Guards against the repo's configuration drifting from its documentation.

Documented commands run from the repo root, so an extra defined only on a member
package is unreachable (`docs/TRAINING.md` once documented `uv sync --extra serve`
while the root forwarded only `train`).
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


def _inspect_cli(tool: str) -> tuple[set[str], dict[str, set[str]]]:
    """Build the real parser and read back its subcommands and option choices.

    Introspection, not a regex over the source, so it checks what the parser
    actually accepts.
    """
    import argparse
    import importlib

    cli = importlib.import_module(f"{tool}.cli")
    subcommands: set[str] = set()
    choices: dict[str, set[str]] = {}

    def collect(parser) -> None:
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                subcommands.update(action.choices)
                for child in action.choices.values():
                    collect(child)
            elif action.choices:
                for option in action.option_strings:
                    choices.setdefault(option, set()).update(action.choices)

    real_parse = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        collect(self)
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        with pytest.raises(SystemExit):
            cli.main([])
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    return subcommands, choices


def _documented_flag_values(flag: str) -> dict[str, set[str]]:
    """Every `--flag value` the docs name, and which file named it."""
    found: dict[str, set[str]] = {}
    sources = [ROOT / "README.md", ROOT / "Makefile"] + sorted((ROOT / "docs").glob("*.md"))
    for src in sources:
        if src.is_file():
            for value in re.findall(rf"{flag}[= ]([a-z][a-z0-9-]*)", src.read_text()):
                found.setdefault(value, set()).add(src.name)
    return found


def test_every_documented_backend_exists():
    """A `--backend x` in the docs must be a choice the CLI accepts."""
    _, choices = _inspect_cli("levbench")
    accepted = choices["--backend"]
    for value, where in sorted(_documented_flag_values("--backend").items()):
        assert value in accepted, (
            f"{sorted(where)} document `--backend {value}`, but the CLI accepts "
            f"only {sorted(accepted)}."
        )


CODE_SPANS = re.compile(r"```[a-z]*\n(.*?)```|`([^`\n]+)`", re.S)


def _documented_commands(tool: str) -> dict[str, set[str]]:
    """Every `uv run <tool> <sub>` the docs tell you to run.

    Only inside fenced blocks and inline backticks; prose like "levbench must
    not import lev" would otherwise match.
    """
    found: dict[str, set[str]] = {}
    sources = [ROOT / "README.md", ROOT / "Makefile", ROOT / "CONTRIBUTING.md"]
    sources += sorted((ROOT / "docs").glob("*.md"))
    pattern = re.compile(rf"(?:^|\s)(?:uv run )?{tool} ([a-z][a-z0-9-]*)", re.M)
    for src in sources:
        if not src.is_file():
            continue
        text = src.read_text()
        # A Makefile is all code apart from its `## help text`, which is prose
        # and mentions commands the way a sentence does.
        code = (
            [re.sub(r"##.*", "", text)]
            if src.suffix != ".md"
            else [block or span for block, span in CODE_SPANS.findall(text)]
        )
        for chunk in code:
            for name in pattern.findall(chunk):
                found.setdefault(name, set()).add(src.name)
    return found


@pytest.mark.parametrize("tool", ["lev", "levbench"])
def test_every_documented_subcommand_is_registered(tool):
    """Every command the docs name has to parse."""
    registered, _ = _inspect_cli(tool)

    for name, where in sorted(_documented_commands(tool).items()):
        if name.startswith("-"):
            continue
        assert name in registered, (
            f"{sorted(where)} document `{tool} {name}`, but the CLI registers "
            f"only {sorted(registered)}."
        )
