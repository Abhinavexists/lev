"""Test workspace extras and documented commands from the repository root."""

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


def _doc_sources() -> list[Path]:
    sources = [ROOT / "README.md", ROOT / "Makefile", ROOT / "CONTRIBUTING.md"]
    return [p for p in sources + sorted((ROOT / "docs").glob("*.md")) if p.is_file()]


def _documented_values(pattern: str) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for src in _doc_sources():
        for value in re.findall(pattern, src.read_text()):
            found.setdefault(value, set()).add(src.name)
    return found


def _documented_extras() -> dict[str, set[str]]:
    return _documented_values(r"--extra[= ]([a-z][a-z0-9-]*)")


def test_root_forwards_every_member_extra():
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


def test_every_documented_backend_exists():
    _, choices = _inspect_cli("levbench")
    accepted = choices["--backend"]
    for value, where in sorted(_documented_values(r"--backend[= ]([a-z][a-z0-9-]*)").items()):
        assert value in accepted, (
            f"{sorted(where)} document `--backend {value}`, but the CLI accepts "
            f"only {sorted(accepted)}."
        )


CODE_SPANS = re.compile(r"```[a-z]*\n(.*?)```|`([^`\n]+)`", re.S)


def _documented_commands(tool: str) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    pattern = re.compile(rf"(?:^|\s)(?:uv run )?{tool} ([a-z][a-z0-9-]*)", re.M)
    for src in _doc_sources():
        text = src.read_text()
        code = (
            [text]
            if src.suffix != ".md"
            else [block or span for block, span in CODE_SPANS.findall(text)]
        )
        for chunk in code:
            for name in pattern.findall(chunk):
                found.setdefault(name, set()).add(src.name)
    return found


@pytest.mark.parametrize("tool", ["lev", "levbench"])
def test_every_documented_subcommand_is_registered(tool):
    registered, _ = _inspect_cli(tool)

    for name, where in sorted(_documented_commands(tool).items()):
        if name.startswith("-"):
            continue
        assert name in registered, (
            f"{sorted(where)} document `{tool} {name}`, but the CLI registers "
            f"only {sorted(registered)}."
        )
