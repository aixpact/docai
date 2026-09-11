"""Smoke tests: every cookbook example's main() runs without error.

Loaded by file path (rather than imported as a package) since `cookbook/`
is example code, not part of the installed `docai_poc` package. Each
script imports its sibling `_fake_client` module via a bare `import
_fake_client`, so the cookbook directory is added to `sys.path` for the
duration of these tests.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

COOKBOOK_DIR = Path(__file__).parent.parent / "cookbook"
SCRIPTS = sorted(COOKBOOK_DIR.glob("[0-9]*.py"))


@pytest.fixture(autouse=True)
def _cookbook_on_path() -> Iterator[None]:
    sys.path.insert(0, str(COOKBOOK_DIR))
    try:
        yield
    finally:
        sys.path.remove(str(COOKBOOK_DIR))


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", SCRIPTS, ids=[s.name for s in SCRIPTS])
def test_cookbook_script_runs(script: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_module(script)
    module.main()
    assert capsys.readouterr().out.strip() != ""


def test_cookbook_has_scripts() -> None:
    assert len(SCRIPTS) >= 6
