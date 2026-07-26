"""Static checks for the Colab notebooks in notebooks/.

These guard the two failure modes that matter:

1. Drift — each notebook vendors kgeval modules as ``%%writefile`` cells so it
   runs standalone on Colab; the cell body must match ``src/kgeval`` byte for
   byte, or the notebook silently teaches old code.
2. Leakage — licensed task data (Wojood / WojoodRelations / Konooz) and local
   cache paths must never appear in a shareable notebook, in either source or
   output cells. Committed notebooks DO carry outputs (executed on Colab T4
   against the toy corpora — no licensed data), and every output must be
   error-free.

No GPU/torch needed: everything here is JSON + text checks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat
import pytest

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"
SRC = REPO / "src"

NOTEBOOKS = [
    "01-adaptner-nested-ner.ipynb",
    "02-relation-extraction.ipynb",
]

FORBIDDEN = [
    "resources/task-specific-resources",
    ".data-cache",
    "/kaggle/input",
    "kaggle-staging",
]

WRITEFILE_RE = re.compile(r"^%%writefile kgeval/([a-z_]+)\.py\n", re.MULTILINE)


def _load(name: str) -> dict:
    return json.loads((NB_DIR / name).read_text(encoding="utf-8"))


def _cells(name: str) -> list[dict]:
    return _load(name)["cells"]


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_notebook_validates(nb_name: str):
    nb = nbformat.read(NB_DIR / nb_name, as_version=4)
    nbformat.validate(nb)
    assert nb["cells"], "notebook has no cells"


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_notebook_outputs_have_no_errors(nb_name: str):
    """Committed notebooks carry executed outputs (toy-data Colab runs); every
    output must be a clean one — no tracebacks ship to readers."""
    n_errors = 0
    for cell in _cells(nb_name):
        if cell["cell_type"] != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                n_errors += 1
    assert n_errors == 0, f"{nb_name} has {n_errors} error outputs committed"


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_no_licensed_data_paths(nb_name: str):
    raw = json.dumps(_load(nb_name), ensure_ascii=False)
    for needle in FORBIDDEN:
        assert needle not in raw, f"{nb_name} mentions {needle!r}"


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_env_overrides_present(nb_name: str):
    """The config cell must expose the env overrides the local smoke run uses."""
    raw = json.dumps(_load(nb_name), ensure_ascii=False)
    assert "KGEVAL_MODEL" in raw
    assert "KGEVAL_FULL" in raw


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_vendored_modules_match_src(nb_name: str):
    """Every %%writefile kgeval/<mod>.py cell must equal src/kgeval/<mod>.py."""
    written: dict[str, str] = {}
    for cell in _cells(nb_name):
        if cell["cell_type"] != "code":
            continue
        src_text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        m = WRITEFILE_RE.match(src_text)
        if not m:
            continue
        module = m.group(1)
        assert module not in written, f"{module} written twice"
        written[module] = src_text[m.end():]
    assert written, f"{nb_name} vendors no kgeval modules"
    for module, body in written.items():
        expected = (SRC / "kgeval" / f"{module}.py").read_text(encoding="utf-8")
        assert body == expected, (
            f"{nb_name}: vendored kgeval/{module}.py differs from src "
            f"(re-vendor by copying the current file into the cell)"
        )
