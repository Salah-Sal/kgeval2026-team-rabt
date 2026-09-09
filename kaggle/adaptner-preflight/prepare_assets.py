#!/usr/bin/env python
"""Whitelist-copy the Kaggle input bundle into a gitignored staging dir.

Never point a Kaggle upload at the working tree: only the paths listed in
WHITELIST are copied, junk files are excluded, and a sha256 manifest is written
into the bundle for kernel-side verification.

  uv run python kaggle/adaptner-preflight/prepare_assets.py --clean
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_STAGING = REPO / "output/staging/kgeval2026-bundle"

DATASET_ID = "salah1992/kgeval2026-bundle"
DATASET_TITLE = "KGEval 2026 bundle (private)"

# (source relative to repo root, destination relative to staging root)
WHITELIST: list[tuple[str, str]] = [
    ("resources/task-specific-resources/codebench/datasets/Wojood", "datasets/Wojood"),
    ("resources/task-specific-resources/codebench/datasets/WojoodRelations", "datasets/WojoodRelations"),
    ("resources/task-specific-resources/codebench/datasets/dev-konooz", "datasets/dev-konooz"),
    ("resources/task-specific-resources/codebench/datasets/blinded-test-data", "datasets/blinded-test-data"),
    ("resources/task-specific-resources/codebench/datasets/re-test", "datasets/re-test"),
    ("src/kgeval", "code/src/kgeval"),
]

EXCLUDED_DIRS = {"__pycache__", ".pytest_cache"}


def _ignore(_dir: str, names: list[str]) -> set[str]:
    return {
        n
        for n in names
        if n.startswith("~$") or n == ".DS_Store" or n in EXCLUDED_DIRS
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", default=str(DEFAULT_STAGING))
    parser.add_argument("--clean", action="store_true", help="remove existing staging first")
    args = parser.parse_args()

    staging = Path(args.staging)
    missing = [src for src, _ in WHITELIST if not (REPO / src).exists()]
    if missing:
        raise SystemExit(f"missing whitelist sources (run ./fetch-data.sh?): {missing}")

    if args.clean and staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    for src, dst in WHITELIST:
        shutil.copytree(REPO / src, staging / dst, ignore=_ignore, dirs_exist_ok=True)

    (staging / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": DATASET_TITLE, "id": DATASET_ID, "licenses": [{"name": "other"}]},
            indent=2,
        )
        + "\n"
    )

    total = 0
    rows = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "bundle-manifest.tsv":
            size = path.stat().st_size
            total += size
            rows.append(f"{path.relative_to(staging)}\t{size}\t{sha256(path)}")
    (staging / "bundle-manifest.tsv").write_text(
        "path\tbytes\tsha256\n" + "\n".join(rows) + "\n"
    )

    print(f"staged {len(rows)} files, {total / 1e6:.1f} MB -> {staging}")
    print(f"dataset id: {DATASET_ID} (PRIVATE — licensed competition data)")


if __name__ == "__main__":
    main()
