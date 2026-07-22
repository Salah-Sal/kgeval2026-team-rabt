#!/usr/bin/env python
"""Assemble the AdaptNER Codabench zip from the Kaggle tags export.

The test kernel exports tags only (no licensed tokens); this script clones the
pristine local blind-test files, swaps in the predicted 21-tag rows, validates
the result independently, and cross-checks the kernel sentinel's stats. The
zip lands in output/submissions/ (gitignored — it contains Konooz text).

  uv run python scripts/assemble_adaptner_submission.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kgeval.adaptner_submission import validate_submission, write_submission  # noqa: E402

TEST_DIR = REPO / "data/blinded-test-data"
KERNEL_OUT = REPO / "output/adaptner-test"
OUT_ZIP = REPO / "output/submissions/adaptner_teamrabt_v1.zip"


def main() -> int:
    tags = json.loads((KERNEL_OUT / "adaptner_test_tags.json").read_text())
    sentinel = json.loads((KERNEL_OUT / "ADAPTNER_TEST_COMPLETED.json").read_text())
    if not sentinel.get("ok"):
        print("REFUSING: kernel sentinel not ok:", sentinel.get("ok"))
        return 1

    info = write_submission(TEST_DIR, tags, OUT_ZIP)
    print(f"[write] {info}")
    if info["token_lines"] != sentinel["rehearsal_token_lines"]:
        print(f"MISMATCH: local {info['token_lines']} token lines vs kernel "
              f"rehearsal {sentinel['rehearsal_token_lines']}")
        return 1

    report = validate_submission(OUT_ZIP, TEST_DIR)
    print("[validate]\n" + report.pretty())
    if not report.ok:
        return 1
    print(f"\nREADY: {OUT_ZIP}")
    print(f"  pred spans {sentinel['pred_spans_total']} over "
          f"{len(sentinel['domains'])} domains, rehearsal PASS in-kernel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
