#!/usr/bin/env python
"""Assemble an AdaptNER ensemble-variant Codabench zip from the Kaggle export.

The ensemble kernel exports adaptner_ens_tags.json — tags per variant per
domain, no licensed tokens. This script clones the pristine local blind-test
files, swaps in the chosen variant's rows, validates independently, and
cross-checks the kernel sentinel. Optionally, `union4` adds the train-only
baseline model (adaptner-test export) as a fourth union member — computed
locally from tags via kgeval.spans.union_tag_rows.

  uv run python scripts/assemble_adaptner_ensemble.py union3
  uv run python scripts/assemble_adaptner_ensemble.py meanprob
  uv run python scripts/assemble_adaptner_ensemble.py union4   # 3 seeds + v1 model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kgeval.adaptner_submission import validate_submission, write_submission  # noqa: E402
from kgeval.konooz import domain_files, read_column_file  # noqa: E402
from kgeval.spans import spans_from_rows, union_tag_rows  # noqa: E402

TEST_DIR = REPO / "data/blinded-test-data"
ENS_OUT = REPO / "output/adaptner-ensemble"
V1_OUT = REPO / "output/adaptner-test"
SUB_DIR = REPO / "output/submissions"

SEEDS = ("seed13", "seed42", "seed77")


def resegment(flat_rows: list[list[str]], domain_path) -> list[list[list[str]]]:
    """Split a flat per-domain row list back into per-sentence row lists using
    the local reference file's sentence lengths."""
    cf = read_column_file(domain_path)
    out, i = [], 0
    for sent in cf.sentences:
        out.append(flat_rows[i : i + len(sent)])
        i += len(sent)
    if i != len(flat_rows):
        raise ValueError(f"{domain_path}: {len(flat_rows)} rows vs {i} reference tokens")
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    variant = sys.argv[1]

    tags_all = json.loads((ENS_OUT / "adaptner_ens_tags.json").read_text())
    sentinel = json.loads((ENS_OUT / "ADAPTNER_ENSEMBLE_COMPLETED.json").read_text())
    if not sentinel.get("ok"):
        print("REFUSING: ensemble sentinel not ok")
        return 1

    if variant in tags_all:
        flat = tags_all[variant]
        expected_lines = sentinel["variants"][variant]["rehearsal_token_lines"]
    elif variant == "union4":
        # 3 train+val seeds + the train-only v1 model, union with min_votes=1
        v1_tags = json.loads((V1_OUT / "adaptner_test_tags.json").read_text())
        flat = {}
        n_spans = 0
        for path in domain_files(TEST_DIR):
            dom = path.stem
            members = [resegment(tags_all[s][dom], path) for s in SEEDS]
            members.append(resegment(v1_tags[dom], path))
            merged = []
            for per_sentence in zip(*members):
                res = union_tag_rows(list(per_sentence), min_votes=1)
                merged.extend(res.rows)
                n_spans += len(res.spans)
            flat[dom] = merged
        expected_lines = sentinel["variants"]["union3"]["rehearsal_token_lines"]
        print(f"[union4] {n_spans} spans (union3 was "
              f"{sentinel['variants']['union3']['blind_spans']['total']})")
    else:
        print(f"unknown variant {variant!r}; kernel exports: {sorted(tags_all)} "
              "(plus local: union4)")
        return 2

    out_zip = SUB_DIR / f"adaptner_teamrabt_v2_{variant}.zip"
    info = write_submission(
        TEST_DIR, flat, out_zip, member_name=f"teamrabt_pred_v2_{variant}.txt"
    )
    print(f"[write] {info}")
    if info["token_lines"] != expected_lines:
        print(f"MISMATCH: local {info['token_lines']} token lines vs kernel "
              f"rehearsal {expected_lines}")
        return 1

    report = validate_submission(out_zip, TEST_DIR)
    print("[validate]\n" + report.pretty())
    if not report.ok:
        return 1

    total = sum(
        len(spans_from_rows(rows).spans)
        for dom in flat
        for rows in resegment(flat[dom], TEST_DIR / f"{dom}.txt")
    )
    print(f"\nREADY: {out_zip}")
    print(f"  variant {variant}: {total} predicted spans")
    return 0


if __name__ == "__main__":
    sys.exit(main())
