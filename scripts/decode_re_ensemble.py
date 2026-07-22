#!/usr/bin/env python
"""Decode the RE ensemble's averaged logits into a Codabench zip locally.

The ensemble kernel exports re_ens_logits.json (averaged logits, type pairs,
and the constraint table's inadmissible-label indices per pair — no licensed
text). This script re-decodes at any penalty and writes a versioned zip,
validated against the local test.jsonl.

  uv run python scripts/decode_re_ensemble.py              # kernel's penalty
  uv run python scripts/decode_re_ensemble.py --penalty 2  # alternative decode
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kgeval.re_data import load_jsonl  # noqa: E402
from kgeval.re_submission import validate_predictions, write_predictions  # noqa: E402

ENS_OUT = REPO / "output/re-ensemble"
TEST_JSONL = REPO / "data/re-test/test.jsonl"
SUB_DIR = REPO / "output/submissions"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalty", type=float, default=None,
                    help="constraint penalty (default: the kernel's)")
    args = ap.parse_args()

    export = json.loads((ENS_OUT / "re_ens_logits.json").read_text())
    sentinel = json.loads((ENS_OUT / "RE_ENSEMBLE_COMPLETED.json").read_text())
    if not sentinel.get("ok"):
        print("REFUSING: ensemble sentinel not ok")
        return 1

    vocab = export["label_vocab"]
    masked_id = vocab.index(export["masked_label"])
    penalty = export["constraint_penalty_used"] if args.penalty is None else args.penalty
    inadmissible = export["inadmissible_by_pair"]

    labels = []
    for logits, (s, o) in zip(export["logits"], export["type_pairs"]):
        row = list(logits)
        if penalty:
            for i in inadmissible[f"{s}|{o}"]:
                row[i] -= penalty
        row[masked_id] = float("-inf")
        labels.append(vocab[max(range(len(vocab)), key=row.__getitem__)])

    test_records = load_jsonl(TEST_JSONL)
    ids = [str(r["triple_id"]) for r in test_records]
    if ids != export["triple_ids"]:
        print("MISMATCH: local test.jsonl order differs from kernel export")
        return 1

    tag = f"p{penalty:g}".replace(".", "_")
    out_zip = SUB_DIR / f"re_teamrabt_v2_{tag}.zip"
    write_predictions(test_records, labels, out_zip, label_whitelist=set(vocab))
    report = validate_predictions(out_zip, test_records, label_whitelist=set(vocab))
    print("[validate]\n" + report.pretty())
    if not report.ok:
        return 1

    if penalty == export["constraint_penalty_used"]:
        kernel_hist = sentinel["label_histogram"]
        local_hist = dict(Counter(labels).most_common())
        if kernel_hist != local_hist:
            print("MISMATCH: label histogram differs from kernel decode")
            return 1
        print("[check] label histogram matches kernel decode exactly")

    print(f"\nREADY: {out_zip}")
    print(f"  penalty {penalty}, {len(labels)} labels, "
          f"{len(set(labels))} distinct, top: {Counter(labels).most_common(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
