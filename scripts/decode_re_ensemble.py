#!/usr/bin/env python
"""Decode the RE ensemble's averaged logits into a Codabench zip locally.

The ensemble kernel exports re_ens_logits.json (averaged logits, type pairs,
and the constraint table's inadmissible-label indices per pair — no licensed
text). This script re-decodes at any penalty and writes a versioned zip,
validated against the local test.jsonl.

  uv run python scripts/decode_re_ensemble.py              # kernel's penalty
  uv run python scripts/decode_re_ensemble.py --penalty 2  # alternative decode
  uv run python scripts/decode_re_ensemble.py --allow-no-relation
      # full 41-class decode (no_relation logit not masked), as the organizers
      # requested on 2026-09-04; writes re_wasl_41class_<tag>.zip and reports
      # how many predictions differ from the masked decode
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
    ap.add_argument("--allow-no-relation", action="store_true",
                    help="do not mask the no_relation logit (41-class decode)")
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

    labels, masked_labels = [], []
    for logits, (s, o) in zip(export["logits"], export["type_pairs"]):
        row = list(logits)
        if penalty:
            for i in inadmissible[f"{s}|{o}"]:
                row[i] -= penalty
        unmasked = vocab[max(range(len(vocab)), key=row.__getitem__)]
        row[masked_id] = float("-inf")
        masked = vocab[max(range(len(vocab)), key=row.__getitem__)]
        masked_labels.append(masked)
        labels.append(unmasked if args.allow_no_relation else masked)

    test_records = load_jsonl(TEST_JSONL)
    ids = [str(r["triple_id"]) for r in test_records]
    if ids != export["triple_ids"]:
        print("MISMATCH: local test.jsonl order differs from kernel export")
        return 1

    tag = f"p{penalty:g}".replace(".", "_")
    stem = "re_wasl_41class" if args.allow_no_relation else "re_teamrabt_v2"
    out_zip = SUB_DIR / f"{stem}_{tag}.zip"
    write_predictions(test_records, labels, out_zip, label_whitelist=set(vocab),
                      allow_no_relation=args.allow_no_relation)
    report = validate_predictions(out_zip, test_records, label_whitelist=set(vocab),
                                  allow_no_relation=args.allow_no_relation)
    print("[validate]\n" + report.pretty())
    if not report.ok:
        return 1

    if penalty == export["constraint_penalty_used"]:
        # The masked decode must always reproduce the kernel's histogram, which
        # ties the 41-class file to the very same logits the official
        # submission came from.
        kernel_hist = sentinel["label_histogram"]
        local_hist = dict(Counter(masked_labels).most_common())
        if kernel_hist != local_hist:
            print("MISMATCH: masked label histogram differs from kernel decode")
            return 1
        print("[check] masked label histogram matches kernel decode exactly")

    if args.allow_no_relation:
        changed = Counter(m for m, u in zip(masked_labels, labels) if m != u)
        n_changed = sum(changed.values())
        n_neg = labels.count(export["masked_label"])
        print(f"[41-class] {n_neg} no_relation predictions ({n_neg / len(labels):.2%}); "
              f"{n_changed} predictions differ from the masked decode "
              f"(all changes are to no_relation: {n_changed == n_neg})")
        print("  replaced labels: " + ", ".join(f"{l} {c}" for l, c in changed.most_common(12)))

    print(f"\nREADY: {out_zip}")
    print(f"  penalty {penalty}, {len(labels)} labels, "
          f"{len(set(labels))} distinct, top: {Counter(labels).most_common(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
