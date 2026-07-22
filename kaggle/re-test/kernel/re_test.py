#!/usr/bin/env python
"""KGEval 2026 — RE blind test inference (design §4.5).

Loads the trained marker classifier from the attached baseline kernel output,
types the test pairs via the same sentence_id→gold→lexicon chain used in
training, predicts with the validated soft-constraint configuration
(penalty 3.0, no_relation masked), and exports the submission zip directly:
predictions.txt is `triple_id<TAB>label` only — no licensed sentence text —
so unlike AdaptNER the finished zip is itself a safe artifact.
"""

import glob
import json
import os
import sys
import time
from collections import Counter

OUT_DIR = "/kaggle/working"
CONSTRAINT_PENALTY = 3.0  # sweep optimum, re-baseline-v3 (0.9378 -> 0.9470)


def locate_bundle() -> str:
    hits = sorted(glob.glob("/kaggle/input/**/code/src/kgeval", recursive=True))
    if not hits:
        raise SystemExit("bundle with code/src/kgeval not found under /kaggle/input")
    src = os.path.dirname(hits[0])
    print(f"[env] kgeval source: {src}")
    sys.path.insert(0, src)
    return os.path.dirname(os.path.dirname(src))


BUNDLE = locate_bundle()


def locate_checkpoint() -> tuple[str, dict]:
    hits = sorted(glob.glob("/kaggle/input/**/best_re_model.pt", recursive=True))
    if not hits:
        raise SystemExit("best_re_model.pt not found — is the RE baseline kernel attached?")
    ckpt = hits[0]
    with open(os.path.join(os.path.dirname(ckpt), "re_history.json")) as f:
        payload = json.load(f)
    print(f"[ckpt] {ckpt} (best_val_micro_f1={payload.get('best_val_micro_f1')})")
    return ckpt, payload


def main() -> int:
    t0 = time.time()

    import torch
    from transformers import AutoModel, AutoTokenizer

    from kgeval.re_constraints import ConstraintTable, observed_pairs
    from kgeval.re_data import load_jsonl
    from kgeval.re_model import MarkerClassifier
    from kgeval.re_submission import validate_predictions, write_predictions
    from kgeval.re_train import build_examples, make_batches, predict
    from kgeval.re_typing import assign_types, build_lexicon, read_csv_nested
    from kgeval.splits import grouped_split

    ckpt, payload = locate_checkpoint()
    cfg = payload["config"]
    label_vocab = payload["label_vocab"]
    masked = payload["masked_label"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    encoder = AutoModel.from_pretrained(cfg["model_name"])
    model = MarkerClassifier(encoder, len(label_vocab), dropout=cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    # typing sources: identical to training (gold sentences from every Wojood
    # split + surface lexicon)
    train_records = load_jsonl(f"{BUNDLE}/datasets/WojoodRelations/train.jsonl")
    gsid_sentences = {}
    for split in ("train", "val", "test"):
        gsid_sentences.update(
            read_csv_nested(f"{BUNDLE}/datasets/Wojood/Wojood1_1_nested/{split}.csv")
        )
    lexicon = build_lexicon(gsid_sentences)
    print(f"[typing] {len(gsid_sentences)} Wojood sentences, lexicon {len(lexicon)} entries "
          f"({time.time() - t0:.0f}s)")

    # constraint table exactly as swept: observed pairs of the grouped-split
    # train portion (the configuration that scored 0.9470 on dev)
    train_recs, _val_recs, split_info = grouped_split(train_records)
    train_types, _ = assign_types(train_recs, gsid_sentences, lexicon)
    table = ConstraintTable(observed_pairs(train_recs, train_types))

    test_records = load_jsonl(f"{BUNDLE}/datasets/re-test/test.jsonl")
    test_types, test_tstats = assign_types(test_records, gsid_sentences, lexicon)
    test_ex, test_fb = build_examples(test_records, test_types)
    print(f"[typing] test {dict(test_tstats)} fallbacks {test_fb}")

    label_index = {l: i for i, l in enumerate(label_vocab)}
    label_index.setdefault("", 0)  # unlabeled test records; ids unused at predict time
    batches = make_batches(test_ex, tokenizer, label_index, cfg["batch_size"], cfg["max_len"])
    n_truncated = sum(b.n_truncated_markers for b in batches)

    preds = predict(model, batches, label_vocab, device, masked_label=masked,
                    constraints=table, penalty=CONSTRAINT_PENALTY)
    labels = [preds[str(rec["triple_id"])] for rec in test_records]

    zip_path = os.path.join(OUT_DIR, "re_submission.zip")
    write_predictions(test_records, labels, zip_path, label_whitelist=set(label_vocab))
    report = validate_predictions(zip_path, test_records, label_whitelist=set(label_vocab))
    print("[submission validation]\n" + report.pretty())

    result = {
        "checkpoint": os.path.basename(os.path.dirname(ckpt)),
        "constraint_penalty": CONSTRAINT_PENALTY,
        "split_info": split_info,
        "n_test": len(test_records),
        "typing_test": dict(test_tstats),
        "marker_fallbacks_test": test_fb,
        "n_truncated_markers": n_truncated,
        "label_histogram": dict(Counter(labels).most_common()),
        "n_distinct_labels": len(set(labels)),
        "submission_validation_pass": report.ok,
        "total_seconds": round(time.time() - t0, 1),
    }
    result["ok"] = report.ok and len(labels) == len(test_records)
    with open(os.path.join(OUT_DIR, "RE_TEST_COMPLETED.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
