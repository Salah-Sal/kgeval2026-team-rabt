#!/usr/bin/env python
"""KGEval 2026 — RE typed-marker baseline (design §4.1–4.2, §4.5).

AraBERTv02 + Zhou&Chen punctuation typed markers, FFNN over the two
start-marker states, 41-way training with no_relation masked at inference.
Entity types via sentence_id join → gold Wojood entities (all splits) →
lexicon → UNK. Exports safe artifacts only: metrics + checkpoint.
"""

import glob
import json
import os
import sys
import time

SMOKE = False  # True → tiny subset + 2 epochs, pipeline check only
OUT_DIR = "/kaggle/working"


def locate_bundle() -> str:
    hits = sorted(glob.glob("/kaggle/input/**/code/src/kgeval", recursive=True))
    if not hits:
        raise SystemExit("bundle with code/src/kgeval not found under /kaggle/input")
    src = os.path.dirname(hits[0])
    print(f"[env] kgeval source: {src}")
    sys.path.insert(0, src)
    return os.path.dirname(os.path.dirname(src))


BUNDLE = locate_bundle()


def main() -> int:
    t0 = time.time()

    import torch
    import transformers

    from kgeval.re_data import label_counts, load_jsonl
    from kgeval.re_train import RETrainConfig, build_examples, run_re_training
    from kgeval.re_typing import assign_types, build_lexicon, read_csv_nested
    from kgeval.splits import grouped_split

    print(f"[env] torch {torch.__version__} transformers {transformers.__version__} "
          f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")

    records = load_jsonl(f"{BUNDLE}/datasets/WojoodRelations/train.jsonl")
    label_vocab = sorted(label_counts(records))  # all 41, split-independent
    train_recs, val_recs, split_info = grouped_split(records)
    print(f"[data] {len(records)} records, {len(label_vocab)} labels, split {split_info}")

    # typing sources: gold entities from every Wojood split (typing only — the
    # NER holdout discipline is unaffected)
    gsid_sentences = {}
    for split in ("train", "val", "test"):
        gsid_sentences.update(
            read_csv_nested(f"{BUNDLE}/datasets/Wojood/Wojood1_1_nested/{split}.csv")
        )
    lexicon = build_lexicon(gsid_sentences)
    print(f"[typing] {len(gsid_sentences)} Wojood sentences, lexicon {len(lexicon)} entries "
          f"({time.time() - t0:.0f}s)")

    train_types, train_tstats = assign_types(train_recs, gsid_sentences, lexicon)
    val_types, val_tstats = assign_types(val_recs, gsid_sentences, lexicon)
    train_ex, train_fb = build_examples(train_recs, train_types)
    val_ex, val_fb = build_examples(val_recs, val_types)
    print(f"[typing] train {dict(train_tstats)} fallbacks {train_fb}")
    print(f"[typing] val {dict(val_tstats)} fallbacks {val_fb}")

    cfg = RETrainConfig(wall_limit_s=3 * 3600)
    if SMOKE:
        cfg.max_epochs = 2
        train_ex, val_ex = train_ex[:512], val_ex[:128]

    model, tokenizer, result = run_re_training(train_ex, val_ex, label_vocab, cfg, OUT_DIR)

    # constraint-penalty sweep on dev (design §4.3): predict-only, tuned post hoc
    from kgeval.re_constraints import ConstraintTable, observed_pairs
    from kgeval.re_data import detect_no_relation
    from kgeval.re_scoring import score as re_score
    from kgeval.re_train import make_batches, predict

    table = ConstraintTable(observed_pairs(train_recs, train_types))
    label_index = {l: i for i, l in enumerate(label_vocab)}
    neg = detect_no_relation(set(label_vocab))
    val_batches = make_batches(val_ex, tokenizer, label_index, cfg.batch_size, cfg.max_len)
    val_gold = [(ex.triple_id, ex.label) for ex in val_ex]
    device = next(model.parameters()).device
    sweep = {}
    for penalty in (0.0, 1.0, 2.0, 3.0, 4.0):
        preds = predict(model, val_batches, label_vocab, device, masked_label=neg,
                        constraints=table, penalty=penalty)
        s = re_score(val_gold, preds, positives_only=True)
        sweep[str(penalty)] = {"micro_f1": round(s.micro_f1, 5),
                               "macro_f1": round(s.macro_f1, 5)}
        print(f"[constraints] penalty {penalty}: micro {s.micro_f1:.4f} "
              f"macro {s.macro_f1:.4f}")
    result["constraint_penalty_sweep"] = sweep

    result["smoke"] = SMOKE
    result["n_labels"] = len(label_vocab)
    result["split_info"] = split_info
    result["typing_train"] = dict(train_tstats)
    result["typing_val"] = dict(val_tstats)
    result["marker_fallbacks_train"] = train_fb
    result["marker_fallbacks_val"] = val_fb
    result["total_seconds"] = round(time.time() - t0, 1)
    # design expects ≥92 (RIFRE 92.89 on this task); below 0.85 = broken run
    result["ok"] = True if SMOKE else result["best_val_micro_f1"] > 0.85
    with open(os.path.join(OUT_DIR, "RE_CAMPAIGN_COMPLETED.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != "history"},
                     indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
