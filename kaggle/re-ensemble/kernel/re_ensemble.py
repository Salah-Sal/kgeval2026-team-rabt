#!/usr/bin/env python
"""KGEval 2026 — RE test-phase ensemble: 5-fold × seed logit averaging.

Five marker classifiers, each trained on 4/5 of WojoodRelations train
(grouped by sentence_id, seed 13+fold) with early stopping on its own
held-out fold — proper per-fold validation, unlike the single grouped split
of the baseline. Blind-test decoding averages the five models' logits, then
applies the soft domain/range constraint penalty and the no_relation mask.

Out-of-fold (OOF) predictions pooled over the five folds give an honest
full-train estimate of single-model quality and re-validate the constraint
penalty on ~7× more validation data than the baseline sweep.

Deviation from the v1 submission config, recorded here: the constraint
table is built from ALL training records (v1 used only the grouped-split
train portion) — a superset of observed type pairs, so strictly fewer
penalties fire at test time.

Safe exports only: re_submission_ens.zip (triple_id + label — no licensed
text), re_ens_logits.json (averaged logits + type pairs, enabling local
re-decoding at any penalty without a GPU), per-fold re_history.json,
RE_ENSEMBLE_COMPLETED.json.
"""

import glob
import json
import os
import sys
import time
from collections import Counter

OUT_DIR = "/kaggle/working"
K = 5
BASE_SEED = 13
CONSTRAINT_PENALTY = 3.0  # baseline sweep optimum; OOF sweep re-checks below


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

    from kgeval.re_constraints import ConstraintTable, observed_pairs
    from kgeval.re_data import detect_no_relation, label_counts, load_jsonl
    from kgeval.re_scoring import score as re_score
    from kgeval.re_submission import validate_predictions, write_predictions
    from kgeval.re_train import RETrainConfig, build_examples, make_batches, run_re_training
    from kgeval.re_typing import assign_types, build_lexicon, read_csv_nested
    from kgeval.splits import grouped_kfold

    print(f"[env] torch {torch.__version__} transformers {transformers.__version__} "
          f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = load_jsonl(f"{BUNDLE}/datasets/WojoodRelations/train.jsonl")
    label_vocab = sorted(label_counts(records))
    neg = detect_no_relation(set(label_vocab))
    label_index = {l: i for i, l in enumerate(label_vocab)}
    masked_id = label_vocab.index(neg)

    gsid_sentences = {}
    for split in ("train", "val", "test"):
        gsid_sentences.update(
            read_csv_nested(f"{BUNDLE}/datasets/Wojood/Wojood1_1_nested/{split}.csv")
        )
    lexicon = build_lexicon(gsid_sentences)
    print(f"[typing] {len(gsid_sentences)} Wojood sentences, lexicon {len(lexicon)} "
          f"entries ({time.time() - t0:.0f}s)")

    # constraint table from ALL training records (see module docstring)
    all_types, _ = assign_types(records, gsid_sentences, lexicon)
    table = ConstraintTable(observed_pairs(records, all_types))

    test_records = load_jsonl(f"{BUNDLE}/datasets/re-test/test.jsonl")
    test_types, test_tstats = assign_types(test_records, gsid_sentences, lexicon)
    test_ex, test_fb = build_examples(test_records, test_types)
    print(f"[typing] test {dict(test_tstats)} fallbacks {test_fb}")

    from transformers import AutoTokenizer
    base_cfg = RETrainConfig(wall_limit_s=1.5 * 3600)
    tokenizer = AutoTokenizer.from_pretrained(base_cfg.model_name)
    test_label_index = dict(label_index)
    test_label_index.setdefault("", 0)  # unlabeled test records; ids unused
    test_batches = make_batches(
        test_ex, tokenizer, test_label_index, base_cfg.batch_size, base_cfg.max_len
    )

    def predict_logits(model, batches):
        """triple_id → (raw logits row [41], (subj_type, obj_type)), CPU fp32."""
        model.eval()
        out = {}
        with torch.no_grad():
            for b in batches:
                logits = model(
                    b.enc["input_ids"].to(device), b.enc["attention_mask"].to(device),
                    b.subj_pos.to(device), b.obj_pos.to(device),
                )
                for tid, row, tp in zip(b.triple_ids, logits.float().cpu(), b.type_pairs):
                    out[tid] = (row, tp)
        return out

    def decode(logits_by_tid, penalty):
        preds = {}
        for tid, (row, (s, o)) in logits_by_tid.items():
            r = row.clone()
            if penalty:
                mask = torch.tensor(table.penalty_mask(label_vocab, s, o))
                r = r - penalty * mask.float()
            r[masked_id] = float("-inf")
            preds[tid] = label_vocab[int(r.argmax())]
        return preds

    folds = grouped_kfold(records, k=K, seed=BASE_SEED)
    result: dict = {"k": K, "folds": {}, "constraint_table": "all_train_records"}
    oof_logits: dict[str, tuple] = {}
    oof_gold: list[tuple[str, str]] = []
    test_logit_sum: dict[str, tuple] = {}

    for fi, (train_recs, val_recs) in enumerate(folds):
        cfg = RETrainConfig(**{**base_cfg.__dict__, "seed": BASE_SEED + fi})
        train_types_f, _ = assign_types(train_recs, gsid_sentences, lexicon)
        val_types_f, _ = assign_types(val_recs, gsid_sentences, lexicon)
        train_ex, _ = build_examples(train_recs, train_types_f)
        val_ex, _ = build_examples(val_recs, val_types_f)
        print(f"[fold {fi}] train {len(train_ex)} val {len(val_ex)} seed {cfg.seed}")
        model, _tok, train_result = run_re_training(
            train_ex, val_ex, label_vocab, cfg,
            os.path.join(OUT_DIR, f"fold{fi}"), tokenizer=tokenizer,
        )
        result["folds"][fi] = {
            "n_train": len(train_ex), "n_val": len(val_ex), "seed": cfg.seed,
            "best_epoch": train_result["best_epoch"],
            "best_val_micro_f1": train_result["best_val_micro_f1"],
            "stopped": train_result["stopped"],
            "train_seconds": train_result["total_seconds"],
        }

        val_batches = make_batches(val_ex, tokenizer, label_index,
                                   cfg.batch_size, cfg.max_len)
        oof_logits.update(predict_logits(model, val_batches))
        oof_gold.extend((ex.triple_id, ex.label) for ex in val_ex)

        for tid, (row, tp) in predict_logits(model, test_batches).items():
            if tid in test_logit_sum:
                test_logit_sum[tid] = (test_logit_sum[tid][0] + row, tp)
            else:
                test_logit_sum[tid] = (row, tp)

        del model
        torch.cuda.empty_cache()

    # ---- OOF: honest single-model estimate + penalty re-validation --------
    oof_positives = [(t, l) for t, l in oof_gold]
    sweep = {}
    for penalty in (0.0, 1.0, 2.0, 3.0, 4.0):
        preds = decode(oof_logits, penalty)
        s = re_score(oof_positives, preds, positives_only=True)
        sweep[str(penalty)] = {"micro_f1": round(s.micro_f1, 5),
                               "macro_f1": round(s.macro_f1, 5)}
        print(f"[oof sweep] penalty {penalty}: micro {s.micro_f1:.4f} "
              f"macro {s.macro_f1:.4f}")
    result["oof_penalty_sweep"] = sweep
    result["oof_micro_f1_at_3"] = sweep["3.0"]["micro_f1"]

    # ---- ensemble decode of the blind test --------------------------------
    test_avg = {tid: (row / K, tp) for tid, (row, tp) in test_logit_sum.items()}
    preds = decode(test_avg, CONSTRAINT_PENALTY)
    labels = [preds[str(rec["triple_id"])] for rec in test_records]

    zip_path = os.path.join(OUT_DIR, "re_submission_ens.zip")
    write_predictions(test_records, labels, zip_path, label_whitelist=set(label_vocab))
    report = validate_predictions(zip_path, test_records, label_whitelist=set(label_vocab))
    print("[submission validation]\n" + report.pretty())

    # averaged logits are model output over ids/types only — safe to export
    export = {
        "label_vocab": label_vocab,
        "masked_label": neg,
        "constraint_penalty_used": CONSTRAINT_PENALTY,
        "triple_ids": [str(rec["triple_id"]) for rec in test_records],
        "type_pairs": [list(test_avg[str(rec["triple_id"])][1]) for rec in test_records],
        "logits": [
            [round(float(x), 4) for x in test_avg[str(rec["triple_id"])][0]]
            for rec in test_records
        ],
    }
    # constraint table snapshot: inadmissible label indices per test type pair —
    # lets local tooling re-decode at any penalty without rebuilding the table
    distinct_pairs = sorted({tuple(tp) for _, tp in test_avg.values()})
    export["inadmissible_by_pair"] = {
        f"{s}|{o}": [i for i, flag in enumerate(table.penalty_mask(label_vocab, s, o))
                     if flag]
        for s, o in distinct_pairs
    }
    with open(os.path.join(OUT_DIR, "re_ens_logits.json"), "w") as f:
        json.dump(export, f, separators=(",", ":"), ensure_ascii=False)

    result["constraint_penalty"] = CONSTRAINT_PENALTY
    result["n_test"] = len(test_records)
    result["typing_test"] = dict(test_tstats)
    result["marker_fallbacks_test"] = test_fb
    result["label_histogram"] = dict(Counter(labels).most_common())
    result["n_distinct_labels"] = len(set(labels))
    result["submission_validation_pass"] = report.ok
    result["total_seconds"] = round(time.time() - t0, 1)
    folds_healthy = all(
        f["best_val_micro_f1"] >= 0.90 for f in result["folds"].values()
    )
    result["ok"] = (report.ok and len(labels) == len(test_records)
                    and folds_healthy and result["oof_micro_f1_at_3"] >= 0.93)
    with open(os.path.join(OUT_DIR, "RE_ENSEMBLE_COMPLETED.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != "label_histogram"},
                     indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
