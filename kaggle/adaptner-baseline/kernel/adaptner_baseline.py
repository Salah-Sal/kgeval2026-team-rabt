#!/usr/bin/env python
"""KGEval 2026 — AdaptNER baseline campaign (design §3.1–3.2).

AraBERTv02 + 21 BIO heads, A10 unified loss, early stopping on Wojood-val
span micro-F1, final regression numbers on Wojood test. Exports safe
artifacts only: metrics (CAMPAIGN_COMPLETED.json, history.json) and the
fine-tuned checkpoint (our weights) — no dataset text ever leaves the kernel.
"""

import glob
import json
import os
import sys
import time

SMOKE = False  # True → tiny subset + 2 epochs, pipeline check only
OUT_DIR = "/kaggle/working"


def locate_bundle() -> str:
    """2026 image mounts datasets at /kaggle/input/datasets/<owner>/<slug>;
    discover by glob so a layout change doesn't strand the kernel."""
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

    from kgeval import ner_data
    from kgeval.convert import convert_corpus
    from kgeval.ner_train import TrainConfig, evaluate, run_training
    from kgeval.wojood import read_nested_txt

    print(f"[env] torch {torch.__version__} transformers {transformers.__version__} "
          f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")

    nested = f"{BUNDLE}/datasets/Wojood/Wojood1_1_nested"
    train_docs, train_stats = convert_corpus(read_nested_txt(f"{nested}/train.txt"))
    val_docs, _ = convert_corpus(read_nested_txt(f"{nested}/val.txt"))
    test_docs, _ = convert_corpus(read_nested_txt(f"{nested}/test.txt"))
    print(f"[data] train {train_stats.n_sentences}/{train_stats.n_tokens} "
          f"val {len(val_docs)} test {len(test_docs)} sentences")

    cfg = TrainConfig(wall_limit_s=7.5 * 3600)
    if SMOKE:
        cfg.max_epochs = 2
        train_docs, val_docs, test_docs = train_docs[:256], val_docs[:64], test_docs[:64]

    model, tokenizer, result = run_training(train_docs, val_docs, cfg, OUT_DIR)

    device = next(model.parameters()).device
    test_batches = ner_data.make_batches(
        ner_data.build_examples(test_docs, tokenizer, cfg.max_len),
        tokenizer, cfg.batch_size, cfg.max_len,
    )
    test_score, _ = evaluate(model, test_batches, test_docs, device)
    print("[test] Wojood-test regression:\n" + test_score.pretty())

    result["smoke"] = SMOKE
    result["test_micro"] = test_score.micro
    result["test_per_type"] = {
        t: v for t, v in sorted(test_score.per_type.items())
    }
    result["test_repairs"] = test_score.n_pred_repairs
    result["total_seconds"] = round(time.time() - t0, 1)
    # SMOKE gates on reaching this line (pipeline mechanics), not model quality;
    # full runs gate on val F1 — published nested-Wojood F1 is ~0.88–0.92, so
    # below 0.85 means a broken run
    result["ok"] = True if SMOKE else result["best_val_f1"] > 0.85
    with open(os.path.join(OUT_DIR, "CAMPAIGN_COMPLETED.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
