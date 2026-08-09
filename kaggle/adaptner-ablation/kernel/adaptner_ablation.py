#!/usr/bin/env python
"""KGEval 2026 — AdaptNER loss-ablation campaign (E1–E4).

Quantifies each component of the A10 unified loss for the system paper
(Faisal's 2026-08-01 experiment grid): E1 CE only; E2 +Dice; E3 +Tversky;
E4 full hybrid without the variance penalty. E5 (hybrid + variance
penalty 5.0) is the submitted baseline (adaptner-baseline-v2: val 0.9208,
test 0.9240) and is not re-run. Setup otherwise identical to the
baseline: train on Wojood train, early stop on Wojood-val span micro-F1
(patience 5, <=40 epochs), seed 13, then Wojood-test regression numbers.
Exports safe artifacts only: per-experiment metrics, training history,
and checkpoints (our weights) — no dataset text ever leaves the kernel.
"""

import glob
import json
import os
import sys
import time

SMOKE = False  # True → tiny subset + 2 epochs, pipeline check only
OUT_DIR = "/kaggle/working"
SESSION_BUDGET_S = 11.0 * 3600  # margin under the 12 h session cap

EXPERIMENTS = [
    # (name, loss_lambdas = (CE, Dice, Tversky, Focal), var_penalty)
    ("E1", (1.0, 0.0, 0.0, 0.0), 0.0),
    ("E2", (0.5, 0.5, 0.0, 0.0), 0.0),
    ("E3", (0.4, 0.3, 0.3, 0.0), 0.0),
    ("E4", (0.4, 0.2, 0.2, 0.2), 0.0),
]

E5_REFERENCE = {
    "name": "E5",
    "note": "submitted single model (adaptner-baseline-v2); not re-run",
    "loss_lambdas": [0.4, 0.2, 0.2, 0.2],
    "var_penalty": 5.0,
    "best_val_f1": 0.9208,
    "test_f1": 0.9240,
}


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

    if SMOKE:
        train_docs, val_docs, test_docs = train_docs[:256], val_docs[:64], test_docs[:64]

    runs = []
    for name, lambdas, vp in EXPERIMENTS:
        remaining = SESSION_BUDGET_S - (time.time() - t0)
        if not SMOKE and remaining < 30 * 60:
            print(f"[{name}] SKIPPED — {remaining/60:.0f} min left in session budget")
            runs.append({"name": name, "loss_lambdas": list(lambdas),
                         "var_penalty": vp, "skipped": "session_budget"})
            continue

        cfg = TrainConfig(loss_lambdas=lambdas, var_penalty=vp)
        if SMOKE:
            cfg.max_epochs = 2
        else:
            cfg.wall_limit_s = remaining - 900  # export margin
        exp_dir = os.path.join(OUT_DIR, name)
        os.makedirs(exp_dir, exist_ok=True)
        print(f"[{name}] loss_lambdas={lambdas} var_penalty={vp}")

        model, tokenizer, result = run_training(train_docs, val_docs, cfg, exp_dir)

        device = next(model.parameters()).device
        test_batches = ner_data.make_batches(
            ner_data.build_examples(test_docs, tokenizer, cfg.max_len),
            tokenizer, cfg.batch_size, cfg.max_len,
        )
        test_score, _ = evaluate(model, test_batches, test_docs, device)
        test_p, test_r, test_f1 = test_score.micro

        best = next((h for h in result["history"]
                     if h["epoch"] == result["best_epoch"]), {})
        summary = {
            "name": name,
            "loss_lambdas": list(lambdas),
            "var_penalty": vp,
            "best_epoch": result["best_epoch"],
            "best_val_p": best.get("val_p"),
            "best_val_r": best.get("val_r"),
            "best_val_f1": result["best_val_f1"],
            "test_p": round(test_p, 5),
            "test_r": round(test_r, 5),
            "test_f1": round(test_f1, 5),
            "test_repairs": test_score.n_pred_repairs,
            "final_epoch": result["final_epoch"],
            "stopped": result["stopped"],
            "seconds": result["total_seconds"],
            # a variant may legitimately score below E5, but under 0.85 on
            # in-domain Wojood means a broken run, not a finding
            "suspect": (not SMOKE) and result["best_val_f1"] <= 0.85,
        }
        runs.append(summary)
        print(f"[{name}] done: " + json.dumps(summary))

        del model
        torch.cuda.empty_cache()

    attempted = [r for r in runs if "skipped" not in r]
    campaign = {
        "smoke": SMOKE,
        "e5_reference": E5_REFERENCE,
        "runs": runs,
        "n_attempted": len(attempted),
        "n_skipped": len(runs) - len(attempted),
        "total_seconds": round(time.time() - t0, 1),
    }
    # SMOKE gates on reaching this line with all 4 loop iterations executed;
    # full runs additionally require every experiment attempted and none suspect
    campaign["ok"] = (
        len(attempted) == len(EXPERIMENTS)
        and all(not r.get("suspect") for r in attempted)
    ) if not SMOKE else len(runs) == len(EXPERIMENTS)
    with open(os.path.join(OUT_DIR, "CAMPAIGN_COMPLETED.json"), "w") as f:
        json.dump(campaign, f, indent=2)
    print(json.dumps(campaign, indent=2))
    return 0 if campaign["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
