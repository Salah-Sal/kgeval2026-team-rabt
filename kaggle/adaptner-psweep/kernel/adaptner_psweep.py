#!/usr/bin/env python
"""KGEval 2026 — AdaptNER variance-penalty sensitivity sweep (P1, P2, P10).

Camera-ready follow-up to reviewer WUYr (2026-09-04 rebuttal): the
submitted recipe fixes the variance-penalty weight at 5.0, taken unchanged
from El Mahdaouy et al. (2023), and the August ablation tested only its
presence (E5, 5.0) versus absence (E4, 0.0). This kernel trains the full
loss mixture with the weight at 1.0, 2.0 and 10.0 under the E1–E5 setup:
Wojood train, early stop on Wojood-val span micro-F1 (patience 5,
<=40 epochs), seed 13, then Wojood val/test micro and macro-F1 with
per-type counts, so the run feeds Appendix H directly without a chained
eval kernel. Never touches Konooz or Codabench. Exports safe artifacts
only: metrics, training history, and checkpoints (our weights) — no
dataset text leaves the kernel.
"""

import glob
import json
import os
import sys
import time

SMOKE = False  # True → tiny subset + 2 epochs, pipeline check only (v1 passed 2026-09-04)
OUT_DIR = "/kaggle/working"
SESSION_BUDGET_S = 11.0 * 3600  # margin under the 12 h session cap

FULL_MIX = (0.4, 0.2, 0.2, 0.2)  # (CE, Dice, Tversky, Focal), as E4/E5

# Bracketing extremes first so a session-budget skip costs the middle point.
EXPERIMENTS = [
    # (name, loss_lambdas, var_penalty)
    ("P1", FULL_MIX, 1.0),
    ("P10", FULL_MIX, 10.0),
    ("P2", FULL_MIX, 2.0),
]

REFERENCE = [
    # from the August 3 ablation (adaptner-ablation-v1 / -eval-v1) and the
    # submitted baseline (adaptner-baseline-v2); not re-run here
    {"name": "E4", "var_penalty": 0.0, "val_micro_f1": 0.9202, "val_macro_f1": 0.7201,
     "test_micro_f1": 0.9240, "test_macro_f1": 0.7226},
    {"name": "E5", "var_penalty": 5.0, "val_micro_f1": 0.9208, "val_macro_f1": 0.8324,
     "test_micro_f1": 0.9240, "test_macro_f1": 0.8462,
     "note": "submitted single model (adaptner-baseline-v2)"},
]


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


def macro_f1(per_type: dict) -> float:
    """Mean per-type F1 over types present in gold (tp+fn > 0); identical to
    the ablation eval kernel so the numbers are comparable with E1–E5."""
    f1s = []
    for tp, fp, fn in per_type.values():
        if tp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def split_metrics(score) -> dict:
    p, r, f1 = score.micro
    return {
        "micro_p": round(p, 5),
        "micro_r": round(r, 5),
        "micro_f1": round(f1, 5),
        "macro_f1": round(macro_f1(score.per_type), 5),
        "repairs": score.n_pred_repairs,
        "per_type": {t: list(v) for t, v in sorted(score.per_type.items())},
    }


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

        # run_training returns the best-val checkpoint already loaded
        model, tokenizer, result = run_training(train_docs, val_docs, cfg, exp_dir)

        device = next(model.parameters()).device
        metrics = {}
        for split, docs in (("val", val_docs), ("test", test_docs)):
            batches = ner_data.make_batches(
                ner_data.build_examples(docs, tokenizer, cfg.max_len),
                tokenizer, cfg.batch_size, cfg.max_len,
            )
            score, _ = evaluate(model, batches, docs, device)
            metrics[split] = split_metrics(score)

        summary = {
            "name": name,
            "loss_lambdas": list(lambdas),
            "var_penalty": vp,
            "best_epoch": result["best_epoch"],
            "best_val_f1": result["best_val_f1"],
            "val": metrics["val"],
            "test": metrics["test"],
            "final_epoch": result["final_epoch"],
            "stopped": result["stopped"],
            "seconds": result["total_seconds"],
            # a variant may legitimately score below E5, but under 0.85 on
            # in-domain Wojood means a broken run, not a finding
            "suspect": (not SMOKE) and result["best_val_f1"] <= 0.85,
        }
        runs.append(summary)
        print(f"[{name}] done: val micro {metrics['val']['micro_f1']} macro "
              f"{metrics['val']['macro_f1']} | test micro {metrics['test']['micro_f1']} "
              f"macro {metrics['test']['macro_f1']} | epochs {result['final_epoch']+1} "
              f"best {result['best_epoch']} {result['stopped']} {result['total_seconds']:.0f}s")

        del model
        torch.cuda.empty_cache()

    attempted = [r for r in runs if "skipped" not in r]
    campaign = {
        "smoke": SMOKE,
        "reference": REFERENCE,
        "runs": runs,
        "n_attempted": len(attempted),
        "n_skipped": len(runs) - len(attempted),
        "total_seconds": round(time.time() - t0, 1),
    }
    # SMOKE gates on reaching this line with all loop iterations executed;
    # full runs additionally require every experiment attempted and none suspect
    campaign["ok"] = (
        len(attempted) == len(EXPERIMENTS)
        and all(not r.get("suspect") for r in attempted)
    ) if not SMOKE else len(runs) == len(EXPERIMENTS)
    with open(os.path.join(OUT_DIR, "CAMPAIGN_COMPLETED.json"), "w") as f:
        json.dump(campaign, f, indent=2)
    print(json.dumps({k: v for k, v in campaign.items() if k != "runs"}, indent=2))
    return 0 if campaign["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
