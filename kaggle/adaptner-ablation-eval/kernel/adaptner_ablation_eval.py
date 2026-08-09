#!/usr/bin/env python
"""KGEval 2026 — per-type / macro-F1 eval for the loss-ablation checkpoints.

The ablation campaign exported only micro P/R/F1 per experiment, but the
imbalance-aware loss components (Dice/Tversky/Focal, variance penalty) are
motivated by rare types — macro-F1 and per-type counts are where their
contribution would show. This kernel chains the ablation kernel's output
(E1–E4 checkpoints) and the baseline kernel's output (E5 checkpoint) as
inputs and evaluates all five on Wojood val and test with per-type
(tp, fp, fn) counts. Exports metrics only (EVAL_COMPLETED.json); no
dataset text leaves the kernel.
"""

import glob
import json
import os
import sys
import time

OUT_DIR = "/kaggle/working"


def locate_bundle() -> str:
    hits = sorted(glob.glob("/kaggle/input/**/code/src/kgeval", recursive=True))
    if not hits:
        raise SystemExit("bundle with code/src/kgeval not found under /kaggle/input")
    src = os.path.dirname(hits[0])
    print(f"[env] kgeval source: {src}")
    sys.path.insert(0, src)
    return os.path.dirname(os.path.dirname(src))


def locate_checkpoints() -> dict[str, str]:
    """Map experiment name -> checkpoint path from chained kernel outputs."""
    ckpts = {}
    for p in sorted(glob.glob("/kaggle/input/**/best_model.pt", recursive=True)):
        parent = os.path.basename(os.path.dirname(p))
        if parent in ("E1", "E2", "E3", "E4"):
            ckpts[parent] = p
        elif "baseline" in p:
            ckpts["E5"] = p
    print(f"[ckpt] found: { {k: v for k, v in sorted(ckpts.items())} }")
    if not ckpts:
        for p in sorted(glob.glob("/kaggle/input/*/*", recursive=False)):
            print("  [tree]", p)
        raise SystemExit("no best_model.pt found in chained kernel outputs")
    return ckpts


BUNDLE = locate_bundle()


def macro_f1(per_type: dict) -> float:
    """Mean per-type F1 over types present in gold (tp+fn > 0)."""
    f1s = []
    for tp, fp, fn in per_type.values():
        if tp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def main() -> int:
    t0 = time.time()

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    from kgeval import ner_data
    from kgeval.convert import convert_corpus
    from kgeval.ner_model import MultiHeadTagger
    from kgeval.ner_train import TrainConfig, evaluate
    from kgeval.wojood import read_nested_txt

    print(f"[env] torch {torch.__version__} transformers {transformers.__version__} "
          f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")

    ckpts = locate_checkpoints()

    nested = f"{BUNDLE}/datasets/Wojood/Wojood1_1_nested"
    val_docs, _ = convert_corpus(read_nested_txt(f"{nested}/val.txt"))
    test_docs, _ = convert_corpus(read_nested_txt(f"{nested}/test.txt"))
    print(f"[data] val {len(val_docs)} test {len(test_docs)} sentences")

    cfg = TrainConfig()  # loss params are irrelevant for eval
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = MultiHeadTagger(AutoModel.from_pretrained(cfg.model_name)).to(device)

    batches = {}
    for split, docs in (("val", val_docs), ("test", test_docs)):
        batches[split] = ner_data.make_batches(
            ner_data.build_examples(docs, tokenizer, cfg.max_len),
            tokenizer, cfg.batch_size, cfg.max_len,
        )

    results = {}
    for name in sorted(ckpts):
        model.load_state_dict(
            torch.load(ckpts[name], map_location=device, weights_only=True)
        )
        entry = {}
        for split, docs in (("val", val_docs), ("test", test_docs)):
            score, _ = evaluate(model, batches[split], docs, device)
            p, r, f1 = score.micro
            entry[split] = {
                "micro_p": round(p, 5),
                "micro_r": round(r, 5),
                "micro_f1": round(f1, 5),
                "macro_f1": round(macro_f1(score.per_type), 5),
                "repairs": score.n_pred_repairs,
                "per_type": {t: list(v) for t, v in sorted(score.per_type.items())},
            }
        results[name] = entry
        print(f"[{name}] val micro {entry['val']['micro_f1']} macro "
              f"{entry['val']['macro_f1']} | test micro {entry['test']['micro_f1']} "
              f"macro {entry['test']['macro_f1']}")

    out = {
        "models_evaluated": sorted(results),
        "results": results,
        "total_seconds": round(time.time() - t0, 1),
        "ok": len(results) == 5,
    }
    with open(os.path.join(OUT_DIR, "EVAL_COMPLETED.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
