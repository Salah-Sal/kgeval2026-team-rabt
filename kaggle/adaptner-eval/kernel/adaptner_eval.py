#!/usr/bin/env python
"""KGEval 2026 — run a trained AdaptNER checkpoint over dev-Konooz.

Dev-Konooz carries all-O placeholder tags (design doc appendix), so there is
no score here. What this kernel proves instead:
  1. prediction statistics on genuine Konooz text per domain/type — a
     collapse detector (a healthy Wojood-trained model should tag plenty of
     entities; near-zero predicted spans means something is broken), and
  2. the submission writer + independent validator running against the dev
     reference — the exact chain the blind test submission will use.
The rehearsal zip contains Konooz text, so it is deleted in-kernel; only
metrics leave. Attach the baseline kernel's output via kernel_sources; the
checkpoint is discovered by glob and the model rebuilt from history.json.
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


BUNDLE = locate_bundle()


def locate_checkpoint() -> tuple[str, dict]:
    hits = sorted(glob.glob("/kaggle/input/**/best_model.pt", recursive=True))
    if not hits:
        raise SystemExit("best_model.pt not found — is the baseline kernel attached?")
    ckpt = hits[0]
    history = os.path.join(os.path.dirname(ckpt), "history.json")
    with open(history) as f:
        payload = json.load(f)
    config = payload["config"]
    print(f"[ckpt] {ckpt} (head={config.get('head', 'multi')}, "
          f"best_val_f1={payload.get('best_val_f1')})")
    return ckpt, config


def main() -> int:
    t0 = time.time()

    import torch
    from transformers import AutoModel, AutoTokenizer

    from kgeval import ner_data
    from kgeval.adaptner_submission import validate_submission, write_submission
    from kgeval.konooz import domain_files, read_column_file
    from kgeval.ner_model import MultiHeadTagger, SigmoidTagger
    from kgeval.ner_train import predict_docs
    from kgeval.spans import spans_from_rows

    ckpt, cfg = locate_checkpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    encoder = AutoModel.from_pretrained(cfg["model_name"])
    head = cfg.get("head", "multi")
    if head == "multi":
        model = MultiHeadTagger(
            encoder, dropout=cfg["dropout"], loss_lambdas=tuple(cfg["loss_lambdas"]),
            tversky_alpha=cfg["tversky_alpha"], focal_gamma=cfg["focal_gamma"],
            var_penalty=cfg["var_penalty"],
        )
    else:
        model = SigmoidTagger(encoder, dropout=cfg["dropout"])
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device)

    result: dict = {"checkpoint": os.path.basename(os.path.dirname(ckpt)),
                    "head": head, "domains": {}}
    tags_by_domain: dict[str, list[list[str]]] = {}
    type_counts: dict[str, int] = {}
    total_spans = 0
    total_repairs = 0
    for path in domain_files(f"{BUNDLE}/datasets/dev-konooz"):
        cf = read_column_file(path)
        docs_bare = [[[tok] for tok in sent] for sent in cf.sentences]
        batches = ner_data.make_batches(
            ner_data.build_examples(docs_bare, tokenizer, cfg["max_len"]),
            tokenizer, cfg["batch_size"], cfg["max_len"],
        )
        pred = predict_docs(model, batches, [len(s) for s in cf.sentences], device)
        n_spans = 0
        repairs = 0
        for rows in pred:
            res = spans_from_rows(rows)
            n_spans += len(res.spans)
            repairs += res.n_repairs
            for _s, _e, typ in res.spans:
                type_counts[typ] = type_counts.get(typ, 0) + 1
        result["domains"][path.stem] = {
            "tokens": cf.n_tokens, "pred_spans": n_spans, "repairs": repairs,
        }
        print(f"[dev-konooz] {path.stem}: {n_spans} predicted spans "
              f"({cf.n_tokens} tokens, {repairs} repairs)")
        total_spans += n_spans
        total_repairs += repairs
        tags_by_domain[path.stem] = [row for rows in pred for row in rows]

    result["pred_spans_total"] = total_spans
    result["repairs_total"] = total_repairs
    result["pred_spans_by_type"] = dict(sorted(type_counts.items()))
    print(f"[dev-konooz] total predicted spans {total_spans} "
          f"(repairs {total_repairs}) by type {result['pred_spans_by_type']}")

    # submission-chain rehearsal on the dev reference; zip never leaves the kernel
    zip_path = os.path.join(OUT_DIR, "rehearsal.zip")
    write_submission(f"{BUNDLE}/datasets/dev-konooz", tags_by_domain, zip_path)
    report = validate_submission(zip_path, f"{BUNDLE}/datasets/dev-konooz")
    print("[submission rehearsal]\n" + report.pretty())
    result["submission_rehearsal_pass"] = report.ok
    os.remove(zip_path)

    result["total_seconds"] = round(time.time() - t0, 1)
    # collapse floor: dev-Konooz has ~3.2K tokens of entity-rich prose; a
    # healthy model should find at least ~100 entities across 10 domains
    result["ok"] = result["submission_rehearsal_pass"] and total_spans >= 100
    with open(os.path.join(OUT_DIR, "ADAPTNER_EVAL_COMPLETED.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
