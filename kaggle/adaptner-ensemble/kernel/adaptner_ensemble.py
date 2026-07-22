#!/usr/bin/env python
"""KGEval 2026 — AdaptNER test-phase ensemble campaign.

Three seeds trained on Wojood train+val combined (the organizers' ruling
allows training on train+dev only; Konooz never touches training or
selection) with a FIXED 29-epoch schedule — epoch 28 (0-indexed) was the
val-selected best of the train-only baseline, and its val curve is flat
26–33, so fixed-epoch carries low risk. select="last": no early stopping,
no checkpoint selection.

Per seed: full Wojood-test regression eval (in-domain sanity, NOT model
selection), then blind Konooz inference collecting per-seed tags and
accumulated softmax probabilities. Ensemble variants built in-kernel:
  seed13/seed42/seed77 — single models
  meanprob            — argmax of seed-averaged softmax probs
  union3              — span-level union (min_votes=1, the recall lever)
  vote2               — majority vote (min_votes=2, the precision lever)
Every variant is scored on Wojood-test (gold available) and rehearsed
through write_submission+validate_submission against the bundled test
reference; rehearsal zips are deleted (they contain licensed tokens).

Safe exports only: adaptner_ens_tags.json (tags per variant per domain,
zero Konooz text), per-seed final checkpoints + history.json (our weights),
ADAPTNER_ENSEMBLE_COMPLETED.json (metrics + rehearsal verdicts).
"""

import glob
import json
import os
import sys
import time

SEEDS = (13, 42, 77)
EPOCHS = 29  # fixed schedule: 0..28, baseline best_epoch 28 (0-indexed)
MONITOR_SENTENCES = 512  # per-epoch curve only; select="last" ignores it
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

    from kgeval import ner_data
    from kgeval.adaptner_submission import validate_submission, write_submission
    from kgeval.convert import convert_corpus
    from kgeval.konooz import domain_files, read_column_file
    from kgeval.ner_scoring import score_tag_docs
    from kgeval.ner_train import TrainConfig, predict_docs, run_training
    from kgeval.spans import spans_from_rows, union_tag_rows
    from kgeval.wojood import read_nested_txt

    print(f"[env] torch {torch.__version__} transformers {transformers.__version__} "
          f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nested = f"{BUNDLE}/datasets/Wojood/Wojood1_1_nested"
    train_docs, train_stats = convert_corpus(read_nested_txt(f"{nested}/train.txt"))
    val_docs, _ = convert_corpus(read_nested_txt(f"{nested}/val.txt"))
    test_docs, _ = convert_corpus(read_nested_txt(f"{nested}/test.txt"))
    trainval_docs = train_docs + val_docs
    monitor_docs = test_docs[:MONITOR_SENTENCES]
    print(f"[data] train+val {len(trainval_docs)} sentences "
          f"({train_stats.n_tokens}+ tokens), Wojood-test {len(test_docs)}, "
          f"monitor {len(monitor_docs)}")

    base_cfg = TrainConfig(max_epochs=EPOCHS, select="last", patience=EPOCHS,
                           wall_limit_s=2.6 * 3600)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_cfg.model_name)

    # batches built once, reused across seeds — identical structure guarantees
    # the per-batch probability accumulators align across seeds
    test_batches = ner_data.make_batches(
        ner_data.build_examples(test_docs, tokenizer, base_cfg.max_len),
        tokenizer, base_cfg.batch_size, base_cfg.max_len,
    )
    test_gold = [[row[1:] for row in rows] for rows in test_docs]
    test_lengths = [len(rows) for rows in test_docs]

    blind_dir = f"{BUNDLE}/datasets/blinded-test-data"
    blind: dict[str, dict] = {}
    for path in domain_files(blind_dir):
        cf = read_column_file(path)
        docs_bare = [[[tok] for tok in sent] for sent in cf.sentences]
        blind[path.stem] = {
            "batches": ner_data.make_batches(
                ner_data.build_examples(docs_bare, tokenizer, base_cfg.max_len),
                tokenizer, base_cfg.batch_size, base_cfg.max_len,
            ),
            "lengths": [len(s) for s in cf.sentences],
            "tokens": cf.n_tokens,
        }

    def predict_probs(model, batches):
        """Per-batch softmax probs [B, T, 21, 3], CPU fp32."""
        model.eval()
        out = []
        with torch.no_grad():
            for b in batches:
                logits = model(
                    b.enc["input_ids"].to(device), b.enc["attention_mask"].to(device)
                )
                out.append(torch.softmax(logits.float(), dim=-1).cpu())
        return out

    def rows_from_probs(probs, batches, lengths):
        return ner_data.assemble_predictions(
            [p.argmax(dim=-1) for p in probs], batches, lengths
        )

    result: dict = {"seeds": {}, "epochs": EPOCHS, "variants": {}}
    # per-seed predicted rows: [seed][domain] -> per-sentence rows
    seed_blind_rows: dict[int, dict[str, list]] = {}
    seed_test_rows: dict[int, list] = {}
    probs_acc_test: list | None = None
    probs_acc_blind: dict[str, list] = {}

    for seed in SEEDS:
        cfg = TrainConfig(**{**base_cfg.__dict__, "seed": seed})
        seed_dir = os.path.join(OUT_DIR, f"seed{seed}")
        print(f"[seed {seed}] training {EPOCHS} epochs on train+val")
        model, _tok, train_result = run_training(
            trainval_docs, monitor_docs, cfg, seed_dir, tokenizer=tokenizer,
        )

        # in-domain regression (sanity only; weights are fixed by schedule)
        probs_t = predict_probs(model, test_batches)
        rows_t = rows_from_probs(probs_t, test_batches, test_lengths)
        score = score_tag_docs(test_gold, rows_t)
        p, r, f1 = score.micro
        print(f"[seed {seed}] Wojood-test micro P {p:.4f} R {r:.4f} F1 {f1:.4f}")
        seed_test_rows[seed] = rows_t
        probs_acc_test = (
            probs_t if probs_acc_test is None
            else [a + b for a, b in zip(probs_acc_test, probs_t)]
        )

        seed_blind_rows[seed] = {}
        for dom, d in blind.items():
            probs_b = predict_probs(model, d["batches"])
            seed_blind_rows[seed][dom] = rows_from_probs(probs_b, d["batches"], d["lengths"])
            probs_acc_blind[dom] = (
                probs_b if dom not in probs_acc_blind
                else [a + b for a, b in zip(probs_acc_blind[dom], probs_b)]
            )

        result["seeds"][seed] = {
            "final_epoch": train_result["final_epoch"],
            "stopped": train_result["stopped"],
            "monitor_best_f1": train_result["best_val_f1"],
            "train_seconds": train_result["total_seconds"],
            "wojood_test_micro": {"p": round(p, 5), "r": round(r, 5), "f1": round(f1, 5)},
        }
        del model
        torch.cuda.empty_cache()

    # ---- ensemble variants ------------------------------------------------
    def union_docs(rows_per_seed_docs: list[list], min_votes: int):
        """Sentence-wise union over N models' per-sentence rows; returns
        (rows_docs, n_dropped_total)."""
        out, dropped = [], 0
        for per_sentence in zip(*rows_per_seed_docs):
            res = union_tag_rows(list(per_sentence), min_votes=min_votes)
            out.append(res.rows)
            dropped += res.n_dropped
        return out, dropped

    test_variants: dict[str, list] = {f"seed{s}": seed_test_rows[s] for s in SEEDS}
    test_variants["meanprob"] = rows_from_probs(probs_acc_test, test_batches, test_lengths)
    per_seed_test = [seed_test_rows[s] for s in SEEDS]
    test_variants["union3"], _ = union_docs(per_seed_test, min_votes=1)
    test_variants["vote2"], _ = union_docs(per_seed_test, min_votes=2)

    blind_variants: dict[str, dict[str, list]] = {
        f"seed{s}": seed_blind_rows[s] for s in SEEDS
    }
    blind_variants["meanprob"] = {
        dom: rows_from_probs(probs_acc_blind[dom], d["batches"], d["lengths"])
        for dom, d in blind.items()
    }
    for name, mv in (("union3", 1), ("vote2", 2)):
        per_dom = {}
        dropped_total = 0
        for dom in blind:
            per_seed = [seed_blind_rows[s][dom] for s in SEEDS]
            per_dom[dom], dropped = union_docs(per_seed, min_votes=mv)
            dropped_total += dropped
        blind_variants[name] = per_dom
        result["variants"].setdefault(name, {})["blind_dropped_overlaps"] = dropped_total

    all_ok = True
    for name, rows_docs in test_variants.items():
        score = score_tag_docs(test_gold, rows_docs)
        p, r, f1 = score.micro
        v = result["variants"].setdefault(name, {})
        v["wojood_test_micro"] = {"p": round(p, 5), "r": round(r, 5), "f1": round(f1, 5)}
        print(f"[variant {name}] Wojood-test P {p:.4f} R {r:.4f} F1 {f1:.4f}")

    tags_export: dict[str, dict[str, list]] = {}
    for name, per_dom in blind_variants.items():
        v = result["variants"].setdefault(name, {})
        flat = {dom: [row for rows in per_dom[dom] for row in rows] for dom in per_dom}
        tags_export[name] = flat
        spans_by_dom = {}
        total_spans = 0
        for dom, rows_docs in per_dom.items():
            n = sum(len(spans_from_rows(rows).spans) for rows in rows_docs)
            spans_by_dom[dom] = n
            total_spans += n
        v["blind_spans"] = {"total": total_spans, "by_domain": spans_by_dom}
        print(f"[variant {name}] blind spans {total_spans}")

        zip_path = os.path.join(OUT_DIR, f"rehearsal_{name}.zip")
        info = write_submission(blind_dir, flat, zip_path)
        report = validate_submission(zip_path, blind_dir)
        os.remove(zip_path)
        v["rehearsal_pass"] = report.ok
        v["rehearsal_token_lines"] = info["token_lines"]
        if not report.ok:
            print(f"[variant {name}] REHEARSAL FAILED\n" + report.pretty())
        all_ok = all_ok and report.ok and total_spans >= 1000

    with open(os.path.join(OUT_DIR, "adaptner_ens_tags.json"), "w") as f:
        json.dump(tags_export, f, separators=(",", ":"))

    # union must contain every seed's spans; a violation means a bug upstream
    u = result["variants"]["union3"]["blind_spans"]["total"]
    m = max(result["variants"][f"seed{s}"]["blind_spans"]["total"] for s in SEEDS)
    result["union_superset_check"] = u >= m
    seeds_healthy = all(
        result["seeds"][s]["wojood_test_micro"]["f1"] >= 0.915 for s in SEEDS
    )
    result["total_seconds"] = round(time.time() - t0, 1)
    result["ok"] = all_ok and result["union_superset_check"] and seeds_healthy
    with open(os.path.join(OUT_DIR, "ADAPTNER_ENSEMBLE_COMPLETED.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
