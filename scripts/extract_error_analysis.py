"""Gold-grounded error extraction for the paper's error-analysis section.

Four parts, each independent (failures don't cascade):
  A. RE fold-0 OOF: rebuild grouped fold 0 (deterministic, seed 13), run the
     fold-0 checkpoint on its held-out 3,476 records, decode at penalty 0 and
     3, dump every misclassification with sentence/gold/pred.
     Sanity gate: penalty-0 micro-F1 must reproduce ~0.9581.
  B. NER Wojood-test: run the submitted v1 baseline checkpoint on Wojood test,
     classify every error as confusion / boundary / miss / spurious.
     Sanity gate: micro-F1 must reproduce ~0.9240.
  C. RE blind-test constraint flips: CPU decode of the exported ensemble
     logits at penalty 0 vs 3; list flipped predictions with sentences.
  D. NER blind-test seed agreement: per-domain 1-vote/2-vote/3-vote span
     rates from the exported per-seed tags + example disputed spans.

Inputs: task data under data/ and fetched kernel outputs under output/
(the baseline and ensemble campaign checkpoints/exports; see the kernel
READMEs under kaggle/).

Outputs JSON under output/error-analysis/ (gitignored: the files
embed licensed dataset sentences, so they must never reach a remote;
this script is the recovery recipe, ~45 s on Apple silicon).
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
DATA = REPO / "data"
CACHE = REPO / "output"
OUT = REPO / "output/error-analysis"
OUT.mkdir(parents=True, exist_ok=True)

import torch  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
summary: dict = {"device": DEVICE}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- A: RE OOF
def part_a():
    from transformers import AutoModel, AutoTokenizer

    from kgeval.re_constraints import ConstraintTable, observed_pairs
    from kgeval.re_data import detect_no_relation, label_counts, load_jsonl
    from kgeval.re_scoring import score as re_score
    from kgeval.re_train import RETrainConfig, build_examples, make_batches, predict
    from kgeval.re_model import MarkerClassifier
    from kgeval.re_typing import assign_types, build_lexicon, read_csv_nested
    from kgeval.splits import grouped_kfold

    records = load_jsonl(DATA / "WojoodRelations/train.jsonl")
    label_vocab = sorted(label_counts(records))
    neg = detect_no_relation(set(label_vocab))
    label_index = {l: i for i, l in enumerate(label_vocab)}

    gsid_sentences = {}
    for split in ("train", "val", "test"):
        gsid_sentences.update(
            read_csv_nested(DATA / f"Wojood/Wojood1_1_nested/{split}.csv")
        )
    lexicon = build_lexicon(gsid_sentences)

    all_types, _ = assign_types(records, gsid_sentences, lexicon)
    table = ConstraintTable(observed_pairs(records, all_types))

    folds = grouped_kfold(records, k=5, seed=13)
    _, val_recs = folds[0]
    log(f"A: fold0 val {len(val_recs)} records (expect 3476)")
    val_types, _ = assign_types(val_recs, gsid_sentences, lexicon)
    val_ex, _ = build_examples(val_recs, val_types)

    cfg = RETrainConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    batches = make_batches(val_ex, tokenizer, label_index, cfg.batch_size, cfg.max_len)

    encoder = AutoModel.from_pretrained(cfg.model_name)
    model = MarkerClassifier(encoder, len(label_vocab), dropout=cfg.dropout)
    state = torch.load(CACHE / "re-ensemble/fold0/best_re_model.pt",
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)

    gold = {ex.triple_id: ex.label for ex in val_ex}
    gold_pairs = [(ex.triple_id, ex.label) for ex in val_ex]
    p0 = predict(model, batches, label_vocab, DEVICE, masked_label=neg)
    p3 = predict(model, batches, label_vocab, DEVICE, masked_label=neg,
                 constraints=table, penalty=3.0)
    s0 = re_score(gold_pairs, p0, positives_only=True)
    s3 = re_score(gold_pairs, p3, positives_only=True)
    log(f"A: micro p0 {s0.micro_f1:.4f} (gate ~0.9581) p3 {s3.micro_f1:.4f}; "
        f"macro p0 {s0.macro_f1:.4f} p3 {s3.macro_f1:.4f}")

    by_rec = {str(r["triple_id"]): r for r in val_recs}
    types_by_tid = {ex.triple_id: (ex.subj_type, ex.obj_type) for ex in val_ex}
    errors, fixed_by_penalty, broken_by_penalty = [], [], []
    for tid, g in gold.items():
        rec, tp = by_rec[tid], types_by_tid[tid]
        row = {"triple_id": tid, "sentence": rec["sentence"],
               "subject": rec["subject"], "object": rec["object"],
               "subj_type": tp[0], "obj_type": tp[1],
               "gold": g, "pred_p0": p0[tid], "pred_p3": p3[tid]}
        if p3[tid] != g:
            errors.append(row)
        if p0[tid] != g and p3[tid] == g:
            fixed_by_penalty.append(row)
        if p0[tid] == g and p3[tid] != g:
            broken_by_penalty.append(row)
    confusions = Counter((e["gold"], e["pred_p3"]) for e in errors)
    out = {
        "n_val": len(val_recs),
        "micro_f1_p0": s0.micro_f1, "micro_f1_p3": s3.micro_f1,
        "macro_f1_p0": s0.macro_f1, "macro_f1_p3": s3.macro_f1,
        "n_errors_p3": len(errors),
        "n_fixed_by_penalty": len(fixed_by_penalty),
        "n_broken_by_penalty": len(broken_by_penalty),
        "top_confusions": [[g, p, c] for (g, p), c in confusions.most_common(20)],
        "errors": errors,
        "fixed_by_penalty": fixed_by_penalty,
        "broken_by_penalty": broken_by_penalty,
    }
    (OUT / "re_oof_fold0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    summary["A"] = {k: out[k] for k in
                    ("micro_f1_p0", "micro_f1_p3", "n_errors_p3",
                     "n_fixed_by_penalty", "n_broken_by_penalty")}
    del model, encoder
    log(f"A: done — {len(errors)} errors, {len(fixed_by_penalty)} fixed / "
        f"{len(broken_by_penalty)} broken by penalty")


# ---------------------------------------------------------- B: NER Wojood-test
def part_b():
    from transformers import AutoModel, AutoTokenizer

    from kgeval import ner_data
    from kgeval.convert import convert_corpus
    from kgeval.ner_model import MultiHeadTagger
    from kgeval.ner_train import TrainConfig, evaluate
    from kgeval.spans import spans_from_rows
    from kgeval.wojood import read_nested_txt

    cfg = TrainConfig()
    test_docs, _ = convert_corpus(
        read_nested_txt(DATA / "Wojood/Wojood1_1_nested/test.txt"))
    log(f"B: {len(test_docs)} test sentences")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    batches = ner_data.make_batches(
        ner_data.build_examples(test_docs, tokenizer, cfg.max_len),
        tokenizer, 32, cfg.max_len)

    encoder = AutoModel.from_pretrained(cfg.model_name)
    model = MultiHeadTagger(encoder, dropout=cfg.dropout)
    state = torch.load(CACHE / "adaptner-baseline/best_model.pt",
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)

    score, pred_docs = evaluate(model, batches, test_docs, DEVICE)
    p, r, f1 = score.micro
    log(f"B: Wojood-test micro-F1 {f1:.4f} (gate ~0.9240) P {p:.4f} R {r:.4f}")

    confusion, boundary, miss, spurious = [], [], [], []
    for si, rows in enumerate(test_docs):
        words = [row[0] for row in rows]
        gold_spans = spans_from_rows([row[1:] for row in rows]).spans
        pred_spans = spans_from_rows(pred_docs[si]).spans
        fn = gold_spans - pred_spans
        fp = pred_spans - gold_spans
        if not fn and not fp:
            continue
        sent = " ".join(words)
        used_fp = set()
        for g in sorted(fn):
            gs, ge, gt = g
            ex = {"sentence": sent, "span": " ".join(words[gs:ge]),
                  "gold_type": gt, "start": gs, "end": ge, "sent_idx": si}
            hit = next((q for q in sorted(fp) if q not in used_fp
                        and q[0] == gs and q[1] == ge), None)
            if hit:  # exact boundaries, wrong type
                used_fp.add(hit)
                ex["pred_type"] = hit[2]
                confusion.append(ex)
                continue
            hit = next((q for q in sorted(fp) if q not in used_fp
                        and q[2] == gt and q[0] < ge and gs < q[1]), None)
            if hit:  # right type, wrong boundaries
                used_fp.add(hit)
                ex["pred_span"] = " ".join(words[hit[0]:hit[1]])
                ex["pred_start"], ex["pred_end"] = hit[0], hit[1]
                boundary.append(ex)
                continue
            miss.append(ex)
        for q in sorted(fp):
            if q in used_fp:
                continue
            qs, qe, qt = q
            spurious.append({"sentence": sent, "span": " ".join(words[qs:qe]),
                             "pred_type": qt, "start": qs, "end": qe,
                             "sent_idx": si})
    cat_counts = {"confusion": len(confusion), "boundary": len(boundary),
                  "miss": len(miss), "spurious": len(spurious)}
    conf_pairs = Counter((e["gold_type"], e["pred_type"]) for e in confusion)
    miss_types = Counter(e["gold_type"] for e in miss)
    spur_types = Counter(e["pred_type"] for e in spurious)
    bound_types = Counter(e["gold_type"] for e in boundary)
    out = {
        "micro": {"p": p, "r": r, "f1": f1},
        "per_type": {t: list(v) for t, v in sorted(score.per_type.items())},
        "categories": cat_counts,
        "top_confusion_pairs": [[a, b, c] for (a, b), c in conf_pairs.most_common(20)],
        "top_miss_types": miss_types.most_common(21),
        "top_spurious_types": spur_types.most_common(21),
        "top_boundary_types": bound_types.most_common(21),
        "confusion": confusion, "boundary": boundary,
        "miss": miss, "spurious": spurious,
    }
    (OUT / "ner_wojood_test.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    summary["B"] = {"micro_f1": f1, **cat_counts}
    del model, encoder
    log(f"B: done — {cat_counts}")


# ------------------------------------------------- C: RE blind constraint flips
def part_c():
    d = json.loads((CACHE / "re-ensemble/re_ens_logits.json").read_text())
    vocab = d["label_vocab"]
    masked = vocab.index(d["masked_label"])
    inad = {k: set(v) for k, v in d["inadmissible_by_pair"].items()}
    test = {str(r["triple_id"]): r for r in
            (json.loads(l) for l in
             (DATA / "re-test/test.jsonl").read_text().splitlines() if l.strip())}

    def decode(row, pair, penalty):
        s, o = pair
        bad = inad.get(f"{s}|{o}", set())
        best, best_v = None, None
        for i, v in enumerate(row):
            if i == masked:
                continue
            vv = v - (penalty if i in bad else 0.0)
            if best_v is None or vv > best_v:
                best, best_v = i, vv
        return vocab[best]

    flips = []
    for tid, pair, row in zip(d["triple_ids"], d["type_pairs"], d["logits"]):
        l0 = decode(row, pair, 0.0)
        l3 = decode(row, pair, 3.0)
        if l0 != l3:
            rec = test.get(tid, {})
            flips.append({"triple_id": tid,
                          "subj_type": pair[0], "obj_type": pair[1],
                          "pred_unconstrained": l0, "pred_penalty3": l3,
                          "sentence": rec.get("sentence", ""),
                          "subject": rec.get("subject", ""),
                          "object": rec.get("object", "")})
    (OUT / "re_blind_constraint_flips.json").write_text(
        json.dumps(flips, ensure_ascii=False, indent=1))
    summary["C"] = {"n_flips": len(flips)}
    log(f"C: done — {len(flips)} blind predictions flipped by the penalty")


# ------------------------------------------------ D: NER blind seed agreement
def part_d():
    from kgeval.spans import spans_from_rows

    tags = json.loads(
        (CACHE / "adaptner-ensemble/adaptner_ens_tags.json").read_text())
    seeds = ["seed13", "seed42", "seed77"]
    domains = sorted(tags["seed13"].keys())
    per_domain = {}
    disputed_examples = []
    for dom in domains:
        ref = (DATA / f"blinded-test-data/{dom}.txt").read_text().splitlines()
        sent_lens, cur = [], 0
        for line in ref:
            if line.strip() == "":
                if cur:
                    sent_lens.append(cur)
                cur = 0
            else:
                cur += 1
        if cur:
            sent_lens.append(cur)
        toks = [l.split(" ")[0] for l in ref if l.strip() != ""]
        assert sum(sent_lens) == len(tags["seed13"][dom]), dom

        vote_hist = Counter()
        offset = 0
        for sl in sent_lens:
            votes = Counter()
            for s in seeds:
                votes.update(spans_from_rows(tags[s][dom][offset:offset + sl]).spans)
            for span, v in votes.items():
                vote_hist[v] += 1
                if v == 1 and len(disputed_examples) < 60:
                    st, en, ty = span
                    words = toks[offset:offset + sl]
                    disputed_examples.append(
                        {"domain": dom, "type": ty, "votes": 1,
                         "span": " ".join(words[st:en]),
                         "sentence": " ".join(words)})
            offset += sl
        per_domain[dom] = dict(vote_hist)
    (OUT / "ner_blind_seed_agreement.json").write_text(json.dumps(
        {"per_domain_vote_histogram": per_domain,
         "disputed_examples": disputed_examples},
        ensure_ascii=False, indent=1))
    summary["D"] = {d: v for d, v in per_domain.items()}
    log("D: done — per-domain vote histograms written")


if __name__ == "__main__":
    for name, fn in (("A", part_a), ("B", part_b), ("C", part_c), ("D", part_d)):
        try:
            t = time.time()
            fn()
            log(f"{name} took {time.time() - t:.0f}s")
        except Exception as e:  # keep going; partial results are still useful
            import traceback
            traceback.print_exc()
            summary[name] = {"error": str(e)}
    (OUT / "DONE.json").write_text(json.dumps(summary, indent=1))
    log("ALL DONE")
