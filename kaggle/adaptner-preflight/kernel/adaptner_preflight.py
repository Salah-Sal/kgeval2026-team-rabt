#!/usr/bin/env python
"""KGEval 2026 — AdaptNER preflight.

Proves on the target GPU: private bundle mounted, kgeval parses all three data
families, AraBERTv02 downloads, tokenizer word-alignment works, and the
21-head model completes real train steps. Exports ONLY safe artifacts
(PREFLIGHT_COMPLETED.json: counts, timings, checks — no dataset text).
"""

import glob
import json
import os
import sys
import time

OUT_DIR = "/kaggle/working"
MODEL_NAME = "aubmindlab/bert-base-arabertv02"
EXPECTED_TRAIN_TOKENS = 390_999
EXPECTED_TRAIN_SENTENCES = 23_125


def locate_bundle() -> str:
    """Find the input bundle regardless of the mount layout Kaggle chose.

    The 2026 image no longer mounts datasets flat at /kaggle/input/<slug>, so
    search recursively and print the layout for diagnosis either way.
    """
    for depth in range(1, 4):
        pattern = "/kaggle/input/" + "*/" * depth
        entries = sorted(glob.glob(pattern))
        print(f"[env] input depth {depth}: "
              f"{[e[len('/kaggle/input/'):].rstrip('/') for e in entries[:20]]}")
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
    checks: dict[str, bool] = {}
    result: dict = {"kernel": "kgeval2026-adaptner-preflight", "checks": checks}

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    from kgeval.convert import convert_corpus
    from kgeval.konooz import domain_files, read_column_file
    from kgeval.re_data import label_counts, load_jsonl
    from kgeval.wojood import read_nested_txt

    result["torch"] = torch.__version__
    result["transformers"] = transformers.__version__
    checks["gpu"] = torch.cuda.is_available()
    result["gpu"] = torch.cuda.get_device_name(0) if checks["gpu"] else None
    print(f"[env] torch {result['torch']} transformers {result['transformers']} gpu {result['gpu']}")

    # --- data family 1: Wojood nested ---
    corpus = read_nested_txt(f"{BUNDLE}/datasets/Wojood/Wojood1_1_nested/train.txt")
    docs, stats = convert_corpus(corpus)
    result["wojood_train_tokens"] = stats.n_tokens
    result["wojood_train_sentences"] = stats.n_sentences
    result["same_type_collisions"] = stats.n_same_type_collisions
    checks["wojood_counts_match_expected"] = (
        stats.n_tokens == EXPECTED_TRAIN_TOKENS
        and stats.n_sentences == EXPECTED_TRAIN_SENTENCES
    )
    print(f"[wojood] {stats.n_sentences} sentences / {stats.n_tokens} tokens "
          f"/ {stats.n_same_type_collisions} same-type collisions "
          f"({time.time() - t0:.0f}s)")

    # --- data family 2: Konooz dev 22-column ---
    dev_tokens = sum(
        read_column_file(f).n_tokens for f in domain_files(f"{BUNDLE}/datasets/dev-konooz")
    )
    result["konooz_dev_tokens"] = dev_tokens
    checks["konooz_dev_readable"] = dev_tokens > 3000
    print(f"[konooz] dev tokens {dev_tokens}")

    # --- data family 3: RE jsonl ---
    re_train = load_jsonl(f"{BUNDLE}/datasets/WojoodRelations/train.jsonl")
    result["re_train_records"] = len(re_train)
    result["re_labels"] = len(label_counts(re_train))
    checks["re_readable"] = len(re_train) == 17_381 and result["re_labels"] == 41
    print(f"[re] {len(re_train)} triples / {result['re_labels']} labels")

    # --- model: tokenizer alignment + train steps ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    device = "cuda" if checks["gpu"] else "cpu"
    n_types = 21
    heads = torch.nn.ModuleList([torch.nn.Linear(encoder.config.hidden_size, 3) for _ in range(n_types)])
    encoder.to(device)
    heads.to(device)
    params = list(encoder.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-5)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # first 64 entity-bearing sentences, first-subtoken labeling
    def tag_id(tag: str) -> int:
        return 0 if tag == "O" else (1 if tag.startswith("B-") else 2)

    sample = [rows for rows in docs if any(t != "O" for r in rows for t in r[1:])][:64]
    batches = []
    for i in range(0, len(sample), 8):
        chunk = sample[i : i + 8]
        words = [[r[0] for r in rows] for rows in chunk]
        enc = tokenizer(
            words, is_split_into_words=True, truncation=True, max_length=256,
            padding="max_length", return_tensors="pt",
        )
        labels = torch.full((len(chunk), 256, n_types), -100, dtype=torch.long)
        for bi, rows in enumerate(chunk):
            seen = set()
            for pos, wid in enumerate(enc.word_ids(bi)):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                for ci in range(n_types):
                    labels[bi, pos, ci] = tag_id(rows[wid][1 + ci])
        batches.append((enc, labels))
    checks["alignment_labels_present"] = all(
        (labels != -100).any().item() for _, labels in batches
    )

    losses = []
    step_times = []
    encoder.train()
    for step in range(12):
        enc, labels = batches[step % len(batches)]
        ts = time.time()
        optimizer.zero_grad()
        hidden = encoder(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).last_hidden_state
        loss = sum(
            ce(heads[ci](hidden).view(-1, 3), labels[:, :, ci].reshape(-1).to(device))
            for ci in range(n_types)
        ) / n_types
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - ts)
        losses.append(float(loss))
        print(f"[train] step {step} loss {losses[-1]:.4f} ({step_times[-1]:.2f}s)")

    result["loss_first"] = losses[0]
    result["loss_last"] = losses[-1]
    result["sec_per_step_batch8_len256"] = sum(step_times[3:]) / len(step_times[3:])
    checks["loss_finite"] = all(l == l and l != float("inf") for l in losses)
    checks["loss_decreased"] = losses[-1] < losses[0]

    result["total_seconds"] = round(time.time() - t0, 1)
    result["ok"] = all(checks.values())
    with open(os.path.join(OUT_DIR, "PREFLIGHT_COMPLETED.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
