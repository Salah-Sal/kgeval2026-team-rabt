"""Local evaluation splits.

RE: split by `sentence_id`, never by row — ~2.75 triples share a sentence and
row-level splits leak (design doc §1.2). NER: leave-subcorpus-out folds over
Wojood's `Sub_corpus` column as the domain-shift proxy (§3.4).
"""

from __future__ import annotations

import random
from collections import Counter


def grouped_split(
    records: list[dict],
    val_frac: float = 0.125,
    seed: int = 13,
    group_key: str = "sentence_id",
) -> tuple[list[dict], list[dict], dict]:
    """Deterministic grouped split: whole groups go to one side."""
    groups = sorted({r[group_key] for r in records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    by_group = Counter(r[group_key] for r in records)
    target = val_frac * len(records)
    val_groups: set = set()
    picked = 0
    for g in groups:
        if picked >= target:
            break
        val_groups.add(g)
        picked += by_group[g]
    train = [r for r in records if r[group_key] not in val_groups]
    val = [r for r in records if r[group_key] in val_groups]
    info = {
        "seed": seed,
        "group_key": group_key,
        "n_groups": len(groups),
        "n_val_groups": len(val_groups),
        "n_train": len(train),
        "n_val": len(val),
    }
    return train, val, info


def grouped_kfold(
    records: list[dict],
    k: int = 5,
    seed: int = 13,
    group_key: str = "sentence_id",
) -> list[tuple[list[dict], list[dict]]]:
    """Deterministic grouped K-fold: every group lands in exactly one fold's
    val side, folds balanced by record count (greedy smallest-fold assignment
    over the shuffled groups). Returns [(train, val)] per fold."""
    if k < 2:
        raise ValueError("k must be >= 2")
    groups = sorted({r[group_key] for r in records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    by_group = Counter(r[group_key] for r in records)
    fold_of: dict = {}
    fold_sizes = [0] * k
    for g in groups:
        fi = min(range(k), key=lambda i: fold_sizes[i])
        fold_of[g] = fi
        fold_sizes[fi] += by_group[g]
    folds = []
    for fi in range(k):
        train = [r for r in records if fold_of[r[group_key]] != fi]
        val = [r for r in records if fold_of[r[group_key]] == fi]
        folds.append((train, val))
    return folds


def loso_folds(sentence_subcorpus: dict[str, str]) -> dict[str, set[str]]:
    """Sub_corpus name → set of global_sentence_ids (leave-subcorpus-out folds)."""
    folds: dict[str, set[str]] = {}
    for gsid, sub in sentence_subcorpus.items():
        folds.setdefault(sub, set()).add(gsid)
    return folds
