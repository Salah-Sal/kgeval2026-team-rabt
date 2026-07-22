"""Tensorization for the 21-head tagger.

Pipeline: docs (rows of [token, tag*21] from convert_corpus) → chunked
Examples → pre-collated length-sorted Batches (built once, order shuffled per
epoch) → argmax predictions re-assembled into per-sentence 21-tag rows that
plug straight into ner_scoring / adaptner_submission.

Only the first subtoken of each word carries labels (design §3.2); other
positions are -100 across all 21 columns, so one mask serves every head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .chunking import chunk_spans
from .columns import ENTITY_TYPES

N_TYPES = len(ENTITY_TYPES)
_PREFIX = ("O", "B", "I")


def tag_id(tag: str) -> int:
    return 0 if tag == "O" else (1 if tag.startswith("B-") else 2)


def id_to_tag(i: int, type_name: str) -> str:
    return "O" if i == 0 else f"{_PREFIX[i]}-{type_name}"


@dataclass
class Example:
    sent_idx: int
    word_start: int
    words: list[str]
    tag_ids: list[list[int]] | None  # per word: 21 ints; None for unlabeled text
    n_subtokens: int


@dataclass
class Batch:
    enc: object  # BatchEncoding (keeps .word_ids); tensors inside
    labels: torch.Tensor  # [B, T, 21], -100 on non-first-subtoken/pad positions
    first_pos: list[list[int]]  # per example: word index → seq position (-1 if truncated away)
    examples: list[Example]


def build_examples(
    docs: list[list[list[str]]], tokenizer, max_len: int = 512
) -> list[Example]:
    """docs: sentences of [token, tag*21] rows (or bare [token] for unlabeled)."""
    budget = max_len - 2  # room for CLS/SEP
    examples: list[Example] = []
    for si, rows in enumerate(docs):
        words = [r[0] for r in rows]
        enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
        costs = [0] * len(words)
        for wid in enc.word_ids():
            if wid is not None:
                costs[wid] += 1
        labeled = len(rows[0]) > 1
        for start, end in chunk_spans(costs, budget):
            chunk = rows[start:end]
            examples.append(
                Example(
                    sent_idx=si,
                    word_start=start,
                    words=[r[0] for r in chunk],
                    tag_ids=[[tag_id(t) for t in r[1:]] for r in chunk] if labeled else None,
                    n_subtokens=sum(costs[start:end]),
                )
            )
    return examples


def make_batches(
    examples: list[Example], tokenizer, batch_size: int, max_len: int = 512
) -> list[Batch]:
    """Length-sorted fixed batches, dynamically padded per batch."""
    batches: list[Batch] = []
    ordered = sorted(examples, key=lambda ex: ex.n_subtokens)
    for i in range(0, len(ordered), batch_size):
        group = ordered[i : i + batch_size]
        enc = tokenizer(
            [ex.words for ex in group],
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        n, t = enc["input_ids"].shape
        labels = torch.full((n, t, N_TYPES), -100, dtype=torch.long)
        first_pos: list[list[int]] = []
        for bi, ex in enumerate(group):
            pos_of_word = [-1] * len(ex.words)
            seen: set[int] = set()
            for pos, wid in enumerate(enc.word_ids(bi)):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                pos_of_word[wid] = pos
                if ex.tag_ids is not None:
                    labels[bi, pos] = torch.as_tensor(ex.tag_ids[wid])
            first_pos.append(pos_of_word)
        batches.append(Batch(enc=enc, labels=labels, first_pos=first_pos, examples=group))
    return batches


def assemble_predictions(
    pred_ids: list[torch.Tensor], batches: list[Batch], sentence_lengths: list[int]
) -> list[list[list[str]]]:
    """argmax ids per batch [B, T, 21] → per-sentence 21-tag rows.

    Words the tokenizer dropped (zero subtokens, or truncated pathological
    chunks) stay all-O rather than going missing — row counts always match.
    """
    docs = [[["O"] * N_TYPES for _ in range(n)] for n in sentence_lengths]
    for preds, batch in zip(pred_ids, batches, strict=True):
        for bi, ex in enumerate(batch.examples):
            for wi, pos in enumerate(batch.first_pos[bi]):
                if pos < 0:
                    continue
                row = docs[ex.sent_idx][ex.word_start + wi]
                for ci in range(N_TYPES):
                    row[ci] = id_to_tag(int(preds[bi, pos, ci]), ENTITY_TYPES[ci])
    return docs
