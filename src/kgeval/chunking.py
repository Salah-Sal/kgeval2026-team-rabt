"""Split long sentences into sub-budget chunks at word boundaries.

Wojood has sentences longer than the 512-subtoken encoder window (design doc
§3.2: split at token boundaries, re-join at write-out). Chunking is pure
arithmetic over per-word subtoken costs so it can be unit-tested without a
tokenizer; predictions are re-assembled by (sentence index, word offset).
"""

from __future__ import annotations

from typing import Sequence


def chunk_spans(word_costs: Sequence[int], budget: int) -> list[tuple[int, int]]:
    """Greedy [start, end) word ranges whose summed cost stays within budget.

    A single word over budget still gets its own chunk (the tokenizer truncates
    it; only the first subtoken is labeled anyway).
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    spans: list[tuple[int, int]] = []
    start = 0
    acc = 0
    for i, cost in enumerate(word_costs):
        if acc and acc + cost > budget:
            spans.append((start, i))
            start = i
            acc = 0
        acc += cost
    if word_costs:
        spans.append((start, len(word_costs)))
    return spans
