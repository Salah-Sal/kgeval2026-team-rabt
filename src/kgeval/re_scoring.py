"""Micro-F1 (exact label match per triple) for the RE track.

Forced-choice single-label prediction ⇒ micro-F1 equals accuracy when every
gold instance receives exactly one prediction; computed via TP counts anyway
so missing predictions are penalized correctly. Local validation mimics the
test condition: positives only (design doc §4.5).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .re_data import detect_no_relation


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


@dataclass
class REScore:
    n: int = 0
    correct: int = 0
    micro_f1: float = 0.0
    macro_f1: float = 0.0
    per_label: dict[str, tuple[float, float, float, int]] = field(default_factory=dict)
    confusions: list[tuple[str, str, int]] = field(default_factory=list)

    def pretty(self, top_confusions: int = 10) -> str:
        lines = [
            f"micro-F1 {self.micro_f1:.4f}  macro-F1 {self.macro_f1:.4f}  "
            f"({self.correct}/{self.n} correct)"
        ]
        for label in sorted(self.per_label):
            p, r, f1, support = self.per_label[label]
            lines.append(f"  {label:<40} P {p:.3f} R {r:.3f} F1 {f1:.3f}  (n={support})")
        if self.confusions:
            lines.append("  top confusions (gold → pred × count):")
            for g, p_, c in self.confusions[:top_confusions]:
                lines.append(f"    {g} → {p_} × {c}")
        return "\n".join(lines)


def score(
    gold: list[tuple[str, str]],
    pred: dict[str, str],
    positives_only: bool = True,
) -> REScore:
    """gold: (triple_id, label) pairs; pred: triple_id → label."""
    neg = detect_no_relation({l for _, l in gold})
    if positives_only and neg:
        gold = [(tid, l) for tid, l in gold if l != neg]

    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    confusion: Counter = Counter()
    correct = 0
    for tid, g in gold:
        p = pred.get(tid, "<MISSING>")
        if p == g:
            tp[g] += 1
            correct += 1
        else:
            fn[g] += 1
            fp[p] += 1
            confusion[(g, p)] += 1

    labels = sorted(set(tp) | set(fn) | (set(fp) - {"<MISSING>"}))
    per_label = {}
    f1s = []
    for label in labels:
        p_, r_, f1_ = _prf(tp[label], fp[label], fn[label])
        support = tp[label] + fn[label]
        per_label[label] = (p_, r_, f1_, support)
        if support:  # macro over labels present in gold
            f1s.append(f1_)

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    _, _, micro = _prf(total_tp, total_fp, total_fn)
    return REScore(
        n=len(gold),
        correct=correct,
        micro_f1=micro,
        macro_f1=sum(f1s) / len(f1s) if f1s else 0.0,
        per_label=per_label,
        confusions=[(g, p, c) for (g, p), c in sorted(confusion.items(), key=lambda kv: -kv[1])],
    )
