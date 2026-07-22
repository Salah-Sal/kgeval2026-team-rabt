"""Entity-level micro-F1 (exact span + type) for the AdaptNER track.

Mirrors the announced metric: an entity counts as correct only if span
boundaries and type both match. To be reconciled against the official
evaluation script from the starting kit the moment it is on disk (design doc
§1.3 open item 1).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .spans import Span, spans_from_rows


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


@dataclass
class NERScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    per_type: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    n_pred_repairs: int = 0

    @property
    def micro(self) -> tuple[float, float, float]:
        return _prf(self.tp, self.fp, self.fn)

    def pretty(self) -> str:
        p, r, f1 = self.micro
        lines = [
            f"micro-P {p:.4f}  micro-R {r:.4f}  micro-F1 {f1:.4f}  "
            f"(TP {self.tp} FP {self.fp} FN {self.fn}; IOB repairs in pred: {self.n_pred_repairs})"
        ]
        for typ in sorted(self.per_type):
            tp, fp, fn = self.per_type[typ]
            tp_, tr_, tf_ = _prf(tp, fp, fn)
            lines.append(f"  {typ:<9} P {tp_:.3f} R {tr_:.3f} F1 {tf_:.3f}  (gold {tp + fn})")
        return "\n".join(lines)


def score_span_sets(gold: list[set[Span]], pred: list[set[Span]]) -> NERScore:
    """gold/pred are parallel per-sentence span sets."""
    if len(gold) != len(pred):
        raise ValueError(f"sentence count mismatch: gold {len(gold)} vs pred {len(pred)}")
    score = NERScore()
    type_counts: Counter = Counter()
    for g, p in zip(gold, pred):
        for span in g & p:
            score.tp += 1
            type_counts[(span[2], "tp")] += 1
        for span in p - g:
            score.fp += 1
            type_counts[(span[2], "fp")] += 1
        for span in g - p:
            score.fn += 1
            type_counts[(span[2], "fn")] += 1
    types = {t for (t, _k) in type_counts}
    score.per_type = {
        t: (type_counts[(t, "tp")], type_counts[(t, "fp")], type_counts[(t, "fn")])
        for t in types
    }
    return score


def score_tag_docs(
    gold_docs: list[list[list[str]]], pred_docs: list[list[list[str]]]
) -> NERScore:
    """gold/pred are parallel lists of sentences, each a list of 21-tag rows."""
    gold_sets, pred_sets = [], []
    repairs = 0
    for rows in gold_docs:
        gold_sets.append(spans_from_rows(rows).spans)
    for rows in pred_docs:
        res = spans_from_rows(rows)
        pred_sets.append(res.spans)
        repairs += res.n_repairs
    score = score_span_sets(gold_sets, pred_sets)
    score.n_pred_repairs = repairs
    return score
