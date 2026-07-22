"""Convert Wojood variable-width nested tags → the fixed 21-column format.

Rule (design doc §1.1): each type column receives that type's B-/I- tag; on a
same-type collision (an inner and an outer entity of the same type on one
token) the OUTER (first-listed) tag wins. Same-type nesting is unrepresentable
in the official format; Wojood contains only ~576 such nested pairs corpus-wide,
so the loss is negligible — but it is counted and reported, never silent.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .columns import ENTITY_TYPES, TYPE_INDEX
from .wojood import NestedCorpus, Sentence, read_nested_txt


@dataclass
class ConvertStats:
    n_sentences: int = 0
    n_tokens: int = 0
    n_multi_tag_tokens: int = 0
    n_same_type_collisions: int = 0
    collisions_by_type: Counter = field(default_factory=Counter)
    nesting_pairs: Counter = field(default_factory=Counter)  # (outer, inner) token counts


def nested_tags_to_columns(tags: list[str], stats: ConvertStats | None = None) -> list[str]:
    """One token's outer→inner tag list → its 21 column tags."""
    columns = ["O"] * len(ENTITY_TYPES)
    seen_types: list[str] = []
    for tag in tags:
        if tag == "O":
            continue
        typ = tag.split("-", 1)[1]
        ci = TYPE_INDEX[typ]
        if columns[ci] == "O":
            columns[ci] = tag
        elif stats is not None:
            stats.n_same_type_collisions += 1
            stats.collisions_by_type[typ] += 1
        if stats is not None:
            for outer in seen_types:
                stats.nesting_pairs[(outer, typ)] += 1
        seen_types.append(typ)
    return columns


def sentence_to_rows(sentence: Sentence, stats: ConvertStats | None = None) -> list[list[str]]:
    """One nested sentence → rows of [token, tag×21]."""
    rows = []
    for token, tags in sentence:
        if stats is not None:
            stats.n_tokens += 1
            if sum(1 for t in tags if t != "O") > 1:
                stats.n_multi_tag_tokens += 1
        rows.append([token] + nested_tags_to_columns(tags, stats))
    if stats is not None:
        stats.n_sentences += 1
    return rows


def convert_corpus(corpus: NestedCorpus) -> tuple[list[list[list[str]]], ConvertStats]:
    stats = ConvertStats()
    docs = [sentence_to_rows(s, stats) for s in corpus.sentences]
    return docs, stats


def convert_file(in_path: str | Path, out_path: str | Path, eol: str = "\n") -> ConvertStats:
    """Wojood nested .txt → 22-field column file (for training-side ingestion)."""
    corpus = read_nested_txt(in_path)
    docs, stats = convert_corpus(corpus)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        for si, rows in enumerate(docs):
            for row in rows:
                f.write(" ".join(row) + eol)
            if si != len(docs) - 1:
                f.write(eol)  # blank line between sentences
    return stats
