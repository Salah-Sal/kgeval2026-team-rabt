"""Readers for the Wojood nested release (training side of AdaptNER).

`train/val/test.txt` are token-per-line with a variable-length space-separated
tag list (1–5 tags), outer entity listed first, e.g. `ملز I-MONEY B-CURR`.
Blank lines separate sentences. The `.csv` twin adds `global_sentence_id` and
`Sub_corpus` — the join keys used by the RE task and by leave-subcorpus-out
model selection.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from .columns import ENTITY_TYPES

# token, then 1..n tags; tags are O or [BI]-TYPE with TYPE from the 21-type set.
_TAG_RE = re.compile(r"^(O|[BI]-([A-Z]+))$")

# (token, [tags outer→inner]) per token.
Token = tuple[str, list[str]]
Sentence = list[Token]


@dataclass
class NestedCorpus:
    sentences: list[Sentence]
    anomalies: list[str] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return sum(len(s) for s in self.sentences)


def read_nested_txt(path: str | Path, strict: bool = True) -> NestedCorpus:
    """Parse a Wojood nested .txt file.

    strict=True raises on any malformed tag line; strict=False records the
    anomaly and keeps the line (unknown tags are preserved verbatim so the
    caller can inspect them).
    """
    sentences: list[Sentence] = []
    current: Sentence = []
    anomalies: list[str] = []

    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split()
            token, tags = parts[0], parts[1:]
            if not tags:
                anomalies.append(f"{path}:{lineno}: token line without tags: {line!r}")
                tags = ["O"]
            for tag in tags:
                m = _TAG_RE.match(tag)
                if not m or (m.group(2) is not None and m.group(2) not in ENTITY_TYPES):
                    anomalies.append(f"{path}:{lineno}: unknown tag {tag!r}")
            if "O" in tags and len(tags) > 1:
                anomalies.append(f"{path}:{lineno}: O combined with entity tags: {tags}")
            current.append((token, tags))
    if current:
        sentences.append(current)

    if strict and anomalies:
        preview = "\n".join(anomalies[:5])
        raise ValueError(
            f"{len(anomalies)} malformed tag lines in {path}; first cases:\n{preview}"
        )
    return NestedCorpus(sentences=sentences, anomalies=anomalies)


def read_sentence_subcorpus(csv_path: str | Path) -> dict[str, str]:
    """Map global_sentence_id → Sub_corpus from a Wojood .csv (file order kept)."""
    mapping: dict[str, str] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gsid = row["global_sentence_id"]
            if gsid not in mapping:
                mapping[gsid] = row["Sub_corpus"]
    return mapping


def read_csv_sentences(csv_path: str | Path) -> dict[str, list[str]]:
    """Map global_sentence_id → token list (for the RE sentence_id join)."""
    sentences: dict[str, list[str]] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentences.setdefault(row["global_sentence_id"], []).append(row["token"])
    return sentences
