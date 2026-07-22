"""Loaders and label utilities for the WojoodRelations track."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REQUIRED_KEYS = {"sentence_id", "sentence", "triple_id", "subject", "object", "relation"}

_NO_RELATION_RE = re.compile(r"^no[_\-]?relation$", re.IGNORECASE)


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = REQUIRED_KEYS - rec.keys()
            if missing:
                raise ValueError(f"{path}:{lineno}: missing keys {missing}")
            records.append(rec)
    return records


def label_counts(records: list[dict]) -> Counter:
    """Counts of non-empty relation strings, verbatim."""
    return Counter(r["relation"] for r in records if r["relation"])


def detect_no_relation(labels: set[str] | Counter) -> str | None:
    """The corpus's exact negative-class string (e.g. 'no_relation'), if present."""
    for label in labels:
        if _NO_RELATION_RE.match(label):
            return label
    return None


def positive_labels(records: list[dict]) -> list[str]:
    counts = label_counts(records)
    neg = detect_no_relation(counts)
    return sorted(l for l in counts if l != neg)
