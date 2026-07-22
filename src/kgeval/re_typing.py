"""Entity typing for RE typed markers (design §4.2).

Chain per record: sentence_id → Wojood global_sentence_id (nested CSV) →
string-match the subject/object against that sentence's gold entities → type.
Fallbacks: corpus-wide entity→type lexicon, then 'UNK' (markers degrade
gracefully to untyped). Every assignment records its source so join coverage
lands in the experiment ledger, per the design doc: measure, don't assume.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .convert import sentence_to_rows
from .spans import spans_from_rows
from .wojood import Sentence

UNK_TYPE = "UNK"


def read_csv_nested(csv_path: str | Path) -> dict[str, Sentence]:
    """global_sentence_id → nested sentence [(token, [tags outer→inner])]."""
    sentences: dict[str, Sentence] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tags = row["tags"].split() or ["O"]
            sentences.setdefault(row["global_sentence_id"], []).append(
                (row["token"], tags)
            )
    return sentences


def entity_types(sentence: Sentence) -> dict[str, Counter]:
    """Entity surface string (space-joined tokens) → Counter of its types."""
    rows = sentence_to_rows(sentence)
    tokens = [r[0] for r in rows]
    out: dict[str, Counter] = {}
    for start, end, typ in spans_from_rows([r[1:] for r in rows]).spans:
        text = " ".join(tokens[start:end])
        out.setdefault(text, Counter())[typ] += 1
    return out


def build_lexicon(sentences: dict[str, Sentence]) -> dict[str, Counter]:
    """Corpus-wide entity string → type counts (fallback for unjoined ids)."""
    lexicon: dict[str, Counter] = {}
    for sentence in sentences.values():
        for text, counts in entity_types(sentence).items():
            lexicon.setdefault(text, Counter()).update(counts)
    return lexicon


def type_mention(
    mention: str,
    sentence_entities: dict[str, Counter] | None,
    lexicon: dict[str, Counter],
) -> tuple[str, str]:
    """→ (type, source) with source ∈ {sentence, lexicon, unk}."""
    if sentence_entities and mention in sentence_entities:
        return sentence_entities[mention].most_common(1)[0][0], "sentence"
    if mention in lexicon:
        return lexicon[mention].most_common(1)[0][0], "lexicon"
    return UNK_TYPE, "unk"


def assign_types(
    records: list[dict],
    gsid_sentences: dict[str, Sentence],
    lexicon: dict[str, Counter],
) -> tuple[list[tuple[str, str]], Counter]:
    """Per record (subject_type, object_type) + Counter of assignment sources."""
    types: list[tuple[str, str]] = []
    stats: Counter = Counter()
    ents_cache: dict[str, dict[str, Counter]] = {}
    for rec in records:
        gsid = str(rec["sentence_id"])
        sentence = gsid_sentences.get(gsid)
        if sentence is not None and gsid not in ents_cache:
            ents_cache[gsid] = entity_types(sentence)
        ents = ents_cache.get(gsid)
        stats["joined" if sentence else "unjoined"] += 1
        s_type, s_src = type_mention(rec["subject"], ents, lexicon)
        o_type, o_src = type_mention(rec["object"], ents, lexicon)
        stats[f"subj_{s_src}"] += 1
        stats[f"obj_{o_src}"] += 1
        types.append((s_type, o_type))
    return types, stats
