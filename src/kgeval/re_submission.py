"""RE submission writer + validator.

Contract (design doc §1.2): a zip containing exactly `predictions.txt`, lines
`triple_id<TAB>predicted_relation`, same count and order as the released test
file. Labels are emitted verbatim in the train.jsonl prefixed namespace.
Val/test contain no no-relation instances, so the writer refuses the negative
class unless `allow_no_relation=True` (kept as insurance until the official
evaluation script is reconciled — §4.5).
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .re_data import detect_no_relation

MEMBER_NAME = "predictions.txt"


@dataclass
class Report:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def fail(self, msg: str, cap: int = 25) -> None:
        self.ok = False
        if len(self.errors) < cap:
            self.errors.append(msg)

    def pretty(self) -> str:
        lines = ["PASS" if self.ok else "FAIL"]
        lines += [f"  ERROR: {e}" for e in self.errors]
        lines += [f"  warn:  {w}" for w in self.warnings]
        for k, v in self.stats.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def write_predictions(
    reference_records: list[dict],
    labels: list[str],
    out_zip: str | Path,
    label_whitelist: set[str] | None = None,
    allow_no_relation: bool = False,
) -> dict:
    """labels[i] is the prediction for reference_records[i] (count+order preserved)."""
    if len(labels) != len(reference_records):
        raise ValueError(f"{len(labels)} labels for {len(reference_records)} reference records")
    neg = detect_no_relation(set(labels))
    if neg and not allow_no_relation:
        n = sum(1 for l in labels if l == neg)
        raise ValueError(
            f"{n} predictions are {neg!r}; val/test contain none (pass allow_no_relation=True to override)"
        )
    if label_whitelist is not None:
        foreign = sorted({l for l in labels if l not in label_whitelist})
        if foreign:
            raise ValueError(f"labels outside the whitelist: {foreign[:10]}")
    text = "".join(
        f"{rec['triple_id']}\t{label}\n" for rec, label in zip(reference_records, labels)
    )
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(MEMBER_NAME, text)
    return {"zip": str(out_zip), "member": MEMBER_NAME, "n": len(labels)}


def validate_predictions(
    zip_path: str | Path,
    reference_records: list[dict],
    label_whitelist: set[str] | None = None,
    allow_no_relation: bool = False,
) -> Report:
    rep = Report()
    try:
        with zipfile.ZipFile(zip_path) as z:
            members = [n for n in z.namelist() if not n.endswith("/")]
            if members != [MEMBER_NAME]:
                rep.fail(f"zip must contain exactly [{MEMBER_NAME!r}] at the root, found {members}")
                return rep
            text = z.read(MEMBER_NAME).decode("utf-8")
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as e:
        rep.fail(f"cannot read zip: {e}")
        return rep

    lines = text.splitlines()
    while lines and lines[-1].strip() == "":
        lines.pop()
    if len(lines) != len(reference_records):
        rep.fail(f"{len(lines)} prediction lines for {len(reference_records)} reference records")
    n_neg = 0
    seen_labels: set[str] = set()
    for i, (line, rec) in enumerate(zip(lines, reference_records), start=1):
        parts = line.split("\t")
        if len(parts) != 2:
            rep.fail(f"line {i}: expected 'triple_id<TAB>relation', got {line[:60]!r}")
            continue
        tid, label = parts
        if str(tid) != str(rec["triple_id"]):
            rep.fail(f"line {i}: triple_id {tid!r} != reference {rec['triple_id']!r} (order must match)")
        seen_labels.add(label)
        if label_whitelist is not None and label not in label_whitelist:
            rep.fail(f"line {i}: label {label!r} not in the train.jsonl inventory")
    neg = detect_no_relation(seen_labels)
    if neg:
        n_neg = sum(1 for l in lines if l.split("\t")[-1] == neg)
        msg = f"{n_neg} predictions are {neg!r} (val/test contain none)"
        if allow_no_relation:
            rep.warnings.append(msg)
        else:
            rep.fail(msg)
    rep.stats["lines"] = len(lines)
    rep.stats["distinct_labels"] = len(seen_labels)
    return rep
