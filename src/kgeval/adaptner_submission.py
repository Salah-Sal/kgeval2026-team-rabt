"""AdaptNER submission writer + independent validator.

Contract (design doc §1.1): one zip containing one file whose name contains
`_pred_`; all 10 domain files concatenated in sorted filename order; each token
line = `token TAG×21` in the fixed column order; token text byte-identical to
the released test files; blank lines separate sentences; mirror the reference
EOL (dev uses CRLF).

The writer never constructs structure — it clones the reference files line by
line and only replaces the 21 tag fields on token lines. That makes token
order, blank-line placement, and EOL correct by construction. Whether the
official sample inserts a blank line *between* domain files is an open contract
item; the writer auto-inserts one only when the previous file does not already
end blank (no sane CoNLL reader merges sentences then), and the validator
accepts and records that optional separator.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .columns import ENTITY_TYPES, KONOOZ_DOMAINS, NUM_TYPES, valid_column_tags
from .konooz import domain_files, read_column_file

TagRow = list[str]  # 21 tags in contract order


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


def write_submission(
    reference_dir: str | Path,
    tags_by_domain: dict[str, list[TagRow]],
    out_zip: str | Path,
    member_name: str = "teamrabt_pred_v1.txt",
) -> dict:
    """tags_by_domain: domain stem → per-token 21-tag rows, flat in file order."""
    if "_pred_" not in member_name:
        raise ValueError(f"member name must contain '_pred_': {member_name!r}")
    files = domain_files(reference_dir)
    stems = [f.stem for f in files]
    if set(tags_by_domain) != set(stems):
        raise ValueError(
            f"domain mismatch: reference has {stems}, predictions for {sorted(tags_by_domain)}"
        )
    if tuple(stems) != KONOOZ_DOMAINS:
        # Not fatal — the constant is a cross-check; the reference dir is authoritative.
        print(f"NOTE: reference domains {stems} differ from expected {list(KONOOZ_DOMAINS)}")

    chunks: list[bytes] = []
    inserted_blanks = 0
    n_token_lines = 0
    for f in files:
        ref = read_column_file(f, validate_tags=False)
        rows = iter(tags_by_domain[f.stem])
        used = 0
        for line in ref.lines:
            if line.strip() == "":
                chunks.append(ref.eol.encode("utf-8"))
                continue
            token = line.split(" ")[0]
            try:
                tags = next(rows)
            except StopIteration:
                raise ValueError(f"{f.stem}: fewer prediction rows than token lines") from None
            if len(tags) != NUM_TYPES:
                raise ValueError(f"{f.stem}: prediction row has {len(tags)} tags, need {NUM_TYPES}")
            chunks.append((token + " " + " ".join(tags) + ref.eol).encode("utf-8"))
            used += 1
            n_token_lines += 1
        leftover = sum(1 for _ in rows)
        if leftover:
            raise ValueError(f"{f.stem}: {leftover} prediction rows beyond the {used} token lines")
        if ref.lines and ref.lines[-1].strip() != "" and f != files[-1]:
            chunks.append(ref.eol.encode("utf-8"))  # keep domains sentence-separated
            inserted_blanks += 1

    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(member_name, b"".join(chunks))
    return {
        "zip": str(out_zip),
        "member": member_name,
        "domains": stems,
        "token_lines": n_token_lines,
        "inter_domain_blanks_inserted": inserted_blanks,
    }


def validate_submission(zip_path: str | Path, reference_dir: str | Path) -> Report:
    """Re-check a submission zip against the released reference files, independently
    of how it was produced."""
    rep = Report()
    try:
        with zipfile.ZipFile(zip_path) as z:
            members = [n for n in z.namelist() if not n.endswith("/")]
            if len(members) != 1:
                rep.fail(f"zip must contain exactly one file, found {members}")
                return rep
            member = members[0]
            if "_pred_" not in Path(member).name:
                rep.fail(f"member name {member!r} does not contain '_pred_'")
            text = z.read(member).decode("utf-8")
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as e:
        rep.fail(f"cannot read zip: {e}")
        return rep

    crlf = text.count("\r\n")
    lf_only = text.count("\n") - crlf
    rep.stats["member"] = member
    rep.stats["eol"] = f"CRLF×{crlf}, bare-LF×{lf_only}"
    sub_lines = text.splitlines()
    # ignore trailing blank lines at EOF (recorded, not fatal)
    n_trailing = 0
    while sub_lines and sub_lines[-1].strip() == "":
        sub_lines.pop()
        n_trailing += 1

    idx = 0
    n_tokens = 0
    n_entity_tags = 0
    files = domain_files(reference_dir)
    for fi, f in enumerate(files):
        ref = read_column_file(f, validate_tags=False)
        ref_lines = list(ref.lines)
        # strip the reference file's own trailing blanks; separators between
        # files are handled uniformly below
        while ref_lines and ref_lines[-1].strip() == "":
            ref_lines.pop()
        if fi > 0:
            # optional single blank separator between domain files
            if idx < len(sub_lines) and sub_lines[idx].strip() == "":
                idx += 1
        for line in ref_lines:
            if idx >= len(sub_lines):
                rep.fail(f"{f.stem}: submission ended early (reference has more lines)")
                return rep
            sub = sub_lines[idx]
            if line.strip() == "":
                if sub.strip() != "":
                    rep.fail(f"line {idx + 1}: expected blank (sentence break in {f.stem}), got {sub[:60]!r}")
                idx += 1
                continue
            ref_token = line.split(" ")[0]
            fields = sub.split(" ")
            if len(fields) != 1 + NUM_TYPES:
                rep.fail(f"line {idx + 1} ({f.stem}): {len(fields)} fields, expected {1 + NUM_TYPES}")
                idx += 1
                continue
            if fields[0] != ref_token:
                rep.fail(f"line {idx + 1} ({f.stem}): token {fields[0]!r} != reference {ref_token!r}")
            for ci, tag in enumerate(fields[1:]):
                if tag not in valid_column_tags(ENTITY_TYPES[ci]):
                    rep.fail(
                        f"line {idx + 1} ({f.stem}): tag {tag!r} invalid in column {ci} ({ENTITY_TYPES[ci]})"
                    )
                elif tag != "O":
                    n_entity_tags += 1
            n_tokens += 1
            idx += 1
    leftovers = [l for l in sub_lines[idx:] if l.strip() != ""]
    if leftovers:
        rep.fail(f"{len(leftovers)} unexpected non-blank lines after the last reference token")

    rep.stats["token_lines_checked"] = n_tokens
    rep.stats["entity_tags"] = n_entity_tags
    rep.stats["trailing_blank_lines"] = n_trailing
    if n_entity_tags == 0:
        rep.warnings.append("submission is all-O (format-valid, but no entities predicted)")
    return rep
