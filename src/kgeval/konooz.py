"""Reader for the fixed 22-field column format (Konooz dev/test + submissions).

Each non-blank line is `token TAG×21`, single-space separated, tag columns in
the fixed order of `columns.ENTITY_TYPES`. Blank lines separate sentences.
The released dev files use CRLF line endings; the reader detects and preserves
the EOL so writers can mirror the reference bytes exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .columns import ENTITY_TYPES, NUM_TYPES, valid_column_tags


@dataclass
class ColumnFile:
    path: str
    eol: str                          # "\r\n" or "\n" (dominant terminator)
    lines: list[str]                  # raw lines without terminators
    sentences: list[list[str]]        # tokens per sentence
    tags: list[list[list[str]]]       # per sentence, per token, 21 tags
    ends_with_blank_line: bool

    @property
    def n_tokens(self) -> int:
        return sum(len(s) for s in self.sentences)


def read_column_file(path: str | Path, validate_tags: bool = True) -> ColumnFile:
    data = Path(path).read_bytes()
    text = data.decode("utf-8")
    eol = "\r\n" if "\r\n" in text[:20000] or "\r\n" in text else "\n"
    # splitlines() handles both terminators; keep raw lines for structure cloning.
    lines = text.split(eol)
    trailing_empty = lines and lines[-1] == ""
    if trailing_empty:
        lines = lines[:-1]  # final terminator, not a real blank line

    sentences: list[list[str]] = []
    tags: list[list[list[str]]] = []
    cur_tok: list[str] = []
    cur_tags: list[list[str]] = []
    for lineno, line in enumerate(lines, start=1):
        if line.strip() == "":
            if cur_tok:
                sentences.append(cur_tok)
                tags.append(cur_tags)
                cur_tok, cur_tags = [], []
            continue
        fields = line.split(" ")
        if len(fields) != 1 + NUM_TYPES:
            raise ValueError(
                f"{path}:{lineno}: expected {1 + NUM_TYPES} space-separated fields, "
                f"got {len(fields)}: {line!r}"
            )
        token, row_tags = fields[0], fields[1:]
        if validate_tags:
            for ci, tag in enumerate(row_tags):
                if tag not in valid_column_tags(ENTITY_TYPES[ci]):
                    raise ValueError(
                        f"{path}:{lineno}: tag {tag!r} invalid in column "
                        f"{ci} ({ENTITY_TYPES[ci]})"
                    )
        cur_tok.append(token)
        cur_tags.append(row_tags)
    if cur_tok:
        sentences.append(cur_tok)
        tags.append(cur_tags)

    ends_with_blank = bool(lines) and lines[-1].strip() == ""
    return ColumnFile(
        path=str(path),
        eol=eol,
        lines=lines,
        sentences=sentences,
        tags=tags,
        ends_with_blank_line=ends_with_blank,
    )


def domain_files(reference_dir: str | Path) -> list[Path]:
    """The domain .txt files of a Konooz release in the contract's sorted order."""
    files = sorted(Path(reference_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt domain files under {reference_dir}")
    return files
