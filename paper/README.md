# System-description paper (ACL LaTeX)

`main.tex` is the master document for the Team Wasl paper (renamed from
Rabt on 2026-08-05 at the organizers' request). It is kept in sync with
Salah's working repo; edit here (branch or main, either is fine) and we
will reconcile.

## Build

```sh
brew install tectonic   # or: cargo install tectonic; see tectonic.dev
./build.sh              # produces main.pdf (~3 s; first run downloads packages)
```

The paper requires a XeTeX engine: the Arabic examples in the error
analysis use fontspec + bidi with the Amiri font, which tectonic resolves
from its TeX bundle out of the box. With a plain TeX Live installation,
use `latexmk -xelatex main.tex` after `tlmgr install amiri bidi`.

## Conventions

- Document mode is `[preprint]` (authors shown, page numbers on). Switch
  to `[final]` for submission: system papers are not anonymous, and the
  guidelines list page numbers among the formatting deviations that can
  get a paper rejected unreviewed.
- Guidelines (2026-08-05): at most 4 pages of main content excluding
  references, EMNLP 2026 ACL template. The current draft lands main
  content on 4 pages, with extended results in the appendix.
- Arabic runs are typeset with the `\arx{...}` macro (RTL, Amiri).
- `references.bib` entries were fetched verbatim from the ACL Anthology.
  The one hand-written entry is W2NER (`li-etal-2022-unified-w2ner`,
  AAAI 2022), currently uncited; verify it against the published version
  before citing.
