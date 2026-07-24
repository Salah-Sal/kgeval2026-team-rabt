# System-description paper (ACL LaTeX)

`main.tex` is the master document for the Team Rabt paper. It is kept in
sync with Salah's working repo; edit here (branch or main, either is fine)
and we will reconcile.

## Build

```sh
brew install tectonic   # or: cargo install tectonic; see tectonic.dev
./build.sh              # produces main.pdf (~3 s; first run downloads packages)
```

Any TeX Live installation also works: `latexmk -pdf main.tex`.

## Conventions

- Red **[FINAL: ...]** markers: items blocked on the July 29 leaderboard
  close (final ranks, conclusion, leaderboard context).
- Red **[CHECK: ...]** markers: claims that need verification before they
  can stand (currently one, in Section 6.2).
- `references.bib` entries were fetched verbatim from the ACL Anthology.
  The one hand-written entry is W2NER (`li-etal-2022-unified-w2ner`,
  AAAI 2022): verify authors/pages against the published version.
- Document mode is `[preprint]` (authors shown, page numbers on). Switch
  to `[final]` for camera-ready. Page limit and template are not yet
  confirmed by the organizers; the current text is ~6.5 pages of main
  matter, which fits a long-paper budget but needs trimming if the limit
  is 4 pages.
