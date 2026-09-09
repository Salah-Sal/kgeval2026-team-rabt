# AdaptNER preflight (Kaggle)

Bounded kernel that proves, on the target T4 GPU, everything the training
campaign depends on: bundle mounting, `kgeval` parsing of all three data
families, AraBERTv02 download, tokenizer word-alignment, and one real
train step of the 21-head model. Run this before any long campaign
(free-compute skill: preflight-then-campaign).

## Data rules

The input bundle contains **licensed competition data** (Wojood,
WojoodRelations, Konooz dev). It must be a **private** Kaggle dataset, always —
never make it public, never attach it to a public kernel. The kernel exports
safe artifacts only (`PREFLIGHT_COMPLETED.json` — counts, timings, checks; no
dataset text). Staging lives under the gitignored `output/`.

## Commands (from the repo root)

```sh
# 1. Build the staging dir (whitelist copier; excludes ~$ lock files etc.)
uv run python kaggle/adaptner-preflight/prepare_assets.py --clean

# 2. Create the private dataset (first time) / push a new version (later)
#    -r zip is REQUIRED: the default dir-mode is "skip", which silently ignores
#    every directory (v1 of this bundle uploaded only bundle-manifest.tsv).
#    -t keeps tabular files verbatim (no CSV auto-conversion).
uv run kaggle datasets create -p output/staging/kgeval2026-bundle -r zip -t
uv run kaggle datasets version -p output/staging/kgeval2026-bundle -m "<change note>" -r zip -t

# 3. Launch
uv run kaggle kernels push -p kaggle/adaptner-preflight/kernel

# 4. Poll / fetch
uv run kaggle kernels status salah1992/kgeval-2026-adaptner-preflight
uv run kaggle kernels output salah1992/kgeval-2026-adaptner-preflight \
  -p output/adaptner-preflight --force
```

The fetch is trusted only if `PREFLIGHT_COMPLETED.json` exists and every entry
under `checks` is true. The bundle ships a `code/src/kgeval` snapshot — after
changing `src/kgeval`, push a new dataset version before relaunching kernels.

## Mount layout (2026 image)

Datasets are NOT mounted flat at `/kaggle/input/<slug>` anymore; the bundle
appears at `/kaggle/input/datasets/<owner>/<slug>/` (verified in the v4 log).
Kernel drivers should still discover it via the recursive-glob bootstrap
(which necessarily lives inline in each driver — it runs before `kgeval` is
importable) rather than hardcoding, in case the layout shifts again.
