# AdaptNER baseline campaign (Kaggle)

Trains the design-doc §3.1–3.2 baseline on the T4: AraBERTv02 + 21 BIO heads,
A10 unified loss (0.4·CE + 0.2·Dice + 0.2·Tversky(α=0.5) + 0.2·Focal(γ=2),
cross-head variance penalty p=5), lr 1e-5 AdamW, batch 16 dynamic-padded,
early stopping (patience 5) on Wojood-val span micro-F1, then Wojood-test
regression numbers. All model/training code lives in `src/kgeval`
(`ner_data`, `ner_model`, `ner_train`) and ships inside the private bundle's
`code/` snapshot — the kernel is a thin driver.

## Data rules

Same as the preflight: the bundle is licensed competition data and stays a
**private** Kaggle dataset. The kernel exports safe artifacts only — metrics
(`CAMPAIGN_COMPLETED.json`, `history.json`) and `best_model.pt` (our
weights). Predictions over dataset text stay in the ephemeral workspace.

## Workflow

Run in `SMOKE = True` mode first (tiny subset, 2 epochs, ~5 min) after any
code change; flip to `False` for the real campaign only after the smoke
sentinel lands with `"ok": true`.

```sh
# after changing src/kgeval: re-stage + re-version the bundle first
uv run python kaggle/adaptner-preflight/prepare_assets.py --clean
uv run kaggle datasets version -p .data-cache/kaggle-staging/kgeval2026-bundle -m "<note>" -r zip -t

uv run kaggle kernels push -p kaggle/adaptner-baseline/kernel
uv run kaggle kernels status salah1992/kgeval-2026-adaptner-baseline
uv run kaggle kernels output salah1992/kgeval-2026-adaptner-baseline \
  -p output/adaptner-baseline --force
```

The fetch is trusted only if `CAMPAIGN_COMPLETED.json` exists with
`"ok": true` (full runs require best val F1 > 0.85; published nested-Wojood
baselines sit at ~0.88–0.92).
