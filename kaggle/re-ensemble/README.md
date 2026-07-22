# RE test-phase ensemble campaign (Kaggle)

5-fold × seed logit-averaging ensemble over the typed-marker classifier:
each fold trains on 4/5 of WojoodRelations train (grouped by sentence_id,
seed 13+fold) with early stopping on its own held-out fold; blind-test
decoding averages the five logit vectors, then applies the soft constraint
penalty (3.0) and the no_relation mask. Pooled out-of-fold predictions
re-validate the penalty sweep on the full training set.

Deviation from v1, on purpose: the constraint table uses ALL training
records (v1 used the grouped-split train portion only).

## Data rules

Bundle stays a **private** Kaggle dataset. Exports are safe artifacts only:
`re_submission_ens.zip` (triple_id + label), `re_ens_logits.json` (averaged
logits + type pairs — enables local re-decoding at any penalty), per-fold
`re_history.json`, `RE_ENSEMBLE_COMPLETED.json`.

## Workflow

```sh
# src/kgeval changed (grouped_kfold) → bundle must be v8+
uv run python kaggle/adaptner-preflight/prepare_assets.py --clean
uv run kaggle datasets version -p output/staging/kgeval2026-bundle -m "<note>" -r zip -t

uv run kaggle kernels push -p kaggle/re-ensemble/kernel
uv run kaggle kernels status salah1992/kgeval-2026-re-ensemble
uv run kaggle kernels output salah1992/kgeval-2026-re-ensemble \
  -p output/re-ensemble --force
```

Expected wall time ≈ 2–3 h. Trust the fetch only if
`RE_ENSEMBLE_COMPLETED.json` has `"ok": true` (validation pass, every fold
≥ 0.90 val micro-F1, OOF micro-F1 at penalty 3.0 ≥ 0.93). The submission
zip is exported ready; copy to `output/submissions/` under a
versioned name before upload.
