# AdaptNER test-phase ensemble campaign (Kaggle)

Trains 3 seeds (13/42/77) on **Wojood train+val combined** with a fixed
29-epoch schedule (`select="last"` — epoch 28 was the val-selected best of
the train-only baseline and its curve is flat there), then runs blind Konooz
inference per seed and builds the ensemble variants in-kernel:

| variant | recipe | lever |
|---|---|---|
| seed13/42/77 | single models | reference |
| meanprob | argmax of seed-averaged softmax | balanced |
| union3 | span union, min_votes=1 | recall (P−R gap is 9.2 pts) |
| vote2 | majority vote, min_votes=2 | precision |

Every variant is scored on Wojood-test (in-domain sanity — Konooz is never
used for training/validation/selection per the organizers' ruling) and
rehearsed through `write_submission` + `validate_submission`; rehearsal zips
are deleted in-kernel.

## Data rules

Bundle stays a **private** Kaggle dataset. Exports are safe artifacts only:
`adaptner_ens_tags.json` (tags per variant per domain, zero Konooz text),
per-seed checkpoints + `history.json`, `ADAPTNER_ENSEMBLE_COMPLETED.json`.

## Workflow

```sh
# src/kgeval changed (spans union + select="last") → re-version the bundle first
uv run python kaggle/adaptner-preflight/prepare_assets.py --clean
uv run kaggle datasets version -p output/staging/kgeval2026-bundle -m "<note>" -r zip -t

uv run kaggle kernels push -p kaggle/adaptner-ensemble/kernel
uv run kaggle kernels status salah1992/kgeval-2026-adaptner-ensemble
uv run kaggle kernels output salah1992/kgeval-2026-adaptner-ensemble \
  -p output/adaptner-ensemble --force
```

Expected wall time ≈ 4.5 h (3 × ~80 min training + evals + inference).
Trust the fetch only if `ADAPTNER_ENSEMBLE_COMPLETED.json` has `"ok": true`
(all rehearsals pass, every seed ≥ 0.915 Wojood-test micro-F1, union ⊇ seeds).
Local assembly: `scripts/assemble_adaptner_ensemble.py <variant>`.
