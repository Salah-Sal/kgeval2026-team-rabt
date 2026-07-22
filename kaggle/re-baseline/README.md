# RE typed-marker baseline (Kaggle)

Design §4.1–4.2, §4.5: AraBERTv02 + Zhou & Chen punctuation typed markers
(`@ * type * subj @` / `# ^ type ^ obj #`), FFNN over the concatenated
start-marker hidden states, 41-way CE training with all 8,092 negatives kept,
no_relation masked at inference (forced choice over the 40 positives).
Entity types: sentence_id → gold Wojood entities (all three splits, typing
only) → corpus lexicon → UNK. Local validation on the grouped split
(seed 13, val_frac 0.125), scored positives-only.

Code lives in `src/kgeval` (`re_markers`, `re_typing`, `re_model`,
`re_train`); the kernel is a thin driver. Same data rules and workflow as
`kaggle/adaptner-baseline/`: private bundle, safe artifacts only
(`RE_CAMPAIGN_COMPLETED.json`, `re_history.json`, `best_re_model.pt`),
smoke-first (`SMOKE = True` → verify → flip to `False`).

```sh
uv run kaggle kernels push -p kaggle/re-baseline/kernel
uv run kaggle kernels status salah1992/kgeval-2026-re-baseline
uv run kaggle kernels output salah1992/kgeval-2026-re-baseline \
  -p output/re-baseline --force
```

Full runs gate on best val micro-F1 > 0.85 (design expects ≥92; RIFRE 92.89
on the published split).
