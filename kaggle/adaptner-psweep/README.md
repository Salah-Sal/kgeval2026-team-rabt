# AdaptNER variance-penalty sensitivity sweep (Kaggle)

Camera-ready follow-up to reviewer WUYr's request (OpenReview review of
Submission 5, answered in the 2026-09-04 rebuttal): the submitted recipe
fixes the variance-penalty weight at 5.0, taken unchanged from El Mahdaouy
et al. (2023), and the August ablation tested only its presence (E5)
versus absence (E4). This kernel adds three points on the same axis:

| Exp | loss_lambdas (CE, Dice, Tversky, Focal) | var_penalty |
|---|---|---|
| E4 | (0.4, 0.2, 0.2, 0.2) | 0.0 (reference, August 3 ablation; not re-run) |
| P1 | (0.4, 0.2, 0.2, 0.2) | 1.0 |
| P2 | (0.4, 0.2, 0.2, 0.2) | 2.0 |
| E5 | (0.4, 0.2, 0.2, 0.2) | 5.0 (reference, submitted baseline; not re-run) |
| P10 | (0.4, 0.2, 0.2, 0.2) | 10.0 |

Everything else matches the ablation campaign: Wojood train split, early
stopping on Wojood-val span micro-F1 (patience 5, <=40 epochs), seed 13,
then Wojood val and test micro/macro-F1 with per-type counts computed in
the same kernel (same macro definition as `adaptner-ablation-eval`, so no
chained eval run is needed). Development-side only; never touches Konooz
or Codabench. Expected cost 3 to 5 T4 hours (3 x 60 to 105 min, from the
ablation timings) inside one session; run order P1, P10, P2 so that a
session-budget skip costs the middle point, not an extreme.

## Data rules

Same as the baseline: the bundle is licensed competition data and stays a
**private** Kaggle dataset. The kernel exports safe artifacts only:
`CAMPAIGN_COMPLETED.json`, per-experiment `P*/history.json` and
`P*/best_model.pt` (our weights). No dataset text leaves the kernel.

## Workflow

Kaggle derives the kernel slug from the title, so the metadata `id` is the
title slug (`kgeval-2026-adaptner-variance-penalty-sweep`), not the
directory name. Run in `SMOKE = True` mode first (tiny subset, 2 epochs each, about 10
min); flip to `False` for the sweep only after the smoke sentinel lands
with `"ok": true`. Bundle re-versioning is only needed after `src/kgeval`
changes (none since bundle-v8).

```sh
uv run kaggle kernels push -p kaggle/adaptner-psweep/kernel
uv run kaggle kernels status salah1992/kgeval-2026-adaptner-variance-penalty-sweep
uv run kaggle kernels output salah1992/kgeval-2026-adaptner-variance-penalty-sweep \
  -p output/adaptner-psweep --force
```

The fetch is trusted only if `CAMPAIGN_COMPLETED.json` exists with
`"ok": true` (full runs: all three experiments attempted, none with
best-val F1 <= 0.85, which on in-domain Wojood indicates a broken run
rather than a legitimate result). Results go to the ledger as
`adaptner-psweep-v1` and to Appendix H of `paper/latex/main.tex`.
