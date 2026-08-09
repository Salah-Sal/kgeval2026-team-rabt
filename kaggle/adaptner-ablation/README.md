# AdaptNER loss-ablation campaign (Kaggle)

Runs Faisal's E1–E4 ablation grid (his 2026-08-01 experiment note) to
quantify each component of the A10 unified loss for the system paper:

| Exp | loss_lambdas (CE, Dice, Tversky, Focal) | var_penalty |
|---|---|---|
| E1 | (1.0, 0, 0, 0) | 0.0 |
| E2 | (0.5, 0.5, 0, 0) | 0.0 |
| E3 | (0.4, 0.3, 0.3, 0) | 0.0 |
| E4 | (0.4, 0.2, 0.2, 0.2) | 0.0 |
| E5 | (0.4, 0.2, 0.2, 0.2) | 5.0 — **not re-run**: this is the submitted baseline (adaptner-baseline-v2, val 0.9208 / test 0.9240) |

Everything else matches the baseline campaign: Wojood train split, early
stopping on Wojood-val span micro-F1 (patience 5, ≤40 epochs), seed 13,
then Wojood-test regression numbers per experiment. These are
development-side results for the paper's ablation table; they never touch
Konooz or Codabench. Expected cost ~6 T4 hours (4 × ~90 min) inside one
session; a session-budget guard skips remaining experiments rather than
dying mid-run.

## Data rules

Same as the baseline: the bundle is licensed competition data and stays a
**private** Kaggle dataset. The kernel exports safe artifacts only —
`CAMPAIGN_COMPLETED.json`, per-experiment `E*/history.json` and
`E*/best_model.pt` (our weights). No dataset text leaves the kernel.

## Workflow

Run in `SMOKE = True` mode first (tiny subset, 2 epochs each, ~10 min);
flip to `False` for the campaign only after the smoke sentinel lands with
`"ok": true`. Bundle re-versioning is only needed after `src/kgeval`
changes (none since bundle-v8).

```sh
uv run kaggle kernels push -p kaggle/adaptner-ablation/kernel
uv run kaggle kernels status salah1992/kgeval-2026-adaptner-ablation
uv run kaggle kernels output salah1992/kgeval-2026-adaptner-ablation \
  -p output/adaptner-ablation --force
```

The fetch is trusted only if `CAMPAIGN_COMPLETED.json` exists with
`"ok": true` (full runs: all four experiments attempted, none with
best-val F1 ≤ 0.85, which on in-domain Wojood indicates a broken run
rather than a legitimate ablation result).
