# AdaptNER ablation eval (Kaggle)

Per-type / macro-F1 pass over the loss-ablation checkpoints. The campaign
kernel exports only micro P/R/F1 per experiment, but the imbalance-aware
loss components (Dice/Tversky/Focal, variance penalty) are motivated by
rare types, so macro-F1 and per-type counts are where their contribution
would show. The kernel chains the ablation kernel's output (E1–E4
checkpoints) and the baseline kernel's output (E5, the submitted recipe)
as inputs via `kernel_sources` and evaluates all five on Wojood val and
test with per-type (tp, fp, fn) counts; these numbers feed the paper's
ablation table. Exports metrics only (`EVAL_COMPLETED.json`); no dataset
text leaves the kernel.

```sh
uv run kaggle kernels push -p kaggle/adaptner-ablation-eval/kernel
uv run kaggle kernels status salah1992/kgeval-2026-adaptner-ablation-eval
uv run kaggle kernels output salah1992/kgeval-2026-adaptner-ablation-eval \
  -p output/adaptner-ablation-eval --force
```
