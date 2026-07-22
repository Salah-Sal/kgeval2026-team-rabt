# AdaptNER dev-Konooz eval (Kaggle)

Loads the baseline campaign's `best_model.pt` (attached via `kernel_sources`)
and runs it over dev-Konooz. Dev-Konooz is **all-O placeholder tags** (design
doc appendix) — there is no score to compute; the kernel instead reports
predicted-span statistics per domain/type (a model-collapse detector on real
Konooz text) and rehearses the submission writer + independent validator
against the dev reference. The rehearsal zip contains Konooz text, so it is
deleted in-kernel — only metrics are exported
(`ADAPTNER_EVAL_COMPLETED.json`). Domain-shift model selection stays on
Wojood-test + leave-subcorpus-out (design §3.4).

Run after each new baseline campaign version; the checkpoint is discovered by
glob, and the model is rebuilt from the config recorded in `history.json`
(works for both the multi-head and sigmoid variants).

```sh
uv run kaggle kernels push -p kaggle/adaptner-eval/kernel
uv run kaggle kernels status salah1992/kgeval-2026-adaptner-eval
uv run kaggle kernels output salah1992/kgeval-2026-adaptner-eval \
  -p output/adaptner-eval --force
```
