# AdaptNER blind-test inference (Kaggle)

Runs the baseline campaign's `best_model.pt` (attached via `kernel_sources`)
over the blind Konooz test release (`datasets/blinded-test-data`, 10 domains,
~50K tokens, all-O placeholder columns). Exports **safe artifacts only**:

- `adaptner_test_tags.json` — per-domain 21-tag rows, our model's output,
  zero Konooz text. The submission zip is assembled locally with
  `kgeval.adaptner_submission.write_submission()` against the pristine local
  test files, then validated before upload to Codabench.
- `ADAPTNER_TEST_COMPLETED.json` — span/repair statistics per domain/type,
  in-kernel rehearsal verdict (the rehearsal zip is built, validated, and
  deleted in-kernel because it contains licensed tokens).

```sh
uv run kaggle kernels push -p kaggle/adaptner-test/kernel
uv run kaggle kernels status salah1992/kgeval-2026-adaptner-test
uv run kaggle kernels output salah1992/kgeval-2026-adaptner-test \
  -p output/adaptner-test --force
```

Local assembly after fetching the output:

```sh
uv run python scripts/assemble_adaptner_submission.py
```
