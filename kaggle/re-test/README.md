# RE blind-test inference (Kaggle)

Loads `best_re_model.pt` from the RE baseline campaign (attached via
`kernel_sources`), types the released test pairs with the same
sentence_id→gold→lexicon chain used in training, and predicts with the
validated soft-constraint configuration (penalty 3.0, `no_relation` masked).

Unlike AdaptNER, the finished submission zip is itself a safe artifact —
`predictions.txt` is `triple_id<TAB>label` with no licensed sentence text —
so the kernel exports `re_submission.zip` directly, plus
`RE_TEST_COMPLETED.json` (typing stats, label histogram, validator verdict).

```sh
uv run kaggle kernels push -p kaggle/re-test/kernel
uv run kaggle kernels status salah1992/kgeval-2026-re-test
uv run kaggle kernels output salah1992/kgeval-2026-re-test \
  -p output/re-test --force
```
