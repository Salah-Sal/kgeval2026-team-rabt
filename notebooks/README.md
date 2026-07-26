# Colab notebooks — Rabt at KnowledgeGraphEval 2026

Two self-contained, pedagogical notebooks that walk from raw text to the two
systems described in the paper (`paper/latex/main.tex`), one per subtask:

- **`01-adaptner-nested-ner.ipynb`** — Subtask 1: multi-head nested BIO
  tagging (21 per-type O/B/I heads over AraBERTv02, UM6P unified imbalance
  loss, IOB repair decoding, seed-ensemble vote).
- **`02-relation-extraction.ipynb`** — Subtask 2: typed-marker relation
  classification (Zhou & Chen punctuation markers, 41-way head, `no_relation`
  masking, soft type-constraint decoding, grouped fold ensembling).

Both notebooks run end-to-end on a free Colab T4 with **no external files**:
each builds a small synthetic Arabic toy corpus in-notebook so every mechanic
(data format → tensorization → model → loss → decode → score → ensemble) is
visible and runnable. A clearly marked **real data** section in each notebook
then shows how to point the same code at Wojood / WojoodRelations obtained
from the shared-task organizers — that path reproduces the paper's actual
numbers. The licensed datasets are never embedded in the notebooks.

## Running on Colab

1. Open <https://colab.research.google.com> → **File → Upload notebook** and
   pick the `.ipynb`.
2. **Runtime → Change runtime type → T4 GPU** (the paper's own compute tier).
3. **Runtime → Run all.** The default `QUICK` path finishes in minutes on the
   toy corpus; the optional real-data and full-replication sections say up
   front how long they take (~90 min for the NER baseline / ~68 min for the
   RE 5-fold ensemble on T4).

## The two flags

Each notebook has one config cell near the top:

- `FULL_REPLICATION` (env `KGEVAL_FULL=1`): use the paper's exact
  hyperparameters instead of the quick path. Default off.
- `MODEL_NAME` (env `KGEVAL_MODEL`): encoder to fine-tune. Default
  `aubmindlab/bert-base-arabertv02` (the paper's encoder). The local smoke
  test overrides this with a tiny model.

## Local checks (repo side)

```sh
uv sync --group dev --group notebooks --group train
uv run pytest tests/test_notebooks.py          # static guards (fast, CPU)
```

The test suite enforces: valid notebooks, error-free outputs, no
licensed-data paths (sources AND outputs are scanned), and — important — the
`%%writefile kgeval/*.py` cells that vendor the package into each notebook
match `src/kgeval/` byte for byte. **If you change `src/kgeval/`, re-vendor
the changed module into both notebooks** (copy the file content into the
corresponding `%%writefile` cell), or the drift test fails.

Full local execution smoke (CPU, downloads a tiny English encoder — validates
mechanics, not quality). Run on a scratch copy so artifacts (`kgeval/`,
`toy_data/`, `runs/`) don't land in `notebooks/`:

```sh
mkdir -p .data-cache/notebook-smoke && cp notebooks/01-adaptner-nested-ner.ipynb .data-cache/notebook-smoke/
KGEVAL_MODEL=google/bert_uncased_L-2_H-128_A-2 uv run jupyter-execute \
    --timeout=1200 .data-cache/notebook-smoke/01-adaptner-nested-ner.ipynb
```

(`prajjwal1/bert-tiny` does not load under transformers ≥ 5 — its repo ships
only a slow-tokenizer vocab; the `google/bert_uncased_L-2_H-128_A-2` variant is
the same 2-layer tiny BERT with proper fast-tokenizer files.)

## Conventions

- Committed notebooks **carry executed outputs** — produced on a Colab T4
  against the synthetic toy corpora only, so no licensed data can appear.
  GitHub renders them inline, which is the point for a teaching artifact.
- Refreshing outputs: run the notebook on Colab (Runtime → Run all) and
  download, or use the `colab` CLI (`colab --auth adc new -s run --gpu T4`,
  `colab --auth adc exec -s run -f <notebook> --timeout 3000`, then
  `colab --auth adc stop -s run`). Never commit outputs from a real-data run.
- Code cells are written to be top-to-bottom runnable in a fresh runtime.
