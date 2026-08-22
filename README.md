# Team Wasl: KnowledgeGraphEval 2026 (ArabicNLP 2026)

Code companion to our system description paper (submitted 2026-08-22,
under review for the ArabicNLP 2026 proceedings; Anthology link to
follow). Cite via `CITATION.cff`.

System code for both subtasks of the KnowledgeGraphEval 2026 shared task.
Team: Salah Abdo, Faisal Muhammad Adam. We competed as Team Rabt and were
renamed to Wasl (وَصْل, joining) on 2026-08-05 at the organizers' request,
after a name collision with another team; the official results tables and
the artifact names (this repo, the Kaggle kernels, the Hugging Face org)
still say Rabt.

## Subtasks and results

**Subtask 1 — AdaptNER** (nested Arabic NER under domain shift: train on
Wojood, blind-test on Konooz, 21 entity types). AraBERTv02 encoder with 21
per-type BIO heads, trained with a unified loss (CE + Dice + Tversky + Focal,
variance-penalized across heads); exact-match IOB2 span decoding with
orphan-I repair (local scorer reconciled against submission rehearsals;
official Codabench numbers are authoritative). Official blind test:
**0.7483 micro-F1, 4th of the eight officially ranked teams** (3-seed
majority-vote ensemble; single model 0.7385). In-domain Wojood test:
0.9329 ensemble / 0.9240 single (published ArabicNER baseline: 88.4).

**Subtask 2 — Relation Extraction** (41-way relation classification over
given entity pairs, WojoodRelations). Typed punctuation markers
(Zhou & Chen) inserted around subject/object with entity types recovered via
sentence-id join against gold Wojood, a FFNN over the start-marker states,
and soft domain/range constraints (logit penalty 3.0) at inference,
ensembled over 5 grouped folds by logit averaging. Official blind test:
**0.9571 micro-F1, 2nd of the eight officially ranked teams** (single
model 0.9519); pooled out-of-fold 0.95995.

A post-submission loss ablation (`kaggle/adaptner-ablation*/`) found that
plain cross-entropy slightly outperforms the submitted unified-loss
recipe in domain (single-seed runs) and that the variance penalty acts
as a guard against rare-type collapse rather than a lever over CE; the
paper reports this negative result in full.

## Layout

```
src/kgeval/     the system: data loaders, models, training, span decoding,
                submission writers + independent format validators
tests/          pytest suite — fully offline, synthetic fixtures (no task data)
kaggle/         kernel drivers used to run training/inference/ablations on
                Kaggle T4s; each directory has a README with
                push/monitor/fetch commands
scripts/        local submission assembly (merges Kaggle tag exports with the
                released test files) and the paper's error-analysis extraction
paper/          system-description paper, ACL LaTeX (see paper/README.md)
notebooks/      pedagogical Colab companions (one per subtask) — self-contained
                walkthroughs of both systems on a synthetic Arabic toy corpus,
                with a FULL_REPLICATION flag restoring the paper's configs
companions/     reading material: the paper companion PDF (light + dark) and
                the companion slide deck (pptx)
```

## Setup

Python via [uv](https://docs.astral.sh/uv/):

```sh
uv sync            # core + dev (pytest, kaggle CLI)
uv run pytest      # offline test suite
uv sync --group train   # torch/transformers, only needed off-Kaggle
```

## Data

The task datasets (Wojood, WojoodRelations, Konooz dev/test) are licensed by
the organizers and are **not redistributed here** — obtain them from the
shared task and place them under `data/` (gitignored). Per the organizers'
rules, Konooz is used for development and evaluation only; all training is on
Wojood train+dev.

## Status

Private until the system-description paper is submitted; it then goes
public as the paper's code companion.
