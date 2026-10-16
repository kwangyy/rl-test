# Data: what is in git, and how to get the rest

Raw and prepared datasets live under `verifier/data/`; run artifacts live
under `verifier/experiments/` and fresh evaluation results under
`verifier/evaluation/results/`. The raw datasets and retrieval index are about
21 GB, so they are downloaded and built once per machine with the steps below.

## 1. Download the raw datasets

Run from the repo root. Needs `curl`, `unzip` and `tar` (Git Bash on
Windows has all three).

```bash
# FEVER 1.0: claims (train, shared-task dev) and the 2017 Wikipedia dump
mkdir -p verifier/data/fever && cd verifier/data/fever
curl -LO https://fever.ai/download/fever/train.jsonl              # 33 MB
curl -LO https://fever.ai/download/fever/shared_task_dev.jsonl    # 4.3 MB
curl -LO https://fever.ai/download/fever/wiki-pages.zip           # 1.7 GB
unzip -q wiki-pages.zip && rm wiki-pages.zip                      # -> wiki-pages/wiki-001..109.jsonl
cd ../../..

# SciFact: claims and the abstract corpus
mkdir -p verifier/data/scifact && cd verifier/data/scifact
curl -LO https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz   # 3 MB
tar -xzf data.tar.gz                                              # -> data/claims_*.jsonl, corpus.jsonl
cd ../../..

# AVeriTeC: train and dev claims with QA evidence (official repo)
mkdir -p verifier/data/averitec && cd verifier/data/averitec
curl -LO https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/train.json   # 10 MB
curl -LO https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json     # 1.8 MB
cd ../../..
```

Sources: [fever.ai](https://fever.ai/dataset/fever.html),
[SciFact](https://github.com/allenai/scifact),
[AVeriTeC](https://github.com/MichSchli/AVeriTeC). The sizes above match
the copies the results in this repo were produced from. FEVER is
CC BY-SA 3.0 (Wikipedia-derived); check each dataset's license before
redistributing.

## 2. Build the retrieval index

```bash
python -m retrieval.build_bm25_index     # writes verifier/data/fever/bm25_index/ and verifier/data/fever/pages.sqlite
python -m retrieval.eval_retrieval       # optional check: page recall@5 should match prompted-baseline-results.md
```

The index is about 2.3 GB and `pages.sqlite` about 5.4 GB. The build runs once.

## 3. Expected layout

```
verifier/data/
  fever/          train.jsonl, shared_task_dev.jsonl, wiki-pages/, bm25_index/, pages.sqlite   (not in git)
  scifact/data/   claims_{train,dev,test}.jsonl, corpus.jsonl, cross_validation/               (not in git)
  averitec/       train.json, dev.json                                                         (not in git)
  train/sft/      model-independent SFT labeled_prompts.jsonl sets                         (in git)
  train/grpo/     GRPO labeled_prompts.jsonl sets and disagreement samples                 (in git)
  eval/           historical dev IDs; evaluation-v1 audit, calibration and holdout manifests (in git)
  traces/         archived agent search/verify action histories                             (in git)
verifier/experiments/<configuration>/runs/<run>/artifacts/
  meta.json, logs/train_log.json, checkpoints/adapter_config.json, optional rollouts.jsonl (in git)
  checkpoints/*.safetensors and checkpoint-*/                                             (not in git)
verifier/evaluation/results/archive/        archived per-claim evaluation outputs                           (in git)
verifier/evaluation/results/<condition>/<run>/  fresh predictions, metrics, manifest and logs                (not in git)
```

## 4. What is left out of git, and how to get it back

| Left out | Size | How to regenerate |
|---|---|---|
| LoRA adapters (`adapter_model.safetensors`) | 65 MB each | Rerun `verifier/experiments/<config>/train.py --run-name <new-run>` after downloading raw data and restoring any required SFT adapter; settings come from the historical metadata |
| Trainer checkpoints (`checkpoint-*/`) | 220 MB each | Written during training; only needed to resume a run |
| Tokenizer and chat template copies, `training_args.bin`, TRL model cards | 20 MB per run | Written next to every adapter; identical to the base model's |
| Duplicate 4B SFT training JSONL | 30 MB | The same model-independent `verifier/data/train/sft/fever-oracle-full/labeled_prompts.jsonl` is shared by the 2B and 4B configurations |

Paths inside historical `meta.json` files are absolute paths from the
machine that ran them (`/mnt/c/...` under WSL2). They describe those old
runs and may point to the former `data/` artifact layout; use the current
`verifier/experiments/<configuration>/runs/<run>/artifacts/` location for files in
this branch.

The historical terminal `.log` files were removed because their useful
results are in the per-claim JSONL, `meta.json`, structured `train_log.json`
or the experiment documents. Future raw terminal logs under `verifier/data/` are
ignored; new SoC job logs belong under their experiment run's
`artifacts/logs/` for training, or under `verifier/evaluation/results/<condition>/<run>/logs/` for evaluation; terminal logs are ignored.

## 5. Fixed SFT/GRPO claim split

`python -m utils.build_disjoint_sft_grpo_split --check` verifies the frozen 5,895-row oracle set and its claim-disjoint 4,695/1,200 split. The 1,200-row side reuses `sft/fever-oracle-matched-grpo-1200/labeled_prompts.jsonl`; the 4,695-row complement is `sft/fever-oracle-disjoint-4695/labeled_prompts.jsonl`. `split_manifest.json` records SHA-256 hashes, row and label counts, and zero overlap with the frozen 900 dev IDs. To regenerate the 4,695-row side from those committed inputs, run `python -m utils.build_disjoint_sft_grpo_split` from the repo root.

## 6. Frozen development, calibration and holdout sets

Frozen ID lists live in `verifier/data/eval/fever-evaluation-v1/`:

| File | Role | Claims |
|---|---|---:|
| `../fever-dev-900-ids.json` | Unchanged development benchmark for prompt, model, retrieval and policy decisions | 900 |
| `development-used-ids.json` | All previously used dev claims, including those 900 | 945 |
| `calibration-900-ids.json` | Fit confidence adjustments after selecting the model/policy | 900 (300 per label) |
| `holdout-3000-ids.json` | Compare finalized verifier and agent configurations | 3,000 (1,000 per label) |

The [usage audit](../data/eval/fever-evaluation-v1/usage-audit.json) found 7,138 used train IDs and 945 used dev IDs across local artifacts and reachable Git history, including deleted files. Calibration and holdout were sampled from official FEVER dev, excluding recorded IDs, normalized text matches to used claims or the full official training pool, and duplicate candidates. The [manifest](../data/eval/fever-evaluation-v1/manifest.json) records the selection method, seed and file hashes. Unused status covers available project records only.

Never use calibration or holdout for model/controller training, teacher targets or prompt/policy tuning. If holdout results guide further changes, treat that set as development and freeze a new version for independent validation. Report the balanced label sampling, since it affects precision and calibration. Agent comparisons keep the verifier, corpus/tools and budget matched.

Run checks from the repo root after downloading the raw FEVER train/dev files:

```bash
# Routine check: unchanged files, reproducible splits and no reserved IDs in training
python -m utils.build_fever_eval_splits --check

# Before the first calibration/holdout run: also require no recorded usage
python -m utils.build_fever_eval_splits --check-unused

# Check a proposed training/controller dataset; each row needs id or claim_id
python -m utils.build_fever_eval_splits --check-training path/to/training.jsonl
```

`--check-unused` includes `--check` and rescans current artifacts and reachable history. It is expected to fail after either reserved set appears in recorded runs, including intended final evaluations; continue using `--check` afterward. The builder refuses to overwrite frozen sets. Runtime leakage guards compare claim IDs.

Select a full set with `--claim-ids <ID-file>` in verifier, agent and retrieval evaluators; this overrides `--n` (retrieval page recall excludes NEI). For example, once the configuration is finalized:

```bash
python -m verifier.evaluation.eval_verifier --model local --retrieval titles \
  --prompt-style direct --max-new-tokens 1024 \
  --claim-ids verifier/data/eval/fever-evaluation-v1/holdout-3000-ids.json \
  --output verifier/evaluation/results/baseline-direct-cap1024/holdout-v1-run1/predictions.jsonl
```

For Slurm, export `CLAIM_IDS=<ID-file>` to `verifier/evaluation/run.sbatch`; it overrides `N_CLAIMS`.
