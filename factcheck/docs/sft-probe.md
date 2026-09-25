# SFT capacity probe: 2B or 4B?

Decision experiment, set up 2026-09-25. Not yet run. Fill in the table
at the bottom and apply the rule in section 3.

## 1. Question

Stage 0 measured a 32-point gap between the 2B and the 180B model when
the gold evidence is in context (63% vs 95%). That gap is between two
models of very different size. It says nothing about how much of it a
2B model *can* close. Before GRPO time is spent, find the 2B model's
ceiling with the cheapest training signal there is: supervised
fine-tuning with the answer on the page.

If SFT with gold evidence in context cannot get the 2B model to roughly
80% accuracy, the limit is capacity, not training method, and stage 1
should run on Qwen3.5-4B.

## 2. Design

| | |
|---|---|
| Train data | FEVER train, balanced by label, default 6,000 claims |
| Context | "Oracle" retrieval: pages of the smallest gold evidence set are guaranteed, title-match pages fill to 5, order shuffled per claim. NEI claims get plain title-match pages. Same prompt and sentence cut as stage 0. |
| Target | `{"verdict": <gold>, "evidence": [<gold ids>], "confidence": 1.0}` |
| Method | LoRA r=16 on all linear layers, lr 1e-4, 1 epoch, batch 8 (1 x 8 accumulation), 2,048 max tokens. 4B uses 4-bit base weights (QLoRA). |
| Eval | `stage0.py --adapter`, the same 900 dev claims as stage 0, in two conditions: `oracle` (ceiling) and `titles` (the real pipeline) |
| Metrics | Accuracy, FEVER score, evidence F1, per-label recall, token-probability ECE, format adherence. Accuracy on the gold-in-context subset under `titles` is the number that compares directly to stage 0's 0.632. |

The stated confidence is 1.0 in every training target, so after SFT the
stated-confidence ECE is meaningless by construction. Ignore it here.
Token-probability ECE is still informative.

## 3. Runs and decision rule

Four models, two retrieval conditions each. Order matters: the 2B runs
decide whether the 4B runs are needed at all.

```
cd factcheck/src

# 1. smoke test: a few minutes, checks the whole pipeline
python sft_probe.py --n-train 60 --max-steps 5
python stage0.py --model local --adapter ../data/sft_probe/Qwen_Qwen3.5-2B_n60 --n 30 --retrieval oracle

# 2. 2B probe
python sft_probe.py --model Qwen/Qwen3.5-2B --n-train 6000
python stage0.py --model local --adapter ../data/sft_probe/Qwen_Qwen3.5-2B_n6000 --n 900 --retrieval oracle
python stage0.py --model local --adapter ../data/sft_probe/Qwen_Qwen3.5-2B_n6000 --n 900 --retrieval titles

# 3. untrained baselines under oracle retrieval (2B has titles numbers already)
python stage0.py --model local --n 900 --retrieval oracle
python stage0.py --model local:Qwen/Qwen3.5-4B --n 900 --retrieval oracle
python stage0.py --model local:Qwen/Qwen3.5-4B --n 900 --retrieval titles

# 4. 4B probe, only if step 2 falls short
python sft_probe.py --model Qwen/Qwen3.5-4B --n-train 6000 --qlora
python stage0.py --model local --adapter ../data/sft_probe/Qwen_Qwen3.5-4B_n6000_qlora --n 900 --retrieval oracle
python stage0.py --model local --adapter ../data/sft_probe/Qwen_Qwen3.5-4B_n6000_qlora --n 900 --retrieval titles
```

Standard error on 900-claim accuracy is about 1.6 points, so treat
differences under 4 points as noise.

| Outcome of 2B SFT, oracle accuracy | Decision |
|---|---|
| >= 0.80 | Stay on 2B. Capacity is not the limit; the gap is trainable. Skip the 4B runs. |
| 0.70 to 0.80 | Run 4B. Pick 4B only if it beats 2B by 5+ points *and* its training time per step is under 3x the 2B time; otherwise 2B, since GRPO needs many more steps than SFT. |
| < 0.70 | Check before blaming the model: training loss still falling at the end (then raise `--n-train`), format failures (see per-key format rate), evidence F1 near zero (then the target ids are wrong). If none of those, run 4B and expect to switch. |

Also record from `meta.json`: `train_seconds` and `peak_gpu_gb`. GRPO on
the chosen model will need several times the SFT memory (reference
model, 8 rollouts per prompt) and far more steps. If 4B QLoRA already
sits above 12 GB or runs slower than 3x the 2B rate, 4B GRPO is not
realistic on one 16 GB card in six weeks whatever its accuracy.

## 4. Known risks in running this

- Qwen3.5's linear-attention layers run on slow reference kernels on
  Windows (see `models.md`). Training may be several times slower than
  a normal 2B model. If a 6,000-example epoch is projected past about
  six hours, either move to WSL2 first or cut `--n-train` to 3,000.
- bitsandbytes 4-bit on Windows works for standard linear layers; the
  linear-attention projections have not been checked. If the 4B QLoRA
  load fails, run it under WSL2.
- `target_modules="all-linear"` may pick up vision-tower layers if the
  causal-LM load keeps them. They get no gradient and only cost a little
  memory. If memory is tight, list the text-tower module names instead.
- TRL's `SFTConfig` field names change between versions. The script
  targets the `max_length` / `completion_only_loss` names; if it errors
  on a field, check the installed TRL version's `SFTConfig`.

## 5. Results

Fill in. Stage-0 rows are from `stage0-results.md`.

| Model | Trained | Retrieval | Acc | FEVER | Ev F1 | R-SUP | R-REF | R-NEI | Acc, gold in ctx | ECE token | Format | Train s | Peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2B | no | titles | 0.459 | 0.340 | 0.423 | 0.940 | 0.300 | 0.137 | 0.632 | 0.308 | 0.843 | - | - |
| 2B | no | oracle | | | | | | | | | | - | - |
| 2B | SFT | oracle | | | | | | | | | | | |
| 2B | SFT | titles | | | | | | | | | | | |
| 4B | no | titles | | | | | | | | | | - | - |
| 4B | no | oracle | | | | | | | | | | - | - |
| 4B | SFT | oracle | | | | | | | | | | | |
| 4B | SFT | titles | | | | | | | | | | | |
| 180B | no | titles | 0.720 | 0.640 | 0.717 | 0.907 | 0.907 | 0.347 | 0.952 | 0.214 | 0.999 | - | - |

Decision: ______ (date, who, one sentence why)
