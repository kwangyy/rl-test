# Multi-turn GRPO: first matched runs (2026-11-01)

RL-2 (Michel), on the RL / agent boundary. Branch `mt-grpo` (from `agent` at `8fd8da7`).
This page records why each run exists, what it changed, and what it showed, so the
numbers can be read without the session that produced them.

## Question

Can GRPO teach the 2B verifier to **search for its own evidence** (call `search`, read the
results, answer with the verifier JSON), and how does that compare with single-turn GRPO
that is handed the evidence? This is the AG-2 deliverable "multi-turn vs single-turn GRPO
on the 900 claims" ([live-backend.md](live-backend.md)).

## Setup shared by every run

- **Claims:** the fixed 1,000-claim GRPO set `verifier/data/train/sft/fever-oracle-matched-grpo-1000`
  (copied from `feat/controller`; disjoint from SFT4500 and from the frozen dev IDs).
- **GRPO:** 8 samples per claim, 2 claims per step, 500 steps (one epoch), learning rate 2e-5,
  beta 0, LoRA of the SFT4500 shape, thinking off.
- **Reward:** `core.reward` with calibration off (`--reward no_calibration`): verdict
  correctness with the harm matrix + 0.5 × evidence score; format failure −2.
  The multi-turn reward also charges 0.02 per search (`L3_COST`).
- **Search tool (multi-turn):** ranked BM25, 3 candidate pages, 5 sentences per search,
  at most 2 searches, 2 extra turns that answer "No searches left" to further calls.
- **Evaluation:** the 900 frozen dev IDs. Multi-turn models use `verifier/evaluation/eval_multiturn.py`
  (same tool, prompt and TRL tool-call parser as training, greedy decoding, scored by the
  shared `score()` / `summarize()`). Single-turn models use the shared evaluator with title
  retrieval and the direct prompt, 1,024-token cap.

## Step 1: does the starting model call the tool? (no training, lr 0)

| Start | Rollouts that searched | Valid final JSON | Ended on an unanswered call |
|---|---:|---:|---:|
| SFT4500 (JSON SFT) | 1 / 120 | 116 / 120 | 0 / 120 |
| Base Qwen3.5-2B | 120 / 120 | 90 / 120 | 24 / 120 |

30 train claims × 4 samples each (job 1055567, A100). Greedy evaluation of SFT4500 on 32 dev
claims also made no searches. **Single-turn JSON SFT removed tool use**, so a GRPO run from
SFT4500 would have nothing to reinforce. Both GRPO arms below start from base.
(H100 runs fail in `fla`'s gated-delta backward with this Triton; use A100.)

## Step 2: matched runs from base (900 dev claims)

| Model | Accuracy | FEVER | Evidence F1 (non-NEI) | Recall S / R / NEI | Format | Searches per claim |
|---|---:|---:|---:|---|---:|---:|
| Base, untrained, searching itself | 0.448 | **0.382** | 0.385 | 0.71 / 0.35 / 0.29 | 0.73 | 1.68 |
| Multi-turn GRPO from base (`run1`) | 0.438 | **0.034** | 0.000 | 0.22 / 0.99 / 0.10 | 1.00 | **0.00** |
| Single-turn GRPO from base, oracle context (`run1`) | **0.594** | 0.251 | 0.369 | 0.83 / 0.95 / **0.00** | 1.00 | — |

Configs: `grpo-multiturn-from-base-qwen3p5-2b-no-calibration-bm25k5-kl0-1000-samples` (job
1055678, 33 min, 10.5 GB) and `grpo-direct-from-base-qwen3p5-2b-no-calibration-kl0-1000-samples`
(job 1055679, 99 min, 16.3 GB). Results under `verifier/evaluation/results/<config>/`.
The untrained base row is `base-qwen3p5-2b/multiturn-bm25-k5-calls2/run1`.

**Both arms learned a shortcut the reward allows.**

- **Multi-turn stopped searching.** Searches per rollout fell from 1.8 at step 1 to 0 by step 17
  and stayed there (`logs/train_log.json`, `rewards/searches/mean`). The model answers from the
  claim alone, nearly always REFUTED. Under the harm matrix a wrong REFUTED scores 0, a wrong
  SUPPORTED on a gold-NEI claim scores −0.75, and a search that runs over budget risks −2, so
  "no search, say REFUTED" is a safe policy with a positive average reward.
- **Single-turn never abstains.** Accuracy rose to 59.4%, but NEI recall is 0.3%. Its training
  prompts always contain the gold pages (oracle context), and a wrong REFUTED/SUPPORTED on an
  NEI claim is cheaper than missing a supported verdict.

## Step 3: multi-turn rerun requiring one search (running)

Config `grpo-multiturn-from-base-qwen3p5-2b-no-calibration-bm25k5-min1search-kl0-1000-samples`
adds `--min-searches 1`: an answer given before any real search scores as a format failure.
This is the tool prompt's own instruction ("Use the search tool…") and base already searches
in every rollout, so training starts inside the allowed region. The reward is otherwise
unchanged. Job 1056636; evaluation 1056637 runs after it.

Training rollouts so far: exactly one search per rollout, 100% valid answers, no cut-offs,
and evidence F1 on non-NEI claims up from 0.31 (steps 0–10) to 0.45 (steps 17–27). But
predicted verdicts are drifting to REFUTED (45 → 126 of ~150 per 10 steps) and NEI is
disappearing (17 → 3), and the mean reward has not risen after step 10. Searching is
enforced, but the verdict shortcut from step 2 remains.

## Open decision: a reward change for GRPO only

Both shortcuts come from the same place: a correct SUPPORTED/REFUTED verdict earns full
credit without the evidence that justifies it, and REFUTED is the cheapest wrong answer.
Candidate changes, each added as a new opt-in `RewardConfig` ablation so `FULL`, the existing
ablations, the evaluator and the agent loop are untouched:

1. **Evidence-gated correctness:** a correct SUPPORTED/REFUTED earns correctness credit only
   when it cites at least one gold sentence (the FEVER score's requirement). Gold-NEI claims
   are scored as now.
2. **Symmetric REFUTED harm:** raise the cost of a wrong REFUTED to match a wrong SUPPORTED,
   so REFUTED stops being the safe default. Changes the project's harm asymmetry.
3. Both.

**Decision (2026-11-02, Michel): option 1.** Implemented as `RewardConfig.evidence_gate`
(default off) and the ablation `no_calibration_evidence_gated`; `verifier/tests/test_reward.py`
checks that `FULL` and the older ablations are unchanged, that only this ablation gates, and
that gold-NEI claims and wrong verdicts score as before.

## Step 4: both arms with the evidence-gated reward (queued)

| Arm | Config |
|---|---|
| Multi-turn, one search required | `grpo-multiturn-from-base-qwen3p5-2b-no-calibration-evidence-gated-bm25k5-min1search-kl0-1000-samples` |
| Single-turn, oracle context | `grpo-direct-from-base-qwen3p5-2b-no-calibration-evidence-gated-kl0-1000-samples` |

Everything else matches steps 2–3. Expected if the gate works: NEI recall rises above the
step-2 values (0.10 and 0.00), the REFUTED share falls toward the gold share, and FEVER moves
closer to accuracy, because unsupported right verdicts no longer pay.

## Reproduce

```
# on SoC, repo root on branch mt-grpo; A100 only
SFT=<SFT4500 run1-json-sft checkpoints>   # LoRA shape only for base-start runs
sbatch --time=36:00:00 --export=ALL,TRAIN_SCRIPT=verifier/experiments/<config>/train.py,SFT_CHECKPOINTS=$SFT,RUN_NAME=<new run> \
  verifier/experiments/run_train.sbatch
sbatch --export=ALL,ADAPTER=<run checkpoints>,EVAL_CONDITION=<config>,RUN_NAME=<new run> \
  verifier/evaluation/eval_multiturn.sbatch
```

Commits: `69bfdbc` (`--train-data`, evaluator), `84cc17d` (evaluator sbatch), `a5272ae` (base
configs), `b60714a` (`--min-searches`). The SoC checkout is `~/rl-test-agent`; the FEVER index
there was built by job 1055351.
