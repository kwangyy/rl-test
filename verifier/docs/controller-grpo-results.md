# Controller GRPO: runs and results

RL-2 (Michel), following [controller-grpo-handover.md](controller-grpo-handover.md). Branch
`controller-grpo`, built on `controller/grpo-handover` (`9c550c6`).

The controller is trained; the verifier `sft-direct-from-base-qwen3p5-2b-5895-samples/run1`
is loaded read-only and never updated. The start and KL reference is the imitation
controller `sft-controller-from-base-qwen3p5-2b-flash-follow-18291-samples/run1`
(dev 900 FEVER 0.684; flash 0.690; fixed loop 0.666).

## Method (handover section 6)

- **Rollouts.** Each unused train claim (`agent.action_cache.train_claims`, so no SFT/GRPO,
  calibration, dev or holdout IDs) is played 8 times in the prompted loop of
  `agent/prompted.py`: follow preset, at most 8 actions, `final_answer` rule v3, BM25
  search. The controller samples at temperature 1.0; the verifier decodes greedily.
- **Reward,** scored against the gold label only:
  FEVER score (0/1) − 0.01 × `n_ops` − 0.5 × invalid controller replies
  (the recommendation in `controller-status.md` section 5.4).
- **Update.** Advantage = reward − the mean of the claim's 8 rollouts (no std scaling, see the
  pilot), applied to the completion tokens of every controller step of the rollout;
  observations are masked. One update per 8 claims; KL (k3) to the imitation controller with
  beta 0.04, through a fresh LoRA on the merged imitation model.
- **Decompose** uses the flash split cache; a claim with no cached split is not split (no API key
  on SoC). 2 of 512 pilot rollouts hit this.

Code: `verifier/experiments/_shared/grpo_controller.py`; tests:
`verifier/tests/test_grpo_controller.py`.

## Where things are saved

| What | Where | Committed |
|---|---|---|
| Config | `verifier/experiments/grpo-controller-from-sftctl18291-qwen3p5-2b-<setting>-<count>-samples/train.py` | yes |
| Run metadata, claim IDs, spread | `.../runs/<run>/artifacts/meta.json` | yes |
| Per-step log | `.../runs/<run>/artifacts/logs/train_log.json` | yes |
| Every rollout (actions, queries, verdicts, reward; no retrieved text) | `.../runs/<run>/artifacts/rollouts.jsonl` | yes |
| GRPO LoRA (stack on the imitation adapter) | `.../runs/<run>/artifacts/checkpoints/` | no (weights git-ignored; on SoC `~/rl-test-ctl`) |
| Resume state (LoRA + optimizer) | `.../runs/<run>/artifacts/resume/` | no |
| Dev 900 traces | `verifier/data/controller/dev900-follow-controller-traces/<name>.jsonl.gz` + `manifest.json` entry | yes, gzipped |
| Slurm logs | SoC only | no |

## Runs

### Pilot: reward spread (no update)

Config `grpo-controller-from-sftctl18291-qwen3p5-2b-fever-ops001-pilot-64-samples`, run
`pilot-t1p0`, job 1059175 (A100-40, 14.5 min). 64 claims (seed 3) × 8 rollouts.

| | |
|---|---:|
| Claims whose 8 rollouts disagree on FEVER | **17 / 64** |
| Claims with any reward difference | 59 / 64 |
| Sampled FEVER / accuracy | 0.582 / 0.732 |
| Actions per rollout (search / follow / verify) | 4.16 (2.70 / 0.72 / 1.46) |
| Invalid controller replies | 0.6% |
| Ended by abstain / by budget | 14.6% / 2.0% |

The gate passes: a quarter of claims carry outcome signal. Most of the remaining "any
difference" is only a different action count, which std-scaled advantages would inflate to
full size and so push the controller to stop early; the training runs therefore use
reward − group mean without std scaling.

### Short run: 300 claims

Config `grpo-controller-from-sftctl18291-qwen3p5-2b-fever-ops001-nostd-kl004-300-samples`, run
`run1`, job 1059669 (A100-40). 300 claims (seed 4) × 8 rollouts, 38 updates, lr 1e-5.
Check (handover): train reward rises and the invalid-output rate does not.

Completed in 2 h 05 min of training (job 1059669, peak 18.5 GB). 60 of 300 claims had FEVER
spread across their rollouts; 11 decompose calls had no cached split.

| Steps | Train reward | Sampled FEVER | Actions | Invalid replies | Abstain | KL |
|---|---:|---:|---:|---:|---:|---:|
| 1–20 | 0.620 | 0.674 | 3.86 | 0.63% | 14.3% | 0.0003 |
| 21–38 | 0.567 | 0.621 | 3.90 | 0.62% | 14.7% | 0.0007 |

The invalid-output rate does not rise. The train reward does not clearly rise either; each
step trains on different claims, so windows differ mostly by claim difficulty. KL to the
imitation controller stays below 0.001, so after 38 updates at lr 1e-5 the policy has moved
very little. Dev 900 evaluation: job 1064145 (below).

### Full run: 1,000 claims

Config `grpo-controller-from-sftctl18291-qwen3p5-2b-fever-ops001-nostd-kl004-1000-samples`, run
`run1`, 1,000 claims (seed 5) × 8 rollouts, 125 updates, same settings as the short run.
Chain 1059772–1059775: 6 h 41 min of training over three A100-40 jobs, stopping and resuming
cleanly after steps 51 and 102 (the fourth job found the run finished and exited). Peak 19.3 GB.
173 of 1,000 claims had FEVER spread; 5 decompose calls had no cached split.

| Steps | Train reward | Sampled FEVER | Actions | Invalid replies | Abstain | KL |
|---|---:|---:|---:|---:|---:|---:|
| 1–20 | 0.656 | 0.705 | 3.84 | 0.42% | 14.3% | 0.0004 |
| 21–40 | 0.640 | 0.688 | 3.65 | 0.51% | 14.0% | 0.0020 |
| 41–60 | 0.628 | 0.671 | 3.68 | 0.28% | 13.4% | 0.0014 |
| 61–80 | 0.654 | 0.700 | 3.88 | 0.29% | 14.0% | 0.0010 |
| 81–100 | 0.625 | 0.673 | 3.70 | 0.45% | 14.1% | 0.0013 |
| 101–125 | 0.665 | 0.712 | 4.05 | 0.25% | 20.8% | 0.0016 |

## Dev 900 results

Same 900 frozen dev claims, follow preset, 5895/run1 verifier, greedy controller
(`eval_controller.sbatch`). Traces: `verifier/data/controller/dev900-follow-controller-traces/`
(`local-grpo-ctl-300-run1-follow.jsonl.gz`, `local-grpo-ctl-1000-run1-follow.jsonl.gz`, with
manifest entries). A rerun of the imitation controller on SoC (job 1060509) scored 0.686 FEVER /
0.762 accuracy against the handover's 0.684 / 0.764, so the SoC evaluation reproduces the baseline.

| Controller | FEVER | Accuracy | Recall S / R / NEI | Actions | Abstain |
|---|---:|---:|---|---:|---:|
| Fixed loop (handover) | 0.666 | 0.771 | | | |
| Flash, prompted (handover) | 0.690 | 0.781 | | 3.88 | |
| Imitation run1 (handover; the start) | 0.684 | 0.764 | | 3.70 | |
| GRPO, 300 claims | 0.681 | 0.763 | 0.853 / 0.747 / 0.690 | 3.56 | 14.8% |
| GRPO, 1,000 claims | 0.687 | 0.764 | 0.863 / 0.747 / 0.683 | 3.63 | 21.1% |

Exact McNemar on FEVER, same claims:

| Comparison | Only first right / only second right | p |
|---|---:|---:|
| GRPO 1,000 vs imitation | 12 / 10 | 0.83 |
| GRPO 1,000 vs flash | 38 / 41 | 0.82 |
| GRPO 300 vs imitation | 13 / 16 | 0.71 |
| GRPO 300 vs flash | 37 / 45 | 0.44 |

**Result: no measurable change.** Neither GRPO controller differs from the imitation controller
or from flash on dev 900; the 1,000-claim controller gives the same final verdict as imitation on
97.7% of claims. Under the handover's success bar (beat imitation by McNemar) GRPO has not
succeeded. What did change is small and follows the reward: slightly fewer operations
(3.70 → 3.63) from the per-operation cost, and more abstaining in the 1,000-claim run
(21% of episodes against 15%) without a gain in NEI recall.

Why it barely moved: KL to the imitation controller stayed at or below 0.002 for the whole run
(lr 1e-5, beta 0.04), and only 17% of claims had rollouts that disagreed on FEVER, so most
updates carried little outcome signal. This is consistent with `controller-status.md` section 4:
most remaining misses are the verifier's, and the controller's own share (63 of 279 flash misses)
leaves little for a better controller to recover.

Possible next runs, not done: a higher learning rate (about 5e-5) or lower beta so the policy
actually moves; training only on the claims with FEVER spread; or rewarding against the
controller-attributable failures (gold retrieved but never sent to the verifier).

## Running on SoC

Checkout `~/rl-test-ctl` (worktree of `~/rl-test-agent`, whose `.venv` and FEVER index it links).
A100-40 only: training fails in `fla`'s backward on H100, and the `xgpj` A100-80 nodes cannot
load this `.venv`. Jobs stop at 3 h, so training runs as a chain that resumes:

```
cfg=verifier/experiments/<config>/train.py
j=$(sbatch --parsable --export=ALL,TRAIN_SCRIPT=$cfg,RUN_NAME=run1 verifier/experiments/run_controller_grpo.sbatch)
for i in 2 3 4; do j=$(sbatch --parsable --dependency=afterany:$j --export=ALL,TRAIN_SCRIPT=$cfg,RUN_NAME=run1 verifier/experiments/run_controller_grpo.sbatch); done
sbatch --dependency=afterany:$j --export=ALL,CONTROLLER=local:<imitation checkpoints>,GRPO=<run checkpoints> verifier/evaluation/eval_controller.sbatch
```
