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
| Holdout 3,000 traces (imitation and run A, one evaluation each) | `verifier/data/controller/holdout3000-follow-controller-traces/<name>.jsonl.gz` + `manifest.json` | yes, gzipped |

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

## Follow-ups: making the policy move

Both use the same reward, beta 0.04 and resumable chain as above; only unused train claims.

- **A** `grpo-controller-from-sftctl18291-qwen3p5-2b-fever-ops001-nostd-lr5e5-kl004-1000-samples/run1`:
  the 1,000-claim run (same claims, seed 5) at lr 5e-5 instead of 1e-5. Jobs 1064778–1064781, eval 1064782.
- **B** `grpo-controller-from-sftctl18291-qwen3p5-2b-fever-ops001-nostd-lr5e5-kl004-spread2ep-250-samples/run1`:
  only the 250 train claims whose 8 rollouts disagreed on FEVER in the three runs above, two
  shuffled passes (63 updates), lr 5e-5. Jobs 1064783–1064785, eval 1064786.

KL to the imitation controller reached 0.004–0.0075 (A) and 0.010–0.011 (B), against ≤ 0.002 before.

| Controller (dev 900) | FEVER | Accuracy | Recall S / R / NEI | Actions | Abstain | Same verdict as imitation |
|---|---:|---:|---|---:|---:|---:|
| Imitation (start) | 0.684 | 0.764 | 0.85 / 0.76 / 0.69 | 3.70 | 16.1% | |
| Flash | 0.690 | 0.781 | 0.88 / 0.75 / 0.71 | 3.88 | 11.1% | |
| GRPO 1,000, lr 1e-5 | 0.687 | 0.764 | 0.86 / 0.75 / 0.68 | 3.63 | 21.1% | 97.7% |
| **A: GRPO 1,000, lr 5e-5** | **0.700** | **0.779** | 0.86 / 0.76 / 0.71 | 3.66 | 10.0% | 95.1% |
| B: GRPO on spread claims, lr 5e-5 | 0.693 | 0.769 | 0.86 / 0.75 / 0.69 | 3.67 | 20.3% | 94.8% |

| Exact McNemar on FEVER | Only first right / only second right | p |
|---|---:|---:|
| A vs imitation | 30 / 16 | 0.054 |
| A vs flash | 41 / 32 | 0.35 |
| B vs imitation | 31 / 23 | 0.34 |

A is the best controller so far: +1.6 FEVER points over imitation and +1.0 over flash, with
abstaining down from 16% to 10% and NEI recall up (0.69 → 0.71). Against imitation it is
borderline (p = 0.054), one seed, on dev claims that earlier prompt work also used, and one of
several comparisons; confirm on the holdout and with a second seed before calling it a gain.

## Why the gains are small

Imitation's 284 FEVER misses on dev 900, by where the gold evidence went:

| Miss | Claims | Whose |
|---|---:|---|
| Gold evidence was in the verifier's input; verdict or citation wrong | 130 | verifier |
| Gold label NEI, final answer SUPPORTED or REFUTED | 94 | mostly verifier |
| Gold evidence never retrieved | 54 | controller and retrieval |
| Search found the gold evidence, never sent to the verifier | 6 | controller |

At most 60 misses (21%) depend on the controller's choices, so a perfect controller could add
about 6.7 points and a realistic one far less. The other 224 are the frozen verifier's.

Disagreements with imitation run in both directions: GRPO 1,000 at lr 1e-5 gained 12 claims and
lost 10. Most flips come from a different search query (the same action sequence, e.g.
search–verify–commit, retrieving different sentences) or from GRPO searching once more and the
verifier then answering differently. The verifier's answer to slightly different evidence is
noisy, so a controller change moves claims both ways, and only the net count reflects learning.

Possible next runs, not done: a higher learning rate (about 5e-5) or lower beta so the policy
actually moves; training only on the claims with FEVER spread; or rewarding against the
controller-attributable failures (gold retrieved but never sent to the verifier).

## Second seed, higher learning rate, evidence bonus and holdout

Three more 1,000-claim runs with run A's claims, reward, beta 0.04 and 125 updates; one setting
changed in each. All dev 900 traces are in `dev900-follow-controller-traces/`.

- **A2** `grpo-controller-from-sftctl18291-qwen3p5-2b-fever-ops001-nostd-lr5e5-kl004-1000-samples/run2-seed1`:
  run A with `seed` 1. Jobs 1066351–1066354, eval 1066355.
- **C** `grpo-controller-from-sftctl18291-qwen3p5-2b-fever-ops001-nostd-lr1e4-kl004-1000-samples/run1`:
  lr 1e-4. Jobs 1067811–1067814, eval 1067815.
- **D** `grpo-controller-from-sftctl18291-qwen3p5-2b-fever-ops001-evbonus03-nostd-lr5e5-kl004-1000-samples/run1`:
  run A plus `--evidence-bonus 0.3`: reward FEVER − 0.01 × actions − 0.5 × invalid + 0.3 when a
  full gold evidence set was in the input of some verify call (commit 5683e59). Jobs
  1067823–1067826, eval 1067827.

Final KL to the imitation controller: about 0.005 (A2), 0.013–0.018 (C), 0.004–0.007 (D).

"Gold delivered" is the share of non-NEI claims where a full gold evidence set reached a verify
call. The last two columns compare each run with imitation on the same claims (exact McNemar on
FEVER).

| Controller (dev 900) | FEVER | Accuracy | Recall S / R / NEI | Abstain | Gold delivered | Same verdict as imitation | Only imitation / only run right | p |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Imitation (start) | 0.684 | 0.764 | 0.85 / 0.76 / 0.69 | 16.1% | 90.0% | | | |
| A: lr 5e-5 | 0.700 | 0.779 | 0.86 / 0.76 / 0.71 | 10.0% | 91.2% | 95.1% | 16 / 30 | 0.054 |
| A2: lr 5e-5, second seed | 0.691 | 0.766 | 0.86 / 0.75 / 0.69 | 14.8% | 90.7% | 96.6% | 15 / 21 | 0.41 |
| C: lr 1e-4 | 0.687 | 0.764 | 0.86 / 0.75 / 0.69 | 9.3% | 88.8% | 94.2% | 28 / 30 | 0.90 |
| D: lr 5e-5 + 0.3 evidence bonus | 0.682 | 0.760 | 0.85 / 0.74 / 0.69 | 19.8% | 89.7% | 95.3% | 23 / 21 | 0.88 |

- **Seed.** A2 lands at 0.691 against A's 0.700 (A2 vs A: 12 / 20 claims, p = 0.22). Across the
  two seeds lr 5e-5 adds about one FEVER point over imitation, and neither seed is significant.
- **Learning rate.** At 1e-4 the policy moves further (KL about three times A's) but gains and
  losses cancel (30 / 28).
- **Evidence bonus.** No effect. In D's 8,000 training rollouts (1,000 claims × 8), 511 of the
  667 non-NEI claims delivered the gold evidence in all 8 rollouts and 37 in none, so the bonus
  was constant within those groups. In 68 of the remaining 119 claims FEVER already varied, and
  a rollout without the gold evidence never scored FEVER (0 of 636), so the bonus pointed the same
  way as the FEVER reward. Only 51 claims got a new signal, and in all of them every rollout was
  FEVER-wrong: the bonus rewarded delivering evidence that the verifier then misjudged. The
  imitation controller already delivers the gold evidence on 90% of non-NEI claims, so the term
  measures something the policy rarely gets wrong.

### Holdout 3,000

The holdout is `verifier/data/eval/fever-evaluation-v1/holdout-3000-ids.json`, frozen in commit
2251e86 (`fever-evaluation-v1`, see its `manifest.json`): 1,000 claims per label drawn with seed 0
from the official FEVER shared-task dev set, after removing every dev claim already used in project
records (including the dev 900) and claims whose normalized text matches train or used claims. Its
policy is "evaluate finalized configurations only". Run A was chosen on dev 900 before the holdout
was opened, and only imitation and A were evaluated on it, once each (jobs 1067828 and 1072154),
with `verifier.evaluation.eval_controller_offline` (no decomposer API calls; no claim needed one).
Flash was not run: SoC has no API key.

| Controller (holdout 3,000) | FEVER | Accuracy | Recall S / R / NEI | Abstain | Gold delivered | Same verdict | Only imitation / only A right | p |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Imitation | 0.673 | 0.763 | 0.87 / 0.74 / 0.68 | 16.3% | 88.9% | | | |
| A: lr 5e-5 | 0.678 | 0.765 | 0.86 / 0.75 / 0.68 | 12.1% | 90.0% | 96.5% | 49 / 65 | 0.16 |

On unseen claims A is +0.5 FEVER points over imitation, not significant. Its dev gain (+1.6) was
partly the luck of one seed and of a dev set used to pick it. Controller GRPO from the imitation
adapter, with this reward and this frozen verifier, gives at most about one FEVER point.

**Conclusion.** The imitation controller already does most of the controller's job: it delivers
the gold evidence on about 90% of non-NEI claims, and most remaining misses are the verifier's
(see "Why the gains are small"). Learning rate, seed and an evidence-delivery reward do not change
that. Larger gains need a better verifier on the evidence this controller passes it, better NEI
handling (the lowest recall, 0.68–0.69), or better retrieval for the claims whose gold evidence
is never found, rather than more controller GRPO tuning.

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
