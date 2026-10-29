# Controller imitation of qwen3.8-flash, and the handoff to controller GRPO

AG-2, 2026-10-28 to 2026-10-29. The prompted loop (`agent/prompted.py`, `--tools follow`)
with qwen3.8-flash as the controller was run on 6,000 unused FEVER train claims. A
LoRA on Qwen3.5-2B was then trained to copy flash's controller decisions (behavioural
cloning) and run in the same loop on dev 900. The verifier is the frozen
`sft-direct-from-base-qwen3p5-2b-5895-samples/run1` adapter throughout.

The next step, multi-turn GRPO of this controller, is set out in section 5.

## 1. Data

| What | Where | Size |
|---|---|---|
| 6,000 flash follow trajectories (train claims) | `verifier/data/controller/imitation-flash-follow-train-unused-n6000-seed2/` (`manifest.json` has the sampler, command and checksums) | 22 MB gz |
| Imitation set: one row per valid flash step of a FEVER-correct episode | `verifier/data/train/controller/flash-follow-imitation-18291-samples/{train,val}.jsonl` | 18,291 train / 988 val |
| Dev 900 traces, flash and the imitation controller | `verifier/data/controller/dev900-follow-controller-traces/` | 6 MB gz |

The claims come from `agent.action_cache.train_claims(6000, seed=2)`: 2,000 per label,
excluding every train ID used for SFT/GRPO and the reserved calibration and holdout
IDs. Flash on these claims scores FEVER 0.693, accuracy 0.797, with 4,157
FEVER-correct episodes; 98.3% of its 28,645 controller calls were valid. Cost $2.10.

`utils/build_controller_imitation.py` rebuilds the observation flash saw before each
step by replaying the earlier steps through `Episode` and `ToolKit`. Link IDs `L<n>`
depend on the order links were seen, so the tools' state is replayed too. The rebuilt
prompts match flash's recorded `tokens_in` exactly (checked on 1,446 steps). Steps
whose controller output was invalid (315) are skipped, but their effects are replayed.
Claims are split train/val by a seeded shuffle, so a claim's steps share a split.

Train actions: search 7,131, verify 5,176, commit 3,211, follow 2,235, abstain 517,
decompose 21.

```
python -m utils.build_controller_imitation --traces verifier/data/controller/imitation-flash-follow-train-unused-n6000-seed2
```

## 2. Training

`verifier/experiments/sft-controller-from-base-qwen3p5-2b-flash-follow-18291-samples/train.py`:
the verifier SFT recipe (`verifier/experiments/_shared/sft.py`: LoRA rank 16, lr 1e-4,
one epoch, thinking off), with the follow preset's system prompt
(`agent.prompted.SYSTEMS["follow"]`) as the system message. Run1: seed 0, 9,856 s on an
RTX 5070 Ti, 5.2 GB peak, final train loss 0.175. The checkpoint (84 MB) is not
committed; `runs/run1/artifacts/meta.json` and `logs/train_log.json` are.

```
python verifier/experiments/sft-controller-from-base-qwen3p5-2b-flash-follow-18291-samples/train.py --run-name <new run>
```

## 3. Results

Held-out agreement with flash (`python -m agent.eval_controller_agreement`, 988 steps,
greedy): same action 0.790, same JSON 0.576.

| Flash's action | n | Same action | Same JSON |
|---|---:|---:|---:|
| search | 384 | 0.823 | 0.320 |
| verify | 279 | 0.871 | 0.871 |
| commit | 168 | 0.917 | 0.917 |
| follow | 122 | 0.451 | 0.295 |
| abstain | 32 | 0.406 | 0.406 |

The commonest disagreements are verify where flash followed a link (47), verify where
flash searched (39) and commit where flash abstained (9). A different search query is
not necessarily worse.

Dev 900, same loop and verifier (exact McNemar on FEVER score):

| Controller | FEVER | Accuracy | Only this right / only 2B right | p |
|---|---:|---:|---|---:|
| 2B imitation, run1 | **0.684** | 0.764 | | |
| qwen3.8-flash, follow | 0.690 | | 41 / 36 | 0.65 |
| qwen3.8-flash, full | 0.688 | | 43 / 40 | 0.83 |
| fixed loop (one search, verify, commit) | 0.666 | | 34 / 51 | 0.08 |

The 2B controller: recall SUPPORTED 0.847, REFUTED 0.760, NEI 0.687; 2.29 searches,
0.55 follows, 1.41 verifies per claim; ends by commit 0.839, abstain 0.161, budget 0;
99.8% valid controller calls; 10.9 s per claim at `--chunk 64`. No API calls apart
from flash for the rare decompose.

The hand rule of [controller-offline-rl.md](controller-offline-rl.md) scores
0.708–0.711, but over cached actions that include a flash re-judgement, not in this
loop, so it has no paired comparison here.

## 4. What this shows

- The 2B copies flash well enough to tie it. It has not gone past flash, and flash
  was not significantly better than the fixed loop either.
- Every controller difference measured so far on dev 900 is inside noise. The
  verifier, not the controller, accounts for most errors: in the flash follow run the
  gold evidence reached the verifier on 556 of 600 SUPPORTED/REFUTED claims, and 148
  of those were still wrong.
- Headroom for a controller exists but is small. Taking whichever of flash and the 2B
  got each claim right would score 0.730; the oracle over the six cached actions of
  the offline study was 0.819.

## 5. Handoff: multi-turn GRPO of the controller

Start from the run1 imitation adapter (retrain with `train.py`, or copy the 84 MB
checkpoint out of band) and keep the 5895 verifier frozen.

What exists:

- `agent/prompted.py` runs a chunk of claims in lockstep. With `--controller
  local:<dir>`, `LocalController.batch` (`agent/decompose.py`) answers all live
  episodes of a chunk in one generate call. Every row records each step's
  action, arguments and result, the controller's raw text and validity, the final answer and
  `reward` (`core/reward.py` `R.total`: harm-weighted correctness, calibration,
  evidence and a per-operation cost), plus `cost.n_ops`.
- `utils/build_controller_imitation.replay` rebuilds the exact observation before any
  step of a saved trace, which is what a trainer needs to recompute log-probs.

What is missing:

1. **Sampling.** `verifier.model.LocalModel` generates greedily (`do_sample=False`).
   Rollouts need a temperature option on `LocalModel`/`LocalController`, and a way
   to run G rollouts per claim (for example, G copies of each claim in one chunk).
2. **Log-probs.** Recompute the policy's log-probs of each controller completion given
   its observation (system prompt + observation as in `train.py`; loss on completion
   tokens only, observations masked).
3. **The trainer.** One episode reward per rollout; advantage = reward − group mean
   (optionally / group std) over the G rollouts of a claim, applied to the tokens of
   every controller step in that rollout. KL to the imitation adapter.
   `verifier/experiments/_shared/grpo.py` is single-turn; it shows the group
   advantage and KL code but not the multi-turn rollout.

Choices to make:

- **Reward.** FEVER score (0/1) minus λ·`n_ops`, or `R.total`. The offline study used
  FEVER score with λ = 0.01–0.02.
- **Claims.** Train on unused train claims only: the 6,000 above, or a new draw with
  `--train-n N --train-seed 3`. Never the dev 900, calibration or holdout IDs. Screen
  claims by reward spread across the policy's own G samples, as the verifier GRPO's
  `--filter` did. Flash's 1,843 FEVER-0 episodes show where the gold never reached
  the verifier (`final.evidence` vs `gold_evidence`), which is where search and follow
  choices can matter.
- **Compute.** Verifier calls dominate rollout time: about 10 s per claim per rollout
  at chunk 64 on the 16 GB card, so 8 rollouts of 1,000 claims take about a day
  locally. Two processes fit on 16 GB; three overflow it. A SoC GPU is the realistic
  place for this.

Practical notes:

- The decomposer calls flash (`OPENROUTER_API_KEY`) on claims missing from its cache.
  Flash decomposed 39 of 6,000 train claims, so this is rare but needs a key.
- Invalid controller output falls back to a safe action (`decide`); count it in the
  reward or penalise it, or the policy can learn to emit invalid output.
- Evaluate a GRPO checkpoint exactly as section 3 did, against the traces in
  `dev900-follow-controller-traces/` with McNemar.
