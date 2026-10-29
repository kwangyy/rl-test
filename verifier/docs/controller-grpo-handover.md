# Handover: multi-turn GRPO of the controller

From AG-2, 2026-10-29. Branch `controller/grpo-handover`, built on `agent`.

## 1. The task

Train the agent loop's **controller** with multi-turn GRPO. The controller is the
model that decides, step by step, whether to search, follow a link, verify, commit
to a verdict or abstain. The **verifier**, the model that reads the evidence and
returns a verdict, stays frozen.

Start from the imitation controller described below, not from the base model. Use
the SFT verifier `5895/run1` for now. The other team's new verifier is not ready;
when it is, swap it in and re-evaluate (section 7).

## 2. What exists and how good it is

All numbers are on the frozen dev 900 claims, with the follow tool preset, at most 8
actions per claim and the `5895/run1` verifier.

| Controller | FEVER score | Accuracy | Notes |
|---|---:|---:|---|
| Fixed loop: one search, verify, commit | 0.666 | 0.771 | no controller |
| qwen3.8-flash, prompted | 0.690 | 0.781 | API model; the teacher |
| **2B imitation controller (start here)** | **0.684** | 0.764 | local, no API |

The 2B is statistically tied with flash (exact McNemar p = 0.65) and not
significantly above the fixed loop (p = 0.08). It agrees with flash's action on 79%
of 988 held-out steps; it is weakest on follow (45%) and abstain (41%).

Expect modest gains. Every controller difference measured so far has been within
noise. Taking the better of flash and the 2B per claim would score 0.730, which
gives a rough idea of the headroom. Most remaining errors are the verifier's: in
the flash run the gold evidence reached the verifier on 556 of 600 SUPPORTED/REFUTED
claims and 148 of those were still wrong.

Details and the full tables: [controller-imitation.md](controller-imitation.md).
The earlier offline-RL controller study, over six cached actions:
[controller-offline-rl.md](controller-offline-rl.md).

## 3. Names, so nothing gets mixed up

| Name | What it is |
|---|---|
| `sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1` | The **verifier**. Qwen3.5-2B LoRA, direct SFT (no CoT, no GRPO) on 5,895 FEVER claims. Dev 900 alone: 0.761 accuracy, 0.671 FEVER. Every loop number here uses it. |
| `sft-controller-from-base-qwen3p5-2b-flash-follow-18291-samples/runs/run1` | The **imitation controller**. Qwen3.5-2B LoRA, SFT on 18,291 flash decisions. No GRPO yet. |

The README and TODO name `run2` of the verifier config as the main SFT checkpoint
(0.741 / 0.659). The agent-loop work used **run1**. Use run1 so your baselines match.

## 4. Setup

1. `git switch controller/grpo-handover`, then `uv sync` (Python 3.13, CUDA 12.8).
2. Download FEVER and build the BM25/SQLite index: [data.md](data.md) sections 1 and 2.
3. Get the two adapters. Neither is in git (84 MB each); ask AG-2 for copies and put
   them at the paths above, under `artifacts/checkpoints/`. Retraining is the
   fallback: `verifier/experiments/<config>/train.py --run-name <new run>` (about
   40 minutes for the verifier and 2.7 hours for the controller on a 16 GB RTX 5070
   Ti). A retrain will not match the baselines exactly; if you retrain, re-run
   them.
4. `OPENROUTER_API_KEY` in `.env` (git-ignored). The decomposer calls flash on any
   claim it has to split; the cache is not written back. Flash decomposed 39 of
   6,000 train claims, so the cost is negligible, but without a key those episodes
   fail.

Check the setup by reproducing the 2B controller's dev result:

```
python -m agent.prompted --tools follow --chunk 64 \
    --controller local:verifier/experiments/sft-controller-from-base-qwen3p5-2b-flash-follow-18291-samples/runs/run1/artifacts/checkpoints \
    --adapter verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints \
    --claim-ids verifier/data/eval/fever-dev-900-ids.json
```

It should print FEVER score 0.684 (about 2.7 hours locally at 10.9 s per claim). The
committed traces to compare against are in
`verifier/data/controller/dev900-follow-controller-traces/`.

## 5. Files

| Path | What |
|---|---|
| `agent/prompted.py` | The loop. `Episode` holds a claim's state and builds the observation; `decide` parses the controller's JSON and falls back to a safe action on invalid output; `run_chunk` runs a chunk of claims in lockstep; `to_row` writes the trace row with `final`, `cost` and `reward`. |
| `agent/decompose.py` | `LocalController`: a LoRA controller on the GPU, `batch()` answers a whole chunk in one generate call. `big_model("local:<dir>")` builds it. |
| `agent/tools.py` | The follow tool and link bookkeeping (`ToolKit`). |
| `core/reward.py` | `R.total`: harm-weighted correctness, calibration, evidence and a per-operation cost. |
| `utils/build_controller_imitation.py` | `replay(row, retriever)` rebuilds the exact observation before every step of a saved trace. |
| `agent/eval_controller_agreement.py` | Action agreement of a controller with flash on the held-out steps. |
| `verifier/experiments/_shared/sft.py`, `_shared/grpo.py` | SFT and single-turn GRPO trainers. `grpo.py` has the group-advantage and KL code. |
| `verifier/data/controller/imitation-flash-follow-train-unused-n6000-seed2/` | 6,000 flash trajectories on unused train claims (FEVER 0.693; 4,157 FEVER-correct). |
| `verifier/data/train/controller/flash-follow-imitation-18291-samples/` | The imitation set the controller was trained on. |
| `verifier/data/controller/dev900-follow-controller-traces/` | Dev 900 traces for flash and the imitation controller. |

## 6. What to build

1. **Sampled rollouts.** `verifier.model.LocalModel` decodes greedily
   (`do_sample=False`). Add a temperature option to `LocalModel` and
   `LocalController`, and a way to run G rollouts per claim, for example G copies of
   each claim in one chunk. Keep the verifier greedy.
2. **Log-probs.** For each controller step, the log-prob of its completion given
   the system prompt (`SYSTEMS["follow"]`) and the observation, with the same chat
   template as `train.py`. Loss on completion tokens only; observations are masked.
   Either record the observation during rollouts or rebuild it with `replay`.
3. **The trainer.** One reward per rollout; advantage = reward − the group mean
   (optionally divided by the group std) over the G rollouts of a claim, applied to
   the tokens of every controller step in that rollout. KL to the imitation adapter.

Then, in order:

1. **Pilot, about an hour.** 64 train claims × 8 sampled rollouts. Count the claims
   whose 8 rewards differ. If almost none do, GRPO has no signal; raise the
   temperature or pick different claims before spending a day on training.
2. **Short run.** 200–300 claims with reward spread. Check that the train reward
   rises and that the invalid-output rate does not.
3. **Full run, then dev 900** with the command in section 4 and McNemar against both
   committed dev traces.

Choices to make:

- **Reward.** FEVER score (0/1) minus λ·`cost.n_ops` (the offline study used
  λ = 0.01–0.02), or `R.total`. Score against the gold label, never agreement with
  the verifier or with flash. Penalise invalid controller output, or the policy can
  learn to emit it and take the fallback.
- **Claims.** Only unused train claims: the 6,000 above, or a fresh draw with
  `--train-n N --train-seed 3` (`agent.action_cache.train_claims` excludes every
  SFT/GRPO, calibration and holdout ID). Screen by reward spread over the policy's
  own samples. Flash's 1,843 FEVER-0 episodes show where the gold evidence never
  reached the verifier, which is where search and follow choices matter most.
- **Compute.** Verifier calls dominate: about 10 s per claim per rollout at chunk
  64 on a 16 GB card, so 8 rollouts × 1,000 claims is about a day. Two processes
  fit on 16 GB; three overflow it and run no faster. A SoC GPU is faster.

## 7. Swapping in the new verifier

Swapping is the `--adapter` flag, as long as the new verifier returns the same
fields: verdict, evidence IDs and confidence. With flash as the controller, the
prompted loop beat the fixed loop by a similar margin under each FEVER verifier
tried (+2.2 points with 5895, +1.4 with the 5,500-claim SFT), so search and follow
behaviour should carry over. The stopping rules may not: the controller sees the
verifier's verdicts and confidence, and run1 predicts NEI on about 35% of claims.

When the new verifier arrives:

1. Run the fixed loop and the GRPO controller on dev 900 with it.
2. If the gain over the fixed loop has shrunk, fine-tune the GRPO controller briefly
   against the new verifier rather than starting over.

## 8. Rules

- Never train or fit anything on the dev 900, calibration or holdout IDs.
- Never overwrite a run; use a new `--run-name`. Suggested config name:
  `grpo-controller-from-sftctl18291-qwen3p5-2b-<setting>-<count>-samples`.
- Do not commit checkpoints, raw traces from `verifier/data/traces/`, credentials or
  terminal logs. Gzip traces worth keeping, with a manifest, as in
  `verifier/data/controller/`.
- Record results in [controller-imitation.md](controller-imitation.md) or a new doc,
  and in `verifier/TODO.md`.
