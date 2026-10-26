# Offline RL controller over cached actions (2026-10-26)

AG-2. Can a controller trained with RL choose evidence and verify actions
better than fixed or hand-written policies? Everything is measured on cached
outcomes, so training needs no model calls.

## Setup

- **Actions** (`agent/action_cache.py`), each one search plus one verify:
  - `bm25`: the fixed loop, always the first step;
  - `bm25-k20`: same 5 pages, top 20 sentences;
  - `hybrid-k20`: same 5 pages, top 20 by BM25 + MiniLM rank fusion;
  - `pages10-k20`: 10 candidate pages, top 20 sentences;
  - `follow-k20`: 5 pages plus up to 5 linked pages (`retrieval/follow.py`
    from `origin/retrieval`);
  - `flash-bm25`: the `bm25` context, verified by `qwen/qwen3.8-flash`.

  The local verifier is SFT `5895/run1`.
- **Episode** (`agent/controller_env.py`):
  - after `bm25`, the controller either STOPs or takes up to 2 further
    actions;
  - the committed verdict is the latest action's;
  - reward = score − λ × calls, where a big-model call costs 2 local calls.
- **Data:**
  - fitted on 1,500 unused FEVER train claims (balanced; no SFT/GRPO train
    IDs and no reserved IDs, checked with `--check-training`);
  - scored on dev 900;
  - calibration and holdout were not used.
- **Policies** (`agent/train_controller.py`):
  - fixed loop;
  - best fixed sequence;
  - `rule-by-verdict`: the best sequence for each first verdict, a hand-rule
    lookup table;
  - `rule-threshold`: confidence < t → sequence S;
  - REINFORCE: masked MLP, normalized advantage, entropy 0.05;
  - fitted Q-iteration (FQI): offline, bootstrapped from fitted later-step
    values;
  - per-claim oracle on dev.

  REINFORCE and FQI each have 3 seeds.

Every action's own result on dev:

| Action | bm25 | bm25-k20 | hybrid-k20 | pages10-k20 | follow-k20 | flash-bm25 |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.771 | 0.752 | 0.756 | 0.758 | 0.747 | 0.712 |
| FEVER score | 0.666 | 0.656 | 0.663 | 0.667 | 0.657 | 0.634 |

No single action beats `bm25`. The cached `bm25` reproduces the
fixed-policy trace verdict for all 900 claims.

## Results on dev 900 (reward = FEVER score)

| λ | Fixed | Best rule (`rule-by-verdict`) | Threshold rule | REINFORCE (3 seeds) | FQI (3 seeds) | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.666 (1.00 calls) | **0.708** (2.65) | 0.673 (1.31) | 0.696 ± 0.011 (2.80) | 0.691 ± 0.006 (2.95) | 0.819 (1.23) |
| 0.01 | 0.666 (1.00) | **0.711** (2.30) | 0.673 (1.31) | 0.698 (2.39) | 0.689 (2.37) | 0.819 |
| 0.02 | 0.666 (1.00) | 0.684 (1.69) | 0.673 (1.31) | 0.684 (2.01) | 0.684 (2.04) | 0.819 |
| 0.05 | 0.666 | 0.666 | 0.666 | 0.666 | 0.669 | 0.819 |

Cells give the FEVER score with mean calls per claim in brackets. The ±
values are the population standard deviation of the reward across seeds.
The best rule fitted at λ = 0.01 is "send SUPPORTED and REFUTED first
verdicts to flash, keep NEI".

With accuracy as the reward, nothing moves. The fixed loop scores 0.771; the
threshold rule reaches 0.776 at λ ≤ 0.02, while REINFORCE and FQI score
0.757–0.771. The oracle reaches 0.890.

## What this says

1. **RL learns the hand rule, not more.** REINFORCE and FQI both send most
   SUPPORTED/REFUTED claims to flash and gain 2–3 FEVER points over the fixed
   loop. The fitted lookup rule gains 4 with fewer calls. No RL seed beats it
   on reward at any λ.
2. **REINFORCE is unstable across seeds.** At λ = 0.02 one seed collapses to
   STOP (0.666), while another reaches 0.702 at 2.6 calls.
3. **The oracle gap stays closed to every learner:** 0.71 against 0.82. The
   features (the 2B's verdict probabilities, retrieval scores, agreement
   between actions) do not reveal which claims a different action would fix.
   This repeats the cascade result ([cascade-payoff.md](cascade-payoff.md))
   with more actions and a sequential policy.
4. **Accuracy is not improvable with these actions.** The FEVER-score gain
   comes from flash citing the gold evidence better, not from more correct
   verdicts.

## What could change it

- **Features the policy cannot see yet:** a claim/evidence text encoding,
  whether the claim's entity matched a page title or a disambiguation page,
  and the verdicts of a cheap probe. The current state is mostly the 2B's
  own output, which is known to be uninformative about its errors.
- **More training claims:** 1,500 claims with about 26 sequences each is
  small for a policy over 79 features. More claims cost about 1 GPU hour and
  $1.50 per 3,000.
- **Actions that add information the 2B lacks**, such as a second hop on the
  cited entity or a disambiguation-page expansion. Most of the cached actions
  re-rank the same pages.

## Caveats

- One verifier checkpoint, one train sample (seed 1), dev only. The holdout
  set is untouched and should be used once, after the configuration is fixed.
- Oracle numbers use dev gold labels and are upper bounds.
- One flash call on train returned no usage data after 5 retries; it is
  recorded as an invalid answer.

## Reproduce

```bash
SFT=verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints
python -m agent.action_cache --train-n 1500 --seed 1 --adapter $SFT \
  --actions bm25,bm25-k20,hybrid-k20,pages10-k20,follow-k20
python -m agent.action_cache --train-n 1500 --seed 1 --actions flash-bm25
# same two commands with --claim-ids verifier/data/eval/fever-dev-900-ids.json for dev
OMP_NUM_THREADS=4 python -m agent.train_controller \
  --train verifier/data/controller/train_unused-n1500-seed1 \
  --dev verifier/data/controller/dev_fever-dev-900-ids-ac9cfead \
  --metric fever_score --lam 0 0.01 0.02 0.05 --seeds 3 \
  --output verifier/evaluation/results/controller-offline/fever_score-run1.json
```

The action cache (`verifier/data/controller/`, 16 MB) is not committed. The
commands above rebuild it.
