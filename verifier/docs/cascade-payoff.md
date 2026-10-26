# Cascade payoff: 2B first, big model on hand-off (2026-10-26)

AG-2. Question: if the 2B answers first and hands some claims to the big
model, how much accuracy does that buy per hand-off, and does a learned router
beat simple rules? This measures the headroom for a controller before any
controller training.

Setup: FEVER direct prompt, titles retrieval, 160-token cap, greedy. Runs
were made fresh at the same commit on the frozen 900 dev IDs and the 900
eval-v1 calibration IDs. The holdout set was not touched. Models:
- SFT `5895/run1` and the base Qwen3.5-2B, run locally;
- `qwen/qwen3.8-flash` via OpenRouter.

Every threshold, margin and learned weight is fitted on calibration and
scored on dev. The AVeriTeC section reuses the gold-QA runs from
[averitec-transfer.md](averitec-transfer.md); AVeriTeC has no calibration
split, so it is scored with the thresholds fitted on FEVER calibration.

Code: `verifier/evaluation/cascade_payoff.py`. Run directories:
`verifier/evaluation/results/{sft,baseline,big}-direct-cap160/cascade-{dev,cal}900/`.
Full results: `results/cascade-payoff/*.json`.

## FEVER dev (900), SFT 2B → flash

| Policy | Routed | Accuracy | FEVER score | Big-model cost |
|---|---:|---:|---:|---:|
| SFT 2B alone | 0% | 0.761 | 0.671 | $0 |
| flash alone | 100% | 0.714 | 0.634 | $0.196 |
| Route if 2B token probability < t (best t on calibration) | 9–27% | 0.763 | 0.670 | $0.02–0.06 |
| **Route every 2B SUPPORTED/REFUTED, keep 2B NEI** | 64% | **0.767** | **0.707** | $0.127 |
| Learned router (logistic regression on 2B outputs) | 66–74% | 0.758 | 0.702 | $0.13–0.15 |
| Oracle router (hand off only claims flash fixes) | 10–11% | 0.864 | 0.779 | $0.02 |

The SFT 2B beats flash on FEVER because flash rarely says NEI. Recall on
gold NEI is 0.727 for SFT and 0.333 for flash. On gold SUPPORTED and REFUTED
claims flash is far better: 0.907 / 0.903 against 0.823 / 0.733. The best
rule follows that split. It also cites evidence better, which is why FEVER
score gains more than accuracy.

- **The confidence threshold buys nothing.** SFT's verdict-token probability
  does not separate the claims flash would fix.
- **The learned router does not beat the rule.** Fitted on 900 calibration
  claims, it learns roughly the same label split.
- **The oracle gap stays open:** 7 points of FEVER score and 10 of accuracy.
  It is not reachable from the 2B's own output.

## FEVER dev, base 2B → flash

The base 2B adds little: 0.459 accuracy alone against flash's 0.714. Every
fitted policy ends up routing about 90% of claims, and the oracle router
(0.764) is only 5 points above flash alone.

## AVeriTeC dev (500, gold QA evidence)

| Small model | Small alone | Threshold rule (FEVER-fitted t) | Route SUP/REF | Route NEI | Learned router | flash alone | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT 2B | 0.298 | 0.408 (30%) | 0.298 (18%) | 0.814 (82%) | 0.504 (54%) | 0.814 | 0.894 (60%) |
| Base 2B | 0.692 | 0.814 (99%) | 0.764 (87%) | 0.742 (13%) | 0.776 (47%) | 0.814 | 0.872 (18%) |

Percentages are the share of claims routed. A lower threshold sends fewer
base-2B claims to flash: the FEVER-fitted 20% and 40% thresholds reach 0.770
at 29% routed and 0.812 at 64% (the `fractions` rows in
`results/cascade-payoff/averitec-baseline.json`).

**The FEVER rule does not transfer.** On AVeriTeC the SFT 2B's errors are
false NEIs, so the opposite rule is the right one. Rules and routers fitted
on FEVER pick the wrong claims to hand off.

## What this says

1. **For routing between these two models, hand-written rules are enough.**
   One label rule gets FEVER from 0.671 to 0.707 FEVER score. A learned
   router fitted on the same features does no better.
2. **The remaining headroom needs new information, not a better router.**
   The oracle gap (0.707 → 0.779) is invisible in the 2B's output. To close
   it, a controller would have to *do* something: re-search, follow a link,
   sample again or ask for a second opinion. The routing decision alone
   can't close it.
3. **Any rule is dataset-specific.** The best FEVER rule is the wrong rule
   on AVeriTeC, so a fixed rule needs a check on the target data.
4. **Cost is not the argument for the 2B.** Running flash on all 900 claims
   costs $0.20 ($0.0002 a claim). A cascade saves cents. The case for a
   local 2B is latency, offline use or keeping data local, not price.

## Caveats

- One seed, greedy decoding, one SFT checkpoint, direct prompt only.
- On FEVER, retrieval misses cap both models: 83 dev claims have no gold
  evidence in context, and FEVER score is 0 for all of them.
- AVeriTeC uses gold evidence, and its thresholds are fitted on FEVER.

## Reproduce

```bash
R=verifier/evaluation/results
python -m verifier.evaluation.cascade_payoff \
  --small-cal $R/sft-direct-cap160/cascade-cal900/predictions.jsonl \
  --big-cal   $R/big-direct-cap160/cascade-cal900/predictions.jsonl \
  --small     $R/sft-direct-cap160/cascade-dev900/predictions.jsonl \
  --big       $R/big-direct-cap160/cascade-dev900/predictions.jsonl
# AVeriTeC: --metrics correct, --small/--big = $R/averitec-{sft,baseline,big}-direct-cap160/run1/predictions.jsonl
# Model runs: python -m verifier.evaluation.eval_verifier --model local|qwen/qwen3.8-flash \
#   --retrieval titles --prompt-style direct --claim-ids <ids> --output <run dir>/predictions.jsonl [--adapter <sft>]
```
