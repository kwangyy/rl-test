# Prompted controller: tools, controller model and verifier on dev 900

AG-2, 2026-10-27 to 2026-10-29. Runs of `agent/prompted.py` on the frozen dev 900
claims, at most 8 actions per claim, BM25 search (5 candidate pages, top 100
sentences), final-answer rule v3. Unless stated, the controller is qwen3.8-flash
(prompt v2) and the verifier is `sft-direct-from-base-qwen3p5-2b-5895-samples/run1`.
FEVER score: the verdict is correct, and for SUPPORTED/REFUTED a complete gold
evidence set is among the cited sentences. P-values are exact McNemar on per-claim
FEVER score.

Traces and a manifest with per-file statistics:
`verifier/data/controller/dev900-prompted-comparison-traces/`. The flash follow trace
is in `dev900-follow-controller-traces/`; the fixed loop with 5895 is
`verifier/data/traces/fever_dev_fixed-sft-direct-from-base-qwen3p5-2b-5895-samples_run1_bm25-p5-k100_n900.jsonl`.

## 1. Tool presets

| Preset | Controller actions besides decompose, verify, commit, abstain | FEVER | Accuracy | Gold reached (of 600) | vs follow |
|---|---|---:|---:|---:|---|
| fixed loop | none: one search, verify, commit | 0.666 | 0.771 | 524 | 45 / 67, p = 0.047 |
| nosearch | none; the claim is searched once up front | 0.661 | 0.754 | | 45 / 71, p = 0.02 |
| full | search | 0.688 | 0.771 | 536 | 35 / 37, p = 0.91 |
| **follow** | search, follow a link from the results | **0.690** | **0.781** | 556 | |
| pages | search, follow, open_page, find_pages | 0.681 | 0.773 | 567 | 27 / 35, p = 0.37 |

"vs follow" counts the claims only that row gets right / only follow gets right.
"Gold reached" counts SUPPORTED/REFUTED claims where a complete gold set was in some
verifier input.

- **Searching again matters.** Without the search action the controller does no
  better than the fixed loop and is worse than follow (p = 0.02).
- **Follow is the best preset, but not significantly better than full.**
- **Opening pages brings more gold evidence but not more correct answers.** 883 of
  944 `open_page` calls re-opened pages that search had already returned. Gold
  reached rose from 556 to 567, but claims where the gold was reached and the
  answer was still wrong rose from 148 to 163.

## 2. Controller model

| Controller (follow preset) | Claims | FEVER | Accuracy | vs flash on the same claims |
|---|---:|---:|---:|---|
| qwen3.8-flash | 900 | 0.690 | 0.781 | |
| 2B imitation of flash ([controller-imitation.md](controller-imitation.md)) | 900 | 0.684 | 0.764 | 36 / 41, p = 0.65 |
| claude-sonnet-5.5 | 100 | 0.700 | 0.760 | 4 / 4, p = 1.0 (flash 0.700) |

The Sonnet run used 100 dev claims (`dev100-ids.json` in the traces folder). A much
stronger controller model did not change the score on those claims.

## 3. Verifier

Same controller (flash, full preset) and the fixed loop, with three verifiers. The
5500 verifier is Chien Hung's `sft-direct-from-base-qwen3p5-2b-5500-samples` config
retrained here (`runs/local-retrain1`, seed 0, 7,335 s; checkpoint not committed).

| Verifier | Alone (titles retrieval) acc / FEVER | Fixed loop | Prompted, flash | Gain |
|---|---|---:|---:|---:|
| SFT 5895, run1 | 0.761 / 0.671 | 0.666 | 0.688 | +2.2 |
| SFT 5500, local-retrain1 | 0.760 / 0.667 | 0.666 | 0.680 | +1.4 (p = 0.21) |
| base Qwen3.5-2B | 0.459 / 0.340 | 0.369 | 0.407 | +3.8 |

The 5500 verifier's standalone results are in
`verifier/evaluation/results/sft5500-direct-cap160/local-retrain1/`.

- **The two SFT verifiers are equivalent** (5895 vs 5500 prompted: 0.688 vs 0.680).
- **The controller's gain carries across verifiers.** It is of the same size with
  either SFT verifier and larger with the untrained base model, which needs more
  help. The verifier sets the level: swapping the base model for SFT is worth 28
  points; any controller change measured here is worth at most 2.4.

## 4. Caveats

- One seed, greedy decoding, one run per row.
- Many pairwise comparisons on the same 900 claims; a single p near 0.05 is weak
  evidence.
- The prompts were developed on these dev claims; the holdout set has not been used.
