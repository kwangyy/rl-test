# Controller workflow and results

This note summarizes the evidence-chunk controller experiment on the frozen 900-claim FEVER dev set. The verifier is the direct-SFT Qwen3.5-2B model; RSI-Jev v2.1 2B supplies controller decisions. The trained experiments update only Jev's option scorer, leaving its language model frozen.

## Workflow

- **Retrieve and split:** Retrieve evidence for each claim and divide it into fixed chunks of at most 10 sentences. Every chunk is processed, regardless of earlier verdicts.
- **Generate chunk candidates:** The verifier produces up to three stochastic outputs on the same chunk input. Each output contains a verdict, cited sentence IDs, and confidence information.
- **Decide whether to retry:** After each observed output, the controller chooses `STOP` or `RETRY` for that chunk. `RETRY` requests another verifier sample on unchanged input; the third draw is the limit. The fixed heuristic uses a verdict-token confidence threshold of 0.5. The trained citation controller learns the same STOP/RETRY decision.
- **Select citations:** The heuristic and citation-only Jev retain valid citations from a stopped draw and discard citations from a retried draw. The full-selection Jev can `KEEP` or `DROP` cited sentences on either action, including removing a sentence kept on an earlier draw. A set prevents duplicate saved citations.
- **Pool evidence:** Combine each chunk's saved citations in retrieval order. A citation may be absent from the final pool if it was not selected; all chunks were still visited.
- **Run final verification:** A fresh Qwen verifier reads the pooled evidence. The controller can stop after the first or second pooled output or request another sample on the same pooled input. If it reaches the third draw, a majority verdict with confidence-based tie-breaking selects the final Qwen output.
- **Score the outcome:** FEVER score, verdict accuracy, and non-NEI evidence F1 use the selected final **Qwen** output. Jev's own terminal verdict is recorded separately as a diagnostic.
- **Information boundary:** The controller sees the claim, current verifier verdict, format status, confidence, and cited sentence text and IDs. It does not see uncited retrieved text, gold labels, future draws, or final Qwen results before making its decisions. Gold labels and future outcomes may be used only for offline training labels, rewards, and post-hoc analysis.

## Training and comparison

- The first RL pilots trained citation-only and full-selection scorer heads on 5,500 train claims.
- A later comparison used a fixed, disjoint 3,000-claim SFT warm-up and 2,500-claim RL continuation. The two RL arms shared the same SFT starting head and training IDs; one used the absolute reward and one used improvement over an always-first-STOP counterfactual. A separate 5,500-claim SFT-only head served as the equal-total-claim control.
- All evaluated policies used the same frozen 900 dev claim IDs and precomputed chunk draws. Different selected citation pools required fresh pooled Qwen generations, so small differences between separate evaluation runs should be interpreted cautiously. The 0.5 heuristic reported below is the matched reference for the listed runs.

## Results

- **Heuristic, threshold 0.5:** 74.67% verdict accuracy, **65.33% FEVER**, and 63.81% non-NEI evidence F1 in the main matched run. It left 240 of 900 claims with an empty evidence pool.
- **Zero-shot Jev, full STOP/RETRY and KEEP/DROP:** 66.56% accuracy, **54.00% FEVER**, and 41.17% evidence F1. It left 434 claims with an empty evidence pool. [Result](../evaluation/results/rsi-jev-evidence-selection-baseline/runs/rsi-jev-v2p1-qwen3p5-2b/keep-drop-direct-cap160/full900-rerun1/metrics.json).
- **Trained full-selection head, 5,500-claim RL pilot:** 72.22% accuracy, **62.89% FEVER**, and 58.32% evidence F1. It improved substantially over zero-shot full selection but remained below the matched heuristic. [Result](../evaluation/results/rsi-jev-evidence-selection-baseline/runs/jev-rl-selection-from-rsi-v2p1-5500-samples/keep-drop-direct-cap160/trained-selection-v1/metrics.json).
- **Trained citation-only head, 5,500-claim RL pilot:** 73.00% accuracy, **64.33% FEVER**, and 60.71% evidence F1; the matched heuristic scored 65.33% FEVER. [Result](../evaluation/results/rsi-jev-evidence-selection-baseline/runs/jev-rl-citation-from-rsi-v2p1-5500-samples/keep-drop-direct-cap160/trained-citation-v1/metrics.json).
- **Citation SFT-only control, 5,500 claims:** 73.56% accuracy, **64.56% FEVER**, and 63.95% evidence F1; the matched heuristic scored 65.33% FEVER. [Result](../evaluation/results/rsi-jev-evidence-selection-baseline/runs/jev-sft-citation-from-rsi-v2p1-5500-samples/keep-drop-direct-cap160/run1/metrics.json).
- **Citation SFT3000 → RL2500, absolute reward:** 73.44% accuracy, **64.44% FEVER**, and 63.61% evidence F1; the matched heuristic scored 65.33% FEVER. [Result](../evaluation/results/rsi-jev-evidence-selection-baseline/runs/jev-rl-citation-from-sft3000-rsi-v2p1-2500-samples/keep-drop-direct-cap160/run1/metrics.json).
- **Citation SFT3000 → RL2500, relative-to-STOP reward:** 74.67% accuracy, **64.11% FEVER**, and 62.12% evidence F1; its matched heuristic scored 64.56% FEVER. [Result](../evaluation/results/rsi-jev-evidence-selection-baseline/runs/jev-rl-citation-from-sft3000-rsi-v2p1-relative-stop-2500-samples/keep-drop-direct-cap160/run1/metrics.json).
- **Heuristic threshold sensitivity:** In the separate matched 0.3/0.7 sweep, threshold 0.3 scored 64.33% FEVER and threshold 0.7 scored 64.89%. Threshold changes can also change the selected evidence pool, not just the number of retries. [Result](../evaluation/results/heuristic-threshold-sweep/runs/sft-direct-from-base-qwen3p5-2b-5500-samples/direct-cap160/thresholds-0p3-0p7-run1/metrics.json).

## Conclusion

- The controller pipeline is operational, and scorer training improved Jev substantially over its zero-shot full-selection baseline.
- None of the evaluated trained heads exceeded the matched 0.5 heuristic on FEVER score. The learned KEEP/DROP component appears especially difficult: full-selection Jev saved fewer complete gold evidence sets and produced more empty pools than the heuristic in the zero-shot run.
- The next analysis should compare **quality per verifier call** and inspect claims where a retry could have corrected an error. A lower call count alone does not establish a better controller when FEVER score falls.

For implementation details and evaluation definitions, see the [controller evaluator](../evaluation/results/rsi_jev_evidence_selection_baseline/README.md) and [training guide](../train/README.md).
