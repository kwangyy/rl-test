# Fact verification RL — scaffold

**Scaffold only.** There is no model here and nothing is trained. The
claims are 12 hand-written sentences, the retriever counts word overlap,
and the two "policies" are a random picker and ten lines of if-else. The
numbers it prints say nothing about whether the task is learnable; do not
quote them.

What it does pin down is the shape of the problem: the interface a real
dataset loader must satisfy, the controller's states and actions, the
reward terms, and the metrics (accuracy, evidence F1, abstention recall,
reliability bins).

## Docs

- [`docs/plan.md`](docs/plan.md) — architecture, what RL is for, the three stages, open decisions.
- [`docs/related-work.md`](docs/related-work.md) — what is already published, what is open, numbers to beat.
- [`docs/models.md`](docs/models.md) — which small model to train and which big model to call, with sizes and prices.
- [`docs/stage0-results.md`](docs/stage0-results.md) — prompted baseline on 900 FEVER dev claims, 2B vs big model, BM25 vs title-match retrieval.
- [`docs/original-proposal.md`](docs/original-proposal.md) — the source proposal, verbatim.

## Run

From the repo root, after the shared setup in the top-level README:

```
cd factcheck/src
python run_scaffold.py
```

## Files

- `src/toy_corpus.py` — 12-claim stand-in for FEVER, word-overlap retriever, split-on-"and" decomposer.
- `src/reward.py` — correctness + evidence F1 + calibration − cost, with an asymmetric harm matrix.
- `src/verification_env.py` — Gymnasium MDP: retrieve / decompose / cross-check / abstain / commit at a stated confidence.
- `src/run_scaffold.py` — random vs heuristic policy over 200 episodes, prints the metrics.

## Known problems, not yet fixed

- **The calibration term is not a proper scoring rule.** It is `+q` when
  right and `-w*q` when wrong, which is linear in the stated confidence
  `q`, so the best strategy is always an extreme confidence and never the
  true probability. It needs a Brier term, `-(q - 1[correct])^2`, with the
  asymmetric harm weights moved onto the verdict term. Fix before any
  training.
- **Retrieval is not entailment.** Word overlap finds sentences *about*
  the claim, including ones that refute it, so the heuristic never
  predicts REFUTED. A step that reads (claim, evidence) and judges
  support is missing from the action set.
- **Evidence F1 is undefined for Not-Enough-Info claims** (no gold
  evidence). Currently scored 0 and excluded; needs a real decision.
