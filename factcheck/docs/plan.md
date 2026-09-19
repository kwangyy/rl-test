# Fact verification RL — plan

Working notes as of 2026-09-19. The source proposal these notes build on
is kept verbatim in [`original-proposal.md`](original-proposal.md). Prior
work is in [`related-work.md`](related-work.md), model choices in
[`models.md`](models.md).

## Task

Input: a claim. Output: a verdict (Supported / Refuted / Not Enough Info),
the evidence sentences that justify it, and a confidence.

Train on FEVER, evaluate transfer on SciFact and AVeriTeC, demo on
AVeriTeC. All three are public with gold labels; this is a loader, not a
data-collection effort.

## Architecture

```
claim
  |
  v
+------------------------ agent loop -------------------------+
|  CONTROLLER: picks the next action                          |
|    sees: evidence so far, retrieval score, verifier's       |
|          verdict + confidence, hops used, budget left       |
|    actions: search | decompose | verify | abstain | commit  |
|        |              |               |                     |
|        v              v               v                     |
|   RETRIEVER      DECOMPOSER       VERIFIER (small LLM)      |
|   BM25 + dense   prompted LLM     (claim, evidence) ->      |
|   frozen         frozen           verdict + evidence ids    |
|                                   + confidence              |
+-------------------------------------------------------------+
  v
verdict + cited sentences + confidence + decision trace
```

The `verify` step is not in the original proposal's action list. The
scaffold showed it is required: retrieval finds sentences *about* a claim,
including ones that refute it, so retrieval score alone cannot produce a
verdict.

## What RL is for

A prompted agent can already run the whole loop with no training. RL is
only needed where there is nothing to imitate. There are two such places.

| Place | Why supervised training cannot do it | What RL optimises |
|---|---|---|
| Verifier (GRPO) | There is no gold confidence label. How sure the model *should* be depends on its own accuracy, so the signal has to be on-policy. Asymmetric harm is a cost on outcomes and cannot be written as a label. | correct verdict + evidence F1 + Brier confidence + harm-weighted errors |
| Controller (RL) | No dataset labels "the right number of searches". It is a sequential decision with delayed reward and a per-step cost. | correctness - lambda * (searches + tokens); when more evidence is worth paying for, when to abstain |

SFT can teach verdict labels. RL is for confidence, stopping and cost.
That is the answer to "why not just fine-tune?".

## Stages

Built so the contribution of RL shows up as a measured difference, and so
an earlier stage is still a complete result if a later one fails.

| Stage | What is trained | Needs | Shows |
|---|---|---|---|
| 0 | nothing, prompted agent | small model run locally + a big model on OpenRouter | baseline accuracy, reliability diagram, searches per claim, format-adherence rate |
| 1 | verifier, single-turn GRPO | 1 GPU | calibration and evidence grounding, before vs after |
| 2 | controller | CPU for a small policy | accuracy-vs-cost curve from a lambda sweep, abstention behaviour |

Stage 0 also tells you whether the small model can hold the output format.
If it cannot, GRPO starts with near-zero reward and nothing to rank, and
an SFT warm-start is needed first.

## Single-turn vs multi-turn GRPO

The difference is who decides what evidence the model sees.

| | Single-turn | Multi-turn |
|---|---|---|
| Rollout | prompt -> one completion | generate -> pause at a search call -> retriever output injected -> resume -> verdict |
| Model learns | read evidence, judge, state confidence, abstain | all that, plus what to query, when to search again, when to stop |
| Retriever miss | stuck; gold evidence not in top-k means guess or NEI | can recover by re-querying |
| Loss masking | none needed | injected retrieval tokens must be masked |
| Credit assignment | one decision, one reward | many decisions share one end reward; noisier |
| Infra | TRL `GRPOTrainer` out of the box | tool-calling rollouts; TRL support is experimental, the papers all used verl |
| Agentic? | no | yes |

Same reward function and same algorithm for both. ProFact's ablation shows
the cost of sparse reward in multi-turn: verdict-only reward drops their
score from 47.8 to 34.4.

## Open decisions

1. **Who the controller is.** The original proposal (section 4) has a
   separate controller that observes the LLM. During discussion this
   drifted to "one LLM, GRPO only", where the LLM picks its own actions.
   The literature search favours the original two-learner design: the
   GRPO-only version is largely pre-empted by ProFact and AWA-RL, while no
   paper was found that puts a small explicit-MDP controller on top of a
   GRPO-trained verifier for fact verification. It is also cheaper (lambda
   sweeps and action ablations retrain in minutes on CPU). Stages 0 and 1
   are identical either way, so this can wait until stage 1 works.
2. **Which GPU is available.** Decides the small model size. Not yet known.
3. **Evidence F1 on Not-Enough-Info claims.** Gold NEI claims have no gold
   evidence, so the term is always 0 on exactly the claims the abstention
   story cares about. Needs a real decision, not the current placeholder.
4. **Optional `escalate` action.** Controller hands uncertain claims to the
   big model at a cost. Makes calibration a working part of the system.
   Novelty for fact-checking not checked. Extra scope; only after the core
   works.

## Must fix before any training

The calibration term in the proposal (and in `src/reward.py`) is `+q` when
right and `-w*q` when wrong. Expected reward is `q * [p(1+w) - w]`, linear
in `q`, so the optimum is always an extreme confidence: maximum if
`p > w/(1+w)`, otherwise minimum. The model never learns to say 0.6 when
it is 60% sure and the reliability diagram collapses to two points. It is
not a proper scoring rule.

Fix: use a Brier term, `-(q - 1[correct])^2`, whose optimum is `q = p`,
and put the asymmetric harm weights on the verdict term instead. Weighting
the Brier term by outcome does not work: the optimum shifts to
`p*w1 / (p*w1 + (1-p)*w0)`, which breaks calibration.
