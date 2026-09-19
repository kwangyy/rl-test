# Related work — what is taken, what is open

Search done 2026-09-18 with three search agents over alphaXiv and the web.

**How far to trust this.** Only ProFact was read in full by hand. Every
other entry is an agent summary that was not cross-checked. ProFact,
AWA-RL, Collapse Law and 2607.18240 are recent enough that they could not
be sanity-checked from memory either. Open the papers before citing them
in a proposal. The scoring-rule maths in `plan.md` was re-derived by hand.

## Taken vs open

| Piece of the plan | Status | Who did it |
|---|---|---|
| GRPO with verdict + evidence-overlap reward for fact-checking | Done | Veri-R1 (2510.01932), including an anti-shortcut guard and FEVER-family -> SciFact transfer |
| Agentic GRPO (decompose -> search -> verdict) on AVeriTeC | Done | ProFact (2606.13262) |
| Proper-scoring-rule confidence reward in RL | Done, QA only | RLCR (2507.16806, Brier); Rewarding Doubt (2503.02623, log score) |
| Abstention as a rewarded action in GRPO | Done, QA only | TruthRL (2509.25760), KnowRL (2506.19807), AWA-RL (2607.10738) |
| Reliability diagram before and after RL | Done, QA only | RLCR, Taming Overconfidence (2410.09724) |
| Calibrated fact-checker with reliability diagrams | Done, but prompted, not trained | 2607.18240 |
| Decomposition errors measured separately | Done | Decomposition Dilemmas (2411.02400) |
| **Calibration + abstention reward, RL-trained, on fact verification** | **Open** | none found |
| **Asymmetric harm-weighted costs in an LLM RL reward** | **Open** | only in non-LLM cost-sensitive RL |
| **Retrieval / compute cost term in a fact-verification RL reward** | **Open** | ProFact reports efficiency but does not reward it |
| **Separate small controller on top of a GRPO-trained verifier** | **Open** | adaptive-retrieval controllers exist, QA only |

"GRPO with verdict + evidence reward for fact-checking" cannot be claimed
as new. Neither can "a small model beats a big one via RL"; Veri-R1
already claims that at 3B.

## Framing that holds up

> Veri-R1 showed GRPO can ground verdicts in evidence. RLCR and TruthRL
> showed RL can train calibration and abstention, but only on QA. Nobody
> has combined them on fact verification, and nobody encodes the fact that
> errors are not equally harmful. We do both, with a cost-aware controller
> on top.

Against ProFact specifically: it learns *what* to ask but follows a fixed
pipeline. It does not learn *when to stop or abstain*, has no notion of
confidence, and its reward cannot leave AVeriTeC.

## ProFact (2606.13262) — read in full

Sun Yat-sen University, 11 Jun 2026. One Qwen model (Qwen2.5-3B/7B,
Qwen3-4B/8B) trained with GRPO over a fixed three-stage rollout: generate
up to 5 questions, search top-3 evidence per question, give a verdict.
Max 12 steps, 8 rollouts per claim, KL 0.001. Retrieval uses
Qwen3-Embedding-0.6B over AVeriTeC's static knowledge store. Trained on
the AVeriTeC train split, evaluated on the dev set.

| Point | Detail | Why it matters here |
|---|---|---|
| Pipeline is fixed | Always decomposes, always searches every question, then verdict. The model never decides when to stop, search more, or abstain. | The controller decisions in the plan are not learned in ProFact at all. |
| Reward needs gold questions and gold QA pairs | Stage rewards are METEOR overlap against AVeriTeC's annotated questions and answers, plus a verdict indicator. | Only AVeriTeC has those annotations, so the reward cannot run on FEVER or SciFact. Evidence-id F1 can. It also scores matching the annotators' wording, not evidence correctness. |
| AVeriTeC only | Dev set only. No test set, no other dataset, no transfer. | FEVER -> SciFact/AVeriTeC transfer is untouched. |
| No calibration work | No confidence output, no abstention analysis, no reliability diagram. "Not Enough Evidence" is one of 4 labels. | The Responsible-AI part of the plan is open against this paper. |
| No cost in the reward | Efficiency comes from pipeline design (fewer stages, context reset), compared only against InFact. | The cost term is open. |
| No limitations section, no hardware details | Neither appears in the text. | Reproducing it may be harder than it looks. |

Useful findings:

- **Bigger was not better.** AVeriTeC score 47.8 at 3B, 48.0 at 7B, 46.2
  at 4B, 46.4 at 8B. The authors attribute it to inverse scaling: larger
  models lean on their priors over the evidence.
- **Process rewards mattered.** Verdict-only reward: 47.8 -> 34.4 at 3B.
- **Their PPO ablation** (31.0 vs GRPO 47.8) is PPO on the LLM itself. It
  says nothing about a small separate controller.

## Numbers to beat (agent-reported unless noted)

| Benchmark | System | Score |
|---|---|---|
| AVeriTeC dev, 3B | ProFact (read by hand) | accuracy 68.8, AVeriTeC score 47.8 |
| AVeriTeC dev, 3B | HerO on the same backbone (from ProFact's table) | accuracy 61.0, AVeriTeC score 43.4 |
| AVeriTeC-25 shared task | CTU AIC / HerO 2 (2507.11004) | 0.332 / 0.271 |
| AVeriTeC-24 shared task | TUDA_MAI / HerO (2410.12377) | 0.63 / 0.57 |
| SciFact | MultiVerS (2112.01640) | abstract F1 72.5 |
| FEVER, claim only, no evidence | Schuster et al. (1908.05267) | 61.7% accuracy |

Schuster et al.'s symmetric test set is a ready-made check that a model is
not guessing labels from claim phrasing.

## Agentic search RL (all QA, agent-reported)

Search-R1 (2503.09516), ReSearch (2503.19470), R1-Searcher (2503.05592),
ZeroSearch (2505.04588), DeepResearcher (2504.03160). None evaluate on
fact verification; none have a cost term or abstention in the reward.
AWA-RL (2607.10738) adds a calibrated abstention reward to Search-R1-style
GRPO, on QA only.

Adaptive retrieval / stopping controllers, all QA: Self-RAG, FLARE,
Adaptive-RAG, DRAGIN, and 2026 budget papers AutoSearch (2604.17337),
GRASP (2607.10463), HALT (2608.02009), MetaRAG (2608.24214), "One Policy,
Any Budget" (2609.00813), "When Should Multi-Round RAG Stop?" (2608.13237).

## Calibration in RL (agent-reported)

| Paper | Reward | Fact verification? | Asymmetric costs? |
|---|---|---|---|
| RLCR (2507.16806) | `1[correct] - (q - 1[correct])^2`; proves a proper scoring rule is required | No | No |
| Rewarding Doubt (2503.02623) | log score | No | No |
| TruthRL (2509.25760) | ternary +1 / 0 abstain / -1 | No | No |
| KnowRL (2506.19807) | format + correctness (+2 / +1 refuse / -1) + atomic-fact reward | No | No |
| Taming Overconfidence (2410.09724) | recalibrates the RLHF reward model | No | No |
| R-Tuning (2311.09677) | SFT, not RL; sure/unsure labels | Yes (FEVER) | No |
| Collapse Law and Repair (2608.00301) | analyses the +1 / -lambda / 0 family; documents collapse to a corner | No | No |
| Why Language Models Hallucinate (2509.04664) | argument that binary grading rewards guessing | No | No |

## Tooling note (agent-reported)

TRL's `GRPOTrainer` reportedly supports multi-turn tool-calling rollouts
natively (`tools=[...]`, `environment_factory=`, experimental, v1.13 docs,
https://huggingface.co/docs/trl/grpo_trainer). Every paper above used verl.
