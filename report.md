# Fact-checking RL project: team split, full plan, and what to do if the numbers disappoint

*Draft for review, 2026-09-25. Builds on `factcheck/REPORT.md`,
`factcheck/docs/plan.md`, `docs/related-work.md` and
`docs/stage0-results.md`. Nothing here has been run; it is a plan. The
four decisions in section 6 were taken the same day and the schedule
below reflects them: two learners, six weeks, both groups train.*

## 1. Where the project is today

- **Task.** Claim in, three things out: verdict (Supported / Refuted /
  Not Enough Info), the Wikipedia sentences it rests on, and a confidence.
- **Done (stage 0).** Retrieval over 5.4M FEVER pages at 87.5% page
  recall@5 with title matching. Prompted baseline on 900 dev claims for
  Qwen3.5-2B (local, to be trained) and Qwen3.8-Flash (180B, frozen).
- **The measured gap.** With gold evidence in context, the big model is
  right 95% of the time, the 2B model 63%. The 2B model says SUPPORTED to
  almost everything (94 / 30 / 14% recall on Supported / Refuted / NEI)
  and its stated confidence carries no information (ECE 0.51).
- **Not done.** Nothing is trained. The controller environment still
  runs on a 12-claim toy corpus. Two reward bugs are known and unfixed
  (calibration term is not a proper scoring rule; evidence F1 undefined
  on NEI). Training infra (WSL2, vLLM, TRL GRPO) is not set up.

## 2. What "properly fleshed out" looks like

The current plan has three stages but leaves gaps that a six-person team
will fall into if they are not named. Below is the full system, then what
has to be added.

### 2.1 The system, end to end

```
claim ──> RETRIEVER (BM25 + title match, later dense) ──> candidate sentences
                │
                v
        CONTROLLER (small policy, stage 2)
        state: evidence so far, retrieval scores, verifier verdict+conf, hops, budget
        actions: search again | decompose | verify | abstain | commit
                │
                v
        VERIFIER (Qwen3.5-2B, GRPO-trained, stage 1)
        (claim, evidence) -> verdict + evidence ids + confidence
                │
                v
        verdict + cited sentences + confidence + decision trace
```

Two learners, one frozen retriever, one frozen decomposer. The verifier is
trained single-turn with GRPO; the controller is trained on top of the
frozen verifier's outputs, on CPU.

### 2.2 Gaps to close before anyone trains

| Gap | Why it matters | Owner (see section 3) |
|---|---|---|
| Brier calibration term | Current term drives confidence to 0 or 1; the reliability diagram would collapse to two points and the calibration story dies | RL-2 |
| Evidence F1 on NEI claims | A third of the data has no gold evidence; the shaped term is 0 exactly where abstention matters. Proposal: on gold-NEI, reward an empty evidence list and penalise cited evidence (it is by definition not sufficient) | RL-2 |
| Frozen evaluation harness | Every number from every sub-team must come from one script with fixed splits, or the two teams' results will not be comparable | AG-3 |
| FEVER symmetric test set (Schuster et al.) | Detects the classic FEVER shortcut: guessing labels from claim wording. Without it, an accuracy gain could be a shortcut gain | AG-3 |
| SFT baseline | "Why RL and not fine-tuning?" needs a number, not an argument. Also measures the 2B model's capacity ceiling (section 4.1) | RL-1 |
| Post-hoc calibration baseline | Temperature scaling on verdict-token probability costs nothing. If RL cannot beat it, the calibration claim is dead | RL-2 |
| Controller on real data | Replace the toy corpus with an offline log of verifier outputs on FEVER train, so the controller trains in minutes on CPU | RL-3 |
| Multi-turn agent loop | The "agentic" half of the project does not exist yet in code. Needed for the prompted agentic baseline, decomposition, re-query, and the live demo | AG-1, AG-2 |
| SciFact / AVeriTeC loaders and knowledge stores | Transfer evaluation is promised in the proposal; loaders do not exist | AG-3, AG-1 |
| Training infra | WSL2, `flash-linear-attention`, vLLM, TRL `GRPOTrainer` with LoRA on a 16GB card | RL-1 |

### 2.3 Stages, restated with deliverables

| Stage | Deliverable | Kill / go criterion |
|---|---|---|
| 0b Agentic prompted baseline | Same 900 claims through the multi-turn loop (search, decompose, re-query) with the frozen 2B and big model. Searches per claim, accuracy, cost. | Informational. Tells us how much a smarter loop buys with no training. |
| 1a SFT verifier (capacity probe) | LoRA SFT on FEVER train with gold evidence in context, evaluated on the same 900 dev claims. Set up in `src/sft_probe.py`; runbook and decision rule in `docs/sft-probe.md`. | If SFT with gold evidence cannot pass ~80%, the 2B model lacks capacity; switch to 4B before spending GPU time on GRPO. Runs in week 1. |
| 1b GRPO verifier | Before/after on the same 900 claims: accuracy, per-label recall, evidence F1, ECE, Brier, reliability diagram. Ablations: no calibration term, no harm weights, format-only reward, random reward. | Go to stage 2 if GRPO beats SFT on calibration or harm-weighted error, even if raw accuracy ties. |
| 2 Controller | Accuracy vs cost curve from a lambda sweep; abstention precision/recall; comparison with fixed policies (always one search; always two). | Success is the learned policy sitting above the fixed policies on the curve for some lambda range. |
| 3 Transfer and demo | SciFact and AVeriTeC dev with the FEVER-trained system. Live demo on AVeriTeC claims with the decision trace. | Report the gap honestly; it is a finding either way. |

### 2.4 Six-week schedule

| Week | RL group | Agentic group | Joint |
|---|---|---|---|
| 1 | SFT capacity probe (2B, then 4B only if needed) and the model decision; Brier term and NEI evidence rule fixed; WSL2 + TRL running | Multi-turn loop skeleton over the existing retriever; SciFact/AVeriTeC loaders; TRL tool-calling rollout spike | Freeze interfaces (section 3.3) and the eval harness; GPU calendar |
| 2 | GRPO run 1 on the chosen model with the full reward; first SFT-vs-GRPO read | Agentic prompted baseline (stage 0b) for 2B and big model; multi-turn GRPO run 1, same base and same reward as RL-1 | Review 1: does GRPO move anything? Apply section 4 |
| 3 | Reward ablations (no calibration, no harm, format-only, random); controller env over offline verifier logs | Multi-turn vs single-turn GRPO on the same 900 claims; symmetric test set in the harness | |
| 4 | Controller training, lambda sweep, fixed-policy baselines | Failure-mode audit tool; transfer runs on SciFact and AVeriTeC start | Review 2: lock which of the three claims leads the report |
| 5 | Controller vs fixed policies; abstention analysis; final reliability diagrams | Transfer results; demo UI with decision trace | Writing starts |
| 6 | Report and slides | Report and slides; demo dry run | Final review |

**Cut for six weeks.** Dense retriever (title match already gives 87.5%
page recall; keep it optional), hyperlink-follow tool, the `escalate`
action, LIAR-PLUS, and a live web-search path for arbitrary claims. The
demo runs on AVeriTeC's knowledge store instead. Everything that feeds
the three claims in section 4.3 stays.

**One GPU, two training groups.** Verifier GRPO (RL-1) and multi-turn
GRPO (AG-2) share the same 16GB card. The controller trains on CPU by
design, and evaluation runs of 900 claims are short, so the collision is
between the two GRPO owners. AG-3 keeps a GPU calendar: RL-1 has
priority in weeks 1 and 2, AG-2 in weeks 2 and 3, ablations run
overnight. Multi-turn GRPO starts from the same base model with the same
reward as the single-turn run, so the only difference is the rollout,
and the comparison answers the open single-turn vs multi-turn question
in `plan.md` cleanly.

## 3. Splitting six people into two sub-teams

The natural seam is **what is learned versus what is orchestrated**. The
RL group owns everything that has a reward and a gradient. The agentic
group owns the environment the learners live in, the tools they call, the
data they are measured on, and the demo. Neither team's work is blocked
on the other's if the interfaces in 3.3 are frozen in week 1.

### 3.1 RL group (3 people)

| Role | Owns | First two weeks | Main deliverable |
|---|---|---|---|
| **RL-1, verifier training** | TRL `GRPOTrainer`, LoRA config, WSL2/vLLM setup, SFT baseline, training runs and their logs | Run the SFT capacity probe (`sft_probe.py`) and make the 2B/4B call in week 1; get one GRPO step running on the chosen model | Before/after table and the SFT-vs-GRPO comparison |
| **RL-2, reward and calibration** | `reward.py`, Brier fix, harm matrix, NEI evidence rule, ECE/Brier/reliability plots, post-hoc temperature-scaling baseline, reward ablations, reward-hacking checks | Rewrite the reward with unit tests that encode the intent (proper scoring rule: optimum confidence equals accuracy; harm asymmetry: wrong SUPPORTED costs more than wrong REFUTED) | Reliability diagrams before/after and the ablation table |
| **RL-3, controller** | Gymnasium env over real FEVER, offline verifier-output dataset, PPO/DQN training, lambda sweep, fixed-policy baselines, abstention analysis | Replace toy corpus; build the offline log format with AG-2 | Accuracy-vs-cost curve, controller vs fixed policies |

### 3.2 Agentic AI group (3 people)

| Role | Owns | First two weeks | Main deliverable |
|---|---|---|---|
| **AG-1, retrieval and tools** | BM25 + title match, dense retriever (Qwen3-Embedding-0.6B), sentence-level retrieval, re-query and hyperlink-follow tools, knowledge stores for SciFact and AVeriTeC, retrieval recall evals | Sentence-level recall numbers; dense index; tool interface | Retrieval eval table across datasets; the tool set the loop calls |
| **AG-2, agent loop and multi-turn GRPO** | Multi-turn loop (search / decompose / verify / abstain / commit), decomposer prompt on the big model, trace logging, agentic prompted baseline (stage 0b), multi-turn GRPO rollouts with retrieval tokens masked | Loop skeleton that runs the 900-claim set end to end and writes traces; TRL tool-calling rollout spike | Stage 0b results; multi-turn vs single-turn GRPO comparison; the trace format the demo and controller consume |
| **AG-3, evaluation, data and demo** | Fixed splits, eval harness, FEVER score, symmetric test set, SciFact/AVeriTeC loaders, failure-mode audit, cost accounting, demo UI | Harness that both teams run; loaders | Every number in the final report, the audit, the live demo |

AG-3 is the referee. No number goes into a slide unless it came from the
harness. This is the single most important coordination rule.

### 3.3 Interfaces to freeze in week 1

Four contracts, each a short Python signature plus a JSONL schema. Once
frozen, each team can work without the other.

1. **Retriever tool.** `search(query, k) -> [{page, sent_id, text, score}]`.
   Owned by AG-1, consumed by everyone.
2. **Verifier call.** `verify(claim, evidence) -> {verdict, evidence_ids,
   confidence, token_prob}`. Owned by RL-1 (the trained model) but the
   signature is set by AG-2 (the loop).
3. **Reward.** `reward(gold, pred, confidence, gold_ids, cited_ids,
   n_ops) -> (float, parts)`. Owned by RL-2. Already exists; the
   signature stays, the body changes.
4. **Trace / episode log.** One JSONL row per claim: actions taken,
   retrieval results, verifier outputs, final answer, cost. Owned by
   AG-2, consumed by RL-3 (controller training data), AG-3 (eval, audit)
   and the demo.

### 3.4 Where the teams meet

- **Week 2, week 4, week 6 joint reviews** with one page of numbers from
  the harness each.
- **The controller is the join point.** RL-3 trains on traces AG-2
  produces with the verifier RL-1 trained. Get a dummy end-to-end run
  going by week 3 even with an untrained verifier.
- **The demo is the other join point.** The trained verifier and
  controller plug into AG-2's loop; AG-3 puts a front on it.

## 4. What if the results are not promising, even though the gap is real

The 32-point gap is real and measured. That does not mean RL on a 2B
model will close it. The honest framing to adopt now, before any run:
**the gap is between a 2B and a 180B model; how much of it a 2B model
can ever close is unknown**. Everything below follows from planning for
that.

### 4.1 First, measure the ceiling cheaply

Before GRPO, run SFT with gold evidence in context (stage 1a). SFT is the
easiest possible training signal. If the 2B model cannot reach roughly
80% accuracy given the answer on the page, the problem is capacity, not
training method. Switch to Qwen3.5-4B with QLoRA at that point, in week
1, not week 5. This one experiment converts "RL did not work" into "the
2B model tops out at X; here is what RL adds on top of that".

The experiment is set up: `factcheck/src/sft_probe.py` builds the
training set (gold pages guaranteed in context, title-match pages as
distractors, gold verdict and evidence ids as target) and trains a LoRA
adapter; `stage0.py --adapter` evaluates it on the same 900 dev claims
under both oracle and real retrieval. The commands, the decision
thresholds and a results table to fill are in `docs/sft-probe.md`.

### 4.2 Failure scenarios and what each one still yields

| Scenario | Likely cause | Diagnostic | What we do | What we still have to show |
|---|---|---|---|---|
| **A. Accuracy does not move under GRPO** | Reward too sparse; format term dominating; KL too tight; LoRA rank too small | Per-term reward curves; fraction of format-valid completions; compare to SFT | Curriculum: train first on claims with gold evidence in context, then widen. Raise LoRA rank. Loosen KL. If SFT also flat, capacity: go 4B | SFT vs GRPO comparison is itself a result: "on this task at this scale, RL adds nothing over SFT" is a legitimate course-report finding if the controls are clean |
| **B. Accuracy moves, calibration does not (ECE flat, confidence still 0/1)** | Brier weight too small relative to correctness; model cannot express graded confidence in text | Reliability diagram; histogram of stated confidence; compare stated vs verdict-token probability | Raise the Brier weight; train on token probability instead of text confidence; compare against temperature scaling | The comparison "RL calibration vs post-hoc temperature scaling" is exactly the question a reviewer would ask. Either answer is informative |
| **C. Abstention over-fires (model says NEI to everything)** | NEI reward is a safe harbour; classic reward hacking | NEI recall vs NEI precision; accuracy-coverage curve | Lower the NEI reward; add a coverage term; report the selective-prediction curve instead of one operating point | An accuracy-coverage curve is the standard selective-prediction result and is a good chart even if no single point is impressive |
| **D. Gain is from format, not judgment** | The 2B model's 84-93% format adherence leaves room; a format-only reward could deliver most of the gain (the "spurious rewards" effect on Qwen) | Format-only-reward ablation; random-reward ablation; accuracy on format-valid outputs only | Report the decomposition: how many points came from format, how many from judgment | Attributing the gain correctly is the whole point of the ablations. If most of it is format, say so; it is an honest and interesting finding about small-model RL |
| **E. Controller adds nothing over "search once, verify, commit"** | On FEVER with 87% page recall@5, one search is usually enough; the fixed pipeline is already on the frontier | Plot fixed policies on the same accuracy-cost axes | Move the controller test to where it should matter: the retrieval-miss subset, and AVeriTeC multi-hop claims | "Adaptive stopping only helps when retrieval is unreliable, and here is the threshold" is a real result about when agentic loops pay for themselves |
| **F. Nothing transfers to SciFact / AVeriTeC** | Domain shift, Wikipedia phrasing overfit | Retrieval recall on each dataset (separates retrieval failure from verifier failure) | Report the gap by component | The proposal already commits to reporting this honestly. It is expected |
| **G. Harm asymmetry does not change behaviour** | Weights too close to 1; correctness term swamps it | Confusion matrix before/after; count of wrong-SUPPORTED on gold-REFUTED | Sweep the harm ratio (1, 1.5, 3, 5); show the confusion matrix shifting | The sweep is the result: "at ratio r the model trades x points of Refuted recall for y points of Supported recall" |

### 4.3 Rules that protect the project from a bad week 6

- **Pre-register success criteria** in the harness, per stage, before
  the first training run. The kill criteria in 2.3 are the draft.
- **Every ablation is a first-class run**, scheduled from the start, not
  something to do "if there is time". Random reward, format-only reward,
  no-calibration, no-harm, SFT-only, temperature scaling. Six controls.
  Without them, a positive result cannot be attributed and a negative
  result cannot be explained.
- **The stage-0 numbers are already a deliverable.** A report whose
  contribution is "we measured the gap, showed that prompting cannot fix
  calibration, and showed what SFT and GRPO each do to it" is a complete
  project even if no number goes the way we hoped.
- **Separate the three claims** so one failing does not sink the others:
  (i) RL improves verdict accuracy; (ii) RL improves calibration and
  abstention; (iii) a cost-aware controller improves the accuracy-cost
  trade-off. The literature says (ii) is the most novel on fact
  verification and (i) is the most likely to be marginal over SFT.
  Lead the report with whichever holds.
- **Keep the failure-mode audit regardless.** Ten confidently-wrong cases
  with their evidence trails, categorised (retrieval miss, shortcut,
  decomposition error, genuine ambiguity in the gold label), is the
  section graders reward for analysis and it does not depend on training
  succeeding.

## 5. Why this matters outside the course: current use in industry and academia

### 5.1 Academic benchmarks that measure this

Two families. The first is **claim verification with evidence**, which
is exactly this project's task. The second is **LLM factuality**, which
measures whether a model's own outputs are true; related, but a different
task.

| Benchmark | What it measures | Relevance here |
|---|---|---|
| **FEVER** (2018), 185K Wikipedia claims | 3-way verdict + evidence sentences; FEVER score requires both | Our training set and metric |
| **FEVER symmetric test set** (Schuster et al. 2019) | Whether the model uses evidence or guesses from claim wording | Our shortcut check |
| **FEVEROUS**, **HoVer**, **Climate-FEVER** | Tables, multi-hop chains, climate claims | Harder FEVER variants; optional transfer targets |
| **SciFact** (1.4K claims over scientific abstracts) | Scientific evidence, abstract-level F1 | Our first transfer target |
| **AVeriTeC** (4.5K real-world claims, web evidence) | 4-way verdict including "conflicting / cherry-picking"; question-answer evidence; shared tasks at FEVER 2024 and 2025 | Our demo dataset. The 2025 shared task allowed open-weight models only, run inside an AWS VM with a single A10G (23GB), at most one minute per claim on average, scored by Ev2R recall with Llama 3.3 70B as grader. Seven teams entered; CTU AIC won with an AVeriTeC score of 33.17. That is a direct endorsement of the small-model, cost-aware framing, and our 16GB card is in the same class |
| **AVerImaTeC** (FEVER 9 at EACL 2026) | Image-text claims with web evidence | The field is moving multimodal; out of scope but worth one sentence in the report |
| **VeriTaS** (2026) | Dynamic, multimodal fact-checking benchmark refreshed to avoid contamination | Shows the contamination worry with static sets like FEVER is taken seriously |
| **Ev2R** | How to evaluate evidence retrieval itself | Relevant to AG-1's metrics |
| **CLEF CheckThat! 2026** | Multilingual check-worthiness and verification | Adjacent shared task |
| **SimpleQA / SimpleQA Verified** | Short closed-book factual questions; graded correct / incorrect / not attempted | The "not attempted" grade is the same abstention idea we reward. Used in frontier-model system cards |
| **FACTS Grounding / FACTS Benchmark Suite** (Google DeepMind) | Whether a long answer is fully grounded in a supplied document | Same spirit as our evidence F1, applied to generation |
| **FActScore, LongFact** | Decompose a generated text into atomic claims and check each against Wikipedia | Our decomposition step is this in reverse; a trained verifier could be the checker inside these pipelines |
| **TruthfulQA, HalluLens, HalluHard** | Hallucination and truthfulness of model outputs | Background; not our task |

### 5.2 Industry use

- **X and Grok.** "@grok, is this true?" became the single most common
  message sent to Grok after it was integrated into X. A working paper by
  Renault, Mosleh and Rand counts 447,083 tweets tagging Grok to request a
  fact check between March and September 2025, and almost 1.4 million
  fact-check requests to Grok and Perplexity combined, 7.6% of all
  interactions with the two bots. The same authors found a substantial
  drop in Community Notes proposals after Grok's launch, and a separate
  study by Zhou and Hou found note requests, contributors, notes written
  and note ratings all fell significantly in the three months after
  launch. Chatbots appear to substitute for, not complement, crowd
  fact-checking. Documented failures (AFP, June 2025): Grok labelled an
  AI-generated video of a giant anaconda "genuine" and cited
  plausible-sounding expeditions, after which users quoted Grok as proof
  the clip was real; during the India-Pakistan conflict it identified old
  footage of Khartoum airport as a missile strike on a Pakistani airbase
  and a building fire in Nepal as "likely" Pakistan's military response.
  This is our stage-0 finding at platform scale: a model that commits
  with high confidence, invents supporting detail, and rarely says "I
  cannot tell". The calibration and abstention part of this project is
  the fix that product needs.
- **Perplexity** is the other chatbot users tag on X for fact checks,
  and its citation-first answer format is the consumer surface where
  evidence grounding matters most.
- **Meta** announced on 7 January 2025 that it would end its US
  third-party fact-checking programme, which had run since 2016 with
  more than 90 certified organisations in over 60 languages, and shut it
  down in the US on 7 April 2025 in favour of a Community Notes model.
  The point for us: platforms are shifting from human fact-checkers to
  crowd and automated systems, so automated verification quality is
  becoming the whole game.
- **Full Fact AI** (UK non-profit) runs claim detection and matching
  over TV, radio and social media, triaging about 300,000 sentences a
  day for partner fact-checkers in the UK, Europe, Africa and the US.
  Their pipeline is check-worthiness, then claim matching against
  already-checked claims, then human verification. A calibrated verifier
  slots in as the triage step that decides what reaches a human.
- **Factiverse** (Norway) sells live fact-checking and an AI editor that
  flags factual errors in text and finds sources, in 100+ languages.
- **Google Fact Check Tools / ClaimReview** aggregate human fact-checks
  into search. **Grokipedia** (xAI, late 2025) is an AI-generated
  encyclopedia, which raises the question of AI checking AI-written
  reference text.
- **Frontier labs** report SimpleQA-style factuality and hallucination
  rates in model cards, and grounding scores on FACTS. The "not
  attempted" rate is now a headline number, which is the abstention
  metric this project trains directly.

The common thread: in every deployed setting the failure mode is not
"the model cannot find evidence", it is "the model commits confidently
when it should not". That is the part of the problem this project is
built around.

## 6. Decisions taken, 2026-09-25

1. **Two-learner design is settled.** A GRPO-trained verifier plus a
   separate small controller. RL-3 owns the controller as written.
2. **The agentic group trains.** Multi-turn GRPO is an AG-2 deliverable,
   not a stretch goal. The cost is GPU contention (section 2.4); the
   payoff is a clean answer to the single-turn vs multi-turn question.
3. **Six weeks, not eight.** Schedule compressed and scope cut as listed
   in section 2.4.
4. **The 2B vs 4B call is made now**, from the SFT capacity probe in
   week 1. Code and runbook are in place (section 4.1); the runs need the
   desktop with the 16GB card and the FEVER data.

## Sources used for section 5

- [Indicator: how X's chatbot performs as a fact-checking tool](https://indicator.media/p/grok-is-this-true-how-x-s-chatbot-performs-as-a-fact-checking-tool)
- [France24: AI 'factchecks' sow misinformation](https://www.france24.com/en/live-news/20250602-hey-chatbot-is-this-true-ai-factchecks-sow-misinformation)
- [Tom Stafford: Grok, a mixed bag](https://tomstafford.substack.com/p/grok-a-mixed-bag)
- [Arab News: Grok during the India-Pakistan conflict](https://www.arabnews.com/node/2603047/pakistan)
- [Wikipedia: Misinformation during the 2026 Iran war](https://en.wikipedia.org/wiki/Misinformation_during_the_2026_Iran_war)
- [Wikipedia: Grokipedia](https://en.wikipedia.org/wiki/Grokipedia)
- [Meta: More Speech and Fewer Mistakes (7 Jan 2025)](https://about.fb.com/news/2025/01/meta-more-speech-fewer-mistakes/)
- [Poynter: Meta ends US third-party fact-checking](https://www.poynter.org/fact-checking/2025/meta-ends-fact-checking-community-notes-facebook/)
- [AVeriTeC shared task (FEVER 2024)](https://aclanthology.org/2024.fever-1.1/)
- [AVeriTeC 2025 task page: rules, VM, time limit, scoring](https://fever.ai/2025/task.html)
- [2nd AVeriTeC shared task overview: open-weights, reproducible, efficient (FEVER 2025)](https://aclanthology.org/2025.fever-1.15/)
- [Team HUMANE at AVeriTeC 2025: HerO 2](https://arxiv.org/pdf/2507.11004)
- [FEVER workshop site](https://fever.ai/)
- [VeriTaS dynamic multimodal benchmark](https://arxiv.org/html/2601.08611)
- [Ev2R: evaluating evidence retrieval](https://arxiv.org/pdf/2411.05375)
- [CLEF-2026 CheckThat! Lab](https://arxiv.org/pdf/2602.09516)
- [SimpleQA Verified](https://arxiv.org/pdf/2509.07968)
- [FACTS Grounding leaderboard](https://www.kaggle.com/benchmarks/google/facts-grounding)
- [FACTS Benchmark Suite, Google DeepMind](https://deepmind.google/blog/facts-benchmark-suite-systematically-evaluating-the-factuality-of-large-language-models/)
- [HalluLens](https://arxiv.org/pdf/2504.17550)
- [Nieman Lab: Full Fact's AI tools](https://www.niemanlab.org/2026/06/full-fact-is-battling-ai-generated-elections-content-with-ai-tools-of-its-own/)
- [Full Fact AI](https://fullfact.org/ai/)
- [Factiverse live fact-checking](https://www.factiverse.ai/blog/introducing-factiverse-live-fact-checking)
