# Original proposal

The source text this experiment started from, kept verbatim (only the
dataset list was reformatted as a table). It was written as a pivot from
an earlier math-reasoning plan, hence the comparisons to math.
[`plan.md`](plan.md) records where the working plan now differs.

## 1. Task definition

Input: a claim (a sentence someone asserted as fact).
Output: a verdict (Supported / Refuted / Not Enough Info) + the evidence sentences/documents that justify it + a confidence score.

This is strictly harder than math in one good way and easier in another: harder because the model must retrieve, not just reason; easier because "harder" doesn't scale unboundedly the way a PDE does — the ceiling on presentability stays low and legible.

## 2. Datasets (all have ground-truth labels → free RLVR reward)

| Dataset | Size | What it adds |
|---|---|---|
| FEVER | ~185K claims over Wikipedia | Your backbone. Clean 3-way labels, huge, well-studied, Wikipedia evidence is easy to retrieve (dense/BM25 both work). |
| SciFact | ~1.4K claims over scientific abstracts | Harder evidence (scientific writing), good for "does it generalize beyond Wikipedia," and lets you use retrieval quality as a decision-difficulty signal. |
| AVeriTeC | ~4.5K real-world claims w/ web evidence | Closest to "real misinformation," requires multi-hop evidence gathering — great fit for your Tree/Graph-of-Thought decomposition step. This is the one to demo live, since it's real claims (political, health, viral social media) rather than synthetic Wikipedia sentences. |
| LIAR-PLUS | ~12.8K claims (PolitiFact) | Comes with 6-way truthfulness scale (not just 3-way) — useful if you want a stretch task on calibration granularity. |

Suggested split for your 8 weeks: train/SFT/GRPO on FEVER (cheap, fast iteration), evaluate generalization on SciFact + AVeriTeC (harder, more realistic), keep AVeriTeC as your live-demo dataset.

## 3. Reward design (this is the part that makes GRPO clean)

Same trick as math: rule-based, no judge needed for the core signal.

Verdict correctness: exact match against gold label → primary reward.

Evidence sufficiency/precision: F1 between retrieved evidence sentence IDs and gold evidence IDs (FEVER provides this natively) → shaped reward component, this is what stops the model from getting the right verdict for the wrong reason (a real problem in FEVER leaderboards historically — models learn to guess labels from claim phrasing alone, ignoring evidence).

Calibration/abstention reward: this is your Responsible-AI-native term. Reward correct NEI predictions when evidence is genuinely insufficient, and — critically — penalize confident-and-wrong more heavily than uncertain-and-wrong. This is the asymmetric cost structure I mentioned that math totally lacks.

Total reward: correctness + λ1·evidence_F1 + λ2·calibration_bonus − λ3·tokens, all four terms computable without an LLM judge. You can still layer an offline larger-model judge later for qualitative checks, but it's not on the critical training path — which is a nice thing to be able to say when asked about cost/reproducibility.

## 4. The controller — states, actions, and why it's more interesting here than in math

State: claim + evidence gathered so far + retrieval confidence (e.g. top-k document overlap/agreement) + model's self-reported confidence + hop count so far.

Actions (richer than the math version because retrieval adds a real branching factor):

- retrieve_more (broaden or re-query — e.g. reformulate the claim, follow a hyperlink, do another retrieval hop)
- decompose (split a compound claim into sub-claims — this is your Graph-of-Thought step; AVeriTeC claims are often compound: "Company X, founded in Y, laid off Z employees in 2026" — three checkable sub-claims)
- cross_check (self-consistency across independently retrieved evidence sets)
- abstain / NEI (declare insufficient evidence — this is the action math never gives you, because math problems are always "solvable")
- verdict (commit to Supported/Refuted with the accumulated evidence)

Reward for the controller specifically: correctness − λ·(retrieval calls + tokens), with the calibration bonus/penalty folded in. This is a genuinely richer MDP than the math version — the action space isn't just "compute more," it includes an information-gathering dimension, which is closer to how sequential decision-making problems look in the real world (search-and-commit under a budget) and is a better fit for the course's "planning under uncertainty" framing.

## 5. Where Responsible AI stops being a bullet point

This is the part I'd lean on hardest in both the report and the presentation, because it's the actual differentiator from a "we fine-tuned a small model on FEVER" project:

Calibrated abstention as a first-class action, not a fallback — the model can say "I don't have enough evidence," and your reward function explicitly rewards this over guessing. You can show a reliability diagram (confidence vs. actual accuracy) before and after your calibration reward term — a very presentable chart.

Evidence transparency — every verdict ships with the exact sentences it's grounded in. You can show, live, a case where the verdict is right but for evidence that doesn't actually support it (a known FEVER failure mode) versus your model after training.

Asymmetric harm framing — explicitly discuss in the report that a false "Refuted" on a true claim and a false "Supported" on a false claim are not equally costly in a misinformation context, and show your reward function encodes that asymmetry. This is a direct, technical instantiation of "accountability and trust" — not just a paragraph gesturing at ethics.

Failure mode audit — pull a handful of cases where the model is confidently wrong, show the evidence trail, and discuss why (e.g., stylistic bias in claim phrasing, retrieval failure, sub-claim decomposition error). Graders explicitly reward "evidence of effort" and "analysis and insights" — this section is where that shows.

## 6. Why this presents better than math, concretely

Pick 3–4 claims live from the audience during Q&A (a recent real headline, a deliberately tricky compound claim, an ambiguous one) and run the pipeline in real time — evidence retrieved, sub-claims shown, verdict + confidence displayed. No one needs to know what a PDE is to follow along, and there's no "wait, is this actually a hard example" derailment because the audience already has priors about whether a claim sounds true.

The controller's decision trace ("retrieved 2 docs → still uncertain → decomposed into 2 sub-claims → cross-checked → committed") is a natural flowchart/graph visual, which plays well as a single slide and is far more intuitive than a search tree over algebraic manipulations.

The accuracy-vs-compute curve and the reliability diagram both survive from your original math plan basically unchanged.

## 7. Adjusted risks (replacing the math-specific ones)

Retrieval quality bottleneck: if retrieval is bad, no amount of controller cleverness fixes it → budget real time to a solid retriever (BM25 baseline + a small dense retriever, e.g. off-the-shelf sentence embeddings) before touching the controller.

Wikipedia-only generalization: FEVER-trained models can overfit to Wikipedia phrasing → mitigate by evaluating (not necessarily training) on SciFact/AVeriTeC, and report the gap honestly — it's a legitimate finding, not a failure.

Sub-claim decomposition errors compound → track this as its own metric, not just folded into final accuracy, so you can show where errors originate.
