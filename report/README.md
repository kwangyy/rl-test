# CS5446 report: what each TODO needs, with numbers and draft text

Source report: `CS5446_Project.zip` (`report.tex`, 20 `\pending{}` TODOs, rendered as `CS5446_Project.pdf`).
Prepared 2026-11-06 after fetching every branch on `origin` (kwangyy/rl-test) and `team` (ChaoChienHung/rl-test).

**How to use this page.** Each section below names one TODO in `report.tex` order. It gives the
numbers to use, a draft paragraph or LaTeX snippet, and the source file and branch. Draft text is
a starting point: check it against your own reading before submitting. Items marked
**[needs owner]** cannot be filled from the repository and need a teammate's input.

Every number was read from a committed doc or recomputed from committed predictions and traces
with the scripts in [`scripts/`](scripts/). The recomputed numbers match the docs wherever both
exist (e.g. Base titles 0.459 / 0.340, SFT-6000 token ECE 0.034, imitation gold delivered 90.0%).

## 0. Branch map: where the results live

| Branch | Latest commit | What it holds for the report |
|---|---|---|
| `controller/grpo-handover` (base of this branch) | 2026-11-06, PR #4 merged | Verifier SFT/direct-GRPO docs, prompted controller, imitation, offline RL, cascade, **controller GRPO incl. holdout 3,000**, decomposer, dev-900 controller traces |
| `team/rl` (ChaoChienHung) | 2026-10-26 | **Native-thinking CoT GRPO** (from SFT 5895 / disjoint 4695 / 4500), continued-SFT-1200 control, CoT SFT 5500, tagged formats, fast/slow, DPO scripts (no DPO results yet), RSI-Jev controller |
| `feat/controller` | 2026-10-27 | RSI-Jev chunk controller summary (`agent/docs/controller-workflow-and-results.md`) |
| `mt-grpo-on-handover` | 2026-11-02 | **Multi-turn verifier GRPO** (learning to search) and the evidence-gated reward |
| `agent` | 2026-10-24 | Retrieval docs incl. follow/Exa extension, agent loop, final-answer rule v3 |
| `demo`, `demo-controller-grpo-traces` | 2026-11-06 | Demo UI and its replay traces (not needed for the written report) |
| `controller-grpo` | 2026-11-04 | Same as merged PR #4 |

All local branches were fast-forwarded; `team/rl` is tracked locally as `team-rl`.

## Summary of what is missing from `report.tex`

| # | Location | TODO | Status |
|---|---|---|---|
| 1 | Abstract | Controller & Verifier results | Draft below (§1) |
| 2 | §2 end | Adaptive controller GRPO | Draft below; one citation to verify |
| 3 | §4.3 | Verifier details | Draft below |
| 4 | §4.4 | Controller details | Draft below |
| 5 | §4.6 | Missing decomposer details | Draft below |
| 6 | §5.1 | Anything missing | Additions below |
| 7 | §5.2 | SciFact / AVeriTeC transfer | **Decision needed**: no transfer results exist. Recommend dropping from §5.2 |
| 8 | §5.3 | Anything missing | Additions below |
| 9–10 | §6.1 | Table 1 + other matrices | Filled (two panels + confusion matrices) |
| 11–12 | §6.2 | Table 2 + other matrices | Filled; budget sweep not run |
| 13 | §6.3 | Ablation summary + learning curves | Filled from existing runs |
| 14–15 | §6.4 | Evidence funnel, error breakdown, case traces | Filled (new funnel computed) |
| 16 | §6.5 | Empty heading | Suggest "Multi-turn GRPO: learning to search" |
| 17 | §7.1 | Check anything else | Additions below |
| 18 | §7.2 | Case study | Draft below |
| 19 | §7.3 | Empty heading | Suggest "Where the remaining errors are" |
| 20 | §9 | Conclusion | Draft below |
| 21 | Team contributions | — | Draft from git history; **Yao Tianhao [needs owner]** |
| 22 | AI assistance disclosure | — | Template; **[needs team]** |

Also fix these **inconsistencies in existing text** (§10 below): §4.2 says λc = 0.02, but controller GRPO used
0.01 per operation and a −0.5 invalid-reply penalty. Eq. (2) omits the −2 invalid-format reward
used in every GRPO run. §8 says "Three RL seeds", but the verifier and controller GRPO runs have
one or two seeds; only the offline RL study has three.

---

## 1. Abstract — "Controller & Verifier results??"

Draft (replace the `\pending` line):

> On 900 label-balanced FEVER development claims with title retrieval, supervised fine-tuning lifts
> Qwen3.5-2B from 45.9% to 76.1% accuracy and from 0.340 to 0.671 strict FEVER score. It also cuts
> mean verdict harm from 0.81 to 0.29. Native-thinking GRPO with a harm-weighted verdict and evidence
> reward does not improve on SFT: 74.1–76.2% accuracy and 0.671–0.672 FEVER, against 74.6% / 0.666 for
> a continued-SFT control on the same 1,200 claims. Direct-output GRPO makes verdict-token confidence overconfident (ECE
> 0.03 → 0.21). For control, a 2B controller trained by imitation of an API model ties that model
> (0.684 vs 0.690 FEVER). Multi-turn GRPO of this controller with a frozen verifier adds at most
> one FEVER point (+0.5 on a 3,000-claim holdout, p = 0.16). An evidence audit shows why: the
> controller already places a complete gold evidence set in the verifier's input for 90% of
> Supported/Refuted claims, so most remaining errors are the verifier's.

---

## 2. §2 Related work — "Adaptive controller GRPO??"

Draft paragraph (add before **Our scope**):

> **Agentic RL for search.** Recent work trains a single language model end-to-end to interleave
> search calls with reasoning using outcome-reward RL. Search-R1 is one example [verify and add
> to `references.bib`: Jin et al., 2025, arXiv:2503.09516]; Veri-R1 [4] applies the same idea to
> claim verification. These approaches update the policy and the judge together, so a gain cannot
> be attributed to better action choice or better judgement. We instead apply GRPO [9] to a
> separate 2B controller over multi-step trajectories, with the verifier frozen and imitation of an
> API controller as the starting policy and KL reference.

Keep only references you have checked. The bib currently has none for agentic search RL.

---

## 3. §4.3 Verifier — "Details"

Facts (sources: `verifier/docs/sft-probe.md`, `grpo-results.md`; `team/rl` run `meta.json`):

- **SFT.** FEVER train, label-balanced, 5,895 examples after 105 drops (gold line beyond the
  sentence cut). Oracle context: pages of the smallest gold set guaranteed, title-match pages fill to 5, order
  shuffled. Target JSON `{"verdict", "evidence": [gold IDs], "confidence": 1.0}`. LoRA r = 16 on all
  linear layers, lr 1e-4, one epoch, effective batch 8, 4,096 max tokens. 2,190 s and 5.2 GB peak for `run1`
  (WSL2); `run2` took 6,720 s.
  4B QLoRA SFT beat 2B by only 0.9 (oracle) / 1.3 (titles) points, below the 5-point bar, so the team kept 2B (2026-10-08).
- **Direct GRPO (historical).** 1,200 oracle-context train claims, 8 samples per prompt, 2 prompts per
  step, lr 2e-5, full reward (harm verdict + 0.5 evidence + 0.5 negative Brier on stated confidence).
  KL β ∈ {0, 0.04, 0.5}. Optional filtering keeps only the 926 claims whose 8 SFT samples' rewards differ.
- **Native-thinking CoT GRPO (current).** Starts from the 5,895-row SFT adapter (`run2`) or from the
  disjoint 4,695-row SFT adapter. Both train on the same fixed 1,200 second-stage claims
  (`fever-oracle-matched-grpo-1200`; split checked by `utils.build_disjoint_sft_grpo_split --check`).
  8 generations, 1,024-token completions, β = 0, 600 steps, seed 0. Reward: harm-weighted verdict +
  0.5 × evidence; calibration off; an unparsable output scores −2. Training took 6.1 h (from 5895) and
  9.7 h (from 4695), with ~20 GB peak on A100.

Draft:

> SFT adapts Qwen3.5-2B with LoRA (rank 16, all linear layers, learning rate $10^{-4}$, one epoch) on
> 5,895 label-balanced FEVER training claims. Each claim is shown with oracle context: the pages of
> its smallest gold evidence set plus title-matched pages up to five, in shuffled order. The target
> is the gold verdict and evidence IDs. Its written confidence is always 1, so SFT cannot teach graded
> confidence. GRPO then samples eight completions per claim and scores each with Eq.~(2). It updates
> the policy towards completions above the group mean. Our main GRPO runs use native thinking: the
> model reasons in a `<think>` block before the JSON answer. The confidence term is off, the KL weight
> is zero, and an unparsable answer scores $-2$. To separate a GRPO effect from more data, both GRPO
> arms train on the same fixed 1,200 claims. One starts from the 5,895-row SFT adapter, whose
> training set overlaps those claims; the other starts from an SFT adapter trained on the disjoint
> 4,695 rows. A continued-SFT control trains the 5,895-row adapter on the same 1,200 claims.

---

## 4. §4.4 Controller — "Details"

Facts (sources: `controller-imitation.md`, `controller-grpo-results.md`, `prompted-controller-comparisons.md`, `agent-loop-notes.md`):

- **Actions** (`follow` preset): `search(query)`, `follow(link)`, `decompose`, `verify(claim | subclaim)`,
  `commit`, `abstain`. At most 8 actions per claim; BM25 search with 5 candidate pages and the top 100 sentences.
- **Observation**: search results as page titles plus link IDs; verify results as verdict,
  confidence and cited sentences; remaining budget. No uncited text, no gold.
- **Final-answer rule v3**: a weaker later verdict cannot override an earlier NEI; implemented in code.
- **Policies**: fixed loop (search once, verify, commit/abstain); prompted API controller
  (`qwen3.8-flash`, prompt v2, reasoning off, ≤200 output tokens); 2B imitation; 2B controller GRPO.
- **Imitation**: flash ran on 6,000 unused train claims (FEVER 0.693, $2.10). From its FEVER-correct
  episodes, 18,291 valid steps (988 validation) train a LoRA with the SFT recipe and the follow system
  prompt. 9,856 s on an RTX 5070 Ti. Held-out action agreement 0.79.
- **Controller GRPO**: 8 rollouts per unused train claim at temperature 1.0, greedy frozen verifier.
  Reward = FEVER (0/1) − 0.01·n_ops − 0.5·invalid replies. Advantage = reward − group mean, without
  std scaling, because most within-group differences are only action counts. Applied to every
  controller step's tokens, with observations masked. KL (k3) to the imitation policy, β = 0.04; one
  update per 8 claims; fresh LoRA on the merged imitation model. Main run (A): 1,000 claims, 125 updates,
  lr 5e-5. The same setup at lr 1e-5 took 6.7 h of training on A100-40.
- **Offline RL** (separate study): REINFORCE and fitted-Q over 6 cached search+verify actions, fitted on
  1,500 unused train claims, 3 seeds each.

Draft:

> The controller acts in a loop of at most eight actions. Its actions are search, follow a link from
> earlier results, decompose, verify the claim or a subclaim, commit and abstain. It sees page titles
> and link IDs from searches and the verifier's verdict, confidence and cited sentences, but no uncited
> text or gold annotations. A deterministic final-answer rule prevents a later, weaker verdict from
> overriding an earlier NEI. We compare four policies with the same frozen SFT verifier. The fixed loop
> searches once and verifies once. The API controller prompts `qwen3.8-flash`. The 2B imitation
> controller is a LoRA on Qwen3.5-2B trained by behavioural cloning on 18,291 steps from flash's
> FEVER-correct episodes on 6,000 unused training claims. Multi-turn GRPO starts from the imitation
> controller and plays eight rollouts per training claim. Each rollout is rewarded with its FEVER
> score minus 0.01 per operation and 0.5 per invalid reply. The advantage is reward minus the group
> mean, without standard-deviation scaling, and a KL penalty ($\beta=0.04$) keeps the policy close
> to the imitation policy.

---

## 5. §4.6 Decomposer — "adding for any missing details"

Facts (`decompose-results.md`, `agent-loop-notes.md` §4):

- Decomposer = `qwen/qwen3.8-flash`, temperature 0, two-example prompt, cached. It split 332 / 900 dev claims (37%) into 2+ parts; ~196 tokens in / 17 out.
- In fixed policies on those 332 claims, verifying sub-claims (`each`) loses 9 points of accuracy against verifying the whole claim (0.642 vs 0.735; 21 fixes / 52 breaks) at 2.1× the searches. Fixes (pooled context, claim context, soft combine) recover only 3 of the 9.
- Controllers rarely choose it: 21 of 18,291 imitation-training actions; 3 of 900 flash dev episodes; 0 of 900 imitation dev episodes.

Draft:

> The decomposer is `qwen3.8-flash` at temperature zero with a two-example prompt. It splits 37% of
> development claims. As a fixed policy, verifying each subclaim and combining the results is
> worse than verifying the whole claim: label accuracy on the 332 split claims falls from 0.735 to
> 0.642. Subclaims lose the entity needed for retrieval, the conjunctive rule compounds NEI, and
> some splits change meaning. Pooled retrieval, claim context and a soft probabilistic combination
> recover only a third of this loss. We therefore keep decomposition as an optional controller
> action, not a default step. Learned and API controllers use it on fewer than 1% of claims.

---

## 6. §5.1 Comparison design — "Anything missing??"

Add:

- The frozen verifier for every controller comparison is `sft-direct-from-base-qwen3p5-2b-5895-samples/run1`;
  controller and verifier decode greedily at evaluation.
- Paired significance uses exact McNemar tests on per-claim FEVER outcomes. Reports give
  "only A right / only B right" counts.
- The holdout (3,000 claims, 1,000 per label, `fever-evaluation-v1`) was opened once, for the
  finalized imitation and GRPO-A controllers only. Everything else is development-set evidence.
- Verifier GRPO comparisons hold the second-stage claims fixed and include a continued-SFT control
  on the same 1,200 claims, so data and RL effects can be separated.
- Multi-turn verifier GRPO is a separate comparison: it reads only what its own search returns.

---

## 7. §5.2 Data splits — SciFact / AVeriTeC

**Fact:** no SciFact or AVeriTeC verifier/controller results exist on any branch. AVeriTeC SFT
data (`averitec-2756`, mixes) and a 50-claim AVeriTeC *retrieval* pilot exist, but the controller
work dropped AVeriTeC. **Recommendation:** delete the TODO, replace the transfer promise with
the sentence below, and add a "FEVER only" point to §8.

Draft replacement and a split table:

> All results are on FEVER. Transfer to SciFact~\cite{wadden2020scifact} and
> AVeriTeC~\cite{schlichtkrull2023averitec} would need their own corpora, label mappings and
> evidence metrics, and is left to future work.

| Split | Claims | Use |
|---|---:|---|
| SFT train (`fever-oracle` 5,895) | 5,895 | Verifier SFT |
| Disjoint SFT (`fever-oracle-disjoint-4695`) | 4,695 | SFT start for the unseen GRPO arm |
| GRPO second stage (`fever-oracle-matched-grpo-1200`) | 1,200 | Both CoT GRPO arms and the continued-SFT control |
| Controller imitation (flash traces, unused train) | 6,000 claims → 18,291 steps | Controller BC |
| Controller GRPO (unused train) | 300 / 1,000 | Controller GRPO runs |
| Offline RL fitting (unused train) | 1,500 | REINFORCE / FQI / rules |
| Dev (`fever-dev-900-ids.json`) | 900 (300/label) | All development comparisons |
| Calibration (`fever-evaluation-v1`) | 900 | Reserved (once misused for cascade fitting, corrected) |
| Holdout (`fever-evaluation-v1`) | 3,000 (1,000/label) | Opened once for final controllers |

---

## 8. §5.3 Evaluation — "Anything missing??"

Add:

- ECE uses 10 equal-width bins over the confidence of the chosen verdict.
- **Token confidence** is the verifier's probability of the chosen verdict token, renormalised over the three labels.
  **Caveat to state in the paper:** for native-thinking (CoT) runs, the logged `label_probs` are not
  read after the reasoning block. In the 4695-start run they often disagree with the printed verdict
  (e.g. NEI printed, P(SUPPORTED) = 0.9998). Brier/ECE for CoT rows are therefore not valid until re-measured.
  This is why Table 1 shows "—" for them.
- Stated confidence after SFT is always 1.0 (training target), so stated ECE = 1 − accuracy;
  CoT GRPO runs emit no stated confidence.
- Invalid outputs count as wrong, with harm 1, in mean harm. They are not dropped.

---

## 9. §6 Results

### 9.1 Table 1 — verifier quality (TODOs 9–10)

FEVER dev 900; recomputed from committed `predictions.jsonl` with `scripts/score_all.py`.
Harm = Eq. (3) mean. R→S and N→S are error counts out of 300 each. Brier/ECE use token confidence.

**Panel A: title retrieval (the deployed pipeline), 1,024-token cap**

| Verifier | Acc | FEVER | Mean harm | R→S | N→S | NEI P / R | Brier | ECE |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| Base, direct | 0.476 | 0.346 | 0.807 | 165 | 229 | 0.53 / 0.14 | 0.331 | 0.311 |
| Base, long CoT prompt | 0.628 | 0.528 | 0.486 | 38 | 111 | 0.61 / 0.44 | 0.320 | 0.314 |
| SFT 5895 (`run2`), direct | 0.746 | 0.663 | 0.302 | 16 | 47 | 0.64 / 0.71 | 0.159 | 0.034 |
| + continued SFT on the 1,200 | 0.746 | 0.666 | 0.294 | 10 | 41 | 0.62 / 0.74 | 0.171 | 0.070 |
| + CoT GRPO on the 1,200 | 0.741 | 0.671 | 0.293 | 9 | 35 | 0.63 / 0.75 | — | — |
| SFT 4695 (disjoint) + CoT GRPO on the 1,200 | **0.762** | 0.672 | 0.289 | 15 | 52 | 0.69 / 0.67 | — | — |
| SFT 4500 (disjoint), direct | 0.767 | **0.676** | **0.280** | 16 | 45 | 0.68 / 0.71 | 0.161 | 0.050 |
| SFT 4500 + CoT GRPO on 1,000 | 0.747 | 0.673 | 0.293 | 7 | 43 | 0.64 / 0.68 | — | — |
| CoT SFT 5500 (teacher rationales) | 0.734 | 0.650 | 0.315 | 11 | 52 | 0.63 / 0.69 | — | — |
| Direct SFT 5500 (same claims) | 0.761 | 0.664 | 0.287 | 14 | 48 | 0.68 / 0.68 | 0.159 | 0.046 |
| *SFT 5895 `run1` (controller's verifier), cap 160* | *0.761* | *0.671* | *0.286* | *15* | *46* | *0.67 / 0.73* | *0.158* | *0.025* |

Invalid outputs: 10 (Base direct), 51 (Base long CoT), 14 / 7 (CoT GRPO from 5895 / 4695), 0 for all direct SFT rows.
The SFT 4695 adapter itself has no committed dev evaluation **[needs owner: Chao Chien Hung]**, so the unseen arm has no matched SFT row yet.

**Panel B: oracle context, historical direct-output GRPO, 160-token cap**

| Verifier | Acc | FEVER | Mean harm | R→S | N→S | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.469 | 0.370 | 0.795 | 157 | 212 | 0.328 | 0.298 |
| SFT 6000 | **0.779** | **0.702** | 0.268 | 18 | 44 | **0.146** | **0.028** |
| SFT on the same 1,200 claims (matched control) | 0.736 | 0.634 | 0.321 | 10 | 61 | 0.168 | 0.040 |
| GRPO from SFT, KL 0 | 0.769 | 0.697 | 0.279 | 16 | 47 | 0.213 | 0.214 |
| GRPO from SFT, KL 0.04, filtered, seed 0 | 0.771 | 0.694 | **0.264** | 13 | **33** | 0.211 | 0.209 |
| same, seed 1 | 0.746 | 0.694 | 0.301 | 17 | 44 | 0.239 | 0.237 |
| GRPO from SFT, KL 0.5, filtered | 0.769 | 0.691 | 0.279 | 17 | 46 | 0.188 | 0.164 |
| GRPO from base, KL 0.04 | 0.718 | 0.580 | 0.340 | 10 | 63 | 0.261 | 0.252 |

LaTeX for Table 1 (replaces the stub; uses `booktabs`, already loaded):

```latex
\begin{table}[htbp]
\centering
\caption{Verifier quality on 900 FEVER development claims. Mean harm uses Eq.~\ref{eq:harm}; Brier and ECE (10 bins) use verdict-token probability. ``--'': token probability not valid for native-thinking outputs. (a) Title retrieval, 1,024-token cap. (b) Oracle context, historical direct-output GRPO with the full reward.}
\label{tab:verifier}
\small
\begin{tabular}{lrrrrr}
\toprule
Verifier & Accuracy & FEVER & Mean harm & Brier & ECE\\
\midrule
\multicolumn{6}{l}{\emph{(a) Title retrieval}}\\
Base (direct) & 0.476 & 0.346 & 0.807 & 0.331 & 0.311\\
SFT 5895 & 0.746 & 0.663 & 0.302 & 0.159 & 0.034\\
\quad + continued SFT 1200 & 0.746 & 0.666 & 0.294 & 0.171 & 0.070\\
\quad + CoT GRPO 1200 & 0.741 & 0.671 & 0.293 & -- & --\\
SFT 4695 + CoT GRPO 1200 & 0.762 & 0.672 & 0.289 & -- & --\\
\midrule
\multicolumn{6}{l}{\emph{(b) Oracle context}}\\
Base & 0.469 & 0.370 & 0.795 & 0.328 & 0.298\\
SFT 6000 & 0.779 & 0.702 & 0.268 & 0.146 & 0.028\\
SFT 1200 (matched) & 0.736 & 0.634 & 0.321 & 0.168 & 0.040\\
SFT + GRPO (KL 0) & 0.769 & 0.697 & 0.279 & 0.213 & 0.214\\
SFT + GRPO (KL 0.04) & 0.771 & 0.694 & 0.264 & 0.211 & 0.209\\
\bottomrule
\end{tabular}
\end{table}
```

**Other matrices.** Confusion matrices (rows gold, columns predicted; S/R/N/invalid), title retrieval:

| Base direct | S | R | N | inv |  | SFT 5895 run2 | S | R | N | inv |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| **S** | 289 | 6 | 5 | 0 | | **S** | 244 | 8 | 48 | 0 |
| **R** | 165 | 97 | 32 | 6 | | **R** | 16 | 213 | 71 | 0 |
| **N** | 229 | 25 | 42 | 4 | | **N** | 47 | 39 | 214 | 0 |

| CoT GRPO from 5895 | S | R | N | inv |  | CoT GRPO from 4695 | S | R | N | inv |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| **S** | 227 | 5 | 60 | 8 | | **S** | 265 | 4 | 29 | 2 |
| **R** | 9 | 215 | 73 | 3 | | **R** | 15 | 221 | 62 | 2 |
| **N** | 35 | 37 | 225 | 3 | | **N** | 52 | 45 | 200 | 3 |

Draft text for §6.1:

> SFT is the decisive step. It raises accuracy by 27–30 points and FEVER by 32 points, and it removes
> the base model's bias towards Supported: Refuted→Supported errors fall from 165 to 16 and
> NEI→Supported from 229 to 47. Mean harm falls from 0.81 to 0.30. No GRPO run reliably improves on
> SFT. Native-thinking GRPO from the 5,895-row adapter matches its continued-SFT control (FEVER 0.671
> vs 0.666; accuracy 0.741 vs 0.746). GRPO from the disjoint 4,695-row adapter reaches the highest
> accuracy among CoT runs (0.762) but the same FEVER score (0.672). The two GRPO arms shift errors
> in different directions. From 5895, GRPO answers NEI more often (NEI recall 0.75, fewest
> NEI→Supported errors at 35). From 4695, it endorses more (Supported recall 0.88, NEI→Supported 52).
> Direct-output GRPO with the full reward makes verdict-token probability overconfident: ECE rises
> from 0.03 to 0.16–0.24 while accuracy is unchanged. Seed variation is large: the two KL-0.04 seeds
> differ by 2.5 accuracy points and 25 points of Refuted recall.

### 9.2 Table 2 — controller quality and resource use (TODOs 11–12)

Dev 900, `follow` preset, ≤8 actions, frozen SFT 5895 `run1` verifier. Recomputed from committed
traces with `scripts/score_traces.py`. Calls = search + follow + decompose + verify operations per claim.

| Controller | FEVER (%) | Acc (%) | Mean harm | Calls | Searches | Verifies | Tokens in (`cost.tokens_in`) | s / claim | Abstain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed loop | 66.6 | 77.1 | 0.275 | 2.00 | 1.00 | 1.00 | 1,408 | 0.42 | 35.1% |
| API controller (qwen3.8-flash) | 69.0 | **78.1** | **0.264** | 3.88 | 2.46 | 1.42 | 5,494 | 10.96† | 11.1% |
| 2B imitation | 68.4 | 76.4 | 0.286 | 3.70 | 2.29 | 1.41 | 5,293 | 10.87‡ | 16.1% |
| 2B imitation + GRPO (A, lr 5e-5) | **70.0** | 77.9 | 0.269 | 3.66 | 2.20 | 1.46 | 5,324 | 1.06§ | 10.0% |
| same, second seed (A2) | 69.1 | 76.6 | 0.286 | 3.46 | 2.03 | 1.43 | 5,114 | 1.12§ | 14.8% |
| *Holdout 3,000: imitation* | *67.3* | *76.3* | *0.279* | *3.62* | | | | | *16.3%* |
| *Holdout 3,000: GRPO A* | *67.8* | *76.5* | *0.279* | *3.52* | | | | | *12.1%* |

† includes OpenRouter latency; flash cost $0.0002–0.0004 per claim. ‡ RTX 5070 Ti, local. § A100-40 on SoC.
**Time is not comparable across rows because the hardware differs.** Report calls and tokens as the cost
axis, or rerun timing on one machine **[needs owner]**.

Paired tests (exact McNemar on FEVER): flash vs fixed 67/45, p = 0.047; imitation vs flash 36/41, p = 0.65;
imitation vs fixed 51/34, p = 0.08; GRPO A vs imitation 30/16, p = 0.054 (dev) and 65/49, p = 0.16 (holdout);
A2 vs imitation 21/15, p = 0.41.

LaTeX:

```latex
\begin{table}[htbp]
\centering
\caption{Controller policies on 900 FEVER development claims with the same frozen SFT verifier, at most eight actions. Calls count search, follow, decomposition and verification operations per claim. Times were measured on different hardware and are indicative only. Bottom rows: the 3,000-claim holdout, evaluated once.}
\label{tab:controller}
\small
\begin{tabular}{lrrrr}
\toprule
Controller & FEVER (\%) & Accuracy (\%) & Calls & Time (s)\\
\midrule
Fixed loop & 66.6 & 77.1 & 2.00 & 0.4\\
API controller (qwen3.8-flash) & 69.0 & 78.1 & 3.88 & 11.0\\
2B imitation & 68.4 & 76.4 & 3.70 & 10.9\\
2B imitation + GRPO & 70.0 & 77.9 & 3.66 & 1.1\\
\midrule
Holdout: 2B imitation & 67.3 & 76.3 & 3.62 & --\\
Holdout: 2B imitation + GRPO & 67.8 & 76.5 & 3.52 & --\\
\bottomrule
\end{tabular}
\end{table}
```

**Other matrices for §6.2**

*Tool presets* (flash controller; `prompted-controller-comparisons.md`): fixed 0.666; nosearch 0.661;
full (search) 0.688; **follow 0.690**; pages (open_page, find_pages) 0.681. Opening pages raises
gold reached from 556 to 567 but also raises gold-reached-yet-wrong from 148 to 163.

*Verifier swap* (fixed → prompted flash): SFT 5895 0.666 → 0.688 (+2.2); SFT 5500 0.666 → 0.680 (+1.4);
base 2B 0.369 → 0.407 (+3.8). The verifier sets the level: swapping base for SFT is worth 28 points,
while a controller change is worth at most 2.4.

*Offline RL over cached actions* (`controller-offline-rl.md`; reward FEVER − λ·calls; 3 seeds): this
is the closest thing to a **budget sweep**.

| λ | Fixed | Best hand rule | REINFORCE | FQI | Oracle |
|---|---:|---:|---:|---:|---:|
| 0 | 0.666 (1.00) | **0.708** (2.65) | 0.696 ± 0.011 (2.80) | 0.691 ± 0.006 (2.95) | 0.819 |
| 0.01 | 0.666 | **0.711** (2.30) | 0.698 (2.39) | 0.689 (2.37) | 0.819 |
| 0.02 | 0.666 | 0.684 (1.69) | 0.684 (2.01) | 0.684 (2.04) | 0.819 |
| 0.05 | 0.666 | 0.666 | 0.666 | 0.669 | 0.819 |

*Cascade routing* 2B → flash (`cascade-payoff.md`): routing all 2B SUPPORTED/REFUTED to flash gives
0.671 → 0.707 FEVER at 64% routed. A learned router reaches 0.702; the oracle router 0.779.

*RSI-Jev STOP/RETRY chunk controller* (`feat/controller`, Chao Chien Hung): the 0.5-threshold
heuristic scores 65.3% FEVER. Zero-shot Jev scores 54.0%. The trained heads score 62.9–64.6%, so
none beats the heuristic.

**Not done:** a matched maximum-budget sweep for the prompted/learned controllers (e.g. 2/4/8 actions)
**[needs owner, or state as a limitation]**.

### 9.3 §6.3 Ablations and training diagnostics (TODO 13)

Ablation summary (dev 900):

| Line | Condition | Reward switches | FEVER | Mean harm | Brier (tok) | Cost |
|---|---|---|---:|---:|---:|---|
| Verifier, direct, oracle | SFT 6000 | — | 0.702 | 0.268 | 0.146 | — |
| | GRPO KL 0 | harm + ev + Brier | 0.697 | 0.279 | 0.213 | 3.0 h |
| | GRPO KL 0.04, spread filter | harm + ev + Brier | 0.694 | 0.264 | 0.211 | 1.0 h |
| | GRPO KL 0.5, spread filter | harm + ev + Brier | 0.691 | 0.279 | 0.188 | 0.8 h |
| | GRPO from base, KL 0.04 | harm + ev + Brier | 0.580 | 0.340 | 0.261 | 3.6 h |
| Verifier, CoT, titles | from SFT 5895 | harm + ev, no Brier, −2 invalid | 0.671 | 0.293 | — | 6.1 h |
| | from SFT 4695 | same | 0.672 | 0.289 | — | 9.7 h |
| Multi-turn verifier, from base | no gate | harm + ev, 0.02/search | 0.034 | — | — | 33 min |
| | min 1 search | same | 0.204 | — | — | — |
| | min 1 search + evidence gate | gated correctness | **0.540** | — | — | ~1.5 h |
| Controller GRPO | lr 1e-5 | FEVER − 0.01·ops − 0.5·invalid | 0.687 | 0.288 | 0.157 | 6.7 h |
| | lr 5e-5 (A) | same | **0.700** | 0.269 | 0.151 | — |
| | lr 5e-5, seed 1 (A2) | same | 0.691 | 0.286 | 0.152 | — |
| | lr 1e-4 (C) | same | 0.687 | 0.284 | 0.153 | — |
| | lr 5e-5 + 0.3 evidence bonus (D) | + gold-delivered bonus | 0.682 | 0.291 | 0.158 | — |
| | spread claims only, 2 epochs (B) | same | 0.693 | 0.279 | 0.154 | — |

The multi-turn harm column is in `multiturn-grpo-results.md` as "harm-weighted error": 0.659 for
untrained base, 0.578 / 0.667 / 0.382 for the three runs. That doc defines it slightly differently;
check before mixing it with Eq. 3.

Not run: harm matrix off (`no_harm`), calibration on vs off at a matched start, and evidence weight
sweeps. List these as limitations unless someone runs them.

**Learning curves by reward component** (CoT GRPO, `team/rl` `train_log.json`; means over 120-step windows):

| Start | Steps | Reward | Correctness | Evidence | Valid final | Mean length | Zero-std groups |
|---|---|---:|---:|---:|---:|---:|---:|
| SFT 5895 | 1–120 | 0.649 | 0.667 | 0.475 | 0.872 | 351 | 46% |
| | 241–360 | 1.045 | 0.790 | 0.518 | 0.998 | 89 | 73% |
| | 481–600 | 1.071 | 0.817 | 0.512 | 0.999 | 104 | 71% |
| SFT 4695 | 1–120 | 0.502 | 0.647 | 0.432 | 0.819 | 462 | 50% |
| | 241–360 | 1.027 | 0.781 | 0.509 | 0.996 | 303 | 65% |
| | 481–600 | 1.051 | 0.812 | 0.484 | 0.998 | 252 | 66% |

Two readings. Most of the reward gain is learning to finish with a valid answer (valid final
0.82–0.87 → ~1.0). Correctness gains on oracle training context (+0.15 to +0.17) do not transfer to the
title-retrieval dev set. From step ~120, two-thirds of groups have zero reward spread and give no gradient.
Thinking shrinks from 351 to ~100 tokens from 5895, but stays at ~250 from 4695.

Multi-turn curves: figure `verifier/docs/figures/multiturn-grpo-training.png` on `mt-grpo-on-handover`.
Controller GRPO curves: per-20-step tables in `controller-grpo-results.md`; KL stays ≤ 0.002 at lr 1e-5.
Plot them from `runs/<run>/artifacts/logs/train_log.json`.

### 9.4 §6.4 Evidence bottlenecks (TODOs 14–15)

**Stage-wise evidence funnel**, 600 SUPPORTED/REFUTED dev claims (new, `scripts/funnel.py`). A complete
gold set means every sentence of at least one annotated set:

| Policy | Gold pages found | Gold set retrieved | Gold set in union of verifier inputs | Gold set in **one** verifier input | Label correct | FEVER |
|---|---:|---:|---:|---:|---:|---:|
| Fixed loop | 524 (87.3%) | 524 (87.3%) | 524 (87.3%) | 524 (87.3%) | 476 (79.3%) | 381 (63.5%) |
| Flash controller | 556 (92.7%) | 556 (92.7%) | 538 (89.7%) | 537 (89.5%) | 490 (81.7%) | 408 (68.0%) |
| 2B imitation | 546 (91.0%) | 546 (91.0%) | 541 (90.2%) | 540 (90.0%) | 482 (80.3%) | 410 (68.3%) |
| 2B imitation + GRPO A | 550 (91.7%) | 550 (91.7%) | 548 (91.3%) | 547 (91.2%) | 487 (81.2%) | 416 (69.3%) |

Error split (same 600 claims):

| Policy | Correct label, incomplete citations | Wrong label despite complete gold in one input |
|---|---:|---:|
| Fixed loop | 95 | 87 |
| Flash | 82 | 84 |
| 2B imitation | 72 | 85 |
| GRPO A | 71 | 86 |

All 900 claims, imitation controller's 284 FEVER misses (`controller-grpo-results.md`):
130 had gold in the verifier input but a wrong verdict or citation (verifier), 94 gold-NEI claims
answered S/R (mostly verifier), 54 gold never retrieved (controller + retrieval), 6 retrieved but never
sent to the verifier (controller). At most 60 misses (21%) depend on the controller.

Retrieval-only context (`retrieval-results.md`, `agent:retrieval-extension-results.md`): the 5-page
candidate-pool ceiling is 0.873. Sentence recall@10 is 0.795 for BM25 and 0.848 for hybrid. Hybrid
+ link-follow reaches 0.865 @10 (+1.67 pp, 95% CI [+0.33, +3.00]).

Draft:

> The controller's evidence search is not the bottleneck. With the learned controller, a complete
> gold evidence set reaches a single verifier input for 90–91% of Supported/Refuted claims, against
> 87% for the fixed loop. Evidence that appears only across separate calls is rare: at most one claim per
> policy. The losses come after delivery. About 85 claims per policy get the wrong label even though
> the full gold set was in the input, and another 71–95 get the right label with incomplete citations,
> which strict FEVER scores as a failure. Across all 900 claims, at most 60 of the imitation
> controller's 284 FEVER misses (21%) depend on the controller's choices.

**Representative traces** (`scripts/cases.py`; dev IDs, imitation or flash controller vs fixed loop):

1. **Recovery, id 139384** (gold NEI) "Recovery features a singer." Fixed loop: SUPPORTED, citing
   *Recovery (James Arthur song)*. Imitation: three searches, two follows, verify → NEI (0.66) →
   commit NEI. FEVER 0 → 1.
2. **Regression, id 217187** (gold S) "A monk lives monastically." Fixed loop cites `Monk:0` (correct).
   Imitation searches "monk monastic life", gets *Monasticism*, cites `Monasticism:7`: right label, wrong
   evidence, FEVER 1 → 0. A different query changed the evidence, not the verdict.
3. **Failed retry, verifier error, id 19983** (gold R) "Sidse Babett Knudsen is a conductor." Gold
   `Sidse_Babett_Knudsen:0` ("Danish actress") is in context, yet the verifier says NEI (0.86) three
   times across two searches and a follow.
4. **Final-answer rule overrides a correct answer, id 29993** (gold R) "The Mirny (sloop-of-war) was
   a ship without allegiance." First verify: NEI (0.63). After a follow, the second verify returns
   REFUTED (0.48) citing the gold sentence. Rule v3 keeps the earlier NEI, so FEVER is 0. This is a
   concrete cost of the NEI-preserving rule.
5. **Decomposition, id 173502** (gold NEI, flash controller) "Sancho Panza is a fictional character in
   a novel written by a Spanish writer born in Prague." Decompose → three subclaims. Two are
   SUPPORTED; the birthplace subclaim is REFUTED (0.48) from *Miguel de Cervantes*, and the final answer is REFUTED.
   The fixed loop said NEI (FEVER 1). The REFUTED verdict is arguably defensible, so this is also an
   annotation-ambiguity case.
6. **Decomposition stops at budget, id 161809** (gold NEI): "…operated by an American gaming corporation
   and it is a rich company." First part SUPPORTED after a follow; "rich company" stays NEI; budget
   exhausted; final NEI (correct).

For each case, record claim ID, gold label, exact verifier input, action sequence, citations and
final output from the trace files listed in the scripts. The exact verifier inputs are the `verify`
steps' `args.evidence` in `verifier/data/controller/dev900-follow-controller-traces/*.jsonl.gz`.

Figure suggestion: a bar or Sankey of the funnel table (pages → retrieved → in input → label → FEVER) for fixed vs imitation+GRPO.

### 9.5 §6.5 (empty heading, TODO 16)

Suggested title: **"Multi-turn GRPO: learning to search and reward shortcuts"**
(`mt-grpo-on-handover:verifier/docs/multiturn-grpo-results.md`, Michel).

| Model (dev 900) | Acc | FEVER | Ev F1 | Recall S / R / NEI | Searches |
|---|---:|---:|---:|---|---:|
| Base, untrained, searching itself | 0.448 | 0.382 | 0.385 | 0.71 / 0.35 / 0.29 | 1.68 |
| Multi-turn GRPO, ungated | 0.438 | 0.034 | 0.000 | 0.22 / 0.99 / 0.10 | 0.00 |
| + one search required | 0.333 | 0.204 | 0.640 | 0.00 / 1.00 / 0.00 | 1.00 |
| **+ evidence-gated correctness** | **0.676** | **0.540** | 0.623 | 0.78 / 0.84 / 0.41 | 1.00 |
| Single-turn GRPO, oracle train, ungated | 0.594 | 0.251 | 0.369 | 0.83 / 0.95 / 0.00 | — |
| Single-turn GRPO, oracle train, gated | 0.443 | 0.419 | 0.131 | 0.34 / 0.02 / 0.96 | — |

Draft:

> When the verifier must search for its own evidence, GRPO from the base model first finds two
> shortcuts that Eq.~(3) allows. Without constraints it stops searching and answers Refuted, the
> cheapest wrong verdict. Required to search, it still answers Refuted on every claim. Making
> correctness credit for Supported/Refuted conditional on citing a gold sentence removes the shortcut.
> The model then keeps all three labels and improves on the untrained base by 22.8 accuracy and 15.8
> FEVER points with one five-sentence search. The same gate applied to single-turn training on oracle
> context causes the opposite collapse, to NEI: it answers NEI on 96% of NEI claims but also on most
> others (Refuted recall 2%).
> One seed; SFT4500 had lost tool use (1/120 rollouts searched), so both arms start from base.

Holdout and second-seed jobs (1058798–1058801) were queued on 2026-11-02, but their results are not
on any branch **[needs owner: Michel]**.

---

## 10. §7 Discussion

### 10.1 §7.1 — "Check anything else to analysis" (TODO 17)

Add these points; each is supported by results above:

- **The harm matrix makes REFUTED the safe wrong answer.** On a gold-NEI claim a wrong Refuted earns
  $r_v = 0$ while a wrong Supported earns −0.75. Multi-turn GRPO exploited exactly this by collapsing
  to Refuted (§6.5). An asymmetric matrix needs either evidence gating or symmetric Refuted harm.
- **Group-relative advantages vanish as the policy sharpens.** In the unfiltered direct GRPO run, about
  88% of groups had zero reward spread. In CoT GRPO, 61–73% did after step 120. KL, spread
  filtering and reward-std removal are all responses to this; none produced a reliable dev gain.
- **The reward mostly taught format, not judgement.** The CoT GRPO reward rose from 0.5–0.65 to ~1.05,
  largely through valid-final-answer rate (0.82–0.87 → 1.0). Dev accuracy matched the SFT control.
- **RL on verdicts sharpens token confidence.** Direct GRPO raised token ECE from 0.03 to 0.16–0.24 with
  the Brier term on, because the Brier term scores *written* confidence (always 1.0 after SFT), not the token
  probability. The confidence channel the reward sees is not the one evaluated.
- **Evidence gating has a context-dependent effect.** It fixed multi-turn training, but under oracle-trained
  single-turn GRPO it taught abstention whenever retrieved context looked unfamiliar.
- **Controller reward gives little outcome signal.** Only 17% of controller training claims had rollouts that
  disagreed on FEVER. An evidence-delivery bonus added signal on just 51 of 1,000 claims, all of
  which the verifier then misjudged.

### 10.2 §7.2 — case study (TODO 18)

Draft:

> Extra actions help only when they add information the verifier can use. Following links raises the
> share of Supported/Refuted claims whose gold set reaches the verifier from 87% to 90–91%, and FEVER
> by 1.8–3.4 points. Opening whole pages raises gold reached further (556 → 567) without more correct
> answers, because the extra context also produces 15 more wrong verdicts. Decomposition can simplify a
> claim but also destroy the relation retrieval needs. "The band that released The Endless River formed
> in London" loses the album that identifies Pink Floyd. "Cars 3 isn't the third Cars movie Brian Fee
> has worked on" lost its negation. Across 332 split claims, per-part verification fixes 21 claims and
> breaks 52. A second verification is not always better either. In claim 29993 the retried verify
> found the correct Refuted with the gold citation, but the NEI-preserving final rule discarded it.
> The oracle over six cached actions reaches 0.819 FEVER, against 0.71 for every learned policy. The
> features available to the controller (mostly the verifier's own outputs) do not reveal which claims
> another action would fix.

### 10.3 §7.3 (empty heading, TODO 19)

Suggested title: **"Where the remaining errors are"**. Draft:

> Errors are dominated by the verifier, not the controller. Swapping the base verifier for SFT changes
> FEVER by 28 points in the controller loop; the best controller change is worth 2.4. With gold evidence
> delivered on 90% of claims, a perfect controller could add at most 6.7 points, and the oracle-gap
> analyses show the 2B verifier's outputs do not identify which claims need more work. NEI is the
> weakest label for every policy (recall 0.68–0.73). Further gains therefore need a stronger judge on
> delivered evidence, better insufficiency detection, or retrieval for the 8–9% of claims whose gold
> evidence is never found.

---

## 11. §8 Limitations — corrections

- Replace "Three RL seeds measure optimization variation" with: "Verifier GRPO and controller GRPO
  have one or two seeds per setting; only the offline-RL study uses three. Single-seed differences
  under about three points are not evidence."
- Add: "All results are on FEVER; no transfer set was evaluated."
- Add: "Token confidence for native-thinking outputs was not measured after reasoning, so calibration is
  reported only for direct outputs."
- Add: "Latency was measured on different hardware (RTX 5070 Ti, A100, API)."
- §4.2: state that controller GRPO uses $\lambda_c = 0.01$ and a 0.5 penalty per invalid controller reply,
  and that every GRPO run scores an unparsable answer −2. Alternatively, add a format term to Eq. (2).

---

## 12. §9 Conclusion (TODO 20)

Draft:

> Supervised fine-tuning, not reinforcement learning, produced the large verifier gains: a 2B model went
> from 34% to 67% strict FEVER score. Its harmful endorsements fell sharply: Refuted→Supported errors
> from 165 to 16 and NEI→Supported from 229 to 47. Rule-based
> GRPO with harm-weighted verdicts and gated evidence credit did not improve on matched SFT. It mostly
> taught output format, lost signal as groups converged, and made token confidence less calibrated when
> trained against written confidence. The reward's structure mattered more than its weights: an asymmetric
> harm matrix invited a Refuted shortcut, which evidence gating removed when the model searched for its own
> evidence. For sequential control, a 2B controller learned by imitation matched an API controller.
> Multi-turn GRPO added about one FEVER point (not significant on a 3,000-claim holdout). The controller
> already delivers gold evidence for 90% of verifiable claims, so the remaining errors are mostly the frozen
> verifier's. Learned control will pay off only alongside a stronger judge or actions that add information
> the judge lacks.

---

## 13. Team contributions (TODO 21)

From the 8 October team brief (`verifier/docs/plan.md`) and commit history across all branches. **Each
person should confirm their own line.**

| Member | Role | Evidence in the repository |
|---|---|---|
| Chao Chien Hung | RL-1 | Verifier SFT, direct and CoT GRPO, disjoint split, CoT SFT data, DPO scripts, RSI-Jev controller and evaluator (most commits in `verifier/experiments`, `evaluation`, `utils`, `agent`) |
| Michel Yildirim | RL-2 | Controller GRPO (incl. holdout), multi-turn verifier GRPO, evidence-gated reward |
| Yao Tianhao | RL-3 | **[needs owner]**: no commits under this name on any branch; real-data controller per the brief |
| Yu Jing | AG-1 | Retrieval: title matching, hybrid ranking, link-follow and Exa extension |
| Chia Kwang Yang | AG-2 | Agent loop, prompted controller, imitation controller, offline RL, cascade study, decomposer, demo backend, report evidence audit |
| Bryan Wu Jiahe | AG-3 | Demo UI (all `demo/` commits), evaluation/data-split support |

## 14. AI assistance disclosure (TODO 22) — [needs team]

Template; each member should add their own use:

> We used AI coding assistants (e.g. Claude Code) to write and debug experiment code, run analyses over
> committed predictions and traces, and draft parts of this report. All reported numbers were produced by
> code in the project repository from saved model outputs, and the authors checked the text and results.
> [Add: any other tools; which sections were drafted with assistance.]

---

## Reproducing the recomputed numbers

From the repository root, with all branches fetched:

```bash
python -I report/scripts/score_all.py . team-rl controller/grpo-handover      # Table 1, confusion data
python -I report/scripts/score_traces.py . controller/grpo-handover \
  verifier/data/traces/fever_dev_fixed-sft-direct-from-base-qwen3p5-2b-5895-samples_run1_bm25-p5-k100_n900.jsonl \
  verifier/data/controller/dev900-follow-controller-traces/*.jsonl.gz       # Table 2
python -I report/scripts/funnel.py . controller/grpo-handover <same traces>  # §6.4 funnel
python -I report/scripts/cases.py . controller/grpo-handover <fixed trace> <controller trace>
```

The scripts read files with `git show <branch>:<path>`, so they need no checkout of the other branches.
Use `team-rl` as the local name for `team/rl`, or pass `team/rl` directly.
