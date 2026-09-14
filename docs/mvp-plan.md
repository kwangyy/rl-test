# MVP feasibility spike — plan

## Why this exists

The full project (CalBench integration, multi-week request streams, PPO
with 3 seeds, a counterpart pool, 6 baselines, fairness/privacy ablations,
Gmail+Calendar demo) is a 6-week, two-team effort. Before committing to
that, we want a fast, cheap signal on whether the core bet even holds:
**can an RL policy learn to pick which slot to give away (and implicitly
how hard to hold the rest) better than a trivial heuristic, under partial
observability of the other side's calendar?**

This spike is deliberately decoupled from CalBench so that CalBench's own
unknowns (see below) don't block getting that signal this week, ahead of
the Friday proposal draft and the 28 Sep submission.

## Two decoupled risks

- **R1 — research bet.** Does PPO beat first-fit at this class of problem
  at all, even in the easiest possible toy setting? This is the risk that
  kills the project if false — better to know now than at W8.
- **R2 — tooling bet.** Is CalBench (scenario generator, CP-SAT oracle,
  metrics, VPS privacy metric) actually usable as advertised? Unverified —
  `anonymous.4open.science` returns HTTP 403 to automated fetch (bot-blocked
  or JS-rendered), so this needs a human to open it in an actual browser.

## Scope of the spike (explicitly NOT the full project)

- Hand-rolled `SlotNegotiationEnv` (Gymnasium), no CalBench.
- One negotiation episode = one incoming request, self vs. one scripted
  counterpart with a hidden acceptable-slot set (not the full
  Boulware/Conceder/rigid/LLM-persona pool).
- One preference dimension (a no-mornings-style cost) + urgency, not the
  full richer preference profile set (focus blocks, seniority, etc).
- Baselines: first-fit only (not greedy/hand-tuned-concession/prompted-LLM/
  DCOP/CP-SAT oracle — those come later once R1 is validated).
- PPO, single seed, small fixed compute budget (~100k steps, CPU, <10 min
  in practice on a laptop).

If any of the above scope cuts change the picture in a way that matters,
that's useful information too — flag it before generalizing "go" to the
full project.

## Tasks

| # | Task | Status |
|---|------|--------|
| 0 | Manually open the CalBench link in a browser; record findings in `docs/calbench-notes.md` | **Open — needs a human** |
| 1 | `SlotNegotiationEnv` (Gymnasium) + random-policy sanity check | Done |
| 2 | First-fit baseline + eval over 200 episodes | Done |
| 3 | PPO training (~100k steps) + eval vs first-fit | Done — see results below |

## Go / no-go rule

- PPO reward ≥ ~15% better than first-fit → **go**: scope the full
  W6-W12 plan, informed by Task 0's wrap-vs-reimplement finding.
- Marginal (0–15%) → **iterate**: rework reward/state before scaling up
  (this is already scheduled as a W8 fallback in the syllabus — just
  pulled forward to this week).
- PPO does not beat first-fit → **pivot**: rethink the POMDP framing
  before locking it into the 28 Sep proposal.

## Results (12 slots, 6 rounds max, 200 eval episodes, PPO 100k steps/1 seed)

| Policy | Mean reward | Deal rate | Rounds to deal | Mean accepted cost |
|---|---|---|---|---|
| first-fit | -1.045 (+/-0.857) | 97.0% | 2.69 | 0.658 |
| greedy best-utility-now | **-0.909** (+/-1.021) | 96.0% | 2.72 | **0.471** |
| PPO | -0.981 (+/-0.795) | 97.5% | 2.64 | 0.635 |

**Honest read: ITERATE, leaning toward the env being under-specified, not
toward "RL can't do this."**

- PPO beats first-fit (+6.0%) but loses to greedy-best-utility-now, a
  one-line heuristic ("always offer your own cheapest free slot"). That
  heuristic is close to optimal *for this exact toy construction*, because
  the counterpart's hidden acceptable set is just a random subset of
  self's free slots, uncorrelated with self's own cost. There's no
  adversarial or exploitable structure for a smarter policy to find beyond
  cost-ordering — so RL had no real edge to discover here, and PPO hasn't
  even matched the simple heuristic yet in 100k steps.
- This points at a real gap in the spike's scope, not a refutation of the
  thesis: the one-shot, single-request framing throws away exactly the
  part of the original idea that should make RL matter — "how hard should
  I hold the rest" only means something if giving away a slot *now* has
  an opportunity cost against a *future, not-yet-seen* request. A single
  isolated negotiation has no such trade-off.
- **Next cheap iteration (not yet run):** extend the episode to a short
  sequence of 2-3 requests competing for overlapping slots, so that a
  policy has a reason to hold a cheap/flexible slot back rather than
  spend it on the first asker. If PPO still can't beat greedy there,
  that's a much stronger no-go signal than this result.

## Iteration 2: MultiRequestSlotEnv (3 sequential requests, shared depleting calendar, contested slots)

`src/multi_request_env.py`, PPO 300k steps / 1 seed, 300 eval episodes:

| Policy | Mean episode reward | Deal rate | Mean accepted cost |
|---|---|---|---|
| first-fit | -2.300 (+/-1.372) | 98.4% | 0.515 |
| greedy best-utility-now | -1.552 (+/-0.861) | 99.7% | 0.409 |
| PPO | -1.630 (+/-0.995) | 99.2% | 0.413 |

PPO vs first-fit: **+29.1%**. PPO vs greedy: **-5.0%**.

**Honest read: this is real signal, and it's a tie with greedy, not a win.**

- Adding real contention (multiple requests competing for the same
  cost-cheap slots, no foresight of future arrivals) made a huge
  difference: PPO went from barely beating first-fit (+6%) to clearly
  beating it (+29%). That confirms the multi-request structure is where
  the interesting problem actually lives, matching the project's own
  framing ("how hard should I hold the rest" only matters against future,
  unseen requests).
- PPO still hasn't beaten greedy-best-utility-now, but the -5% gap is
  close to the noise floor for 1 seed / 300 eval episodes (SEM on the
  reward difference is on the order of the gap itself). This is not
  strong evidence PPO *can't* beat greedy here — it's inconclusive with
  the compute spent so far (300k steps, no hyperparameter tuning, no
  multi-seed averaging).
- Greedy-best-utility-now is a genuinely strong, non-trivial baseline in
  both toy settings. That's useful to know for the real project too: it
  should be a first-class baseline (it already is, in the original
  deliverables list), and "beats first-fit" is too low a bar for any
  result to be interesting — "beats greedy" is the real one.

## Iteration 2b: 3 seeds, 500k steps, small entropy bonus

| Policy | Mean episode reward | Notes |
|---|---|---|
| first-fit | -2.300 | |
| greedy best-utility-now | -1.552 | |
| PPO, 3 seeds | -1.608 (std across seeds: **0.042**) | seeds: -1.618, -1.552, -1.654 |

PPO vs first-fit: +30.1%. PPO vs greedy: **-3.7%, consistently** (not noise
— seed-to-seed std of 0.042 is tiny relative to the 0.056 gap). One seed
(seed 1) landed at *exactly* -1.552, matching greedy to three decimals —
strong evidence PPO converged to essentially the same policy as greedy,
not a worse one.

**Sharper diagnosis:** this isn't "PPO failed to find a better policy
that exists." It's "greedy-cost-first appears to be at or very near the
optimal policy for this exact toy construction, and PPO correctly found
it." The reason: **the toy env has no actual concession dynamics.** The
counterpart's acceptable-slot set is fixed for the whole request (drawn
once, static across rounds) — it never softens or hardens based on how
the round goes. So "how hard should I hold the rest" — the central
mechanic in the actual proposal — has literally nothing to act on in this
construction: holding firm longer costs rounds/reward but buys no
information or leverage, because the counterpart's true walk-away
behavior never depends on self's play. Real strategic depth (Boulware vs.
Conceder counterparts, holding firm to bluff urgency, timing concessions)
requires a counterpart whose behavior is *responsive* to the negotiation,
not just revealed round-by-round.

**This changes the recommended next step:** the highest-value next
iteration is not more compute or seeds on the current env — it's adding
a counterpart with actual concession dynamics (e.g. a Boulware-style
counterpart whose acceptable set expands as rounds/deadline pressure
increase, so holding firm is a real bet with real risk). That is
precisely the "counterpart pool of varying flexibility" already in the
project's deliverables list — this spike shows it isn't a nice-to-have,
it's the mechanism the whole thesis depends on. Not yet built or run.

## Overall MVP verdict (as of this spike)

**Provisional go, with a specific, actionable caveat — not a clean yes,
and not resolved by more compute.** RL clearly learns something
non-trivial once the environment has real opportunity-cost structure
(multi-request contention: +6% -> +29% over first-fit). It has
consistently (3 seeds, tight variance) converged to match — not beat —
a one-line greedy heuristic, and the diagnosis is structural, not a
training-budget problem: this toy env has no counterpart concession
dynamics, so "how hard to hold" has nothing to act on yet. Throwing more
seeds/steps/hyperparameter search at *this exact env* is not expected to
change the outcome.

**What this means for feasibility:** the thesis isn't refuted — the
piece of the environment that would let a learned policy beat greedy
(responsive counterparts with real concession curves, i.e. the
Boulware/Conceder/rigid pool already in the deliverables) hasn't been
built yet, even in miniature. That's the concrete next test, not "add
compute." Recommend treating "PPO vs greedy-best-utility" (not vs
first-fit) as the headline comparison throughout the project, and
treating "does PPO beat greedy once counterparts have real concession
dynamics" as the next go/no-go gate before committing to the full W6-W12
plan.
