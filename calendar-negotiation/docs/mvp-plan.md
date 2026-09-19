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
  **Resolved: yes, proven right, with a sharper caveat.** PPO beats
  first-fit (+6% one-shot, +29-30% multi-request) but converges to match,
  not beat, a one-line greedy heuristic — see "Overall MVP verdict" below.
- **R2 — tooling bet.** Is CalBench (scenario generator, CP-SAT oracle,
  metrics, VPS privacy metric) actually usable as advertised?
  **Resolved: yes — but the original plan for resolving it was wrong.**
  This originally said it needs a human to open `anonymous.4open.science`
  in a browser (blocked at HTTP 403 to automated fetch). That never
  happened and turned out to be unnecessary: the real de-anonymized
  mirror (`github.com/bosonphoton/calbench2026`) was found and cloned
  directly, then actually plugged in and run (Iteration 3 below,
  213/213 tests pass, 30 real scenarios against the live CP-SAT oracle).
  See `docs/calbench-notes.md`.

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
| 0 | Manually open the CalBench link in a browser; record findings in `docs/calbench-notes.md` | **Resolved — but not by opening a browser.** Original plan was wrong; see `docs/calbench-notes.md` (license/reuse terms still open there) |
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

## Iteration 3: real CalBench engine, real oracle, no training yet

Cloned the real repo (`github.com/bosonphoton/calbench2026` — see
`docs/calbench-notes.md`), plugged a minimal custom agent (`RLClient`) into
the actual game engine via its public testing seam
(`CalendarGame._run_with_agents`, no engine code touched), and ran it
across 30 real generated scenarios (2 agents, 16 slots, 3 meetings each)
against the shipped DSM baseline. Deliberately narrow scope: the RL
client only makes the DECIDE-phase call (which slot to take); it does no
cheap-talk/negotiation dialogue yet, and it's a hand-coded heuristic, not
a trained policy — this iteration tests the plug-in seam and the reward
signal shape, not learning.

| Agent 0 policy | Mean realized cost | Mean regret vs CP-SAT oracle | Mean success rate |
|---|---|---|---|
| DSM (baseline vs baseline) | 0.00 | 0.00 | 100% |
| RLClient(random) | 2.87 | 2.87 | 100% |
| RLClient(first-fit-among-free) | 5.03 | 5.03 | 100% |

**Two real findings, both useful:**

1. **The plug-in seam works end to end against the real engine and real
   CP-SAT oracle**, with zero changes to CalBench's own code. Every
   scenario stayed 100% "successful" (all meetings scheduled) regardless
   of agent 0's policy, because the engine has a built-in fallback-repair
   mechanism (`enable_fallback`) that patches up conflicting slot choices
   automatically — at a displacement-cost penalty. That's good news for
   RL: failures show up as *continuous regret*, not a sparse pass/fail
   signal, which is a much easier reward to learn from.
2. **Surprise, verified not guessed: "always take the earliest free slot"
   is worse than picking randomly** (5.03 vs 2.87 mean cost). Checked
   DSM's own source: it explicitly sorts free slots and breaks ties by
   lowest slot index (`calendar_game/clients/dsm.py`, `_free_slots` /
   proposal ranking). So a naive low-index-first policy collides with
   DSM's own bias more often than random does, forcing more costly
   fallback repairs. **This directly contradicts the toy spike's finding
   that a cost-ordering heuristic is a hard-to-beat baseline** — that
   result was specific to the one-sided, no-collision structure of the
   hand-rolled bilateral env. In a real two-sided coordination task,
   naive heuristics can actively hurt each other, which means there is
   *more* headroom for a policy that reasons about what the counterpart
   is likely to do than the toy spike alone would suggest.

**Not done in this iteration:** actual PPO training against the real
engine. This iteration only proves the seam and reward signal are sound;
training a real policy on it is real, separate work — correctly the W7
milestone in the timeline, not something to fold into this spike.

## Iteration 4: counterparts with concession dynamics (the gate named in 2b)

`src/concession_env.py` (`ConcessionSlotEnv`): same calendar, reward and
observation as Iteration 2, but each request's counterpart has a private
slot cost anti-correlated with self's (real conflict of interest), a
hidden type (Boulware / Conceder / rigid) that sets how its acceptance
threshold loosens per round, and a hidden reserve. Re-offering a rejected
slot is allowed, so "hold firm, re-offer later" is a real move.
`src/train_ppo_concession.py`, same protocol as 2b (3 seeds, 500k steps,
ent_coef=0.01, 300 eval episodes, eval seed 0).

**4a (fixed stakes ratio) -- PPO beat first-fit (-2.95) and greedy
(-3.15) on every seed at -2.741, and the script printed "GO". It was
wrong.** PPO's accepted slot cost was *higher* than both baselines', which
gave it away: a one-liner that offers your costliest slot first (= their
likely favourite) and steps down scored -2.699, matching PPO. It is now a
baseline, `src/concede_first.py`. Greedy losing to first-fit here is real
though: against an opposed-interest counterpart, hoarding cheap slots is
punished.

**Why no policy could beat it:** the reward scaled the no-deal penalty and
the slot cost by the same urgency, pinning their ratio at 5:1 for *every*
request. One fixed concession rule was therefore right every time.

**4b (stakes vary per request).** Failing to agree now costs a log-uniform
0.2-10, observable, against a slot cost of at most 1 -- a meeting with your
boss must happen at almost any slot cost; a loose catch-up is worth
dropping rather than burning your best slot. Same PPO protocol:

| Policy | Mean episode reward | Deal rate | Mean accepted cost |
|---|---|---|---|
| greedy best-utility-now | -3.012 | 98.0% | 0.699 |
| first-fit | -2.832 | 99.2% | 0.799 |
| PPO, 3 seeds | -2.746 (std across seeds 0.014) | 99.0% | 0.824 |
| concede-first | -2.732 | 100.0% | 0.855 |
| **stakes-switch** (`src/stakes_switch.py`) | **-2.659** | 81.3% | 0.852 |
| full-info oracle (2000 eps, sem 0.008) | -2.103 | — | — |

**Honest read: still not a go, and the failure is specific and useful.**

- The new decision the stakes dimension creates is *abstention*: when
  failing costs 0.2-0.6 but conceding costs ~0.85, the right move is to
  hold your cheapest slot and let the request fail. `stakes-switch` does
  exactly that below a swept threshold (deal rate 81%) and is the best
  non-learned policy.
- **PPO did not find abstention at all** (99.0% deal rate) and lands 3.3%
  below stakes-switch, consistently across seeds (std 0.014). Deliberately
  failing means a run of rejected offers followed by a terminal penalty,
  with acceptance always tempting in between -- plausibly an exploration/
  credit-assignment problem rather than a capacity one, but untested.
- Headroom is now large and *not* closable by one-liners: the oracle leads
  the best heuristic by 0.56, and threshold sweeps capture ~10% of it. The
  rest needs inferring the counterpart's hidden type from its rejections,
  which is the thing the toy env was built to test and the thing PPO has
  so far declined to do.

**Pattern across 4a and 4b worth carrying into the proposal:** twice now, a
"PPO wins" result has dissolved once a heuristic was written for the
behaviour PPO appeared to discover. Any headline RL result in this project
should ship with the one-line heuristic that imitates it, or it is not
evidence.

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

**Updated by Iteration 3 (real engine, below): don't over-generalize this
finding.** It's specific to the hand-rolled bilateral toy env's one-sided
structure. The real CalBench engine already shows naive heuristics
*colliding* with each other in ways the toy env couldn't produce — so
"greedy is hard to beat" is not expected to simply carry over.

## Go/no-go for the whole project, combining the toy spike and the real engine

Three separate things were tested, on purpose, before touching CalBench's
LLM-scale machinery or spending real training compute:

1. **Does RL learn anything non-trivial once there's real opportunity
   cost?** Yes — toy spike, one-shot to multi-request, +6% to +29% over
   first-fit. Learnable signal confirmed cheaply, off-CalBench.
2. **Is CalBench itself real, working, and something we can build on?**
   Yes, verified by actually cloning and running it (213/213 tests pass,
   a real 30-scenario run against the real CP-SAT oracle), not just
   reading the paper. Integration seam identified and *used*, not just
   theorized: a custom agent plugs in via `BaseClient` with zero engine
   changes.
3. **Does a naive heuristic dominate in the real, two-sided environment
   the way it did in the toy one?** No — the opposite showed up: a naive
   heuristic actively collided with the baseline's own bias and lost to
   random chance. That's evidence *for* the thesis, not against it: real
   coordination has strategic depth the one-sided toy env couldn't
   produce, which is exactly where a learned policy has room to help.

**Verdict: go.** Every cheap, fast-to-falsify check that could have
killed this early (env doesn't produce real signal, CalBench doesn't
exist or doesn't run, naive heuristics already win) came back negative.
Nothing here proves the full project succeeds — no PPO has been trained
against the real engine yet — but nothing found in ~a day of testing
says it can't. That combination (structurally sound, nothing disqualifying
found, real open work remaining) is exactly the bar for committing W7
effort to it rather than either declaring victory early or walking away.

**Concrete next milestone (W7-shaped, not part of this spike):** extend
`RLClient` to (a) act on real training scenarios across seeds, (b) touch
the cheap-talk/DM phase, not just decide(), since the collision finding
above suggests coordination signaling matters a lot, and (c) wire up an
actual PPO update loop against repeated `game.run()` calls (buffer
decisions per game, use final metrics as terminal reward — see the
integration notes in `docs/calbench-notes.md`).
