# CalBench access check (Task 0)

## Status

- The exact code link from the proposal
  (`https://anonymous.4open.science/r/calbench2026-235F/README.md`) is
  **confirmed genuine** — it's cited verbatim in the published paper's own
  code-availability statement. It is also **confirmed blocked for
  automated fetch** (HTTP 403, tried 3x, including retrying after finding
  independent confirmation it's the real link). This needs a human to
  open it in an actual browser — still not done.
- However, CalBench is not an anonymous/unpublished submission: it's a
  public arXiv paper, **Zou, Yao, She, Goodman & Hawkins (Stanford),
  "CalBench: Evaluating Coordination–Privacy Trade-offs in Multi-Agent
  LLMs," arXiv:2605.09823v3** (https://arxiv.org/abs/2605.09823, posted
  28 May 2026, revised 5 Jun 2026). Reading the actual paper answered
  most of what Task 0 needed — more reliably than a README would have.
- ⚠️ One dead end for the record: an automated PDF-text summary earlier
  hallucinated a GitHub URL (`github.com/bosonphoton/calbench`) that does
  **not** exist (confirmed 404 via `gh`). Don't trust that URL if it
  resurfaces anywhere — it was a tool artifact, not something in the
  paper.
- A live leaderboard is real and reachable:
  https://d3mern3a2mjjur.cloudfront.net/leaderboard (OpenSkill ratings
  across 7 LLMs + 4 non-LLM reference protocols).

## What CalBench actually is — and where it differs from the proposal's assumptions

This matters for the "wrap vs reimplement" call and possibly for revising
scope before the 28 Sep proposal.

**1. It's an N-agent (not bilateral) group-scheduling benchmark for LLM
agents, not an RL environment.**
Each game has N=5 agents (canonical setup), each with a private calendar,
scheduling a stream of M=5 meetings. Agents coordinate via **natural-
language cheap-talk** (private DMs, participant groupchat, or all-agent
groupchat), then independently submit an atomic action batch
(RESCHEDULE*/SCHEDULE), validated and resolved centrally. This is
structurally different from "n Calendlys, one bilateral negotiation per
request" — it's closer to a multi-party group-scheduling / DCOP problem
with LLM agents as the primary subject of study.

**2. There is no Gym/Gymnasium interface.** Confirmed absent from the
paper. The proposal's W6-recess deliverable ("Gym wrapper... episode =
one person's calendar over N weeks") is *not* something CalBench ships —
building it is real, non-trivial work: either (a) wrap CalBench's
existing scenario generator + resolution logic in a Gym `step()`/`reset()`
loop yourselves, or (b) reimplement a bilateral-negotiation-shaped
environment that reuses only CalBench's scenario generator, CP-SAT
oracle, and VPS metric code. Recommend deciding explicitly which of
these two "wrap" actually means before the proposal draft, since they're
very different amounts of work.

**3. The shipped non-LLM baselines are IMAP, SD-MAP, and DSM
(DSM-private/DSM-welfare variants)** — not literally "first-fit,"
"greedy-best-utility-now," or a named "DCOP baseline." From the paper:
- **IMAP**: full per-slot cost vectors sent to initiator, who picks joint
  minimum — low-cost, high-disclosure reference point.
- **SD-MAP** (Modi & Veloso 2004): typed proposal-status only, low
  disclosure, feasibility-first.
- **DSM** (Farhadi & Jennings 2021): tunable privacy-cost curve via
  offer-set size and privacy-cost weight θ.
These are useful analogues but the proposal's specific baseline names
(first-fit/Calendly, greedy-best-utility-now, hand-tuned concession,
DCOP) will need to be implemented separately — they aren't drop-in from
CalBench. Table 2 in the paper has their numbers on the canonical task
suite if useful for calibration.

**4. "Prompted frontier LLM" is CalBench's main subject, not an add-on
baseline.** The paper's headline results ARE seven prompted LLMs (Claude
Sonnet 4.6, Gemini 3.1 Pro, Gemini 3 Flash, GPT-5.4 Mini, Llama 4
Maverick, Qwen 3.6 Plus, DeepSeek V4 Pro) evaluated exactly this way.

**5. No Boulware/Conceder/rigid concession-strategy counterparts appear
anywhere in the paper.** The "counterpart pool of varying flexibility"
in the proposal is not something CalBench provides — Team A's planned
"rule-based counterpart pool" work is fully necessary, not partially
redundant with CalBench. (This also matches what our own toy MVP spike
found independently, see docs/mvp-plan.md — concession dynamics turned
out to be exactly the missing piece for RL to have any edge over a
greedy heuristic.)

**6. VPS (the privacy metric) is now precisely understood** —
"Valuations of Possible States" (Maheswaran et al., 2006), §3.2 +
Appendix F of the paper:
- Each (round, target agent, observer agent) pair maintains a belief
  vector over slots, initialized to a uniform prior `p0 = 0.5`.
- Each message triggers a belief update `Bel'[k] = (1-α)Bel[k] + α·e`
  (evidence `e`, strength `α`) — exact update rules per message type are
  tabulated in the paper (Table 10) for the typed protocols.
- Per-round leakage = sum over slots of `|Bel_post[k] - p0|` (slot-
  equivalent units, Eq. 1).
- For LLM agents (unstructured chat), CalBench uses **reflection-
  calibrated VPS**: a measurement-only post-round prompt asks each agent
  how its belief about every other agent changed, and only belief
  movement toward the *true* state counts as leakage (conservative lower
  bound), audited against an LLM-as-judge over the raw chat.
- This is directly reusable/portable to a bilateral RL setting; the
  update-rule machinery would need adapting to whatever message/offer
  format the actual negotiation env uses.

**7. Confirmed real: CP-SAT oracle, scenario generator with a hidden
feasible witness solution + errands (movable, with private semantic
context) + blocked/immovable commitments, uniform vs. varied
(1/2/3, shown as 1/10/100 log-scale) displacement costs, and a
feasibility-difficulty score `d = F / (T! / (T-M)!)` from CP-SAT feasible
counts. OR-Tools (Perron & Didier, 2025) confirmed as the CP-SAT
dependency.**

## Still open — needs a human with a browser

- [ ] Open the actual repo, confirm it installs, check whether the real
      code structure matches the paper (papers and code sometimes drift).
- [ ] Check license / terms for reuse given the paper is public but the
      code link is still styled as an anonymous review link.
- [ ] Concretely scope what "wrap CalBench" means given point 2 above —
      this is a real decision, not just a checkbox, and affects the W6
      "decide wrap vs reimplement" call materially.

## Recommendation

Given points 1-5 above, "wrap CalBench" is more accurately "reuse
CalBench's scenario generator, CP-SAT oracle, and VPS pipeline as
building blocks, and build the bilateral-negotiation Gym env, request
stream, and counterpart pool yourselves" — which is close to what the
proposal's own W6/recess tasks already assumed (Gym wrapper, multi-week
stream, counterpart pool are all listed as things *to build*, not things
CalBench hands you). The main proposal risk this surfaces: baseline names
in the deliverables list don't match what's actually shipped 1:1, so
either rename them to IMAP/SD-MAP/DSM or budget real time to implement
first-fit/greedy/DCOP/concession baselines from scratch (the MVP spike
already has first-fit and greedy-best-utility-now working, at
docs/mvp-plan.md).
