# Agent loop: design notes and open questions (2026-10-17)

AG-2. A review of the three agent loops after retrieval PR #1 (ranked sentence
search) and Bryan's frozen evaluation v1 were merged into `agent`. It records
what each loop does, what was measured today, the problems found, and what to
test next. Revisit this page before changing a controller prompt, the
decomposer or the multi-turn GRPO harness.

## 1. The three loops

| Loop | Code | Shape | What the deciding model sees | ReAct? |
|---|---|---|---|---|
| Fixed policy (stage 0) | `agent/agent_loop.py` | search the claim once, verify, abstain on NEI else commit | nothing; the rule is fixed | No loop at all, by design: it is the baseline |
| Prompted controller (stage 0b) | `agent/prompted.py` | up to 8 steps of search / decompose / verify / commit / abstain | a text summary of the history: search results as **page titles only**; verify results as verdict, confidence and the cited sentences | Act-observe loop without the thought: one JSON action per call, reasoning disabled, `max_tokens=200` |
| Multi-turn GRPO | `verifier/experiments/_shared/grpo_multiturn.py` | the 2B model calls `search` mid-generation, then writes the verdict JSON | the full numbered sentences from each search | Act-observe loop without the thought: `enable_thinking: False`; tool calls carry no text |

Why neither loop writes thoughts:

- **Prompted controller.** It stands in for RL-3's learned controller, which
  picks discrete actions from a compact state, so the two stay comparable. It
  never reads evidence, so the 2B verifier stays the only judge and any gain is
  attributable to search decisions. Reasoning off also keeps cost at a fraction
  of a cent per claim.
- **Multi-turn GRPO.** Memory: the spike already peaked at 41.5 GB on a 16 GB
  card, and thoughts add tokens on every turn of all 8 rollouts. The SFT
  verifier was also trained on direct JSON.

## 2. Fixed-policy baselines on the new search

`search()` now returns BM25-ranked sentences from 5 candidate pages instead of
the first 20 sentences of each page in page order. On 300 of the 900 dev
claims, the returned sentence set was identical for 261 claims (87%), but the
order was identical for none, so every E-ID changes. Rerun on 2026-10-17, RTX
5070 Ti, same 900 dev claims, `bm25-p5-k100`:

| Verifier | Old search acc | New search acc | FEVER | Recall S / R / NEI | s/claim |
|---|---:|---:|---:|---:|---:|
| Base 2B | 0.459 | **0.489** | 0.369 | 0.943 / 0.367 / 0.157 | 0.84 |
| SFT `5895/run1` | 0.759 | **0.771** | 0.666 | 0.833 / 0.753 / 0.727 | 0.42 |

The base model gives the same verdict as its old trace on only 81% of claims.
**Compare new agent results against these traces, not the old ones:**
`verifier/data/traces/fever_dev_fixed-base_bm25-p5-k100_n900.jsonl` and
`fever_dev_fixed-sft-direct-from-base-qwen3p5-2b-5895-samples_run1_bm25-p5-k100_n900.jsonl`.

## 3. Prompted controller: v1 and v2 prompts

30-claim smoke tests (`--n 30`, 10 per label), SFT `run1` verifier, same claims
for both prompts. The prompt version is recorded as `run.controller_prompt`
(absent means v1).

| Prompt | Acc | FEVER | Recall S / R / NEI | Searches | Verifies | Controller tokens/claim |
|---|---:|---:|---:|---:|---:|---:|
| v1: "commit when the verifier is confident" | 0.633 | 0.600 | 0.90 / 0.80 / 0.20 | 1.60 | 1.40 | 1,513 in, 55 out |
| v2: retry once below 0.70; a weaker verdict does not override an earlier NEI | **0.733** | **0.667** | 0.80 / 0.90 / 0.50 | 1.63 | 1.73 | 2,135 in, 61 out |

v1 failure modes, read from the traces:

1. **Search until a yes.** When the verifier said NEI, the controller searched
   again until a verify returned SUPPORTED at 0.48–0.52, then committed it, as
   with "Children get sweet candy from Trick-or-Treating". This turns NEI claims
   into false SUPPORTED, the most heavily penalised error.
2. **Commits at about 0.5 confidence**, with no threshold in the prompt.
3. **Near-repeat queries** ("Pluto Nash cast" / "cast list" / "cast list
   actors" / "cast members"). The controller sees only page titles, so nothing
   tells it why a search failed.

v2 fixed 4 claims and broke 1, but the controller does not follow the
confidence-comparison rule reliably. "The Mighty Ducks…" was committed as
REFUTED at 0.52 after an NEI at 0.64, and happened to be right.

Why the threshold is "retry", not "abstain": on the 900-claim SFT fixed trace,
SUPPORTED/REFUTED verdicts below 0.70 are still right 61% of the time
(108 / 176). Abstaining on all of them loses 108 correct answers to gain 51 NEIs.

| Verifier confidence (S/R verdicts) | n | correct | gold NEI |
|---|---:|---:|---:|
| < 0.6 | 113 | 67 | 35 |
| 0.6–0.7 | 63 | 41 | 16 |
| 0.7–0.8 | 76 | 64 | 9 |
| 0.8–0.9 | 127 | 114 | 11 |
| ≥ 0.9 | 205 | 190 | 11 |

Cost: about $0.00035 per claim with v2, or **about $0.31 per 900-claim run**, at the
prices recorded in the code ($0.15 / $0.47 per M tokens, 2026-09-19); about 40
minutes locally. The prompts were tuned on dev claims, so the final comparison
must use the holdout set.

## 4. Decomposition: the flow and why it has not helped

### Two pipelines

**Offline comparison, `agent/eval_decompose.py`** (the numbers in
[decompose-results.md](decompose-results.md)):

```
claim -> decomposer (qwen3.8-flash, temperature 0, 2 examples, cached)
         fewer than 2 parts -> [claim]; only claims with 2+ parts are compared
      -> search(part, k) for every part   (+ search(claim) for *-pooled)
         each             verify(part) on its own hits -> hard combine
         pool             verify(claim) once on a 5-page union of all searches
         each-pooled      verify(part) on the union, claim's own search first -> combine
         each-pooled-ctx  same, with the original claim attached to each part
combine: any REFUTED -> REFUTED; all SUPPORTED -> SUPPORTED; else NEI
```

**Live, `agent/prompted.py`:** `decompose` is one controller action. The parts
are shown as `[0] [1] …`; `verify {subclaim: i}` judges part *i* against
`Episode.pool()` (earlier citations plus the **latest** search, whatever its
query); `commit` calls `final_answer()`.

### Claim size

FEVER claims are short: median 8 words, 90th percentile 13, longest 31. The
decomposer splits 18% of claims of 7 words or fewer, 41% of 8–10, 73% of 11–14
and 83% of 15+. Correct answers on the 332 split claims (old search, SFT
verifier):

| Words | Split claims | full | each | pool | each-pooled | each-pooled-ctx |
|---|---:|---:|---:|---:|---:|---:|
| ≤ 7 | 77 | **59** | 52 | 56 | 54 | 55 |
| 8–10 | 112 | **80** | 78 | **80** | 75 | 76 |
| 11–14 | 105 | **74** | 58 | 70 | 59 | 63 |
| 15+ | 38 | 31 | 25 | **32** | 29 | 26 |

By part count: 2 parts, full 215/293 against pool 209; 3 parts, 26/35 against
25; 4 parts, 3/4 against 4/4. Of all 900 dev claims, 61% are settled by one
sentence, 33% are NEI and 6% need two or more sentences. **FEVER cannot test
"decomposition helps very big claims"**: it has 46 claims of 15+ words and 4
with four parts. What it does show is that decomposition does not help at the
sizes FEVER has.

Caveat: the `full` column is the old trace labelled `n6000`; the decompose
traces are labelled `5895_run1`. This is probably the same adapter under an
older name, but that has not been checked.

### Flaws that would hide a gain on big claims

1. **The page budget does not grow with the claim.** `round_robin_pages` caps
   the union at 5 pages total, so a 3-part claim gets about 1–2 pages per part.
2. **The live controller discards part verdicts once the full claim has been
   verified.** `final_answer()` prefers any full-claim verdict. In the v2 smoke
   test, "The Adventures of Pluto Nash stars an actor that died in 1961" (gold
   REFUTED) got full-claim NEI 0.68, then parts SUPPORTED 0.92 and REFUTED 0.76.
   The combined parts give REFUTED, which is correct; the commit returned NEI.
3. **Live part verifies use the latest search**, which may have been for a
   different part. In the offline `each` policy each part gets its own search.
4. **The decomposer splits claims that are not plain conjunctions.**
   "…stars an actor that died in 1961" means *some cast member* died in 1961.
   Splitting it into "…stars an actor" and "That actor died in 1961" loses the
   link between the halves and breaks the prompt's own no-pronoun rule. This
   needs sequential decomposition (find the cast, then check each member), not
   parallel parts. `decompose-results.md` also records dropped negations and
   expanded "three regions".
5. **Nothing decides when to decompose.** 58% of the splits are claims one
   sentence already settles.

### How to test it properly

1. Fix the pipeline first: a page budget per part, part verdicts that count in
   `final_answer` (or log both answers), and a separate search for each part in the live loop.
2. Define "big" ahead of time (3+ parts, or 15+ words) and report by bucket
   against `full` on the same claims, verifier and search.
3. Use a dataset with big claims: HoVer (2–4 hops over the same Wikipedia dump)
   or AVeriTeC (already a planned transfer set). Use FEVER only to check that
   short claims are not hurt.
4. Then compare against decomposing only after a full-claim NEI or confidence below 0.70.

## 5. Multi-turn GRPO: rollouts cut off after a tool call

The 48 spike rollouts (`grpo-multiturn-from-base-qwen3p5-2b-kl004-16-samples/run1`)
by shape (C = tool call, T = tool result, A = answer):

| Shape | Rollouts |
|---|---:|
| `C T A` | 28 |
| `C T C T A` | 9 |
| `C T C T C`: asks for a third search, never run | 10 |
| `C T C`: second search never run | 1 |

**11 of 48 (23%) end on an unanswered tool call**, scored as a format failure.
GRPO then learns that searching again is bad, when the harness simply never let
the model answer.

Cause (TRL 1.13 `GRPOTrainer._tool_call_loop`): it executes at most
`max_tool_calling_iterations` rounds of tool calls, which the script set to
`--max-calls` (2). If the model then calls again, the loop exits and the
completion ends on that call. Separately, a tool result that would push the
sequence past `max_completion_length` (2048, counting the masked tool results;
about 860 tokens per search) is dropped and the rollout also ends on the call.
That is the likely cause of the `C T C` rollout.

Fix (2026-10-17): the environment enforces the search budget itself.

- `FeverSearchEnv.search` runs at most `max_calls` real searches. The last one
  ends with "(That was your last search. Reply now with only the JSON object.)";
  later calls return "No searches left. Reply now with only the JSON object."
  without searching.
- TRL gets `max_calls + budget_grace` iterations (`--budget-grace`, default 2),
  so a model that asks again still gets a turn to answer.
- `n_search` (the reward's `n_ops`) counts only real searches. Refused calls
  are logged as `n_refused`, and every rollout records `ended_on_tool_call`; the
  `cut_off` metric is logged with zero reward weight.
- `--budget-grace 0` restores the historical behaviour.

The overlong-result cutoff is not fixed by this. Shrink tool results with
ranked search (`--method bm25 --search-k 5` to `10`, which also needs a new
`--config-name` and `--run-name`), which is also the memory fix the spike
needed.

Smoke result with the fix (2026-10-17, same settings as the spike: base model,
16 claims, 8 generations, `--max-calls 2`, `--budget-grace 2`, page-prefix
search, RTX 5070 Ti). Step 3 of 3 ran out of memory (PyTorch had allocated
40.6 GB, spilling past the 16 GB card, and needed 14.8 GB more), so only the
32 rollouts from steps 1–2 exist:

| Shape | Rollouts |
|---|---:|
| `C T A` | 17 |
| `C T C T A` | 7 |
| `C T C T C T A`: refused once, then answered | 1 |
| `C T C T C T C T A`: refused twice, then answered | 1 |
| `C T C T C T C T C`: refused twice, called again, cut off | 6 |

The fix works only in part: 2 of the 8 rollouts that hit the budget answered
when told to, but 6 ignored "No searches left" twice and kept calling. 6 of 32
(19%) still end on a tool call, against 11 of 48 (23%) before; 10 of 32 have
no valid answer. The difference now is that the harness gives the model a turn
to answer, so a format failure reflects the model ignoring the instruction, and
`n_refused` / `ended_on_tool_call` make that visible. TRL fixes the offered tools
when the rollout starts, so the environment cannot withdraw `search` once the
budget is spent.

Memory is the binding constraint locally: steps took 2.5, 15 and 16+ minutes
as memory spilled into system RAM. Run further multi-turn work with shorter
tool results (`--method bm25 --search-k 5`) and/or `--num-generations 4`, or on SoC.

## 6. Should the loops do ReAct?

The ReAct paper (Yao et al., 2023; PaLM-540B) reports on FEVER roughly 61%
accuracy for ReAct, 59% for acting without thoughts and 56% for chain-of-thought;
ReAct combined with self-consistency reached about 65%. (Quoted from memory;
check the paper before citing.) Thoughts added about 2 points over act-only on
a mostly single-hop dataset. In this repo, CoT prompting helped the base 2B
(47.6% to 62.8%) but hurt the SFT verifier (74.6% to 71.4%, 44 outputs at the
1,024-token cap).

- **Prompted controller: probably a small gain, low risk, cheap.** The
  controller's failures are decisions (repeat queries, low-confidence commits).
  But it sees page titles only, so thoughts have little to work on. Test in two
  steps on the same 30 claims, then 900:
  - **v3a:** show the top 2–3 hits' text per search;
  - **v3b:** v3a plus one short thought before the JSON action.

  Track accuracy, NEI recall, searches per claim, repeated queries, cost, and
  how often the controller commits against the verifier's verdict (the risk
  once it reads evidence). Expected cost is about $0.40–0.60 per 900 claims.
- **Multi-turn GRPO: probably worse now.** Thoughts make the memory and cutoff
  problems worse. Starting from SFT, reasoning cost accuracy in evaluation, and
  GRPO with reasoning tends to grow output length. RL-1's native-thinking CoT
  GRPO runs (root `run.sbatch`) answer "does reasoning help under GRPO" for the
  single-turn verifier first; revisit after those results.

## 7. Next steps

- [x] Multi-turn GRPO: smoke-test the cutoff fix (section 5). The harness now
      offers a final turn; 6 of 32 rollouts still ignore it.
- [ ] Multi-turn GRPO: rerun the smoke test with `--method bm25 --search-k 5`
      and `--num-generations 4` (new `--config-name` / `--run-name`) to check
      memory, remaining cutoffs, and whether training reduces `n_refused`.
- [ ] Prompted controller: enforce the NEI rule in code (final answer is NEI
      unless a later verdict has higher confidence than the earlier NEI), as
      prompt v3 or a policy flag; compare with v2 on the same 30 claims.
- [ ] Prompted controller v3a / v3b as in section 6.
- [ ] Full 900-claim prompted run with the chosen prompt, against the new fixed baselines.
- [ ] Fixed retry policies (retry on NEI, retry below 0.70) before any learned controller.
- [ ] Decomposition: fix `final_answer` precedence and give each part its own
      page budget; rerun `each` / `pool` by length bucket; plan a HoVer test.
- [ ] Report multi-hop claims (6% of dev) as a separate slice in every agent comparison.
