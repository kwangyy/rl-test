# Decomposer: does splitting a claim help? (2026-10-15)

AG-2. Code: `agent/decompose.py` (the decomposer and its cache),
`agent/eval_decompose.py` (this comparison). Verifier: the 5,895-row direct
SFT adapter (`sft-direct-from-base-qwen3p5-2b-5895-samples/run1`), on the
same 900 FEVER dev claims and title-match retrieval as the fixed policy.

## Decomposer

`qwen/qwen3.8-flash`, temperature 0, two-example prompt. Cached at
`verifier/data/decompose/fever_dev_qwen_qwen3.8-flash_n900.jsonl`.

| | |
|---|---|
| Claims split into 2+ sub-claims | 332 / 900 (37%) |
| Sub-claim counts | 1: 568, 2: 293, 3: 35, 4: 4 |
| Split rate by gold label | SUPPORTED 33%, REFUTED 39%, NEI 38% |
| Big-model tokens per claim | 196 in, 17 out |

## Fixed policies on the 332 split claims

- **full**: search the claim, verify the claim (the existing fixed-policy trace).
- **each**: search and verify every sub-claim; any REFUTED refutes, all
  SUPPORTED supports, otherwise NEI.
- **pool**: search every sub-claim, verify the full claim once on the union
  (5 pages, round-robin across the searches). Tests retrieval gain only.

| | full | each | pool |
|---|---|---|---|
| Label accuracy | **0.735** | 0.642 | 0.717 |
| FEVER score | **0.630** | 0.551 | 0.620 |
| Recall SUPPORTED | **0.838** | 0.646 | 0.747 |
| Recall REFUTED | **0.720** | 0.627 | **0.720** |
| Recall NEI | 0.661 | 0.652 | **0.687** |
| Searches / claim | 1.00 | 2.13 | 2.13 |
| Verifies / claim | 1.00 | 2.13 | 1.00 |
| Tokens / claim | 1,438 | 3,120 | 1,683 |
| vs full: fixes / breaks | | 21 / 52 | 8 / 14 |

Decomposing hurts at twice the cost. `each` loses 9 points; `pool` is
within noise of `full` (8 vs 14 flips) and gains nothing.

## Why `each` loses

33 of its 52 breaks turn a correct SUPPORTED or REFUTED into NEI. Three
causes, from reading the breaks:

1. **Sub-claims lose the context retrieval needs.** "The band that released
   The Endless River formed in London" has no entity to search for; the
   full claim found Pink Floyd through the album title. Multi-hop claims are
   where decomposition was supposed to help, and they are where it fails.
2. **The AND rule compounds NEI.** One uncertain sub-claim out of three
   ("Brad Wilk co-founded Rage Against the Machine": NEI at 0.63) sinks a
   claim the verifier gets right whole.
3. **The decomposer changes the claim.** "Cars 3 isn't the third Cars movie
   Brian Fee has worked on" became "... is the third ..." (negation dropped).
   "Arizona is in three regions" became three regions the model chose.
   "Only" / "exclusively" claims split into a weak positive and a negative
   the verifier cannot judge.

## Fixing retrieval, context and the combine (second run)

Two causes above were measured from the traces before trying fixes:
rewording is a minor one (on sub-claims that use only the claim's own words,
`each` still loses 5 points), while in 21% of split claims at least one
sub-claim's own search misses every gold page (all sub-claim searches
together miss only as often as the full claim's search). Three fixes:

- **each-pooled**: every sub-claim is verified on a 5-page union of the full
  claim's search and all sub-claim searches, so no sub-claim is left without
  the right page.
- **each-pooled-ctx**: as each-pooled, and the verifier sees
  `(This is one part of the claim: "<claim>")` after the sub-claim.
- **Soft combine** (scored on every `each*` trace, no extra GPU):
  P(SUPPORTED) = prod P_S, P(REFUTED) = 1 - prod(1 - P_R), NEI the rest.

| | full | each | pool | each-pooled | each-pooled-ctx |
|---|---|---|---|---|---|
| Label accuracy | **0.735** | 0.642 | 0.717 | 0.654 | 0.663 |
| Soft combine | | 0.633 | | 0.639 | 0.675 |
| FEVER score | **0.630** | 0.551 | 0.620 | 0.557 | 0.569 |
| Recall SUPPORTED | **0.838** | 0.646 | 0.747 | 0.707 | 0.707 |
| Recall REFUTED | **0.720** | 0.627 | **0.720** | 0.644 | 0.678 |
| Recall NEI | 0.661 | 0.652 | **0.687** | 0.617 | 0.609 |
| Searches / claim | 1.00 | 2.13 | 2.13 | 3.13 | 3.13 |
| Verifies / claim | 1.00 | 2.13 | 1.00 | 2.13 | 2.13 |
| vs full: fixes / breaks | | 21 / 52 | 8 / 14 | 16 / 43 | 16 / 40 |

All three fixes together recover 3 of the 9 points (0.675 at best) and
still lose 6 to verifying the whole claim, at 2.4x the tokens. Since
each-pooled shows the verifier the same pages as `full` plus more, the
remaining gap is the split itself: the 2B verifier judges the whole claim
better than it judges its pieces, and any combine of piece-wise verdicts
compounds their errors.

## What this means

- Keep `decompose` as a controller action, but not in any fixed policy. A
  learned controller (RL-3) should find it rarely pays; that is a useful
  result for the accuracy-vs-cost story, not a bug.
- Retrieval, context and a soft combine were tried and close only a third
  of the gap (section above). Better decomposition alone is unlikely to make
  it pay with this verifier; revisit only if the verifier is trained on
  sub-claims, or with a much stronger verifier.
- Traces for RL-3's offline env:
  `verifier/data/traces/fever_dev_decompose-{each,pool,each-pooled,each-pooled-ctx}-sft-direct-from-base-qwen3p5-2b-5895-samples_run1_n900.jsonl`
  (332 rows each, schema 2).
