# Stage 0 results — prompted baseline, no training

Run 2026-09-19. 900 FEVER dev claims, 300 per label, seed 0. Top 5 pages,
up to 20 sentences each, one JSON answer per claim, temperature 0,
no thinking. Code: `src/stage0.py`. Raw per-claim rows are in
`data/stage0/` (gitignored). Standard error on a 900-claim accuracy is
about ±1.6 points; on the 300-claim per-label recalls about ±2.8.

## Retrieval

Page recall on 1,000 non-NEI claims (`src/eval_retrieval.py`). A hit
means every page of some complete gold evidence set is in the top k.
Title matching was tuned on train and checked once on dev.

| | dev @1 | dev @5 | dev @10 | train @5 |
|---|---|---|---|---|
| BM25 | 0.313 | 0.569 | 0.656 | 0.464 |
| title match + BM25 | 0.690 | **0.875** | 0.896 | 0.847 |

Title match: spans of the claim that start with a capital and equal a
page title; longest span wins; exact titles before `X (film)`-style
variants; `(disambiguation)` pages never used; at most 3 title pages,
then BM25 fills to 5. Adds about 5 ms per claim.

## Verifier

| | 2B, BM25 | 2B, titles | big, BM25 | big, titles |
|---|---|---|---|---|
| Format adherence | 0.927 | 0.843 | 1.000 | 0.999 |
| Label accuracy | 0.472 | 0.459 | 0.667 | **0.720** |
| FEVER score | 0.266 | 0.340 | 0.476 | **0.640** |
| Evidence F1 (non-NEI) | 0.298 | 0.423 | 0.457 | 0.717 |
| Recall SUPPORTED | 0.913 | 0.940 | 0.780 | 0.907 |
| Recall REFUTED | 0.297 | 0.300 | 0.750 | 0.907 |
| Recall NEI | 0.207 | 0.137 | 0.470 | 0.347 |
| ECE, stated confidence | 0.512 | 0.513 | 0.307 | 0.256 |
| ECE, verdict-token probability | 0.286 | 0.308 | 0.263 | 0.214 |
| Gold in context (of 600 non-NEI) | 317 | 517 | 317 | 517 |
| Accuracy, gold in context | 0.640 | 0.632 | 0.953 | 0.952 |
| Tokens in / out per claim | 1072 / 47 | 1365 / 53 | 1072 / 27 | 1365 / 28 |
| Cost | local | local | $0.156 | $0.196 |

2B = `Qwen/Qwen3.5-2B` in bf16 on the local GPU. Big = `qwen/qwen3.8-flash`
on OpenRouter, served by Alibaba for every call (the only provider
supporting `logprobs` + `seed`; it caps `top_logprobs` at 5).

## What this says

1. **Retrieval was the ceiling; title matching mostly removes it.** Gold
   in context went from 53% to 86% of non-NEI claims, and the big model's
   FEVER score from 0.48 to 0.64 with no model change.
2. **With the evidence in front of it, the big model is right 95% of the
   time, the 2B model 63%.** That 32-point gap is what stage 1 GRPO on
   the verifier has to close. It does not move with retrieval.
3. **The 2B model says SUPPORTED to nearly everything.** 94% recall on
   SUPPORTED, 30% on REFUTED, 14% on NEI. Waving false claims through is
   the error the harm-weighted reward targets.
4. **Stated confidence carries no information.** The 2B model writes 1,
   or 0 on its NEI answers (it reads "confidence" as "how much evidence").
   The big model writes 0.95 or 1.0. Verdict-token probability is better
   but still overconfident. Calibration has to be learned, not prompted.
5. **Format needs a reward term, not SFT.** 84–93% adherence is enough
   for GRPO to have signal. Nearly every failure is the same habit:
   copying whole sentences into the evidence list instead of ids, which
   on long contexts also runs past the 160-token limit.
6. **NEI is where both models are weakest, and got worse with better
   retrieval.** More relevant-looking pages make both models commit.
   Some "wrong" REFUTED answers on gold-NEI claims may be defensible
   from the shown text; not audited.

Stage 1 should use title-match retrieval, so the verifier is trained on
contexts that usually contain the answer.
