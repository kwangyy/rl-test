# Model choices

Parameter counts and availability pulled from the HuggingFace and
OpenRouter APIs on 2026-09-18. Prices and listings change; re-check before
relying on them.

Two models are needed: a **small one that will later be trained**, and a
**big frozen one** for reference.

## Small, trainable

Pick the model you will GRPO later. Stage 0 is the "before" half of the
before/after comparison; if the baseline is a different model, the
difference cannot be attributed to RL.

| Model | Released | Actual params | License | On OpenRouter? |
|---|---|---|---|---|
| `google/gemma-4-E4B-it` | Mar 2026 | 8.0B | Apache-2.0 | No |
| `google/gemma-4-E2B-it` | Mar 2026 | 5.1B | Apache-2.0 | No |
| `Qwen/Qwen3.5-4B` | Feb 2026 | 4.7B | Apache-2.0 | No |
| `Qwen/Qwen3.5-2B` | Feb 2026 | 2.3B | Apache-2.0 | No |

Qwen3.8 has no small release (smallest is 27B). Llama has nothing small
since 3.2. There is no Phi-5.

**Gemma 4 "E4B" is 8B in memory.** The "E" means *effective* 4B: it
computes like a 4B model at inference, but training has to hold all 8B of
weights plus audio and vision towers that will not be used (pipeline tag
is any-to-any). For GRPO treat it as a 7-8B-class model, which rules out a
24GB card unless LoRA and tight settings are used. E2B is 5.1B in memory,
roughly the same class as Qwen3.5-4B.

**Pick: `Qwen3.5-4B`, or `Qwen3.5-2B` if the GPU is tight.**

- 4.7B is the real parameter count. Current, Apache-2.0.
- Direct successor to the Qwen2.5-3B / Qwen3-4B backbones Veri-R1 and
  ProFact used, so numbers stay comparable to theirs.
- RL tooling (TRL, verl) has historically supported the Qwen line first.
- Has a vision tower too, but much smaller than Gemma's.

If Gemma is preferred, use E2B, not E4B. Unverified risk: its architecture
(`Gemma4ForConditionalGeneration`, any-to-any) is new; whether TRL's
`GRPOTrainer` and vLLM handle it cleanly for text-only RL has not been
checked.

**None of these four are on OpenRouter.** Run the small model locally with
Ollama or vLLM; 2-5B does inference fine on a laptop, and the stage 0
baseline then uses the exact weights that get trained later. OpenRouter
providers may serve quantized weights, so an OpenRouter baseline is not
guaranteed to match a local checkpoint anyway.

Rough GRPO sizing (estimates, not measured):

| Size | Feasibility | Risk |
|---|---|---|
| 0.5-1B | almost any GPU | often cannot hold the output format; reward stays near 0 and GRPO has nothing to rank |
| 1.5-2.5B | LoRA on a 16-24GB card | fine single-turn, shaky for multi-turn tool use |
| 3-5B | LoRA on 24GB, tight | same class as Veri-R1 / ProFact |
| 7-8B | 40-80GB | out of course-project range without cluster access |

Older small models that *are* on OpenRouter, if a hosted small model is
needed for quick prototyping: `meta-llama/llama-3.2-3b-instruct`
($0.05/M in, Veri-R1 backbone), `qwen/qwen-2.5-7b-instruct` ($0.10/M,
ProFact backbone), `qwen/qwen3-8b` ($0.12/M, ProFact backbone, thinking
mode by default), `google/gemma-3-4b-it` ($0.05/M).

## Big, frozen

Ceiling reference, decomposer, and later the `escalate` target.

**`qwen/qwen3.8-flash` on OpenRouter.**

| Spec | Value | Source |
|---|---|---|
| Price | $0.15/M input, $0.47/M output | OpenRouter |
| Context | 1M tokens | OpenRouter |
| Supports | `tools`, `structured_outputs`, `response_format`, `logprobs`, `seed` | OpenRouter |
| Weights | `Qwen/Qwen3.8-Flash-Next`, license "other", not gated | HuggingFace |
| Total params | 179,999,981,459 (BF16) | HF safetensors metadata |
| Experts | 512 per layer, 10 per token, plus a shared expert | `config.json` |
| Layers / hidden | 48 / 2560 | `config.json` |
| Active params per token | about 4-5B | estimate from config, not stated on the card |

A hand tally of expert weights from the config gives about 121B; the
remaining ~60B toward HF's 180B was not accounted for (probably the vision
tower and modules not inspected). Treat 180B as authoritative.

It cannot be trained here: 180B in BF16 is about 360GB of weights before
optimizer state, and MoE training needs every expert in memory even though
few are active. Frozen role only.

Alternatives: `google/gemma-4-31b-it` ($0.09/M) to stay in one family;
`qwen/qwen3.8-27b:free` and `google/gemma-4-31b-it:free` cost nothing but
are rate-limited, and the free 27B has no `logprobs` or `seed`.

## Run settings for stage 0

- `temperature=0` and a fixed `seed`; log which provider served each call.
- Track the format-adherence rate of the small model (valid verdict +
  evidence ids + confidence). Low adherence means an SFT warm-start is
  needed before GRPO.
- If `logprobs` are available, log confidence from the verdict token's
  probability alongside the confidence the model states in text. Comparing
  the two is a standard baseline in the calibration papers and costs
  nothing extra.
- Cost is negligible: about 1,000 FEVER claims at ~5k tokens each through
  a $0.05/M model is roughly $0.25.
- Retriever: start with BM25. For dense retrieval, ProFact used
  `Qwen3-Embedding-0.6B`, which runs locally.
