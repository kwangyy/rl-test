# Trace format: one JSONL row per claim

Frozen 2026-10-11 (schema 1). AG-2 writes it, RL-3 trains the controller
on it, AG-3's eval script and the demo read it. Interface 4 of the four
frozen in week 1 ([`plan.md`](plan.md)). The `search` and `verify` signatures
the steps log are in `core/interfaces.py`.

## Rules

- One line per claim per run, UTF-8 JSON. File name
  `<dataset>_<split>_<run>.jsonl` under `verifier/data/traces/`.
- Labels use untrained baseline's spelling: `SUPPORTED`, `REFUTED`, `NOT_ENOUGH_INFO`.
  `reward.py` maps the last to `NEI` internally.
- Evidence is always the global id `"<page>:<line>"`, never the
  prompt-local `E3`, so traces from different prompts and retrievals compare.
- Every step is logged, including the ones that change nothing. The
  controller learns from the full sequence, not only the outcome.
- Fields are only added, never renamed or removed, after sign-off. Bump
  `schema` when adding one.

## Row

| Field | Type | Meaning |
|---|---|---|
| `schema` | int | Format version, starts at 1 |
| `id` | int or str | Claim id from the dataset |
| `dataset`, `split` | str | `fever` / `scifact` / `averitec`; `train` / `dev` / `test` |
| `claim` | str | Claim text |
| `gold` | str | Gold label |
| `gold_evidence` | list of list of str | Gold evidence sets, each a list of `"page:line"`. `[]` for NEI |
| `run` | object | `policy` (e.g. `prompted`, `grpo-mt-r1`, `controller-ppo`), `verifier` (model + adapter path), `reward_version` (git sha of `reward.py`), `seed` |
| `steps` | list | The actions taken, in order (below) |
| `final` | object | `verdict`, `evidence` (`"page:line"`, at most 5), `confidence`, `ended_by` (`commit` / `abstain` / `budget`) |
| `cost` | object | `n_search`, `n_decompose`, `n_verify`, `n_ops` (what `reward.total` takes), `tokens_in`, `tokens_out`, `seconds` |
| `reward` | object or null | `total` and `parts` from `reward.total`; filled by whoever scores the run, null if unscored |

## Steps

Each step is `{"t": <int>, "action": <str>, "args": {...}, "result": {...}, "seconds": <float>}`.

| `action` | `args` | `result` |
|---|---|---|
| `search` | `query`, `k` | `hits`: list of `{page, line, text, score}` (interface 1) |
| `decompose` | `claim` | `subclaims`: list of str |
| `verify` | `claim`, `evidence` (`"page:line"` list) | `verdict`, `evidence`, `confidence`, `token_prob` (`{SUPPORTED, REFUTED, NOT_ENOUGH_INFO}`) (interface 2) |
| `abstain` | `{}` | `{}` |
| `commit` | `{}` | `{}` |

`abstain` and `commit` end the episode and are always the last step,
unless the budget ran out (`ended_by: "budget"`). The controller's
observation features (retrieval scores, verifier confidence, hop count,
budget left) are all derivable from the steps, so they are not stored twice.

## Example (illustrative values, not from a run)

```json
{"schema": 1, "id": 75397, "dataset": "fever", "split": "dev",
 "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
 "gold": "SUPPORTED", "gold_evidence": [["Nikolaj_Coster-Waldau:7", "Fox_Broadcasting_Company:0"]],
 "run": {"policy": "prompted", "verifier": "Qwen/Qwen3.5-2B", "reward_version": "a2317dc", "seed": 0},
 "steps": [
  {"t": 0, "action": "search", "args": {"query": "Nikolaj Coster-Waldau Fox", "k": 5},
   "result": {"hits": [{"page": "Nikolaj_Coster-Waldau", "line": 7, "text": "He then played Detective John Amsterdam in the short-lived Fox television series New Amsterdam ...", "score": 21.4}]},
   "seconds": 0.08},
  {"t": 1, "action": "verify", "args": {"claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.", "evidence": ["Nikolaj_Coster-Waldau:7"]},
   "result": {"verdict": "SUPPORTED", "evidence": ["Nikolaj_Coster-Waldau:7"], "confidence": 0.8,
              "token_prob": {"SUPPORTED": 0.83, "REFUTED": 0.05, "NOT_ENOUGH_INFO": 0.12}},
   "seconds": 0.41},
  {"t": 2, "action": "commit", "args": {}, "result": {}, "seconds": 0.0}],
 "final": {"verdict": "SUPPORTED", "evidence": ["Nikolaj_Coster-Waldau:7"], "confidence": 0.8, "ended_by": "commit"},
 "cost": {"n_search": 1, "n_decompose": 0, "n_verify": 1, "n_ops": 2, "tokens_in": 1840, "tokens_out": 31, "seconds": 0.49},
 "reward": null}
```

## Decisions at sign-off (2026-10-11)

- **Reward is final only.** `reward` is filled once per row from
  `reward.total`; there is no per-step reward. RL-3 can derive shaping from
  the steps if it needs it.
- **Search hits keep the full sentence text**, so a trace is
  self-contained for the demo and the failure audit (about 1 KB per hit).
- **Sentence ids are `line`**, FEVER's line number on the page, the same
  as `eval_verifier.py` and `pages.sqlite`. A hit is `{page, line, text, score}`.

## Schema 2 (2026-10-15): prompted controller

Added for `agent/prompted.py`, where a big model picks the actions. Schema 1
rows (the fixed policy) stay valid; every field below is an addition.

- `run.controller`, `run.max_steps`: the controller model and the action budget.
- `steps[].controller`: the controller call that chose this step:
  `text` (raw reply), `valid` (false when the reply was unusable and the loop
  fell back to search, then verify, then commit), `tokens_in`, `tokens_out`,
  `seconds`. A `decompose` step also has `decompose_tokens_in` / `_out`.
- `steps[].args.subclaim` on `verify`: index into the preceding `decompose`
  result, or null when the full claim was verified.
- `cost.n_controller`, `cost.controller_tokens_in`, `cost.controller_tokens_out`.
  `cost.tokens_in` / `tokens_out` are verifier plus controller.
