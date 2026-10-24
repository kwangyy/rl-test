# Live event stream backend (AG-2), and where the AG-2 plan stands

Written 2026-10-20 on `feat/backend`, branched from `agent` at `b520c2e`.
Section 1 checks each AG-2 item from the original proposal's team table
against the repo. Section 2 is the spec of the backend built on this branch.

## 1. AG-2 items from the original proposal

The proposal's AG-2 row (removed with `proposal.tex` in `6338188`):

> Owns: loop skeleton over all 900 claims; one multi-turn GRPO test (GPU); keep the GPU calendar.
> Wk 2: prompted agentic baseline (GPU); loop streams each step as an event (FastAPI).
> Wk 3–4: multi-turn GRPO runs 1 and 2 (GPU); GRPO verifier into the demo.
> Deliverable: multi-turn vs single-turn GRPO on the 900 claims; live event stream.

`verifier/TODO.md` replaced the dated schedule, so the weeks are no longer deadlines.

| Item | Status | What exists | What is missing |
|---|---|---|---|
| Loop skeleton over all 900 claims | **Done** | `agent/agent_loop.py`, fixed policy (search, verify, commit or abstain). Reproduces the archived verifier runs. Re-run on ranked search 2026-10-17: base 0.489, SFT `5895/run1` 0.771 accuracy ([agent loop notes §2](agent-loop-notes.md#2-fixed-policy-baselines-on-the-new-search)). Committed traces: `verifier/data/traces/fever_dev_fixed-*`, `fever_dev_decompose-*`. | — (the `bm25-p5-k100` traces the notes cite were committed 2026-10-24) |
| Prompted agentic baseline | **Partial** | `agent/prompted.py`: an OpenRouter controller picks search / decompose / verify / commit / abstain, and the 2B verifier judges on the GPU. Prompt v2 on 30 dev claims with the SFT verifier: 0.733 accuracy, 0.667 FEVER ([notes §3](agent-loop-notes.md#3-prompted-controller-v1-and-v2-prompts)). | The full 900-claim run, the NEI rule enforced in code, prompts v3a/v3b, and a holdout check. Only the two 30-claim smoke traces are committed. |
| Loop streams each step as an event (FastAPI) | **Done** | `agent/stream.py` and `agent/server.py`, specified in section 2. Fixed and prompted policies stream live, and saved traces replay as the same events. 16 tests in `verifier/tests/test_agent_stream.py`, including prebuilt UI serving. Live smoke test on the real index, GPU and controller passed 2026-10-24 (section 2.7). | The event format is not yet agreed with AG-3. Multi-turn GRPO is not streamed. |
| One multi-turn GRPO test | **Done (spike)** | `verifier/experiments/_shared/grpo_multiturn.py`, 16 claims, 3 steps. It runs end to end, but peak memory was 41.5 GB on a 16 GB card ([GRPO results §5](grpo-results.md#5-multi-turn-spike-2026-10-12)). Cutoff fix and smoke test done 2026-10-17: 6 of 32 rollouts still end on a tool call ([notes §5](agent-loop-notes.md#5-multi-turn-grpo-rollouts-cut-off-after-a-tool-call)). | Rerun with `--method bm25 --search-k 5` and `--num-generations 4`. |
| Multi-turn GRPO runs 1 and 2 | **Not started** | Nothing beyond the spike and the smoke test. | Both runs, then the deliverable: multi-turn against single-turn GRPO on the same 900 claims. |
| GRPO verifier into the demo | **Plumbing and initial frontend** | The server takes any verifier through `--adapter`, including the `sft,grpo` form for a GRPO LoRA on a merged SFT model. `demo/` provides a React/Bun/Tailwind frontend for live steps and saved-trace replay. | Live frontend verification with a GRPO adapter remains pending. No GRPO run beats the 5,895-row SFT yet ([GRPO results §3](grpo-results.md#3-what-the-numbers-say)), and the CoT GRPO runs are pending. Multi-turn GRPO models call `search` themselves, so they would need their own stream function. |
| Keep the GPU calendar | Not tracked in the repo | — | — |

## 2. Backend spec

### 2.1 What it is for

The demo's Live mode (AG-3) shows one claim being checked as it happens:
the query, the hits, the verifier's verdict, and then commit or abstain. Browse
and Compare mode read saved traces. This backend serves both through one
event format, so the frontend draws a live run and a saved run the same way.

The `demo/` frontend provides saved-trace browsing, replay, live claims,
a step timeline and expandable evidence with final citations marked. Browse loads complete traces on demand; Replay streams their recorded
steps. Matched-claim comparison remains future work.

### 2.2 Run it

```
uv sync                                 # fastapi and uvicorn are dependencies; httpx and pytest are in the dev group
uv run python -m agent.server           # base 2B verifier, http://127.0.0.1:8000
uv run python -m agent.server --adapter verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints
uv run python -m agent.server --no-live # built UI + saved traces: no Bun, GPU, index or API key
uv run python -m pytest verifier/tests/test_agent_stream.py -q
```

Open <http://127.0.0.1:8000> for the committed `demo/dist/` UI; without
`dist/index.html`, the server serves only the API. Rebuild frontend changes
with `bun run --cwd demo build`, which produces `app.js` and `app.css` and
removes obsolete files. See [frontend development](../../README.md#frontend-demo)
for Bun's hot-reloading server and SSE proxy.

| Flag | Default | Meaning |
|---|---|---|
| `--model`, `--adapter` | `Qwen/Qwen3.5-2B`, none | Verifier, as in `agent_loop.py` and `eval_verifier.py` |
| `--method`, `--candidate-pages` | `bm25`, 5 | Search, as in `agent_loop.py` |
| `--controller` | `qwen/qwen3.8-flash` | Big model for `policy=prompted`; needs `OPENROUTER_API_KEY` in `.env` |
| `--max-new-tokens` | 160 | Verifier output budget; invalid replies are logged in the terminal |
| `--host`, `--port` | `127.0.0.1`, 8000 | Bind address |
| `--cors` | `*` | Allowed browser origins |
| `--no-live` | off | Disable `/api/live` |
| `--allow-reserved` | off | Allow calibration and holdout claim ids in `/api/live` |

The index and verifier load on the first live request, not at startup. A
missing FEVER index makes that request end with an `error` event.

### 2.3 Endpoints

All are `GET`, so a browser `EventSource` can open the streams.

| Path | Returns |
|---|---|
| `/` and built asset paths | Prebuilt UI from `demo/dist/`, when its `index.html` exists |
| `/api/health` | `live`, `policies`, which backends have `loaded`, and the model, adapter, search and controller settings |
| `/api/live?id=<FEVER dev id>&policy=fixed\|prompted&search_k=<1–100>` | SSE stream for a dev claim. The row has gold, gold evidence and reward. |
| `/api/live?claim=<text>&policy=...` | SSE stream for free text. The row has `dataset: "live"`, `split`, `gold` and `reward` null, and `gold_evidence: []`. |
| `/api/traces` | `[{name, rows}]` for each `.jsonl` in `verifier/data/traces/` |
| `/api/traces/{name}?offset=&limit=` | `{name, rows, items}`, one summary per claim: `id, claim, gold, verdict, confidence, ended_by, correct, n_steps, n_ops` |
| `/api/traces/{name}/{id}` | The full trace row |
| `/api/traces/{name}/{id}/replay?delay=<0–10 s>` | SSE stream of the saved row, `delay` seconds between events (default 0.5) |

Errors before a stream starts are plain HTTP errors:

- 422: not exactly one of `id` / `claim`, an empty claim, an unknown policy, or `search_k` out of range.
- 403: a reserved claim id.
- 404: an unknown id or trace. Trace names are bare file names, so a path is a 404.
- 503: `--no-live`, or the FEVER dev file is missing.

### 2.4 Events

Each SSE message is `id: <n>`, `event: <type>` and `data: <JSON>`. The JSON
repeats the type as `event`. A stream ends after `final` or `error`.

| `event` | Fields | When |
|---|---|---|
| `queued` | `message` | Live only: another live run holds the GPU. Sent before `run`. |
| `status` | `message` | Live only: the index and verifier are loading (first request). |
| `run` | `claim`, `id`, `policy`, `run` (the trace row's `run` object); `replay: true` on replays | Once, first |
| `action` | `t`, `action`, `args`; `controller` (the call that chose it) for `prompted` | The action is chosen and about to run. Use it to show "searching …". |
| `step` | `step`: one trace step `{t, action, args, result, seconds[, controller]}` | The action finished. Same object as `row.steps[t]`. |
| `final` | `row`: the full trace row ([trace format](trace-format.md)) | Once, last |
| `error` | `message` (`<ExceptionType>: <text>`) | The run failed and the stream ends |

Every `step` is preceded by an `action` with the same `t`, live and in replay. A `prompted` run
stops after `commit` or `abstain`, or with `final.row.final.ended_by:
"budget"` after `prompted.MAX_STEPS` (8) steps. A `fixed` run always has three
steps: `search`, `verify`, then `commit` or `abstain`. The last action follows from the
verifier's answer, so its `action` event arrives just before its `step`.

Example (fixed policy, abbreviated):

```
id: 0
event: run
data: {"event": "run", "claim": "Ada Lovelace wrote an algorithm.", "id": null, "policy": "fixed-bm25-p5-k100", "run": {...}}

id: 1
event: action
data: {"event": "action", "t": 0, "action": "search", "args": {"query": "Ada Lovelace wrote an algorithm.", "k": 100}}

id: 2
event: step
data: {"event": "step", "step": {"t": 0, "action": "search", "args": {...}, "result": {"hits": [{"page": "Ada_Lovelace", "line": 1, "text": "...", "score": 21.4}]}, "seconds": 0.08}}

...

id: 7
event: final
data: {"event": "final", "row": {"schema": 1, ...}}
```

### 2.5 Same rows as the batch loops

A live run must produce the trace row the batch loop would write, so the same
viewer and scorer can read it. Each policy's row code therefore exists only once:

- `agent_loop.run_chunk` now builds each row with `agent_loop.to_row`, which `stream_fixed` also calls.
- `prompted.run_chunk` now executes each round with `prompted.act`, which `stream_prompted` calls once per decision. Both end with `prompted.to_row`.

Checks:

- **The refactor changes nothing.** On a 3-claim chunk with fake search, verifier and controller (search, decompose, sub-claim verify, claim verify, commit), both batch loops give byte-identical rows on `agent` (`b520c2e`) and on this branch, ignoring wall-clock `seconds`.
- **Streamed rows match batch rows.** `test_fixed_stream_matches_batch_row` and `test_prompted_stream_matches_batch_row` compare them field for field, ignoring `seconds`. They also check that each streamed `step` is the matching `row.steps[t]`.
- **No schema change.** `run` holds the same keys as a batch run, and the trace schema is unchanged. Free-text claims only fill existing fields: `dataset: "live"` and null gold.

### 2.6 Concurrency and safety

- **One run at a time.** A process-wide lock covers the GPU: a request that has to wait gets `queued`, then runs. If the client disconnects, the generator closes and the lock is released.
- **Background thread.** Starlette runs the blocking loop in its thread pool, so health and trace requests stay responsive while a claim runs.
- **Reserved claims.** Calibration and holdout ids (`utils.fever_eval.reserved_ids`) are refused with 403, so the demo cannot become an unrecorded look at the holdout. Free-text claims are not checked.
- **Local by default.** The server binds to 127.0.0.1 and has no authentication. Do not expose it beyond a local or tunnelled demo.

### 2.7 Tested, and not tested

Tested on 2026-10-20 with no GPU, FEVER index or API key:

- Frontend checks (2026-10-23, Bun 1.4.2): TypeScript and production builds
  pass; saved traces stream through both the Bun proxy and FastAPI. Headless
  browser checks cover replay controls, steps/evidence, null verdicts and a
  390 px layout without exceptions or overflow. Repeated builds produce
  identical assets and remove obsolete files.
- The 16 backend tests use fakes to cover event order, row equality, free-text
  claims, the 8-step budget, replay, HTTP errors, reserved ids, streamed errors,
  queueing, UI assets and startup without a bundle.
- The real server was run with `uvicorn` against the committed traces. `/api/traces` listed all 7 files, a replay streamed 8 events over 0.6 s at `delay=0.1`, and a live request without the index returned `status` and then `error` naming the missing `bm25_index/params.index.json`.

Live smoke test, 2026-10-24, RTX 5070 Ti, BM25 `bm25-p5-k100`, verifier `sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1`, controller `qwen/qwen3.8-flash`. Run in both the old pip `.venv` (transformers 5.17) and the `uv sync --locked` environment (transformers 5.20); both gave the same verifier output.

- **Fixed policy, dev id 19983.** The first request loads the index and verifier (about 37 s in all). The hits, evidence ids and verdict equal the row in `fever_dev_fixed-sft-…_run1_bm25-p5-k100_n900.jsonl`. Confidence is 0.859 live against 0.845 in the batch trace, a small numeric difference between a batch of one and the batched run; the reward differs only through that confidence.
- **Warm latency, five dev claims.** Search 0.07–0.20 s, verify 0.83–0.94 s per claim.
- **Prompted policy, dev id 19983.** Streams the controller's choices (search, verify, search, verify, commit) with the controller call on each `action`; about 14 s for 5 steps. The controller is not deterministic, so the number of steps varies between runs.
- **Free text.** `dataset: "live"`, null `gold` and `reward`, empty `gold_evidence`.
- **Queueing.** A second request sent during a live run received `queued`, then ran after the first finished.
- **Errors.** 403 for a holdout id, 422 for missing or both of `id`/`claim`, an empty claim, an unknown policy or `search_k=0`, 404 for an unknown id and for a `../` trace name.
- **Traces and replay.** `/api/traces` listed 27 files; a replay of 19983 gives the same `action`/`step` sequence as the live run.
- **GPU memory.** About 7 GB in use with search, verifier and controller loaded.
- **Found and fixed.** The fixed live stream sent no `action` before `commit`/`abstain`, while replay did. `stream_fixed` now sends it, and `test_fixed_stream_matches_batch_row` compares the live and replayed action sequences.
- Transformers falls back to its slow reference kernels because `flash-linear-attention` and `causal-conv1d` are not installed. They are not in `pyproject.toml`; installing them should cut verify latency.

### 2.8 Next

- [x] Live smoke test on the GPU machine (section 2.7).
- [ ] Optional `--save` to append live rows to a `fever_live_<run>.jsonl` trace, for the failure audit.
- [ ] Stream the multi-turn GRPO model: search calls happen during generation, so this needs a generation loop that yields at each tool call, not `stream_prompted`.
- [ ] Confirm the event format with AG-3 as the frontend develops. Change it the way the trace format changes: add fields, never rename them.
