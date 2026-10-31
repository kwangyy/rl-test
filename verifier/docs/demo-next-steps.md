# Demo: next steps once the RL team finishes the controller

Written 2026-10-31 for whoever picks up the demo (AG-2 backend, AG-3 frontend). The RL team's state is in `controller-status.md` (untracked; the detailed docs are on `origin/controller/grpo-handover`). Come back to this when they have a trained controller they want shown.

## 0. Where things stand

- **Frontend (`demo/`)** is in good shape: Browse, Compare, Replay and Live modes, a restyled UI (Geist, Source Serif 4 and Geist Mono, self-hosted and inlined into `dist/app.css`), and a confidence ring in the verdict panel. After any frontend change run `cd demo && bun run check && bun run build`, because FastAPI serves the committed `demo/dist/`.
- **Frontend already understands the controller branch's tools**, but this was written against `agent/tools.py` on `controller/grpo-handover` and has never been run on a real trace that uses them:
  - Icons and step chips for `follow`, `open_page`, `find_pages` and `web_pages` (`demo/components/StepTimeline.tsx`). `find_pages` shows the titles it found. A tool step that returns nothing shows its `result.status`.
  - `run.controller` and `run.tools` are shown in the verdict header and in Compare (`controllerLabel` in `demo/app.tsx`). Compare warns when two runs use different controllers or tool presets.
  - The policy dropdown labels come from `policyLabel` in `demo/app.tsx`. Unknown policy names are title-cased and shown as "<Name>: agent chooses steps".
- **Backend (`agent/server.py`)** only has `policy=fixed|prompted`. `prompted` calls the big OpenRouter model (`--controller`). The 2B controller is not a live policy, and `server.py` and `stream.py` have no changes for it on the handover branch.
- **Nobody has tried live mode with the fine-tuned models.** Only the default stand-in models were smoke tested.

## 1. What the RL team is expected to deliver

Per `controller-status.md`, in rough order:

1. **Imitation controller** (a Qwen3.5-2B LoRA that ties flash on dev 900). It is already trained. The adapter is in Git LFS on `controller/grpo-handover` under `verifier/experiments/sft-controller-from-base-qwen3p5-2b-flash-follow-18291-samples/runs/run1/artifacts/checkpoints/`.
2. **GRPO controller**, started from the imitation adapter, with the verifier frozen at `sft-direct-from-base-qwen3p5-2b-5895-samples/run1`. They run a pilot first and may stop if the reward doesn't vary. Check for a result before building anything specific to GRPO.
3. **A new verifier** to swap in, and a final **holdout comparison** of fixed loop, flash, imitation and GRPO.

Do not assume GRPO exists. The imitation controller is the safe thing to demo.

## 2. Steps

### 2.1 Get their code and weights
- `git fetch origin && git log origin/controller/grpo-handover` (or whichever branch they finish on; also check `origin/feat/controller` and `origin/agent`).
- Adapters are Git LFS, so run `git lfs install` and `git lfs pull` for the paths above. Without LFS the files are pointer stubs and loading fails confusingly.
- Merge or cherry-pick the controller files into the `demo` branch: `agent/prompted.py`, `agent/tools.py`, `agent/action_cache.py`, `agent/decompose.py` and the docs. `agent/prompted.py` changed a lot (+249 lines on their branch). Do not take their `agent/server.py` or `agent/stream.py` without checking, as they are unchanged there and the demo branch's versions are newer.
- Read `verifier/docs/trace-format.md` (the schema 2 and 3 sections) and `controller-grpo-handover.md` on their branch before touching the backend.

### 2.2 Make the 2B controller a live policy
- Add a policy to `POLICIES` in `agent/server.py` (suggest `controller`) and update the `Query` regex on `/api/live`, which is currently `^(fixed|prompted)$`.
- Load the local controller model and adapter the way `agent.prompted` does for `--adapter` (the local controller exposes `big.batch(...)`; the live path is single-claim). Add `--controller-adapter` and `--tools follow|pages` flags next to `--controller`.
- `stream.stream_prompted` yields `action` and `step` events as the loop runs. It must handle the tool actions: `follow`, `open_page` and `find_pages` run through `agent.tools.ToolKit` (one per claim) and their hits must go into the verifier's input the way `run_chunk` does it (opened pages' sentences first). Mirror `run_chunk` in `agent/prompted.py` rather than reinventing it.
- Give the new rows `run.policy` such as `prompted-bm25`, `run.controller` (the model or adapter name) and `run.tools`, because the UI keys its display off those. Add fields, never rename them.
- `/api/health` already lists `policies`, so the UI picks the new policy up automatically.
- The GPU lock allows one live run at a time, and the verifier and controller will both want the GPU. Check that both fit in memory together on the demo machine before promising a live demo.
- Tests: `uv run python -m pytest verifier/tests/test_agent_stream.py -q`. Add a case for a tool action in a stream.

### 2.3 Load their traces into Browse and Compare
- Dev 900 traces for flash, imitation and GRPO are in `verifier/data/controller/dev900-*-traces/` on their branch. Copy the ones worth showing into `verifier/data/traces/` as `fever_dev_<name>.jsonl`; the server lists everything there.
- These are the same 900 claims across runs, which is what Compare needs. The fixed-loop trace for the same verifier is `fever_dev_fixed-sft-direct-from-base-qwen3p5-2b-5895-samples_run1_bm25-p5-k100_n900.jsonl`.
- Open one trace with tool steps and one without, and check the timeline. This is the first real test of the new step rendering.
- If a verifier swap happens, rerun the traces on the new verifier first. Compare only means something with the same verifier and claims.

### 2.4 Smoke test live mode with the real models
- Start with the fine-tuned verifier alone: `uv run python -m agent.server --adapter verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints --save-live <name>`, with `fixed` then `prompted` policies, in the UI.
- Then the 2B controller policy. Watch for a long "Connecting…" on the first request (it loads the index, verifier and now the controller).
- Check that a hand-typed claim works as well as a dev claim.
- Expect roughly 11 s per claim at `--chunk 64` for the 2B controller in their batch runs. A single live claim may differ.

### 2.5 UI follow-ups once real traces exist
- Check the policy dropdown labels read well with the real policy names, and extend `policyLabel` if they don't.
- If the controller records a short thought per step (their doc item v3b), render it in `StepTimeline` the way `result.reasoning` is rendered.
- Optional ideas from the review: a Compare filter for "only one run correct" (the claims their McNemar tests count), and a marker for runs that retrieved gold evidence but never verified it (19 claims on dev 900, `controller-status.md` section 4).
- The `agent_loop` fixed row has `run.controller` unset, so no controller label shows. That is intended.

## 3. Things to watch out for

- **The 2B controller is not a demo of "a better model".** The doc says a controller that can search beats the fixed loop by about 2 points, borderline significant, and most misses are the verifier's. Keep any text in the UI honest about that.
- **Dev 900 numbers are development results.** The holdout is untouched and reserved. `--allow-reserved` exists, so don't use calibration or holdout IDs in the demo.
- **The branch has two kinds of GRPO.** The "multi-turn GRPO" model in `live-backend.md` section 2.8 is a different harness from the controller GRPO in `controller-status.md` section 5.1. "Stream the multi-turn GRPO model" in the old TODO and "demo: stream the multi-turn GRPO model through `agent/server.py`" in the status doc may mean either. Ask before building.
- **Docs on `agent` and on `controller/grpo-handover` disagree in places.** The status doc lists which branch holds which doc (section 6).
