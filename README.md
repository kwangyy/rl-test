# Fact verifier experiments

This repository studies evidence-grounded claim verification on FEVER. A verifier reads a claim and retrieved sentences, then returns a JSON verdict and cited sentence IDs; some configurations also return a stated confidence. The `rl` branch contains historical SFT and GRPO runs plus direct, CoT, and brief reasoning baselines.

- **Current priorities & experiment checklist**: [`verifier/TODO.md`](verifier/TODO.md)
- **Agent evaluation and controller checklist**: [`agent/TODO.md`](agent/TODO.md)
- **Research journal, experimental insights & error analyses**: `verifier/INSIGHT.md` (local / Git-ignored) and [`verifier/docs/`](verifier/docs/)

---

## Project Map

| Path | What it contains |
|---|---|
| [`verifier/train.sbatch`](verifier/train.sbatch) | Single training entry point; select a configuration with `TRAIN_SCRIPT` |
| [`verifier/dpo_rollouts.sbatch`](verifier/dpo_rollouts.sbatch) | Generate one reusable 1,000-claim native-CoT rollout set for the DPO comparison |
| [`verifier/dpo_pilot.sbatch`](verifier/dpo_pilot.sbatch) | Build genuine pairs, train DPO500 or DPO1000, then evaluate |
| [`verifier/grpo_after_dpo.sbatch`](verifier/grpo_after_dpo.sbatch) | Train GRPO500 on the disjoint half after DPO500, then evaluate |
| [`agent/eval.sbatch`](agent/eval.sbatch) | Agent batch-data and controller evaluation entry point; choose the method with `EVAL_METHOD` |
| [`agent/train.sbatch`](agent/train.sbatch) | Controller rollout, SFT data, scorer SFT, and RL entry point; choose `PHASE=rollouts`, `sft_data`, `sft`, or `rl` |
| [`agent/requirements.txt`](agent/requirements.txt) | Optional RSI-Jev controller dependency |
| [`agent/train/`](agent/train/) | Citation controller SFT labels, scorer training, RL training, and reward design |
| [`agent/docs/controller-workflow-and-results.md`](agent/docs/controller-workflow-and-results.md) | English bullet-point summary of the controller workflow, results, and conclusion |
| [`verifier/TODO.md`](verifier/TODO.md) | Current priorities and experiment checklist |
| [`verifier/docs/`](verifier/docs/) | Data guide, experiment results, architecture and background |
| [`verifier/data/train/`](verifier/data/train/) | Committed SFT and GRPO training examples |
| [`verifier/data/eval/`](verifier/data/eval/) | Frozen FEVER dev claim IDs |
| [`verifier/experiments/`](verifier/experiments/) | SFT and GRPO training configurations |
| [`verifier/evaluation/`](verifier/evaluation/) | Shared evaluator, `evaluate.sbatch` and `results/` |
| [`retrieval/`](retrieval/) | FEVER index and search code |
| [`core/`](core/) | Shared reward and interfaces |
| [`agent/`](agent/) | Runnable FEVER search/verify loop and agent-specific work |
| [`agent/evaluation/`](agent/evaluation/) | Agent evaluators and method-specific `results/`; submit jobs with `agent/eval.sbatch` |
| [`agent/evaluation/results/sampling_retry/`](agent/evaluation/results/sampling_retry/) | Three-draw retry evaluator and its `runs/` outputs |
| [`agent/data/build_evidence_batches.py`](agent/data/build_evidence_batches.py) | Exhaustive 10-sentence batch-data producer |
| [`agent/evaluation/results/rsi_jev_controller_baseline/`](agent/evaluation/results/rsi_jev_controller_baseline/) | RSI-Jev 2B or OpenJev controller baseline code; outputs use `rsi-jev-controller-baseline/` |
| [`agent/evaluation/results/rsi_jev_retry_baseline/`](agent/evaluation/results/rsi_jev_retry_baseline/) | Matched 0.3/0.5/0.7 threshold heuristics versus zero-shot RSI-Jev STOP/RETRY on fixed three-draw evidence chunks; prompt and command in its README |
| [`agent/evaluation/results/rsi_jev_evidence_selection_baseline/`](agent/evaluation/results/rsi_jev_evidence_selection_baseline/) | STOP/RETRY plus immediate cited-sentence KEEP/DROP, pooled verifier retry, and record-only controller verdict |
| [`agent/evaluation/results/heuristic_threshold_sweep/`](agent/evaluation/results/heuristic_threshold_sweep/) | Heuristic 0.3/0.7 only, using the fixed chunk source and fresh pooled Qwen samples |
| [`agent/train/`](agent/train/) | Train-only rollouts, pooled STOP/RETRY reward and RSI-Jev scorer pilot |
| [`agent/experiments/`](agent/experiments/) | Isolated controller training runs and the synthetic toy scaffold |
| [`agent/data/traces/`](agent/data/traces/) | Archived fixed-loop FEVER traces |
| `agent/data/evidence_batches/` | Local, Git-ignored Qwen batch outputs shared by controller evaluations; transfer separately from SoC |
| [`agent/docs/`](agent/docs/) | Agent trace schema and method notes |
| [`agent/tests/`](agent/tests/) | Agent evaluation and policy tests |
| [`agent/TODO.md`](agent/TODO.md) | Agent-only experiment checklist |
| [`utils/`](utils/) | Data preparation and experiment helpers |
| [`AGENTS.md`](AGENTS.md) | Conventions for collaborators and coding agents |

Detailed document map: [data and downloads](verifier/docs/data.md), [system plan](verifier/docs/plan.md), [prompted baseline](verifier/docs/prompted-baseline-results.md), [SFT](verifier/docs/sft-probe.md), [CoT teacher rejection audit](verifier/docs/cot-teacher-rejection-audit.md), [GRPO](verifier/docs/grpo-results.md), [agent traces](agent/docs/trace-format.md), [models](verifier/docs/models.md), and [related work](verifier/docs/related-work.md).

The runnable FEVER loop stays at `agent/agent_loop.py`; its traces are under `agent/data/traces/`. The 12-claim synthetic controller scaffold lives under `agent/experiments/toy_scaffold/` and runs with `python -m agent.experiments.toy_scaffold.run`. It is separate from FEVER evaluation and does not establish a controller gain. Agent-specific next steps are in [`agent/TODO.md`](agent/TODO.md).

### Agent controller comparison and train-only pilot

After the frozen 900-claim three-draw chunk source is complete on SoC, submit `EVAL_METHOD=rsi_jev_evidence_selection_baseline JEV_SOURCE_RUN=<completed-three-draw-source> RUN_NAME=<new-run> sbatch --export=ALL agent/eval.sbatch`. The comparison runs heuristic and RSI-Jev on the **same** precomputed chunk outputs. Each controller sees a chunk verifier output and chooses STOP/RETRY. The heuristic and citation-only Jev arms discard RETRY draws and save valid citations only on STOP. The evidence-selection Jev arm can KEEP/DROP citations on both RETRY and STOP; it can also revise earlier KEEP choices when deciding on the next draw. After every chunk is read, a new final Qwen verifier receives the pooled saved evidence. The validation evaluator precomputes three new Qwen draws on this unchanged pooled evidence, stores them separately in `pooled_samples.jsonl`, and lets each controller choose when to stop. An early STOP uses that exact output. On reaching the forced third draw, both policies use majority verdict across all three, with confidence to break ties and select the winning draw. Jev's own ACCEPT/REJECT/NEI is recorded separately; the main FEVER score uses its chosen pooled Qwen output. See the [selection evaluator README](agent/evaluation/results/rsi_jev_evidence_selection_baseline/README.md) for inputs, metrics and result paths.

**Controller design rationale.** The two trainable arms share the same fixed 10-sentence chunk draws, pooled verifier and call limit, so citation selection can be compared with learned STOP/RETRY alone. For selection, each draw exposes only its cited sentences plus citations the controller itself saved from earlier draws in that chunk. KEEP leaves an ID in the set; DROP removes it, including a previous KEEP. This lets RETRY seek a better verdict while revising evidence without exposing uncited retrieval results. The training reward scores the **final unique saved set once per claim**, alongside the final pooled verdict and call cost. It does not pay for a promising interim RETRY citation that the controller later drops. FEVER's annotated evidence sets provide incomplete supervision: an unannotated sentence gets a modest penalty, not a claim that it is factually false. The validation metrics separate actual precomputed GPU calls from simulated early-stop calls.

[`agent/train.sbatch`](agent/train.sbatch) first created three-draw rollouts for the fixed **5,500 FEVER train IDs**, then trained RSI-Jev citation and selection scorer heads; both 5,500-episode runs and frozen 900-claim evaluations are now recorded. The next citation experiment fixes a **3,000 SFT / 2,500 RL** train split under `agent/data/controller_split/`, generates high-margin outcome-derived SFT actions, and compares a 5,500-claim SFT-only control with two 3,000-SFT→2,500-RL reward variants. Read [agent/train/README.md](agent/train/README.md) for data, jobs, reward definitions, checkpoints, and limitations. The 900 dev claims are never training input.
The citation reward ablation compares two RL continuations from the **same 3,000-claim SFT head**: the current absolute outcome reward and `REWARD_MODE=relative_stop`, which scores claim-level improvement over always stopping on the first verifier draw. A separate **5,500-claim SFT-only** control matches the two-stage arms' total claim budget. Evaluate all three on the same frozen 900 dev claims; report FEVER, verdict accuracy, evidence F1, STOP/RETRY rates, simulated calls and extra GPU calls used only to compute the counterfactual training reward. The always-STOP policy has zero incremental reward by construction in the relative mode. See [agent/train/README.md](agent/train/README.md) for the exact formula and Slurm dependency sequence.

For an exploratory validation subset from the completed 900-claim source, set `PILOT_CLAIMS=30` with `EVAL_METHOD=rsi_jev_evidence_selection_baseline` and a new `RUN_NAME`. The evaluator validates the complete source first, then reads its first 30 frozen IDs; `manifest.json` records both counts. The resulting scores are a pipeline check, not a stable 900-claim comparison. Leave `PILOT_CLAIMS` unset for the full evaluation. The evaluator reports eight matched arms: heuristic thresholds 0.3, 0.5, and 0.7, combined Jev, Jev chunk STOP only, Jev citation KEEP only, Jev pooled STOP only, and `jev_citation` (Jev chunk plus pooled STOP with automatic citation retention). The last arm matches the citation-only training variant. Each threshold applies at both chunk and pooled STOP/RETRY; each single-component Jev arm changes just one decision rule. All arms share the same precomputed chunk draws. Pooled Qwen draws are reused only for identical selected evidence IDs within the same claim; a different pool triggers fresh Qwen draws. The train-only reward separately scores the selected verdict, the **final saved evidence set**, pooled-verifier citations, asymmetric pooled STOP/RETRY feedback, and extra calls. Chunk STOP/RETRY choices receive downstream reward only. See [agent/train/README.md](agent/train/README.md) and [agent/TODO.md](agent/TODO.md).

If a five-arm Jev evaluation is already running, run the **0.3/0.7 heuristic arms only** with `EVAL_METHOD=heuristic_threshold_sweep` and a new `RUN_NAME`; it does not reload Jev or repeat its decisions. See the [threshold sweep instructions](agent/evaluation/results/heuristic_threshold_sweep/README.md) for the SoC command and result paths. A running Slurm job keeps the code version it started with and does not gain new arms after a later Git pull.

### Agent sampling retry evaluation

The existing `agent/agent_loop.py` is the one-search, one-verification fixed baseline. The separate [`agent/evaluation/results/sampling_retry/eval_sampling_retry.py`](agent/evaluation/results/sampling_retry/eval_sampling_retry.py) experiment uses the **same frozen 900 FEVER dev IDs** and title-plus-BM25 retrieval context for each claim, then samples the verifier three times from the identical prompt (`temperature=0.7`, `top_p=0.9` by default). It saves all three verdicts, global evidence IDs, stated and verdict-token confidence, raw completions, and whether gold evidence was actually shown. No new evidence is retrieved between draws.

The offline heuristic selects the first valid `SUPPORTED` or `REFUTED` output with verdict-token confidence at least `0.5`; `NOT_ENOUGH_INFO`, invalid output, and missing or lower confidence trigger another draw, up to three. All three draws are generated for every claim even if the simulated policy would stop early. Therefore `metrics.json` reports both **actual three-call cost** and **simulated stopping cost**, along with first-draw versus heuristic accuracy/FEVER, verdict/evidence disagreement, and separate `gold_in_context`, `gold_not_in_context`, and gold-NEI results. The threshold is an exploratory rule, not a calibrated probability guarantee. FEVER gold labels describe the claim against Wikipedia, so a retrieval miss can make the original label impossible to justify from the shown context.

`heuristic.accuracy` remains the actual selected-output accuracy. A separate **hindsight controller diagnostic** credits a claim when the selected output is correct, or when the policy reaches its three-draw limit and all three verifier outputs are wrong. It reports early wrong stops and third-draw errors when a correct answer appeared among the three draws separately. This diagnostic uses gold and future draws after the run, so it is **not** an online controller accuracy, an action label, or an RL reward. The archived 900-claim run has 664 selected-correct, 81 exhausted/all-wrong, and `controller_decision_accuracy_hindsight = 745/900 = 0.828`; its actual selected-output accuracy remains `664/900 = 0.738`. See its `controller_diagnostics.json`; original predictions and metrics were left intact. For another completed run, use `python -m agent.evaluation.results.sampling_retry.reanalyse <run-dir>`.

On SoC, submit from the repository root. The job changes to the repository root before checking data and running the evaluator. The default adapter is the direct 5,500-row SFT checkpoint, with 900 claims and the direct prompt:

```bash
EVAL_METHOD=sampling_retry sbatch --export=ALL agent/eval.sbatch
```

To use another checkpoint, set `CHECKPOINTS` when submitting, for example `EVAL_METHOD=sampling_retry CHECKPOINTS=<adapter-path> PROMPT_STYLE=direct sbatch --export=ALL agent/eval.sbatch`. For a composed SFT+GRPO adapter, export `CHECKPOINTS="<sft-checkpoint>,<grpo-checkpoint>"` before `EVAL_METHOD=sampling_retry sbatch --export=ALL`; set `PROMPT_STYLE=cot_native` for the native-thinking GRPO model. Optional `RUN_NAME`, `N_CLAIMS`, `TEMPERATURE`, `TOP_P`, `THRESHOLD`, `MAX_NEW_TOKENS`, `SEED`, `EVAL_MODEL`, and `RESULT_DIR` override the defaults. The new run directory is `agent/evaluation/results/sampling_retry/runs/<eval_model>/<prompt>-cap<tokens>/<run>/` unless `RESULT_DIR` is set. Read `predictions.jsonl` for each claim's three draws and `metrics.json` for the comparison. The run refuses to overwrite an existing result directory.

After `sbatch` returns a job ID, check `squeue -j <job-id>` for a pending or running job. New jobs create `slurm-agent-eval-<job-id>.out` in the **directory where you ran `sbatch`** only when the job starts; read it there with `tail -f slurm-agent-eval-<job-id>.out`. Historical sampling-retry jobs retain their old log name. After the job leaves the queue, check `sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed` and the log for completion or errors.

The 900 dev claims are for **evaluation only**. Training a learned `STOP/RETRY` controller requires separate FEVER train claims and rollout data. Copying the heuristic's actions as labels would be imitation learning; it would not establish an RL result.

### Agent evidence batching and Jev-style controller evaluation

The [evidence-batch producer](agent/docs/evidence-batches.md) keeps the same title-plus-BM25 retrieval result, splits its sentences into batches of 10, and verifies **every** batch. It saves the full retrieval context, all batch outputs, and the union of valid citations under `agent/data/evidence_batches/`; it does not produce the final verifier result. The union can have **more than 10 sentences**. By default this run uses all 900 frozen dev claims, deterministic direct decoding and the direct 5,500-row SFT adapter. Check current SoC partition/resource limits before submitting:

```bash
RUN_NAME=deterministic-1-draw-per-chunk-900-claims-run2 sbatch --export=ALL agent/eval.sbatch
```

Read `agent/data/evidence_batches/<eval_model>/direct-cap160/<run>/{manifest.json,metrics.json,predictions.jsonl}`. Plain `sbatch agent/eval.sbatch` also works and saves under `job-<Slurm job ID>`. For another checkpoint, set `CHECKPOINTS=<adapter-path>` and optionally `EVAL_MODEL=<name>`. `EVIDENCE_BATCH_SIZE` changes the chunk size. Batch calls, prompt tokens, empty citation unions and gold evidence missing from the union are reported. This intermediate data is Git-ignored and can be reused by multiple controller evaluations.

The completed `full900-batches1` job used commit `176c90a` and originally wrote under the older `agent/data/evidence_batches/runs/` layout. Its organized location is `agent/data/evidence_batches/sft-direct-from-base-qwen3p5-2b-5500-samples/direct-cap160/deterministic-1-draw-per-chunk-900-claims/`; move the SoC output there once with the checked command below. The original manifest remains unchanged as provenance. It processed 900 claims and 3,169 ten-sentence chunks with **one deterministic verifier draw per chunk** (`do_sample=false`), and did not run final verification. Because the generated data directory is Git-ignored, `git pull` will not copy this completed run from SoC; copy it separately if claim-level analysis is needed locally. The three-draw `sampling_retry` experiment instead resamples the *full* retrieval context, so it is not a three-draw-per-chunk dataset.

On SoC, from `~/rl-test`, the following moves this completed run once without overwriting another run. It also removes only empty directories from the old layout:

```bash
(
  set -e
  src=agent/data/evidence_batches/runs/sft-direct-from-base-qwen3p5-2b-5500-samples/direct-cap160/full900-batches1
  dst=agent/data/evidence_batches/sft-direct-from-base-qwen3p5-2b-5500-samples/direct-cap160/deterministic-1-draw-per-chunk-900-claims
  test -f "$src/manifest.json" && test -f "$src/metrics.json" && test -f "$src/predictions.jsonl"
  test ! -e "$dst"
  python3 - "$src" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
manifest = json.loads((run / "manifest.json").read_text())
metrics = json.loads((run / "metrics.json").read_text())
assert manifest["n_claims"] == metrics["n_claims"] == 900
assert manifest["do_sample"] is False and manifest["batches_only"] is True
assert sum(1 for _ in (run / "predictions.jsonl").open()) == 900
PY
  mkdir -p "$(dirname "$dst")"
  mv -- "$src" "$dst"
  rmdir -- "$(dirname "$src")" "$(dirname "$(dirname "$src")")" agent/data/evidence_batches/runs 2>/dev/null || true
  echo "Moved to $dst"
  wc -l "$dst/predictions.jsonl"
)
```

Its aggregate result is 3.52 chunks per claim, 1.17 selected citation sentences per claim, 280/900 empty citation unions, and 431 claims with a complete gold evidence set in the union. The matched full-context sampling-retry run had 517 non-NEI claims with complete gold evidence in the retrieved context; because each union is a subset of that context, the batching/citation step lost complete gold coverage for 86 of those claims. These are **coverage and cost diagnostics**, not controller accuracy or final FEVER score. Claim-level failure analysis requires the Git-ignored `predictions.jsonl` from SoC.

The [Jev-style controller evaluator](agent/evaluation/results/rsi_jev_controller_baseline/README.md) reads the completed batch data. By default, [RSI-Jev v2.1 2B](https://huggingface.co/shgao/rsi-jev-v2.1-qwen3.5-2b) sees the claim and each batch verifier's verdict, format status, confidence and cited sentence IDs. The main comparison is **RSI-Jev controller versus a predeclared heuristic controller** making the same route choice from the same batch summaries. The heuristic uses empty-union, invalid-format, NEI, disagreement and verdict-token confidence below `0.5` to choose full context. Both controllers choose before Qwen runs a new final verification on each selected evidence set. Matching choices share one final Qwen call. Neither route decision sees gold labels or final Qwen outputs. Install the optional `rsi-jev` runtime in the SoC virtual environment before running; see the [controller instructions](agent/evaluation/results/rsi_jev_controller_baseline/README.md):

```bash
JEV_SOURCE_RUN=agent/data/evidence_batches/sft-direct-from-base-qwen3p5-2b-5500-samples/direct-cap160/deterministic-1-draw-per-chunk-900-claims \
EVAL_METHOD=rsi_jev_controller_baseline RUN_NAME=full900-run1 \
  sbatch --partition=gpu-long --time=12:00:00 --export=ALL agent/eval.sbatch
```

The batch producer writes under `agent/data/evidence_batches/`; controller results go under `agent/evaluation/results/rsi-jev-controller-baseline/runs/`. The Python module folder uses underscores because it is imported with `python -m`. Use `RESULT_DIR` to choose another exact new directory. New agent evaluation jobs write `slurm-agent-eval-<job-id>.out` in the submission directory; historical sampling-retry logs retain their old name.

The proposed RL pilot in [`agent/train/`](agent/train/) addresses a different decision: for each **fixed 10-sentence input**, decide whether its current verifier result is sufficient or the **same input** needs another sample, up to three draws. Every evidence chunk is still processed. The replay environment, Jev head optimizer and reward code are implemented; the train-only 5,500-claim rollout and a trained controller checkpoint have not yet been produced. Do not use the frozen 900 dev results for training.

To produce the shared **three stochastic draws per fixed ten-sentence chunk** on the frozen 900 dev claims, first inspect current SoC GPU limits with `sinfo -o "%P %l %a"`, then submit from the repository root:

```bash
EVAL_METHOD=evidence_batches EVIDENCE_BATCHES_ONLY=1 EVIDENCE_DRAWS=3 \
EVIDENCE_BATCH_SIZE=10 RUN_NAME=stochastic-3-draws-per-chunk-900-claims \
  sbatch --time=1-00:00:00 --export=ALL agent/eval.sbatch
```

The job uses `temperature=0.7`, `top_p=0.9` and seed 0 unless overridden. It saves under `agent/data/evidence_batches/<eval-model>/direct-cap160/stochastic-3-draws-per-chunk-900-claims/`. Each claim row contains `shown` plus `chunks[]`; each chunk contains its fixed `evidence`, `sampling_seed`, and three scored `draws`. No union, controller action or final verdict is selected in this dataset. If retrieval returns the same 3,169 chunks as the earlier run, expect 9,507 generated verifier outputs. `EVIDENCE_DRAWS=5` is supported for a later, separately named run; the current STOP/RETRY budget is three. This dev set is evaluation data only.

### DPO500 → GRPO500 versus DPO1000

Both arms start from the **same completed JSON SFT4500 adapter**. The existing SFT4500-disjoint 1,000-claim training pool is frozen by [`dpo-grpo-split.json`](verifier/data/train/sft/fever-oracle-matched-grpo-1000/dpo-grpo-split.json): 500 claim IDs for DPO followed by a disjoint 500 for GRPO. The other arm gives all 1,000 IDs to DPO. The completed SFT4500 → GRPO1000 run is a separate control. All arms use the same frozen 900 dev IDs for evaluation; dev predictions are never preference training data.

Generate **one** set of six native-thinking SFT4500 completions per claim and reuse it to build both DPO datasets. A DPO pair requires an actual correct-verdict, exact-oracle-citation completion and an actual wrong-verdict completion from the *same* prompt. Unfinished generations and evidence-only disagreements are excluded because the stored oracle is only one possible valid FEVER evidence set. Thus `DPO_POOL=500` or `1000` counts candidate **claims**, while the final experiment name records the smaller number of usable **pairs**. Inspect the generated pairs for reasoning faithfulness; the automated grader only checks answer and citations. GRPO samples new completions online on its own disjoint 500 claims and uses the existing verdict/evidence reward without confidence calibration.

On SoC, pull the branch, verify the SFT4500 adapter, `.venv`, FEVER dev file and BM25/SQLite indexes, and check current GPU limits with `sinfo -o "%P %l %a"`. From the repository root:

```bash
python -m utils.build_dpo_grpo_split --check
rollout_job=$(sbatch --parsable --export=ALL,ROLLOUT_NAME=shared1000-v1 verifier/dpo_rollouts.sbatch)
rollout_dir=verifier/data/train/dpo/rollouts-sft4500-pool1000/shared1000-v1
dpo500_job=$(sbatch --parsable --dependency=afterok:$rollout_job \
  --export=ALL,DPO_ROLLOUTS_DIR="$rollout_dir",DPO_POOL=500,RUN_NAME=dpo500-v1 \
  verifier/dpo_pilot.sbatch)
dpo1000_job=$(sbatch --parsable --dependency=afterok:$rollout_job \
  --export=ALL,DPO_ROLLOUTS_DIR="$rollout_dir",DPO_POOL=1000,RUN_NAME=dpo1000-v1 \
  verifier/dpo_pilot.sbatch)
grpo500_job=$(sbatch --parsable --dependency=afterok:$dpo500_job \
  --export=ALL,DPO_REPORT=verifier/data/train/dpo/pool500-dpo500-v1/train_report.json,RUN_NAME=grpo500-v1 \
  verifier/grpo_after_dpo.sbatch)
printf 'rollouts=%s dpo500=%s dpo1000=%s grpo500=%s\n' "$rollout_job" "$dpo500_job" "$dpo1000_job" "$grpo500_job"
```

Use fresh run names for any repeat. Rollouts and pair data are Git-ignored under `verifier/data/train/dpo/`. `pool500-dpo500-v1/pairs/manifest.json` and `pool1000-dpo1000-v1/pairs/manifest.json` give the actual pair counts and rejection categories. The DPO training report records its configuration and checkpoint path; the experiment is named `dpo-cot-from-sft4500-qwen3p5-2b-pool<claims>-<pairs>-samples`. Training metadata and logs are in `verifier/experiments/<config>/runs/<run>/artifacts/`. Dev metrics and predictions are in `verifier/evaluation/results/<config>/cot-native-cap1024/<run>/`. The DPO evaluator loads SFT then DPO adapters; the GRPO evaluator loads SFT, DPO, then GRPO adapters. `OUTPUT_ROOT` and `RESULTS_ROOT` can relocate those roots. Each job fails if required data or adapters are missing; the dependency prevents the GRPO job from starting until DPO500 finishes successfully.

Interpret the comparison as an allocation experiment, because DPO500 may yield substantially fewer than 500 trainable pairs while GRPO500 sees 500 claims. DPO1000 includes both halves and is a separate arm, so its 500 latter IDs are intentionally shared with GRPO500 across arms. Oracle-context training and title-retrieval dev evaluation also have a distribution shift. Record pair yield, label mix, token budget, runtime and the same dev metrics before attributing any gain to DPO or the sequence. The earlier 395 teacher-rejection pairs were withdrawn because their chosen and rejected answers reused the same rejected reasoning.

For independent pair curation after collecting multiple CoT completions for the **same train-only claim and exact user/system prompt**, save each unedited candidate as JSONL with `id`, `prompt`, `system_prompt`, `completion` and `model` (optional `run`). Then run:

```bash
python -m utils.build_dpo_pairs \
  --candidate path/to/train-rollouts-model-a.jsonl \
  --candidate path/to/train-rollouts-model-b.jsonl \
  --output verifier/data/train/dpo/<new-dataset-name>
```

The builder requires the same output format and no frozen dev IDs. Inspect reasoning faithfulness before using the resulting pairs for DPO. The side-by-side reviewer and model comparison are tracked in [verifier/TODO.md](verifier/TODO.md).

---

## Repository Workflow & Pipeline

The core repository workflow consists of data and retrieval index preparation, verifier training, and multi-condition evaluation.

### 1. Data & Retrieval Index Preparation

- **Python Environment**: Requires Python 3.10+ and PyTorch with CUDA/MPS/CPU support (`requirements.txt`).
- **FEVER Raw Data**: Download train/dev JSONL files and Wikipedia dump (~21 GB uncompressed):
  ```bash
  mkdir -p verifier/data/fever && cd verifier/data/fever
  curl -fLO https://fever.ai/download/fever/train.jsonl
  curl -fLO https://fever.ai/download/fever/shared_task_dev.jsonl
  curl -fLO https://fever.ai/download/fever/wiki-pages.zip
  unzip -q wiki-pages.zip && rm wiki-pages.zip
  cd ../../..
  ```
- **SQLite Database & BM25 Index**: Builds `pages.sqlite` (Wikipedia sentences) and `bm25_index/` (BM25 term index):
  ```bash
  python -m retrieval.build_fever_index --db verifier/data/fever/pages.sqlite --bm25-dir verifier/data/fever/bm25_index
  ```

### 2. Course Focus: RL-Driven vs Non-RL Architectural Contributions

As this project serves as a Reinforcement Learning research study, our engineering and methodological contributions are strictly partitioned into **RL-Driven Innovations** (the core research subject) and **Non-RL Structural Foundations** (the enabling infrastructure):

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     RL-Driven Contributions (Course Core)                       │
│  • Rule-Based Verifiable Rewards (RLVR): verdict harm, evidence F1, format pen. │
│  • On-Policy Reasoning Exploration: GRPO group rollouts vs static SFT CoT       │
│  • Advanced RLVR Algorithmic Frontier: Dr. GRPO (σ-removal) & DAPO (sampling)   │
│  • Planned Offline Preference Alignment: DPO on genuine paired CoT rollouts     │
│  • Exploration-Exploitation Search Loop: Dynamic multi-turn retrieval (agent/)  │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ Built on top of
┌──────────────────────────────────────▼──────────────────────────────────────────┐
│               Non-RL Architectural & Structural Foundations                     │
│  • Causal Attention Reordering: Evidence-First Tag format (<evidence> → <verdict>)│
│  • Decoupled Representation Logits: Eliminating textual JSON confidence tax     │
│  • Prefix Bidirectional Attention: Simulating T5 all-to-all encoder in prefill  │
│  • Data Hygiene & Split Manifests: 5,500 clean repair & 4,500/1,000 partitions │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Glossary of Terms & Metrics

To ensure transparent cross-team readability and consistent terminology, the key methods, environments, partitions, and metrics used throughout this repository are defined below:

#### Training Paradigms & Algorithms
- **`sft-direct`**: Direct Supervised Fine-Tuning without intermediate reasoning. The model is trained directly on input claim-evidence pairs to emit the final verdict and evidence citations immediately, offering fast inference but limited attention conditioning.
- **`sft-cot`**: Chain-of-Thought Supervised Fine-Tuning via teacher imitation. The model is fine-tuned to reproduce step-by-step explanatory rationales (distilled from Qwen-27B or Gemini) prior to generating the verdict.
- **`grpo-cot`**: Group Relative Policy Optimization (RLVR) with native `<think>` tokens. Bootstrapped from an SFT checkpoint, the model actively explores reasoning paths via group rollouts ($G=8$) and optimizes against verifiable, rule-based reward functions (verdict harm penalty, evidence F1, and syntax compliance).
- **`dr-grpo`**: A mathematically stabilized GRPO variant that removes the division by group reward standard deviation ($\sigma_R$), preventing gradient variance explosion on homogeneous rollouts and eliminating token-length bias.
- **`dapo`**: Decoupled Clip and Dynamic Sampling Policy Optimization; an advanced RLVR method that discards uninformative unanimous rollout batches ($\sigma_R = 0$) and decouples the upper and lower PPO clipping thresholds.
- **`dpo`**: Direct Preference Optimization; a planned offline experiment using genuine chosen/rejected CoT rollouts for the same training claim and prompt. No valid DPO pair dataset is currently available; the previous teacher-rejection pairs were withdrawn.

#### Retrieval Environments & Conditions
- **`oracle`**: An evaluation setting where the ground-truth Wikipedia article is guaranteed to be present in the verifier's input context alongside title-matched distractor pages, isolating verification reasoning from retrieval quality.
- **`titles`**: A realistic, end-to-end evaluation condition that queries the corpus via Wikipedia article title exact-matching combined with BM25 sentence retrieval.
- **`bm25`**: A baseline information-retrieval condition relying solely on keyword BM25 score ranking across the corpus without article title priors.

#### Dataset Partitions & Experimental Arms
- **`disjoint split`**: A strict claim-level dataset partitioning scheme (e.g., 4,695/1,200 or 4,500/1,000) that guarantees zero claim ID overlap between training stages, preventing data leakage.
- **`seen arm`**: A control experiment setting where Stage 2 training (continuation SFT or RL) operates on claim IDs already seen during Stage 1 SFT, measuring memorization and in-distribution policy optimization.
- **`unseen arm`**: The primary benchmark setting where Stage 2 models are exposed strictly to novel, held-out claim IDs, evaluating out-of-distribution reasoning generalization and zero-shot transfer.

#### Evaluation Metrics & Scoring
- **`accuracy`**: The fraction of evaluation claims where the model's predicted verdict matches the ground-truth label ($V = V^*$).
- **`fever_score`**: The strict official FEVER benchmark metric, which requires **both** an accurate verdict AND a complete citation of the gold evidence sentence set ($V = V^* \land E^* \subseteq E$).
- **`evidence_f1`**: The sentence-level harmonic mean of precision and recall between cited evidence IDs and gold evidence sets on non-NEI claims.
- **`format_adherence` (`format_rate`)**: The percentage of completions that strictly conform to parseable syntax (valid JSON or well-formed Native Tags) without truncated or malformed tokens.
- **`token_conf` / `ECE`**: Calibrated confidence and Expected Calibration Error computed directly from the normalized softmax logits over the model's first verdict token (extracted from the pre-softmax vocabulary head).
- **`stated_conf` / `ECE`**: Confidence and Expected Calibration Error evaluated from a numeric float explicitly verbalized in the model's generated text (observed to suffer from severe overconfidence and poor calibration).

#### Target Classification Labels
- **`SUPPORTED`**: The claim is fully affirmed and proven by the provided evidence (corresponds to FEVER's `SUPPORTS`).
- **`REFUTED`**: The claim is contradicted and disproven by the provided evidence (corresponds to FEVER's `REFUTES`).
- **`NOT_ENOUGH_INFO` (`NEI`)**: The provided evidence lacks sufficient information to either confirm or deny the claim (corresponds to FEVER's `NOT ENOUGH INFO`).

---

### 4. Verifier Training & Methodology Taxonomy

The verifier fine-tunes `Qwen/Qwen3.5-2B` using parameter-efficient LoRA (rank 16) across two distinct training paradigms and two output representations:

#### Method Taxonomy: What Do `sft-direct`, `sft-cot`, and `grpo-cot` Mean?

| Training Method | Description | Input $\to$ Output Flow | Strengths & Trade-offs |
|---|---|---|---|
| **`sft-direct`** | **Direct Supervised Fine-Tuning** (no reasoning tokens). Fine-tunes the base LLM directly on claim and evidence text to emit the final verdict and evidence citations immediately. | $\text{Claim} + \text{Evidence} \to \text{Verdict} + \text{Evidence IDs}$ | **Pros**: Extremely fast inference, low compute, high raw classification accuracy (76.1% on 5500), and strong refutation recall.<br>**Cons**: Narrow attention bottleneck; decoder cannot cite evidence after blindly committing to a verdict; higher false abstention (19.8%). |
| **`sft-cot`** | **Chain-of-Thought Supervised Fine-Tuning** (teacher imitation). Supervised training on teacher-generated explanations (from Qwen-27B / Gemini) prior to emitting the verdict. | $\text{Claim} + \text{Evidence} \to \text{Reasoning} \to \text{Verdict} + \text{Evidence IDs}$ | **Pros**: Provides step-by-step explanatory provenance and sentence matching.<br>**Cons**: Pure supervised imitation suffers from forced rationalization, reducing NEI recall (drops to 50.3%) without reinforcement learning. |
| **`grpo-cot`** | **Group Relative Policy Optimization with Native Thinking** (RLVR). Bootstrapped from an SFT checkpoint, the model explores reasoning trajectories via group rollouts ($G=8$) optimized under verifiable rule-based rewards (verdict harm penalty, evidence F1, format compliance). | $\text{Claim} + \text{Evidence} \to \langle\text{think}\rangle \dots \langle/\text{think}\rangle \to \text{Evidence} + \text{Verdict}$ | **Pros**: Peak repository performance (**76.2% Accuracy, 73.7% Evidence F1** on Unseen Arm); reverses NEI regression (75.0% recall); grounds reasoning via verifiable rewards.<br>**Cons**: Longer training time (~6–10h on A100). |
| **`dr-grpo` / `dapo`** | **Advanced RLVR Variants** (addressing group variance $\sigma_R$ and length bias). | Group-relative advantage with $\sigma_R$ removal (`dr-grpo`) and dynamic sampling of non-unanimous batches (`dapo`). | Eliminates gradient explosion/vanishing on unanimous batches ($\sigma_R = 0$) and strips out token-level length biases. |

#### Dual Output Formats: JSON vs Native Tag Ordering

1. **Path 1 — Standard JSON Format** (Baseline & Historical Compatibility):
   ```json
   {"verdict": "SUPPORTED", "evidence": ["E1", "E3"], "confidence": 1.0}
   ```
   Maintained across `fever-oracle-disjoint-4500` and `fever-oracle-matched-grpo-1000` to preserve continuity with historical benchmarks.

2. **Path 2 — Native Tag Format** (Evidence-First Causal Attention):
   ```text
   <think>
   [Step-by-step evidence verification]
   </think>
   <evidence>E1, E3</evidence>
   <verdict>SUPPORTED</verdict>
   ```
   Maintained across `fever-oracle-tagged-4500` and `fever-oracle-tagged-grpo-1000`. Citing `<evidence>` *before* `<verdict>` allows the causal decoder's self-attention heads to condition the verdict directly on the cited evidence, eliminating JSON syntax token tax.

- **Outputs**: LoRA weights, metadata (`meta.json`), and step logs (`train_log.json`) save under `verifier/experiments/<config>/runs/<run>/artifacts/`.

### 5. Verifier Evaluation & Scoring

Evaluates claim verification on frozen dev claim sets (default 900 claims with title retrieval):
- **Prompt Styles**:
  - `direct`: Output concise JSON directly.
  - `cot_brief`: Output concise evidence reasoning (<=50 words) within the JSON object.
  - `cot_chain` (`cot_long`): Output full evidence reasoning chain within the JSON object.
  - `cot_native`: Output native model thinking tokens (`<think>...</think>`), followed by JSON.
- **Evaluation Results Directory Layout (`verifier/evaluation/results/`)**:
  Evaluation results follow a hierarchical structure mirroring training configurations:
  ```text
  verifier/evaluation/results/
  ├── <eval_model>/               # e.g., base-qwen3p5-2b or sft-direct-from-base-qwen3p5-2b-5500-samples
  │   └── <condition>/            # <prompt_style>-cap<tokens>, e.g., direct-cap1024, cot-brief-cap1024
  │       └── <run>/              # Run directory (e.g. run1, job-12345) containing:
  │           ├── manifest.json   # Reproducibility metadata, command, commit, and file paths
  │           ├── metrics.json    # Accuracy, FEVER score, format compliance, evidence F1, calibration ECE
  │           ├── predictions.jsonl # Claim-by-claim predictions and citation outputs
  │           └── logs/run.log    # Evaluation terminal log
  └── archive/                    # Historical repository of raw per-claim .jsonl prediction files
                                  # from early project baselines and probes (including local_n900_titles.jsonl,
                                  # from which the frozen 900 dev set IDs were established).
  ```
- **Metrics Tracked**:
  - `accuracy`: Overall verdict classification accuracy.
  - `fever_score`: Strict FEVER metric (requires both correct verdict and full gold evidence citation).
  - `format_rate`: JSON parse and field adherence rate.
  - `evidence_f1`: Evidence sentence retrieval F1 on non-NEI claims.
  - `calibration`: Brier score and Expected Calibration Error (ECE) for verdict-token probability and stated text confidence.
- **Offline Scoring Script**:
  ```bash
  python -m verifier.evaluation.score_predictions <predictions.jsonl>
  python -m unittest verifier.tests.test_score_predictions
  ```

### 4. Experiment Catalogue

| Configuration in `verifier/experiments/` | Description |
|---|---|
| `sft-direct-from-base-qwen3p5-2b-5895-samples` | 2B no-CoT SFT, 5,895 usable FEVER claims (main SFT baseline) |
| `sft-direct-from-base-qwen3p5-2b-disjoint-4695-samples` | 2B SFT on the 4,695 disjoint split (unseen GRPO starting checkpoint) |
| `sft-direct-from-sft5895-qwen3p5-2b-1200-samples` | 2B continued SFT on the 1,200-claim second-stage set (seen SFT control) |
| `grpo-cot-from-sft5895-qwen3p5-2b-no-calibration-kl0-1200-samples` | 2B native-thinking CoT GRPO from SFT 5895 (seen arm) |
| `grpo-cot-from-sft4695-qwen3p5-2b-no-calibration-kl0-1200-samples` | 2B native-thinking CoT GRPO from SFT 4695 (unseen arm) |
| `sft-cot-from-base-qwen3p5-2b-5500-samples` | 2B CoT SFT on 5,500 validated and repaired Qwen-27B teacher explanations |
| `sft-direct-from-base-qwen3p5-2b-5500-samples` | 2B direct SFT baseline on the exact same 5,500 claims |
| `sft-direct-from-base-qwen3p5-4b-qlora-5895-samples` | 4B QLoRA SFT on 5,895 examples |

---

## Running on NUS SoC Cluster

If local GPU resources are unavailable (e.g. developing on a laptop or Mac), push your code changes to GitHub and use NUS SoC compute cluster GPU nodes for training and evaluation.

### 1. Prerequisites & Access Setup

1. **Create an SoC account**: Register your account at [MySoC Account Creation](https://mysoc.nus.edu.sg/app/newacct/).
2. **Connect to NUS VPN**: If working outside the NUS network, connect via [NUS VPN](https://nusit.nus.edu.sg/services/wifi_and_network/vpn/).
3. **SSH into the SoC cluster**:
   - **Direct login to Slurm compute nodes** (recommended; required for `sbatch`):
     ```bash
     ssh <soc_account>@xlogin.comp.nus.edu.sg
     ```
   - **Via jump host** (if accessing student general servers):
     ```bash
     ssh -J <soc_account>@stujump.comp.nus.edu.sg <soc_account>@stu.comp.nus.edu.sg
     ```
   > [!IMPORTANT]
   > Always submit Slurm jobs (`sbatch`) from an SoC login node (`xlogin.comp.nus.edu.sg`), not from `stu` or `stu2`.

### 2. Local-to-Cluster Development Workflow

Standard development and experiment cycle:

1. **Commit and push locally**:
   ```bash
   git add .
   git commit -m "commit message"
   git push origin rl
   ```
2. **SSH into the SoC login node**:
   ```bash
   ssh <soc_account>@xlogin.comp.nus.edu.sg
   ```
3. **Navigate to the cluster repository and pull updates**:
   ```bash
   cd "$HOME/rl-test"
   git pull
   ```
4. **Launch GPU jobs via Slurm** (see commands below).

### 3. SoC Partitions & GPU Guidelines

| Partition | Max Wall Time | Usage |
|---|---:|---|
| `gpu` | 3 hours | Evaluation jobs (~90 min requested) |
| `gpu-long` | 3 days | Training jobs (~3–6 hours requested) |
| `normal` | 3 hours | CPU index build (~13 minutes observed) |

- **Single GPU**: Evaluation and training scripts each run **one Python process on one GPU** (`--gpus=1`). Requesting multiple cards will not speed up execution.
- **GPU types**: Use `a100-40:1` (default), `a100-80:1`, `h100-47:1`, or `h100-96:1` which support `bfloat16`. Avoid `nv` (which may allocate older cards like Titan RTX on `xgpe2`).
- **Disk quota and package installs**: SoC may report `Disk quota exceeded` while `df -h /tmp` still shows free space; the login node's per-user `/tmp` quota can be smaller. Set `TMPDIR` to a writable directory under home **before** any large `pip install` or Git-based package install, and before submitting jobs. `pip --no-cache-dir` disables the download cache but does not move pip's temporary Git checkout. Use this as the first remedy for `/tmp` quota errors; if home itself lacks quota, choose another permitted filesystem with enough space.
  ```bash
  mkdir -p "$HOME/.pip-tmp"
  export TMPDIR="$HOME/.pip-tmp"
  ```
- **Optional Jev controller runtime**: Install it into the same `.venv` used by `agent/eval.sbatch` and `agent/train.sbatch`, after the root requirements. The model weights download separately on first use.
  ```bash
  cd "$HOME/rl-test"
  source .venv/bin/activate
  mkdir -p "$HOME/.pip-tmp"
  export TMPDIR="$HOME/.pip-tmp"
  python -m pip install --no-cache-dir -r agent/requirements.txt
  python -c 'import rsijev; print("RSI-Jev: OK")'
  ```

### 4. Cluster Job Submission

#### A. Submit CPU index build job
```bash
export TMPDIR="$HOME/.pip-tmp"
job_id=$(sbatch --parsable --export=ALL retrieval/build_fever_index.sbatch)
echo "Index Job: $job_id"
squeue -j "$job_id"
```

#### B. Submit agent sampling and retry evaluation
```bash
job=$(EVAL_METHOD=sampling_retry sbatch --parsable --export=ALL agent/eval.sbatch)
echo "Agent evaluation job: $job"
squeue -j "$job"
```

#### C. Submit a verifier training configuration (`verifier/train.sbatch`)
```bash
# Example: train the direct 5,895-row SFT configuration with a new run name.
TRAIN_SCRIPT=verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/train.py \
RUN_NAME=run3 sbatch --partition=gpu --gpus=a100-40:1 --time=03:00:00 \
  --export=ALL verifier/train.sbatch
```

#### D. Submit single model evaluation job (`verifier/evaluation/evaluate.sbatch`)
```bash
# Example: evaluate SFT checkpoint under cot_brief prompt
adapter=verifier/experiments/sft-direct-from-base-qwen3p5-2b-5500-samples/runs/run1/artifacts/checkpoints
sbatch --partition=gpu --gpus=a100-40:1 --time=01:30:00 \
  --export=ALL,N_CLAIMS=900,RUN_NAME=run1,PROMPT_STYLE=cot_brief,CHECKPOINTS="$adapter" \
  verifier/evaluation/evaluate.sbatch
```

#### E. Job monitoring and status checks
```bash
squeue -u "$USER"                                                               # Check running/queued jobs
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS         # Check completed job status and resource usage
cat slurm-split-followup-<job_id>_0.out                                         # Inspect terminal output log
```
