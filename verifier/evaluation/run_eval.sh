#!/usr/bin/env bash
set -euo pipefail

# THINKING is the historical on/off switch. PROMPT_STYLE selects an exact style.
# Only PROMPT_STYLE=cot_native enables model-internal thinking.
thinking="${THINKING:-off}"
if [[ -n "${PROMPT_STYLE:-}" ]]; then
  case "$PROMPT_STYLE" in
    direct|cot_chain|cot_brief|cot_sft|cot_native) style="$PROMPT_STYLE" ;;
    *) echo "PROMPT_STYLE must be direct, cot_chain, cot_brief, cot_sft, or cot_native" >&2; exit 2 ;;
  esac
else
  case "$thinking" in
    off) style=direct ;;
    on) style=cot_chain ;;
    *) echo "THINKING must be on or off" >&2; exit 2 ;;
  esac
fi
case "$style" in
  direct) condition=baseline-direct; default_cap=160 ;;
  cot_chain) condition=baseline-cot-long; default_cap=512 ;;
  cot_brief) condition=baseline-cot-brief; default_cap=512 ;;
  cot_sft) condition=baseline-cot-sft; default_cap=320 ;;
  cot_native) condition=baseline-cot-native; default_cap=1024 ;;
esac

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}"
mkdir -p "$TMPDIR"
test -f README.md || { echo "Submit from the repository root (README.md missing)" >&2; exit 2; }
test -f verifier/data/fever/shared_task_dev.jsonl || { echo "Missing FEVER dev claims; see README Data preparation" >&2; exit 2; }
test -f verifier/data/fever/pages.sqlite || { echo "Missing SQLite index; run retrieval/build_fever_index.sbatch" >&2; exit 2; }
test -f verifier/data/fever/bm25_index/doc_ids.json || { echo "Missing or incomplete BM25 index; check retrieval/build_fever_index.sbatch" >&2; exit 2; }
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

claim_ids="${CLAIM_IDS:-}"
selection_args=()
if [[ -n "$claim_ids" ]]; then
  n="$(python -c 'import sys; from utils.fever_eval import read_id_manifest; print(len(read_id_manifest(sys.argv[1])["ids"]))' "$claim_ids")"
  selection_args=(--claim-ids "$claim_ids")
else
  n="${N_CLAIMS:-900}"
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "N_CLAIMS must be a positive integer" >&2; exit 2; }
  if [[ "$n" == 900 ]]; then
    claim_ids=verifier/data/eval/fever-dev-900-ids.json
  fi
fi
job_id="${SLURM_JOB_ID:?}"
run_name="${RUN_NAME:-job-$job_id}"
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "RUN_NAME must use letters, digits, underscore or dash" >&2; exit 2; }

checkpoints="${CHECKPOINTS:-}"
adapter_args=()
generation_args=()
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
  [[ "$MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
    echo "MAX_NEW_TOKENS must be a positive integer" >&2
    exit 2
  }
  generation_args=(--max-new-tokens "$MAX_NEW_TOKENS")
fi
model="Qwen/Qwen3.5-2B"
if [[ -n "$checkpoints" ]]; then
  [[ -f "$checkpoints/adapter_config.json" ]] || { echo "Missing $checkpoints/adapter_config.json" >&2; exit 2; }
  [[ -f "$checkpoints/adapter_model.safetensors" || -f "$checkpoints/adapter_model.bin" ]] || {
    echo "Missing adapter weights in $checkpoints" >&2; exit 2;
  }
  model="$(python - "$checkpoints/adapter_config.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding='utf-8') as f:
    print(json.load(f)['base_model_name_or_path'])
PY
)"
  adapter_args=(--adapter "$checkpoints")
  condition="${condition/baseline-/checkpoint-}"
fi

condition="${condition}-cap${MAX_NEW_TOKENS:-$default_cap}"

if [[ -n "${EVAL_CONDITION:-}" ]]; then
  [[ "$EVAL_CONDITION" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
    echo "EVAL_CONDITION must use letters, digits, underscore or dash" >&2; exit 2;
  }
  condition="$EVAL_CONDITION"
fi

if [[ -n "${RESULT_DIR:-}" && -n "${RESULTS_ROOT:-}" ]]; then
  echo "Set either RESULT_DIR (exact run directory) or RESULTS_ROOT (parent of condition/run), not both" >&2
  exit 2
fi
if [[ -n "${RESULT_DIR:-}" ]]; then
  result_dir="${RESULT_DIR%/}"
  [[ -n "$result_dir" && "$result_dir" != "." ]] || { echo "RESULT_DIR must name a new run directory" >&2; exit 2; }
else
  results_root="${RESULTS_ROOT:-verifier/evaluation/results}"
  result_dir="${results_root%/}/$condition/$run_name"
fi
if [[ -e "$result_dir" || -L "$result_dir" ]]; then
  echo "Result directory already exists: $result_dir (choose another RUN_NAME or RESULT_DIR)" >&2
  exit 2
fi
mkdir -p "$result_dir/logs"
output="$result_dir/predictions.jsonl"
metrics="$result_dir/metrics.json"
log="$result_dir/logs/run.log"
commit="$(git rev-parse HEAD)"
{
  echo "job=$job_id run=$run_name commit=$commit n=$n claim_ids=${claim_ids:-implicit-sampling} thinking=$thinking prompt_style=$style max_new_tokens=${MAX_NEW_TOKENS:-default} retrieval=titles model=$model checkpoints=${checkpoints:-none} result_dir=$result_dir"
  echo "started_at=$(date -u +%FT%TZ)"
  srun python -m verifier.evaluation.eval_verifier --model local --n "$n" --retrieval titles \
    --prompt-style "$style" --output "$output" "${generation_args[@]}" "${adapter_args[@]}" "${selection_args[@]}"
  if [[ -n "$claim_ids" ]]; then
    python - "$output" "$claim_ids" <<'PY'
import json
import sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
expected = json.loads(Path(sys.argv[2]).read_text())['ids']
actual = [row['id'] for row in rows]
if len(actual) != len(expected) or len(set(actual)) != len(expected) or set(actual) != set(expected):
    raise SystemExit('run does not match the selected frozen FEVER dev ID manifest')
PY
  fi
  python -m verifier.evaluation.score_predictions "$output" > "$metrics"
  echo "finished_at=$(date -u +%FT%TZ)"
  echo "predictions=$output metrics=$metrics"
} 2>&1 | tee "$log"
python - "$result_dir/manifest.json" "$job_id" "$run_name" "$commit" "$model" "$checkpoints" "$thinking" "$style" "$n" "$output" "$metrics" "$log" "$condition" "$result_dir" "${MAX_NEW_TOKENS:-}" "$claim_ids" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(_, manifest, job_id, run_name, commit, model, checkpoints, thinking,
 style, n, predictions, metrics, log, condition, result_dir, max_new_tokens, claim_ids) = sys.argv
record = {
    'job_id': job_id, 'run_name': run_name, 'commit': commit,
    'model': model, 'checkpoints': checkpoints or None,
    'thinking': 'on' if style == 'cot_native' else thinking,
    'native_thinking': style == 'cot_native',
    'prompt_style': style, 'retrieval': 'titles',
    'max_new_tokens': int(max_new_tokens) if max_new_tokens else {
        'direct': 160, 'cot_chain': 512, 'cot_brief': 512, 'cot_sft': 320,
        'cot_native': 1024,
    }[style],
    'condition': condition, 'result_dir': result_dir,
    'claim_ids_manifest': claim_ids or None,
    'claim_ids_sha256': hashlib.sha256(Path(claim_ids).read_bytes()).hexdigest() if claim_ids else None,
    'n_claims': int(n), 'predictions': predictions, 'metrics': metrics,
    'log': log,
}
Path(manifest).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
PY
