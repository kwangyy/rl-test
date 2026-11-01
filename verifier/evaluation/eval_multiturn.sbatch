#!/usr/bin/env bash
#SBATCH --job-name=eval-multiturn
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus=a100-40:1
#SBATCH --output=slurm-%x-%j.out

# Multi-turn (search-calling) verifier evaluation; see verifier/evaluation/eval_multiturn.py.
#   sbatch --export=ALL,ADAPTER=<sft>[,<grpo>],EVAL_CONDITION=<condition>,RUN_NAME=<run> \
#     verifier/evaluation/eval_multiturn.sbatch
# Optional: ADAPTER (unset for the base model), N_CLAIMS (smoke test), IDS (manifest; default
# the 900 dev IDs), RESULTS_ROOT.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
test -f README.md || { echo "Submit from the repository root (README.md missing)" >&2; exit 2; }
source .venv/bin/activate
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}" TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"

: "${EVAL_CONDITION:?set EVAL_CONDITION, e.g. grpo-multiturn-from-sft4500-qwen3p5-2b-no-calibration-bm25k5-kl0-1000-samples}"
run_name="${RUN_NAME:-job-${SLURM_JOB_ID:?}}"
output="${RESULTS_ROOT:-verifier/evaluation/results}/$EVAL_CONDITION/multiturn-bm25-k5-calls2/$run_name"
args=(--method bm25 --search-k 5 --max-calls 2 --output "$output")
[[ -n "${ADAPTER:-}" ]] && args+=(--adapter "$ADAPTER")  # unset: the untrained base model
[[ -n "${N_CLAIMS:-}" ]] && args+=(--n "$N_CLAIMS")
[[ -n "${IDS:-}" ]] && args+=(--ids "$IDS")

echo "job=$SLURM_JOB_ID commit=$(git rev-parse HEAD) adapter=${ADAPTER:-none} output=$output"
# Plain python, not srun: srun rejects the job when the submitting shell exported a different
# SLURM_CPUS_PER_TASK, and a single process on one GPU needs no step launcher.
python -m verifier.evaluation.eval_multiturn "${args[@]}"
