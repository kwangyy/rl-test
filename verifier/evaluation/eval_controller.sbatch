#!/usr/bin/env bash
#SBATCH --job-name=eval-ctl
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --gpus=a100-40:1
#SBATCH --output=slurm-%x-%j.out
# Dev 900 evaluation of a controller in the prompted follow loop with the 5895/run1
# verifier, as the handover's section 4. agent.prompted skips claims already in its
# trace, so chain two jobs with --dependency=afterany if one does not finish in 3 h.
#   CONTROLLER=local:<imitation checkpoints>,<grpo run checkpoints>  (or local:<imitation> for the baseline)
# Exits at once when the controller's checkpoints do not exist yet (a training chain still running).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
source .venv/bin/activate
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}" TOKENIZERS_PARALLELISM=false
last="${CONTROLLER:?}"; last="${last##*,}"; last="${last#local:}"
test -f "$last/adapter_model.safetensors" || { echo "no adapter at $last yet; nothing to evaluate"; exit 0; }
echo "job=$SLURM_JOB_ID commit=$(git rev-parse HEAD) node=$(hostname) controller=$CONTROLLER"
python -m agent.prompted --tools follow --chunk 64 --controller "$CONTROLLER" \
  --adapter verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints \
  --claim-ids verifier/data/eval/fever-dev-900-ids.json
