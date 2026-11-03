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
#   CONTROLLER=local:<imitation checkpoints>   GRPO=<grpo run checkpoints>   (omit GRPO for the baseline)
# GRPO is a separate variable because sbatch --export splits values on commas.
# Exits at once when the controller's checkpoints do not exist yet (a training chain still running).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
source .venv/bin/activate
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}" TOKENIZERS_PARALLELISM=false
CONTROLLER="${CONTROLLER:?}${GRPO:+,$GRPO}"
last="${CONTROLLER##*,}"; last="${last#local:}"
test -f "$last/adapter_model.safetensors" || { echo "no adapter at $last yet; nothing to evaluate"; exit 0; }
# A decompose on a claim missing from the cache calls OpenRouter; there is no key on SoC.
# The committed flash prompt-v2 cache covers all 900 IDs; agent.prompted looks for it under
# the manifest-tagged name, so copy it there when that file is missing.
dec=verifier/data/decompose
tagged=$dec/fever_dev_qwen_qwen3.8-flash_n900_promptv2_fever-dev-900-ids-ac9cfead.jsonl
[[ -f "$tagged" ]] || cp "$dec/fever_dev_qwen_qwen3.8-flash_n900_promptv2.jsonl" "$tagged"
echo "job=$SLURM_JOB_ID commit=$(git rev-parse HEAD) node=$(hostname) controller=$CONTROLLER"
# CLAIM_IDS: default the frozen dev 900; verifier/data/eval/fever-evaluation-v1/holdout-3000-ids.json for
# the holdout (finalized configurations only). The offline wrapper never calls the decomposer API.
python -m verifier.evaluation.eval_controller_offline --tools follow --chunk 64 --controller "$CONTROLLER" \
  --adapter verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints \
  --claim-ids "${CLAIM_IDS:-verifier/data/eval/fever-dev-900-ids.json}"
