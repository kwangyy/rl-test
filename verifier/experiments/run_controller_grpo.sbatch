#!/usr/bin/env bash
#SBATCH --job-name=ctl-grpo
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --gpus=a100-40:1
#SBATCH --output=slurm-%x-%j.out
# Controller GRPO, one link of a resumable chain. SoC A100 jobs stop at 3 h, so each job
# stops cleanly before 2 h 50 min with a resume state and the next job continues it;
# a job after the run has finished exits at once. A100-40 only: training fails in fla's
# backward on H100, and the xgpj a100-80 nodes cannot load this .venv. Runs python
# directly (srun rejects some nodes' default CPU count). Submit from the repo root:
#
#   cfg=verifier/experiments/<grpo-controller config>/train.py
#   j=$(sbatch --parsable --export=ALL,TRAIN_SCRIPT=$cfg,RUN_NAME=run1 verifier/experiments/run_controller_grpo.sbatch)
#   for i in 2 3 4; do j=$(sbatch --parsable --dependency=afterany:$j --export=ALL,TRAIN_SCRIPT=$cfg,RUN_NAME=run1 verifier/experiments/run_controller_grpo.sbatch); done
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
test -f README.md || { echo "Submit from the repository root" >&2; exit 2; }
source .venv/bin/activate
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}" TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
case "${TRAIN_SCRIPT:?}" in verifier/experiments/grpo-controller-*/train.py) ;; *) echo "TRAIN_SCRIPT must be a grpo-controller config" >&2; exit 2 ;; esac
echo "job=$SLURM_JOB_ID commit=$(git rev-parse HEAD) node=$(hostname) script=$TRAIN_SCRIPT run=${RUN_NAME:?}"
python "$TRAIN_SCRIPT" --run-name "$RUN_NAME" --resume --time-budget "${TIME_BUDGET:-10200}" ${EXTRA_ARGS:-}
