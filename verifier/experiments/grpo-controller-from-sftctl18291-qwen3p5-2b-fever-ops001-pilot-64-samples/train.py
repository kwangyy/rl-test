"""Controller GRPO pilot (handover section 6, step 1): no update, only the reward-spread gate.

64 unused train claims (seed 3) x 8 sampled rollouts of the imitation controller at
temperature 1.0, verifier 5895/run1 frozen. Reward: FEVER - 0.01 * n_ops - 0.5 per
invalid controller reply. Reports how many claims have rewards that differ; if almost
none do, GRPO has no signal (raise the temperature or change the claims first).

    sbatch --partition=gpu --gpus=a100-40:1 --time=03:00:00 \
      --export=ALL,TRAIN_SCRIPT=verifier/experiments/grpo-controller-from-sftctl18291-qwen3p5-2b-fever-ops001-pilot-64-samples/train.py,RUN_NAME=pilot-t1p0 \
      verifier/experiments/run_train.sbatch
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verifier.experiments._shared.grpo_controller import main

if __name__ == "__main__":
    main({"config_name": Path(__file__).parent.name, "rollouts_only": True, "n_train": 64,
          "train_seed": 3, "group_size": 8, "claims_per_step": 8, "temperature": 1.0,
          "ops_cost": 0.01, "invalid_penalty": 0.5})
