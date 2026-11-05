"""Controller GRPO short run (handover section 6, step 2).

300 unused train claims (seed 4; the pilot used seed 3) x 8 sampled rollouts at
temperature 1.0, 8 claims per update (38 updates), imitation controller as start and
KL reference (beta 0.04), verifier 5895/run1 frozen. Reward: FEVER - 0.01 * n_ops -
0.5 per invalid controller reply. Advantage = reward - group mean, without std
scaling: in the pilot 59 of 64 claims had some reward spread but only 17 had FEVER
spread, so std scaling would turn operation-count noise into full-size advantages.

Check: train reward rises and the invalid-output rate does not.

    sbatch --partition=gpu --gpus=a100-40:1 --time=03:00:00 \
      --export=ALL,TRAIN_SCRIPT=verifier/experiments/grpo-controller-from-sftctl18291-qwen3p5-2b-fever-ops001-nostd-kl004-300-samples/train.py,RUN_NAME=run1 \
      verifier/experiments/run_train.sbatch
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verifier.experiments._shared.grpo_controller import main

if __name__ == "__main__":
    main({"config_name": Path(__file__).parent.name, "n_train": 300, "train_seed": 4,
          "group_size": 8, "claims_per_step": 8, "temperature": 1.0, "ops_cost": 0.01,
          "invalid_penalty": 0.5, "no_std_scale": True, "lr": 1e-5, "beta": 0.04,
          "save_steps": 10})
