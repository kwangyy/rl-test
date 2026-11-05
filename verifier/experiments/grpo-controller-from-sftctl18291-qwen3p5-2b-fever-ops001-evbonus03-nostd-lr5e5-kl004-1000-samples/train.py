"""Controller GRPO with a reward for the controller's own job: + 0.3 when the gold evidence reaches the verifier.

Matched to follow-up A (1,000 claims seed 5, lr 5e-5, beta 0.04); only the reward changes:
FEVER - 0.01 * n_ops - 0.5 per invalid reply + 0.3 when a full gold evidence set was in the
input of some verify call (non-NEI claims). About 80% of the imitation controller's dev misses
happen after the gold evidence reached the verifier or on gold-NEI claims, so the FEVER-only
reward mostly measures the frozen verifier; the bonus rewards finding and passing on evidence,
which the controller controls. The bonus is below the FEVER reward, so a right answer still
pays more than delivering evidence alone.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verifier.experiments._shared.grpo_controller import main

if __name__ == "__main__":
    main({"config_name": Path(__file__).parent.name, "n_train": 1000, "train_seed": 5,
          "group_size": 8, "claims_per_step": 8, "temperature": 1.0, "ops_cost": 0.01,
          "invalid_penalty": 0.5, "evidence_bonus": 0.3, "no_std_scale": True, "lr": 5e-5,
          "beta": 0.04, "save_steps": 25, "resume_every": 5})
