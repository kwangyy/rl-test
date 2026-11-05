"""Controller GRPO follow-up: the 1,000-claim run at lr 1e-4 (A used 5e-5, the first run 1e-5).

The 1e-5 run (grpo-controller-...-nostd-kl004-1000-samples/run1) kept KL to the imitation
controller at or below 0.002 and matched it on dev 900 (FEVER 0.687 vs 0.684). Same claims
(seed 5), reward and beta; only the learning rate changes, so the two runs compare directly.
Run with verifier/experiments/run_controller_grpo.sbatch as a resumable chain.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verifier.experiments._shared.grpo_controller import main

if __name__ == "__main__":
    main({"config_name": Path(__file__).parent.name, "n_train": 1000, "train_seed": 5,
          "group_size": 8, "claims_per_step": 8, "temperature": 1.0, "ops_cost": 0.01,
          "invalid_penalty": 0.5, "no_std_scale": True, "lr": 1e-4, "beta": 0.04,
          "save_steps": 25, "resume_every": 5})
