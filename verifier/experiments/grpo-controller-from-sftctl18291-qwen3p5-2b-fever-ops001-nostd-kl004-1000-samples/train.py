"""Controller GRPO full run (handover section 6, step 3).

Same settings as the 300-claim short run, on 1,000 unused train claims (seed 5; the
pilot used seed 3 and the short run seed 4) x 8 rollouts, 125 updates. About 6 h on an
A100-40, so it runs as a resumable chain of run_controller_grpo.sbatch jobs; then dev 900
with eval_controller.sbatch and McNemar against the committed flash and imitation traces.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verifier.experiments._shared.grpo_controller import main

if __name__ == "__main__":
    main({"config_name": Path(__file__).parent.name, "n_train": 1000, "train_seed": 5,
          "group_size": 8, "claims_per_step": 8, "temperature": 1.0, "ops_cost": 0.01,
          "invalid_penalty": 0.5, "no_std_scale": True, "lr": 1e-5, "beta": 0.04,
          "save_steps": 25, "resume_every": 5})
