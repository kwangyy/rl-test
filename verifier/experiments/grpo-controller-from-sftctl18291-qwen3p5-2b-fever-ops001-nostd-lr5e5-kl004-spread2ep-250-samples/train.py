"""Controller GRPO follow-up B: only the claims that carry outcome signal, two passes, lr 5e-5.

Training claims are the 250 train claims whose 8 rollouts disagreed on FEVER in the pilot
(17), the 300-claim run (60) and the 1,000-claim run (173); in those runs most claims gave all
8 rollouts the same FEVER outcome and so no outcome signal. Two shuffled passes (500 claim
passes, 63 updates), lr 5e-5, otherwise the same reward and beta. All are unused train claims;
no dev, calibration or holdout ID is used. Run as a resumable chain.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verifier.experiments._shared.grpo_controller import main

RUNS = "verifier/experiments/grpo-controller-from-sftctl18291-qwen3p5-2b-"
if __name__ == "__main__":
    main({"config_name": Path(__file__).parent.name,
          "claims_from": [RUNS + "fever-ops001-pilot-64-samples/runs/pilot-t1p0/artifacts/rollouts.jsonl",
                          RUNS + "fever-ops001-nostd-kl004-300-samples/runs/run1/artifacts/rollouts.jsonl",
                          RUNS + "fever-ops001-nostd-kl004-1000-samples/runs/run1/artifacts/rollouts.jsonl"],
          "epochs": 2, "train_seed": 6, "group_size": 8, "claims_per_step": 8, "temperature": 1.0,
          "ops_cost": 0.01, "invalid_penalty": 0.5, "no_std_scale": True, "lr": 5e-5, "beta": 0.04,
          "save_steps": 25, "resume_every": 5})
