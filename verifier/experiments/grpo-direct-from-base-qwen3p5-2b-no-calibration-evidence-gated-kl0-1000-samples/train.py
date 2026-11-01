"""Single-turn direct GRPO from the base model on the fixed 1,000 GRPO claims.

Reward: no_calibration_evidence_gated. A correct SUPPORTED/REFUTED earns correctness
credit only when it cites at least one gold sentence (core.reward evidence_gate); both
matched arms use it. Otherwise identical to grpo-direct-from-base-qwen3p5-2b-no-calibration-kl0-1000-samples. See docs/multiturn-grpo-results.md.

Matched single-turn arm for grpo-multiturn-from-base-qwen3p5-2b-no-calibration-bm25k5-kl0-1000-samples:
same claims, fresh LoRA of the SFT4500 shape on base Qwen3.5-2B, group size 8, 2 claims per
step (500 steps), learning rate, beta 0, thinking off and reward without calibration. This
arm reads the oracle-page context from labeled_prompts.jsonl instead of searching.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "verifier/experiments/_shared/grpo.py"
PARAMS = {'init': 'base', 'reward': 'no_calibration_evidence_gated', 'data': 'verifier/data/train/sft/fever-oracle-matched-grpo-1000',
          'n_train': 1000, 'num_generations': 8, 'prompts_per_step': 2, 'max_steps': 0, 'lr': 2e-05, 'beta': 0.0,
          'model': 'Qwen/Qwen3.5-2B'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="new run label, e.g. run1")
    parser.add_argument("--output-root", help="experiments directory for run artifacts")
    parser.add_argument("--sft", required=True, help="SFT4500 adapter directory (LoRA shape only)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cmd = [sys.executable, str(SCRIPT)]
    for key, value in PARAMS.items():
        flag = "--" + key.replace("_", "-")
        if key == "data":
            value = str(ROOT / value)
        cmd.extend([flag, str(value)])
    cmd.extend(["--config-name", Path(__file__).parent.name])
    cmd.extend(["--run-name", args.run_name, "--seed", str(args.seed), "--sft", args.sft])
    if args.output_root:
        cmd.extend(["--output-root", args.output_root])
    subprocess.run(cmd, cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
