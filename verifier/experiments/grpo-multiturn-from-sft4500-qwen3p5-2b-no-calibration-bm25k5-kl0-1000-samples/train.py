"""Multi-turn GRPO from the 4,500-row JSON SFT adapter on the fixed 1,000 GRPO claims.

Matched to grpo-cot-from-sft4500-qwen3p5-2b-no-calibration-kl0-1000-samples (single-turn
native-thinking CoT GRPO): same starting adapter, claim IDs, group size (8), 2 claims per
step (500 steps, one epoch), learning rate, beta 0 and reward without calibration. The
difference is the loop: the model gets only the claim and calls ranked BM25 search
itself (5 sentences per search, at most 2 searches), with thinking off; the reward also
charges 0.02 per search (core.reward L3_COST).
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "verifier/experiments/_shared/grpo_multiturn.py"
PARAMS = {'init': 'sft', 'reward': 'no_calibration', 'train_data': 'verifier/data/train/sft/fever-oracle-matched-grpo-1000',
          'num_generations': 8, 'prompts_per_step': 2, 'max_steps': 0, 'lr': 2e-05, 'beta': 0.0,
          'max_calls': 2, 'budget_grace': 2, 'max_completion_length': 2048,
          'method': 'bm25', 'search_k': 5, 'candidate_pages': 3, 'save_steps': 50, 'model': 'Qwen/Qwen3.5-2B'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="new run label, e.g. run1")
    parser.add_argument("--output-root", help="experiments directory for run artifacts")
    parser.add_argument("--sft", required=True, help="SFT4500 adapter checkpoint directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", help="checkpoint-<step> directory of this run to resume from")
    args = parser.parse_args()
    cmd = [sys.executable, str(SCRIPT)]
    for key, value in PARAMS.items():
        flag = "--" + key.replace("_", "-")
        if key == "train_data":
            value = str(ROOT / value)
        cmd.extend([flag, str(value)])
    cmd.extend(["--config-name", Path(__file__).parent.name])
    cmd.extend(["--run-name", args.run_name, "--seed", str(args.seed), "--sft", args.sft])
    if args.output_root:
        cmd.extend(["--output-root", args.output_root])
    if args.resume:
        cmd.extend(["--resume", args.resume])
    subprocess.run(cmd, cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
