"""Controller imitation (AG-2): LoRA SFT of Qwen3.5-2B on qwen3.8-flash's controller steps.

Data: utils.build_controller_imitation over the 6,000 flash follow trajectories in
verifier/data/controller/imitation-flash-follow-train-unused-n6000-seed2, FEVER-correct
episodes only, invalid controller steps skipped. The system prompt is the follow
preset's (agent.prompted.SYSTEMS["follow"]), the prompt the observation flash saw,
the completion the action JSON. Same LoRA settings as the verifier SFT, thinking off.

    python verifier/experiments/sft-controller-from-base-qwen3p5-2b-flash-follow-18291-samples/train.py --run-name run1

The adapter is a separate controller; the verifier adapter is unchanged. Evaluate with
python -m agent.prompted --tools follow --controller local:<checkpoints> --adapter <verifier>.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent.prompted import SYSTEMS
from utils.experiment_layout import artifacts_dir
from verifier.experiments._shared.sft import train

DATA = ROOT / "verifier/data/train/controller/flash-follow-imitation-18291-samples"
PARAMS = {"model": "Qwen/Qwen3.5-2B", "rank": 16, "lr": 1e-4, "max_steps": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=PARAMS["max_steps"], help="stop early; for smoke tests")
    a = ap.parse_args()
    with open(DATA / "train.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    artifacts = artifacts_dir("sft", "", a.seed, a.run_name, config_name=Path(__file__).parent.name)
    if artifacts.exists():
        raise SystemExit(f"run already exists: {artifacts}; use a new --run-name")
    checkpoints = artifacts / "checkpoints"
    checkpoints.mkdir(parents=True)
    secs, peak_gb = train(PARAMS["model"], rows, str(checkpoints), False, a.max_steps, PARAMS["lr"],
                          PARAMS["rank"], a.seed, system_prompt=SYSTEMS["follow"])
    meta = {**PARAMS, "max_steps": a.max_steps, "seed": a.seed, "data": DATA.relative_to(ROOT).as_posix(),
            "n_train": len(rows), "system_prompt": "agent.prompted.SYSTEMS['follow']",
            "train_seconds": round(secs), "peak_gpu_gb": round(peak_gb, 2)}
    with open(artifacts / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
