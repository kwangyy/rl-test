"""Action agreement of a local controller with the teacher on held-out imitation steps (AG-2).

Each val row of utils.build_controller_imitation is one teacher (flash) decision and
the observation it saw. The local controller answers the same observation greedily;
this reports how often it picks the same action and the exact same JSON.

    python -m agent.eval_controller_agreement \
        --adapter verifier/experiments/sft-controller-from-base-qwen3p5-2b-flash-follow-18291-samples/runs/run1/artifacts/checkpoints \
        --data verifier/data/train/controller/flash-follow-imitation-18291-samples/val.jsonl

Agreement measures imitation only; score a controller with agent.prompted on dev 900.
"""

import argparse
import json
from collections import Counter, defaultdict

from agent.decompose import LocalController, first_json
from agent.prompted import SYSTEMS

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="controller LoRA dir")
    ap.add_argument("--data", required=True, help="val.jsonl from utils.build_controller_imitation")
    ap.add_argument("--tools", default="follow", help="preset whose system prompt the data used")
    a = ap.parse_args()

    with open(a.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    outs = LocalController(a.adapter).batch([(SYSTEMS[a.tools], r["prompt"]) for r in rows])
    by, conf = defaultdict(Counter), Counter()
    for r, g in zip(rows, outs):
        want, got = json.loads(r["completion"]), first_json(g["text"], dict) or {}
        act, pred = want["action"], got.get("action", "invalid")
        by[act]["n"] += 1
        by[act]["action"] += act == pred
        by[act]["exact"] += want == got
        conf[(act, pred)] += 1
    n = len(rows)
    print(f"{n} steps: action match {sum(v['action'] for v in by.values()) / n:.3f}, "
          f"exact JSON {sum(v['exact'] for v in by.values()) / n:.3f}")
    for act, v in sorted(by.items(), key=lambda x: -x[1]["n"]):
        print(f"  {act:9s} n={v['n']:4d}  action {v['action'] / v['n']:.3f}  exact {v['exact'] / v['n']:.3f}")
    print("confusions (teacher, local):", [(k, c) for k, c in conf.most_common() if k[0] != k[1]][:8])
