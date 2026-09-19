"""Smoke-run the scaffold: random policy vs a hand-written controller
heuristic, plus the two charts the presentation depends on (reliability
bins, ops-vs-accuracy).

No training here on purpose. The question this answers is only "does the MDP
hold together and do the metrics compute", not "does RL win" -- same split as
the negotiation spike's R1/R2.

    python run_scaffold.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from verification_env import ABSTAIN, CROSS_CHECK, DECOMPOSE, RETRIEVE, ClaimVerificationEnv

N_EPISODES = 200


def random_policy(obs, rng):
    return int(rng.integers(8))


def heuristic_policy(obs, rng):
    """Retrieve until evidence is decent, decompose if the claim looks
    compound, cross-check once, then commit -- or abstain when retrieval
    never found anything. The baseline the controller must beat.
    """
    n_evidence, top_score, agreement, _, decomposed, compound, best, _ = obs
    if compound and not decomposed:
        return DECOMPOSE
    if n_evidence < 0.5 and top_score > 0.0:
        return RETRIEVE
    if n_evidence == 0.0:
        return RETRIEVE
    if agreement == 0.0:
        return CROSS_CHECK
    if best < 0.25:
        return ABSTAIN  # nothing convincing was ever retrieved
    return 4 if best > 0.4 else 5  # SUPPORTED, high vs low confidence


def evaluate(policy_fn, n_episodes: int = N_EPISODES, seed: int = 0) -> dict:
    env = ClaimVerificationEnv(seed=seed)
    rng = np.random.default_rng(seed)
    rewards, correct, ops, f1s = [], [], [], []
    bins = defaultdict(lambda: [0, 0])  # confidence -> [n_correct, n]
    abstain_when_nei, nei_total = 0, 0
    compound_correct, compound_total = 0, 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        while True:
            obs, r, terminated, truncated, info = env.step(policy_fn(obs, rng))
            if terminated or truncated:
                rewards.append(r)
                correct.append(info["correct"])
                ops.append(info["cost"])
                f1s.append(info["evidence_f1"])
                b = bins[round(info["confidence"], 1)]
                b[0] += info["correct"]
                b[1] += 1
                if info["gold"] == "NEI":
                    nei_total += 1
                    abstain_when_nei += info["pred"] == "NEI"
                if info["is_compound"]:
                    compound_total += 1
                    compound_correct += info["correct"]
                break

    return {
        "mean_reward": float(np.mean(rewards)),
        "accuracy": float(np.mean(correct)),
        "mean_evidence_f1": float(np.mean(f1s)),
        "mean_ops": float(np.mean(ops)),
        "abstain_recall_on_nei": abstain_when_nei / nei_total if nei_total else float("nan"),
        "compound_accuracy": compound_correct / compound_total if compound_total else float("nan"),
        "reliability": {k: v[0] / v[1] for k, v in sorted(bins.items())},
    }


if __name__ == "__main__":
    for name, fn in [("random", random_policy), ("heuristic", heuristic_policy)]:
        m = evaluate(fn)
        print(f"\n== {name} ==")
        for k, v in m.items():
            if k == "reliability":
                print("  reliability (stated confidence -> actual accuracy):")
                for conf, acc in v.items():
                    print(f"    {conf:.1f} -> {acc:.2f}")
            else:
                print(f"  {k:24s} {v:.3f}")
