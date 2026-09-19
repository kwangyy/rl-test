"""Shared evaluation harness: run a policy over N episodes on
SlotNegotiationEnv and report the metrics the go/no-go decision hinges on.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from slot_negotiation_env import SlotNegotiationEnv


def evaluate_policy(
    policy_fn: Callable[[np.ndarray], int],
    n_episodes: int = 200,
    n_slots: int = 12,
    max_rounds: int = 6,
    seed: int = 0,
) -> dict:
    env = SlotNegotiationEnv(n_slots=n_slots, max_rounds=max_rounds, seed=seed)
    rewards, deals, rounds_used, accepted_costs = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        while True:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                deals.append(bool(info.get("deal", False)))
                if info.get("deal"):
                    rounds_used.append(info["rounds"])
                    accepted_costs.append(env.cost[info["slot"]])
                break
        rewards.append(total_reward)

    rewards = np.array(rewards)
    deals = np.array(deals)
    return {
        "n_episodes": n_episodes,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "deal_rate": float(deals.mean()),
        "mean_rounds_to_agreement": float(np.mean(rounds_used)) if rounds_used else float("nan"),
        "mean_accepted_cost": float(np.mean(accepted_costs)) if accepted_costs else float("nan"),
    }


def print_report(name: str, stats: dict) -> None:
    print(f"\n{name}  (n={stats['n_episodes']})")
    print(f"  mean reward         : {stats['mean_reward']:.3f} +/- {stats['std_reward']:.3f}")
    print(f"  deal rate           : {stats['deal_rate']*100:.1f}%")
    print(f"  mean rounds to deal : {stats['mean_rounds_to_agreement']:.2f}")
    print(f"  mean accepted cost  : {stats['mean_accepted_cost']:.3f}  (lower = gave away cheaper slots)")
