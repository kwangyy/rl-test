"""Evaluation harness for MultiRequestSlotEnv (sequence of requests per
episode, sharing one depleting calendar)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from multi_request_env import MultiRequestSlotEnv


def evaluate_policy_multi(
    policy_fn: Callable[[np.ndarray], int],
    n_episodes: int = 200,
    n_slots: int = 12,
    max_rounds: int = 6,
    n_requests: int = 3,
    seed: int = 0,
) -> dict:
    env = MultiRequestSlotEnv(n_slots=n_slots, max_rounds=max_rounds, n_requests=n_requests, seed=seed)
    ep_rewards, deals, accepted_costs = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        while True:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if "deal" in info:
                deals.append(bool(info["deal"]))
                if info["deal"]:
                    accepted_costs.append(env.cost[info["slot"]])
            if terminated or truncated:
                break
        ep_rewards.append(total)

    ep_rewards = np.array(ep_rewards)
    deals = np.array(deals)
    return {
        "n_episodes": n_episodes,
        "mean_episode_reward": float(ep_rewards.mean()),
        "std_episode_reward": float(ep_rewards.std()),
        "deal_rate": float(deals.mean()) if len(deals) else float("nan"),
        "mean_accepted_cost": float(np.mean(accepted_costs)) if accepted_costs else float("nan"),
    }


def print_report_multi(name: str, stats: dict) -> None:
    print(f"\n{name}  (n={stats['n_episodes']} episodes)")
    print(f"  mean episode reward : {stats['mean_episode_reward']:.3f} +/- {stats['std_episode_reward']:.3f}")
    print(f"  deal rate           : {stats['deal_rate']*100:.1f}%  (across all requests)")
    print(f"  mean accepted cost  : {stats['mean_accepted_cost']:.3f}")
