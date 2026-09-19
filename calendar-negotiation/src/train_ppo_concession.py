"""Iteration 4 gate: does PPO beat the best non-learned baseline once
counterparts have real concession dynamics (ConcessionSlotEnv)? Same
protocol as Iteration 2b (3 seeds, 500k steps, ent_coef=0.01, fixed eval
seed) so the only thing that changed is the counterpart.
"""

from __future__ import annotations

import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from concession_env import ConcessionSlotEnv
from first_fit import first_fit_action
from greedy_utility import greedy_utility_action
from concede_first import concede_first_action
from stakes_switch import stakes_switch_action
from evaluate_multi import evaluate_policy_multi, print_report_multi

N_SLOTS = 12
MAX_ROUNDS = 6
N_REQUESTS = 3
TRAIN_STEPS = 500_000
EVAL_EPISODES = 300
SEEDS = [0, 1, 2]
EVAL_SEED = 0  # fixed across seeds/policies for apples-to-apples comparison


def make_env():
    return ConcessionSlotEnv(n_slots=N_SLOTS, max_rounds=MAX_ROUNDS, n_requests=N_REQUESTS, seed=None)


def main():
    kwargs = dict(n_episodes=EVAL_EPISODES, n_slots=N_SLOTS, max_rounds=MAX_ROUNDS, n_requests=N_REQUESTS, seed=EVAL_SEED, env_cls=ConcessionSlotEnv)
    ff_stats = evaluate_policy_multi(lambda o: first_fit_action(o, N_SLOTS), **kwargs)
    gu_stats = evaluate_policy_multi(lambda o: greedy_utility_action(o, N_SLOTS), **kwargs)
    cf_stats = evaluate_policy_multi(lambda o: concede_first_action(o, N_SLOTS, MAX_ROUNDS), **kwargs)
    ss_stats = evaluate_policy_multi(lambda o: stakes_switch_action(o, N_SLOTS, MAX_ROUNDS), **kwargs)

    ppo_rewards = []
    for seed in SEEDS:
        vec_env = make_vec_env(make_env, n_envs=4, seed=seed)
        model = PPO("MlpPolicy", vec_env, seed=seed, ent_coef=0.01, verbose=0)
        start = time.time()
        model.learn(total_timesteps=TRAIN_STEPS)
        elapsed = time.time() - start
        print(f"seed {seed}: trained {TRAIN_STEPS} steps in {elapsed:.1f}s")

        def ppo_policy(obs, _model=model):
            action, _ = _model.predict(obs, deterministic=True)
            return int(action)

        stats = evaluate_policy_multi(ppo_policy, **kwargs)
        print_report_multi(f"PPO seed={seed}", stats)
        ppo_rewards.append(stats["mean_episode_reward"])

    ppo_rewards = np.array(ppo_rewards)
    print_report_multi("first-fit", ff_stats)
    print_report_multi("greedy-best-utility-now", gu_stats)
    print_report_multi("concede-first", cf_stats)
    print_report_multi("stakes-switch", ss_stats)

    print(f"\nPPO across {len(SEEDS)} seeds: mean={ppo_rewards.mean():.3f}  std={ppo_rewards.std():.3f}  per-seed={list(np.round(ppo_rewards, 3))}")
    best_name, best = max([("first-fit", ff_stats), ("greedy-util", gu_stats), ("concede-first", cf_stats), ("stakes-switch", ss_stats)], key=lambda kv: kv[1]["mean_episode_reward"])
    best_r = best["mean_episode_reward"]
    vs_gu = (ppo_rewards.mean() - gu_stats["mean_episode_reward"]) / abs(gu_stats["mean_episode_reward"]) * 100
    vs_ff = (ppo_rewards.mean() - ff_stats["mean_episode_reward"]) / abs(ff_stats["mean_episode_reward"]) * 100
    print(f"PPO (mean of {len(SEEDS)} seeds) vs first-fit  : {vs_ff:+.1f}%")
    print(f"PPO (mean of {len(SEEDS)} seeds) vs greedy-util: {vs_gu:+.1f}%")
    vs_best = (ppo_rewards.mean() - best_r) / abs(best_r) * 100
    print(f"PPO (mean of {len(SEEDS)} seeds) vs best baseline ({best_name}): {vs_best:+.1f}%")
    if ppo_rewards.min() > best_r:
        print("GO: every seed beat the best baseline.")
    elif vs_best >= 0:
        print("PROVISIONAL GO: mean beats the best baseline but not every seed.")
    else:
        print("NO CLEAR EDGE over the best baseline at 3 seeds / 500k steps.")

if __name__ == "__main__":
    main()
