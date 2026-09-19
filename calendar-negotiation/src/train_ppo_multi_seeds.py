"""Push further on the PPO-vs-greedy question: 3 seeds, more steps, a
small entropy bonus (in case PPO was converging prematurely), same fixed
eval protocol across seeds so the gap vs greedy can be judged against
seed-to-seed variance instead of a single run.
"""

from __future__ import annotations

import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from multi_request_env import MultiRequestSlotEnv
from first_fit import first_fit_action
from greedy_utility import greedy_utility_action
from evaluate_multi import evaluate_policy_multi, print_report_multi

N_SLOTS = 12
MAX_ROUNDS = 6
N_REQUESTS = 3
TRAIN_STEPS = 500_000
EVAL_EPISODES = 300
SEEDS = [0, 1, 2]
EVAL_SEED = 0  # fixed across seeds/policies for apples-to-apples comparison


def make_env():
    return MultiRequestSlotEnv(n_slots=N_SLOTS, max_rounds=MAX_ROUNDS, n_requests=N_REQUESTS, seed=None)


def main():
    kwargs = dict(n_episodes=EVAL_EPISODES, n_slots=N_SLOTS, max_rounds=MAX_ROUNDS, n_requests=N_REQUESTS, seed=EVAL_SEED)
    ff_stats = evaluate_policy_multi(lambda o: first_fit_action(o, N_SLOTS), **kwargs)
    gu_stats = evaluate_policy_multi(lambda o: greedy_utility_action(o, N_SLOTS), **kwargs)

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

    print(f"\nPPO across {len(SEEDS)} seeds: mean={ppo_rewards.mean():.3f}  std={ppo_rewards.std():.3f}  per-seed={list(np.round(ppo_rewards, 3))}")
    vs_gu = (ppo_rewards.mean() - gu_stats["mean_episode_reward"]) / abs(gu_stats["mean_episode_reward"]) * 100
    vs_ff = (ppo_rewards.mean() - ff_stats["mean_episode_reward"]) / abs(ff_stats["mean_episode_reward"]) * 100
    print(f"PPO (mean of {len(SEEDS)} seeds) vs first-fit  : {vs_ff:+.1f}%")
    print(f"PPO (mean of {len(SEEDS)} seeds) vs greedy-util: {vs_gu:+.1f}%")
    if ppo_rewards.min() > gu_stats["mean_episode_reward"]:
        print("GO: every seed beat greedy.")
    elif vs_gu >= 0:
        print("PROVISIONAL GO: mean beats greedy but not every seed -- still noisy at 3 seeds.")
    else:
        print("STILL NO CLEAR EDGE over greedy at 3 seeds / 500k steps.")


if __name__ == "__main__":
    main()
