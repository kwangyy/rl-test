"""Iteration 2 of the go/no-go spike: PPO on MultiRequestSlotEnv (a short
sequence of requests sharing one depleting calendar, so holding a slot
back has a real opportunity cost) vs. first-fit and greedy-best-utility-now.
"""

from __future__ import annotations

import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from multi_request_env import MultiRequestSlotEnv
from first_fit import first_fit_action
from greedy_utility import greedy_utility_action
from evaluate_multi import evaluate_policy_multi, print_report_multi

N_SLOTS = 12
MAX_ROUNDS = 6
N_REQUESTS = 3
TRAIN_STEPS = 300_000
EVAL_EPISODES = 300
SEED = 0


def make_env():
    return MultiRequestSlotEnv(n_slots=N_SLOTS, max_rounds=MAX_ROUNDS, n_requests=N_REQUESTS, seed=None)


def main():
    vec_env = make_vec_env(make_env, n_envs=4, seed=SEED)
    model = PPO("MlpPolicy", vec_env, seed=SEED, verbose=0)

    start = time.time()
    model.learn(total_timesteps=TRAIN_STEPS)
    elapsed = time.time() - start
    print(f"PPO training done: {TRAIN_STEPS} steps in {elapsed:.1f}s")

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    kwargs = dict(n_episodes=EVAL_EPISODES, n_slots=N_SLOTS, max_rounds=MAX_ROUNDS, n_requests=N_REQUESTS, seed=SEED)
    ppo_stats = evaluate_policy_multi(ppo_policy, **kwargs)
    ff_stats = evaluate_policy_multi(lambda o: first_fit_action(o, N_SLOTS), **kwargs)
    gu_stats = evaluate_policy_multi(lambda o: greedy_utility_action(o, N_SLOTS), **kwargs)

    print_report_multi("PPO", ppo_stats)
    print_report_multi("first-fit", ff_stats)
    print_report_multi("greedy-best-utility-now", gu_stats)

    vs_ff = (ppo_stats["mean_episode_reward"] - ff_stats["mean_episode_reward"]) / abs(ff_stats["mean_episode_reward"]) * 100
    vs_gu = (ppo_stats["mean_episode_reward"] - gu_stats["mean_episode_reward"]) / abs(gu_stats["mean_episode_reward"]) * 100
    print(f"\nPPO vs first-fit  : {vs_ff:+.1f}%")
    print(f"PPO vs greedy-util: {vs_gu:+.1f}%  <-- this is the bar that matters")
    if vs_gu >= 15:
        print("GO: PPO clears the 15% bar over the sensible heuristic.")
    elif vs_gu >= 0:
        print("ITERATE: PPO edges out greedy but not convincingly.")
    else:
        print("NO-GO signal: PPO still doesn't beat the sensible heuristic.")


if __name__ == "__main__":
    main()
