"""MVP spike: train PPO on SlotNegotiationEnv for a small, fixed budget and
compare it against the first-fit baseline. This is the go/no-go script —
see docs/mvp-plan.md for the decision rule.
"""

from __future__ import annotations

import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from slot_negotiation_env import SlotNegotiationEnv
from first_fit import first_fit_action
from evaluate import evaluate_policy, print_report

N_SLOTS = 12
MAX_ROUNDS = 6
TRAIN_STEPS = 100_000
EVAL_EPISODES = 200
SEED = 0


def make_env():
    return SlotNegotiationEnv(n_slots=N_SLOTS, max_rounds=MAX_ROUNDS, seed=None)


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

    ppo_stats = evaluate_policy(ppo_policy, n_episodes=EVAL_EPISODES, n_slots=N_SLOTS, max_rounds=MAX_ROUNDS, seed=SEED)
    ff_stats = evaluate_policy(
        lambda obs: first_fit_action(obs, N_SLOTS),
        n_episodes=EVAL_EPISODES,
        n_slots=N_SLOTS,
        max_rounds=MAX_ROUNDS,
        seed=SEED,
    )

    print_report("PPO", ppo_stats)
    print_report("first-fit baseline", ff_stats)

    improvement = (ppo_stats["mean_reward"] - ff_stats["mean_reward"]) / abs(ff_stats["mean_reward"]) * 100
    print(f"\nPPO vs first-fit reward improvement: {improvement:+.1f}%")
    if improvement >= 15:
        print("GO: PPO clears the 15% bar even in the toy setting.")
    elif improvement >= 0:
        print("ITERATE: PPO beats first-fit but not by much — revisit reward/state before scaling up.")
    else:
        print("NO-GO signal: PPO does not beat first-fit in the easiest possible setting.")


if __name__ == "__main__":
    main()
