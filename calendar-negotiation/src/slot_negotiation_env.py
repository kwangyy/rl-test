"""Minimal single-request slot negotiation POMDP.

This is a deliberately small, hand-rolled Gymnasium environment used to
test one thing fast, before any CalBench integration: can an RL policy
learn to pick *which slot to give away* (minimizing its own preference
cost) and implicitly *how hard to hold the rest* (by choosing which slots
to try, and when), under partial observability of the counterpart's
true acceptable slots?

One episode = negotiating ONE incoming meeting request. Self proposes a
slot each round; the counterpart (hidden acceptable-slot set) accepts or
rejects. Self observes its own calendar/costs plus the history of what
it has offered/been rejected on this episode (a sufficient statistic
standing in for a belief state), but never the counterpart's true
acceptable set directly.

Scope note: this is intentionally simpler than the real project (single
counterpart type, one-sided offers, no CalBench scenario generator). It
exists only to answer: does PPO beat first-fit here at all? See
docs/mvp-plan.md.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SlotNegotiationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_slots: int = 12,
        max_rounds: int = 6,
        no_deal_penalty: float = 5.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.max_rounds = max_rounds
        self.no_deal_penalty = no_deal_penalty

        self.action_space = spaces.Discrete(n_slots)
        # [free, cost, offered, rejected] per slot, + [urgency, rounds_left_frac]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_slots * 4 + 2,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng

        # self's calendar: ~60% of slots free
        self.free = (rng.random(self.n_slots) < 0.6).astype(np.float32)
        if self.free.sum() == 0:
            self.free[rng.integers(self.n_slots)] = 1.0

        # preference cost per slot in [0,1]; even-indexed slots stand in for
        # "mornings" and cost more (proxy for a no-mornings preference)
        base_cost = rng.random(self.n_slots).astype(np.float32)
        is_morning = (np.arange(self.n_slots) % 2 == 0).astype(np.float32)
        self.cost = np.clip(base_cost + 0.3 * is_morning, 0.0, 1.0).astype(np.float32)

        # counterpart's hidden acceptable set: random subset of self's free slots
        free_idx = np.where(self.free > 0)[0]
        n_accept = max(1, int(len(free_idx) * rng.uniform(0.2, 0.5)))
        accept_idx = rng.choice(free_idx, size=n_accept, replace=False)
        self._accept_mask = np.zeros(self.n_slots, dtype=np.float32)
        self._accept_mask[accept_idx] = 1.0

        self.urgency = float(rng.uniform(0.0, 1.0))
        self.offered = np.zeros(self.n_slots, dtype=np.float32)
        self.rejected = np.zeros(self.n_slots, dtype=np.float32)
        self.round = 0

        return self._obs(), {}

    def _obs(self):
        rounds_left_frac = (self.max_rounds - self.round) / self.max_rounds
        return np.concatenate(
            [
                self.free,
                self.cost,
                self.offered,
                self.rejected,
                [self.urgency, rounds_left_frac],
            ]
        ).astype(np.float32)

    def step(self, action):
        action = int(action)
        self.round += 1
        terminated = False
        truncated = False
        info: dict = {}

        if self.free[action] < 1.0:
            # offered a slot self doesn't even have free: wasted round
            reward = -0.1
        elif self.offered[action] > 0:
            # re-offering an already-tried slot: always rejected again, wasted round
            reward = -0.1
        else:
            self.offered[action] = 1.0
            if self._accept_mask[action] > 0:
                reward = -self.cost[action] - self.urgency * (self.round / self.max_rounds) * 0.5
                terminated = True
                info["deal"] = True
                info["slot"] = action
                info["rounds"] = self.round
            else:
                self.rejected[action] = 1.0
                reward = -0.05 * (1.0 + self.urgency)

        if not terminated and self.round >= self.max_rounds:
            truncated = True
            reward -= self.no_deal_penalty
            info["deal"] = False
            info["rounds"] = self.round

        return self._obs(), float(reward), terminated, truncated, info
