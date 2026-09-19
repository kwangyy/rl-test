"""Iteration 2 of the MVP spike: a short SEQUENCE of requests sharing one
calendar, arriving one at a time with no foresight of future requests.

This is the piece the one-shot SlotNegotiationEnv left out: giving away a
slot now has an opportunity cost against a request that hasn't arrived
yet. Counterparts are biased toward the same low-cost-to-self slots self
would also prefer to give away, so there is real contention over the same
desirable slots -- not independent, uncorrelated draws. Reward for an
accepted deal is weighted by that request's urgency, so spending a
contested cheap slot on a low-urgency request only shows up as a mistake
later, once a high-urgency request finds the good slots already gone.
This is meant to test whether a learned policy can beat myopic
per-request heuristics once "how hard to hold the rest" actually matters.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiRequestSlotEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_slots: int = 12,
        max_rounds: int = 6,
        n_requests: int = 3,
        no_deal_penalty: float = 5.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.max_rounds = max_rounds
        self.n_requests = n_requests
        self.no_deal_penalty = no_deal_penalty

        self.action_space = spaces.Discrete(n_slots)
        # [free, cost, offered, rejected] per slot + [urgency, rounds_left_frac, requests_remaining_frac]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_slots * 4 + 3,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng

        self.free = (rng.random(self.n_slots) < 0.6).astype(np.float32)
        if self.free.sum() == 0:
            self.free[rng.integers(self.n_slots)] = 1.0

        base_cost = rng.random(self.n_slots).astype(np.float32)
        is_morning = (np.arange(self.n_slots) % 2 == 0).astype(np.float32)
        self.cost = np.clip(base_cost + 0.3 * is_morning, 0.0, 1.0).astype(np.float32)

        self.request_idx = 0
        self._start_request()
        return self._obs(), {}

    def _start_request(self):
        rng = self._rng
        free_idx = np.where(self.free > 0)[0]
        if len(free_idx) == 0:
            self._accept_mask = np.zeros(self.n_slots, dtype=np.float32)
        else:
            # bias acceptance toward low-cost (to self) slots -> real contention
            # over the same desirable slots, not independent random draws
            weights = (1.0 - self.cost[free_idx]) + 0.05
            weights = weights / weights.sum()
            n_accept = max(1, int(round(len(free_idx) * rng.uniform(0.2, 0.5))))
            n_accept = min(n_accept, len(free_idx))
            accept_idx = rng.choice(free_idx, size=n_accept, replace=False, p=weights)
            self._accept_mask = np.zeros(self.n_slots, dtype=np.float32)
            self._accept_mask[accept_idx] = 1.0

        self.urgency = float(rng.uniform(0.0, 1.0))
        self.offered = np.zeros(self.n_slots, dtype=np.float32)
        self.rejected = np.zeros(self.n_slots, dtype=np.float32)
        self.round = 0

    def _obs(self):
        rounds_left_frac = (self.max_rounds - self.round) / self.max_rounds
        requests_remaining_frac = (self.n_requests - self.request_idx) / self.n_requests
        return np.concatenate(
            [
                self.free,
                self.cost,
                self.offered,
                self.rejected,
                [self.urgency, rounds_left_frac, requests_remaining_frac],
            ]
        ).astype(np.float32)

    def step(self, action):
        action = int(action)
        self.round += 1
        terminated = False
        truncated = False
        info: dict = {}
        request_done = False

        if self.free[action] < 1.0 or self.offered[action] > 0:
            reward = -0.1
        else:
            self.offered[action] = 1.0
            if self._accept_mask[action] > 0:
                reward = -self.cost[action] * (0.5 + self.urgency) - self.urgency * (
                    self.round / self.max_rounds
                ) * 0.25
                self.free[action] = 0.0  # slot consumed -- unavailable to future requests
                info["deal"] = True
                info["slot"] = action
                info["rounds"] = self.round
                request_done = True
            else:
                self.rejected[action] = 1.0
                reward = -0.05 * (1.0 + self.urgency)

        if not request_done and self.round >= self.max_rounds:
            reward -= self.no_deal_penalty * (0.5 + self.urgency)
            info["deal"] = False
            info["rounds"] = self.round
            request_done = True

        if request_done:
            self.request_idx += 1
            if self.request_idx >= self.n_requests:
                terminated = True
            else:
                self._start_request()

        return self._obs(), float(reward), terminated, truncated, info
