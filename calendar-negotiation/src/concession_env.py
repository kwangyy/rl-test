"""Iteration 4 of the MVP spike: counterparts with real concession dynamics.

Iteration 2b's diagnosis was that MultiRequestSlotEnv's counterpart is
static -- its acceptable set is drawn once and never moves -- so "how hard
to hold" has nothing to act on and greedy is ~optimal. Here each request's
counterpart instead has:

- a private per-slot cost that is anti-correlated with self's cost (a slot
  that is cheap for self tends to be expensive for them -- a real conflict
  of interest, unlike Iteration 2 where both sides wanted the same slots);
- a hidden type from a small pool (Boulware / Conceder / rigid) that sets
  how its acceptance threshold loosens as rounds pass;
- a hidden reserve that caps how far it will ever concede.

Each request also carries an observable `stakes`: what failing to agree
costs, log-uniform over 0.2-10 while a slot costs at most 1. A meeting
with your boss must happen at almost any slot cost (stakes 10); a loose
catch-up is worth dropping rather than burning your best slot (stakes
0.2). Iteration 4 scaled the no-deal penalty and the slot cost by the
same urgency, pinning their ratio at 5:1 for every request -- so one
fixed concession rule was right every time. Varying the ratio is what
makes "how hard to hold" depend on the request in front of you.

Re-offering a previously rejected slot is allowed (no penalty), because
against a conceding counterpart "offer the same good slot again later" is
exactly the hold-firm move. The observation appends stakes to
MultiRequestSlotEnv's layout without reordering it, so the existing
baselines and eval harness still apply.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from multi_request_env import MultiRequestSlotEnv

STAKES_RANGE = (0.2, 10.0)  # cost of failing to agree, vs a slot cost of at most 1.0

# concession exponent on (round / max_rounds); None = rigid (never concedes)
COUNTERPART_TYPES = {"boulware": 5.0, "conceder": 0.33, "rigid": None}


class ConcessionSlotEnv(MultiRequestSlotEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # parent obs + log-scaled stakes for this request
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_slots * 4 + 4,), dtype=np.float32
        )

    def _obs(self):
        lo, hi = np.log(STAKES_RANGE)
        stakes_frac = (np.log(self.stakes) - lo) / (hi - lo)
        return np.concatenate([super()._obs(), [stakes_frac]]).astype(np.float32)

    def _start_request(self):
        rng = self._rng
        self.cp_type = rng.choice(list(COUNTERPART_TYPES))
        noise = rng.normal(0.0, 0.25, self.n_slots)
        self.cp_cost = np.clip(1.0 - self.cost + noise, 0.0, 1.0).astype(np.float32)
        self.cp_reserve = float(rng.uniform(0.4, 1.0))  # max fraction of their cost range they'll concede
        self.urgency = float(rng.uniform(0.0, 1.0))
        self.stakes = float(np.exp(rng.uniform(*np.log(STAKES_RANGE))))
        self.offered = np.zeros(self.n_slots, dtype=np.float32)
        self.rejected = np.zeros(self.n_slots, dtype=np.float32)
        self.round = 0

    def _accepts(self, slot: int) -> bool:
        free_idx = np.where(self.free > 0)[0]
        lo, hi = self.cp_cost[free_idx].min(), self.cp_cost[free_idx].max()
        exponent = COUNTERPART_TYPES[self.cp_type]
        progress = 0.5 if exponent is None else (self.round / self.max_rounds) ** exponent
        threshold = lo + (hi - lo) * self.cp_reserve * progress
        return bool(self.cp_cost[slot] <= threshold + 1e-6)

    def step(self, action):
        action = int(action)
        self.round += 1
        info: dict = {}
        request_done = False
        terminated = False

        if self.free[action] < 1.0:
            reward = -0.1
        else:
            self.offered[action] = 1.0
            if self._accepts(action):
                reward = -self.cost[action] - self.urgency * (
                    self.round / self.max_rounds
                ) * 0.25
                self.free[action] = 0.0
                info.update(deal=True, slot=action, rounds=self.round)
                request_done = True
            else:
                self.rejected[action] = 1.0
                reward = -0.05 * (1.0 + self.urgency)

        if not request_done and self.round >= self.max_rounds:
            reward -= self.stakes
            info.update(deal=False, rounds=self.round)
            request_done = True

        if request_done:
            self.request_idx += 1
            if self.request_idx >= self.n_requests or self.free.sum() == 0:
                terminated = True
            else:
                self._start_request()

        return self._obs(), float(reward), terminated, False, info
