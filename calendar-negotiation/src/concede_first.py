"""Concede-first baseline for ConcessionSlotEnv: offer the free slot that is
most expensive to self first, then step down one slot per round. Since the
counterpart's cost is anti-correlated with self's, this is roughly "give
them what they want" -- it trades slot cost for near-certain, fast deals.
Added after PPO's Iteration 4 "GO" turned out to match this one-liner."""

from __future__ import annotations

import numpy as np


def concede_first_action(obs: np.ndarray, n_slots: int, max_rounds: int) -> int:
    free = np.where(obs[:n_slots] > 0)[0]
    if len(free) == 0:
        return 0
    order = free[np.argsort(-obs[n_slots : 2 * n_slots][free])]
    rounds_done = round((1.0 - obs[4 * n_slots + 1]) * max_rounds)
    return int(order[min(rounds_done, len(order) - 1)])
