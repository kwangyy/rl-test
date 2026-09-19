"""Greedy best-utility-now baseline: always offer the cheapest (to self)
free, not-yet-offered slot. Unlike first-fit, this uses the preference
cost self already knows about itself — a much higher bar than first-fit,
and the one PPO actually needs to clear to be interesting."""

from __future__ import annotations

import numpy as np


def greedy_utility_action(obs: np.ndarray, n_slots: int) -> int:
    free = obs[:n_slots]
    cost = obs[n_slots : 2 * n_slots]
    offered = obs[2 * n_slots : 3 * n_slots]
    best_i, best_cost = None, None
    for i in range(n_slots):
        if free[i] > 0 and offered[i] == 0:
            if best_cost is None or cost[i] < best_cost:
                best_i, best_cost = i, cost[i]
    return best_i if best_i is not None else 0
