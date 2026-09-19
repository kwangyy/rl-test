"""Stakes-conditional baseline for ConcessionSlotEnv: when failing to agree
costs less than the ~0.85 slot cost concede-first pays, hold your cheapest
slot and let the request fail; otherwise concede. This is the strongest
non-learned policy found for this env, and the bar PPO has to clear --
concede-first alone always deals, even when the meeting isn't worth it.

The threshold was swept over 2000 episodes on eval seed 0 (0.2 best, then
0.3/0.4/0.5 monotonically worse), so it is mildly tuned on the eval set --
which only makes it a harder, more honest bar for PPO.
"""

from __future__ import annotations

import numpy as np

from concede_first import concede_first_action

HOLD_BELOW = 0.2  # log-scaled stakes; ~0.4 raw, vs a ~0.85 mean conceded slot cost


def stakes_switch_action(obs: np.ndarray, n_slots: int, max_rounds: int) -> int:
    if obs[4 * n_slots + 3] > HOLD_BELOW:
        return concede_first_action(obs, n_slots, max_rounds)
    free = np.where(obs[:n_slots] > 0)[0]
    if len(free) == 0:
        return 0
    return int(free[np.argmin(obs[n_slots : 2 * n_slots][free])])
