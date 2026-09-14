"""First-fit baseline: always offer the earliest still-free, not-yet-offered
slot. This mirrors what a plain Calendly-style link does — take the first
available slot, no notion of which slot is more precious to give away."""

from __future__ import annotations

import numpy as np


def first_fit_action(obs: np.ndarray, n_slots: int) -> int:
    free = obs[:n_slots]
    offered = obs[2 * n_slots : 3 * n_slots]
    for i in range(n_slots):
        if free[i] > 0 and offered[i] == 0:
            return i
    return 0  # fallback: nothing left to try, will hit no-deal
