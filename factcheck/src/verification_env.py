"""The controller MDP: search-and-commit under a budget.

One episode = one claim. The policy gathers evidence, optionally decomposes
or cross-checks, then commits to a verdict at a stated confidence -- or
abstains. Reward lands at commit time (see reward.py).

What makes this a different MDP from the math-controller version: the action
space has an *information-gathering* dimension, not just "compute more", and
`abstain` is a legitimate terminal action because some claims genuinely are
not settleable from the corpus. Math problems are always solvable; claims
are not.

Observation (all in [0, 1], fixed width so PPO can eat it directly):
    0  n_evidence gathered / MAX_EVIDENCE
    1  top retrieval score last seen        (retrieval confidence)
    2  mean pairwise agreement of evidence  (filled by cross_check, else 0)
    3  hop count / MAX_OPS
    4  decomposed yet (0/1)
    5  claim looks compound (0/1)           (cheap surface feature)
    6  best-evidence score seen so far
    7  budget remaining / MAX_OPS
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import reward as R
from toy_corpus import CLAIMS, NEI, REFUTED, SENTENCES, SUPPORTED, decompose, retrieve, tokens

MAX_EVIDENCE = 6
MAX_OPS = 8
TOP_K = 2

# 0-3 information actions, 4-7 terminal commits (label x confidence)
RETRIEVE, DECOMPOSE, CROSS_CHECK, ABSTAIN = 0, 1, 2, 3
COMMITS = {
    4: (SUPPORTED, 0.9),
    5: (SUPPORTED, 0.4),
    6: (REFUTED, 0.9),
    7: (REFUTED, 0.4),
}


class ClaimVerificationEnv(gym.Env):
    def __init__(self, claims=None, seed: int = 0):
        self.claims = claims if claims is not None else CLAIMS
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = int(self._rng.integers(len(self.claims)))
        self.claim, self.gold, self.gold_ids, self.is_compound = self.claims[idx]
        self.queries = [self.claim]
        self.cited: list[str] = []
        self.top_score = 0.0
        self.best_score = 0.0
        self.agreement = 0.0
        self.n_ops = 0
        self.decomposed = False
        return self._obs(), {}

    def _obs(self):
        surface_compound = 1.0 if " and " in self.claim else 0.0
        return np.array(
            [
                len(self.cited) / MAX_EVIDENCE,
                self.top_score,
                self.agreement,
                self.n_ops / MAX_OPS,
                float(self.decomposed),
                surface_compound,
                self.best_score,
                1.0 - self.n_ops / MAX_OPS,
            ],
            dtype=np.float32,
        )

    def _retrieve(self):
        query = self.queries[self.n_ops % len(self.queries)]
        hits = retrieve(query, TOP_K, exclude=set(self.cited))
        self.top_score = hits[0][1] if hits else 0.0
        self.best_score = max(self.best_score, self.top_score)
        for sid, _ in hits:
            if len(self.cited) < MAX_EVIDENCE:
                self.cited.append(sid)

    def _cross_check(self):
        """Mean pairwise token overlap between gathered evidence sentences --
        a stand-in for self-consistency across independently retrieved sets.
        Low agreement on plentiful evidence is the signal that should push the
        policy toward abstain rather than a confident commit.
        """
        if len(self.cited) < 2:
            self.agreement = 0.0
            return
        sims = []
        for i, a in enumerate(self.cited):
            for b in self.cited[i + 1 :]:
                ta, tb = tokens(SENTENCES[a]), tokens(SENTENCES[b])
                sims.append(len(ta & tb) / max(len(ta | tb), 1))
        self.agreement = float(np.mean(sims))

    def step(self, action):
        action = int(action)
        self.n_ops += 1

        if action == RETRIEVE:
            self._retrieve()
        elif action == DECOMPOSE:
            if not self.decomposed:
                self.queries = decompose(self.claim)
                self.decomposed = True
        elif action == CROSS_CHECK:
            self._cross_check()
        else:
            pred, conf = (NEI, 0.0) if action == ABSTAIN else COMMITS[action]
            return self._finish(pred, conf)

        if self.n_ops >= MAX_OPS:  # budget exhausted -> forced low-confidence abstain
            return self._finish(NEI, 0.0)
        return self._obs(), 0.0, False, False, {}

    def _finish(self, pred: str, conf: float):
        r, parts = R.total(self.gold, pred, conf, self.gold_ids, self.cited, self.n_ops)
        info = {
            "claim": self.claim,
            "gold": self.gold,
            "pred": pred,
            "confidence": conf,
            "cited": list(self.cited),
            "correct": self.gold == pred,
            "decomposed": self.decomposed,
            "is_compound": self.is_compound,
            **parts,
        }
        return self._obs(), r, True, False, info
