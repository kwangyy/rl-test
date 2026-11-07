"""Offline controller environment over the action cache (AG-2).

Every episode starts with the fixed policy's step (`bm25` search, SFT verify).
The controller then either STOPs or takes one more evidence/verify action it
has not taken yet, up to `budget` extra actions. The committed verdict is the
verdict of the latest action taken. Every outcome is read from
`agent/action_cache.py` output, so an episode costs no model calls.

Reward = score(final verdict) - lam * sum(action costs), where score is label
accuracy or FEVER score. Costs are in units of one local 2B verify call; the
big model costs `big_cost` units.
"""

import json
import os

import numpy as np

from verifier.model import LABELS

FIRST = "bm25"
EXTRA = ("bm25-k20", "hybrid-k20", "pages10-k20", "follow-k20", "flash-bm25")
STOP = 0  # action index 0 is STOP, i + 1 is EXTRA[i]
N_ACTIONS = 1 + len(EXTRA)


def load_cache(set_dir, actions=(FIRST,) + EXTRA):
    """{claim id: {action: row}} for claims present under every action."""
    per = {}
    for a in actions:
        with open(os.path.join(set_dir, f"{a}.jsonl"), encoding="utf-8") as f:
            per[a] = {str(r["id"]): r for r in map(json.loads, f)}
    ids = sorted(set.intersection(*(set(d) for d in per.values())))
    return {i: {a: per[a][i] for a in actions} for i in ids}


def action_cost(action, big_cost):
    return big_cost if action.startswith("flash") else 1.0


def _row_features(r):
    lp = r.get("label_probs") or {}
    p = [lp.get(k, 0.0) for k in LABELS]
    pred = [float(r["pred"] == k) for k in LABELS]
    scores = r.get("hit_scores") or [0.0]
    return p + pred + [float(r["token_conf"] or 0.0), r["n_shown"] / 100, scores[0] / 20,
                       len(r.get("pages") or []) / 10, len(r.get("cited") or []) / 3]


ROW_DIM = 11


def features(claim_rows, taken, step, budget):
    """Observation: one block per action (zeros until taken), agreement with
    the first verdict, steps left and claim length. Uses no gold fields."""
    first = claim_rows[FIRST]
    x = _row_features(first)
    for a in EXTRA:
        if a in taken:
            r = claim_rows[a]
            x += [1.0] + _row_features(r) + [float(r["pred"] == first["pred"])]
        else:
            x += [0.0] * (ROW_DIM + 2)
    x += [(budget - step) / budget, len(first["claim"]) / 200, 1.0]
    return np.array(x, dtype=np.float32)


OBS_DIM = ROW_DIM + len(EXTRA) * (ROW_DIM + 2) + 3


def final_row(claim_rows, taken):
    return claim_rows[taken[-1]] if taken else claim_rows[FIRST]


def episode_reward(claim_rows, taken, metric, lam, big_cost):
    r = final_row(claim_rows, taken)
    return float(bool(r[metric])) - lam * (1.0 + sum(action_cost(a, big_cost) for a in taken))


def mask(taken, step, budget):
    m = np.ones(N_ACTIONS, dtype=bool)
    if step >= budget:
        m[1:] = False
    for i, a in enumerate(EXTRA):
        if a in taken:
            m[i + 1] = False
    return m


def sequences(budget):
    """Every ordered action sequence of length 0..budget without repeats."""
    out = [()]
    frontier = [()]
    for _ in range(budget):
        frontier = [s + (a,) for s in frontier for a in EXTRA if a not in s]
        out += frontier
    return out


def evaluate_policy(cache, choose, metric, lam, big_cost, budget):
    """Run a policy choose(claim_rows, taken, step) -> action index on every claim."""
    tot = {"score": 0.0, "accuracy": 0.0, "fever_score": 0.0, "cost": 0.0, "reward": 0.0, "big_calls": 0.0}
    counts = {a: 0 for a in EXTRA}
    for rows in cache.values():
        taken = []
        for step in range(budget + 1):
            a = choose(rows, taken, step)
            if a == STOP or step >= budget:
                break
            taken.append(EXTRA[a - 1])
        fin = final_row(rows, taken)
        tot["score"] += bool(fin[metric])
        tot["accuracy"] += bool(fin["correct"])
        tot["fever_score"] += bool(fin["fever_score"])
        tot["cost"] += 1.0 + sum(action_cost(x, big_cost) for x in taken)
        tot["big_calls"] += sum(x.startswith("flash") for x in taken)
        tot["reward"] += episode_reward(rows, taken, metric, lam, big_cost)
        for x in taken:
            counts[x] += 1
    n = len(cache)
    out = {k: v / n for k, v in tot.items()}
    out["action_rate"] = {a: c / n for a, c in counts.items()}
    return out
