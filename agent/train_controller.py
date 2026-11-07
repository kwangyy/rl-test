"""Train the evidence/verify controller with policy-gradient RL on cached
outcomes, and compare it with fixed and hand-written policies (AG-2).

Every policy is fitted on unused FEVER train claims and scored on dev 900. The
environment is agent/controller_env.py.

Policies:
    fixed            the fixed loop: one bm25 search, verify, commit
    best-sequence    the single action sequence with the best train reward
    rule-by-verdict  a lookup table: best sequence for each first verdict
                     (SUPPORTED / REFUTED / NEI); the hand-rule family
    rule-threshold   if first-verdict confidence < t take sequence S, else stop
    rl               masked-softmax MLP trained with REINFORCE (mean-reward
                     baseline, entropy bonus), greedy at test; several seeds
    oracle           the best sequence chosen per claim with the gold label

    python -m agent.train_controller --train verifier/data/controller/train_unused-n1500-seed1 \
        --dev verifier/data/controller/dev_fever-dev-900-ids-<hash> --metric fever_score --lam 0.02
"""

import argparse
import json
import os

import numpy as np
import torch

from agent.controller_env import (EXTRA, FIRST, N_ACTIONS, OBS_DIM, STOP, episode_reward, evaluate_policy,
                                  features, load_cache, mask, sequences)
from verifier.model import LABELS


def seq_policy(seq):
    def choose(rows, taken, step):
        return EXTRA.index(seq[step]) + 1 if step < len(seq) else STOP
    return choose


def best_sequence(cache, seqs, metric, lam, big_cost):
    return max(seqs, key=lambda s: (np.mean([episode_reward(r, list(s), metric, lam, big_cost)
                                             for r in cache.values()]), -len(s)))


def rule_by_verdict(cache, seqs, metric, lam, big_cost):
    table = {}
    for lab in LABELS + [None]:
        sub = {i: r for i, r in cache.items() if r[FIRST]["pred"] == lab}
        table[lab] = best_sequence(sub, seqs, metric, lam, big_cost) if sub else ()

    def choose(rows, taken, step):
        return seq_policy(table[rows[FIRST]["pred"]])(rows, taken, step)
    return choose, {str(k): list(v) for k, v in table.items()}


def rule_threshold(cache, seqs, metric, lam, big_cost):
    conf = lambda r: float(r[FIRST]["token_conf"] or 0.0)
    ts = sorted({round(conf(r), 3) for r in cache.values()})[:: max(1, len(cache) // 50)] + [1.01]
    best, best_val = ((), 0.0), -1e9
    for s in seqs[1:]:
        for t in ts:
            v = np.mean([episode_reward(r, list(s) if conf(r) < t else [], metric, lam, big_cost)
                         for r in cache.values()])
            if v > best_val:
                best, best_val = (s, t), v
    s, t = best

    def choose(rows, taken, step):
        return seq_policy(s if conf(rows) < t else ())(rows, taken, step)
    return choose, {"sequence": list(s), "threshold": t}


def oracle(cache, seqs, metric, lam, big_cost):
    pick = {i: max(seqs, key=lambda s: (episode_reward(r, list(s), metric, lam, big_cost), -len(s)))
            for i, r in cache.items()}
    by_rows = {id(r): pick[i] for i, r in cache.items()}

    def choose(rows, taken, step):
        return seq_policy(by_rows[id(rows)])(rows, taken, step)
    return choose


class Policy(torch.nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(OBS_DIM, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, N_ACTIONS))

    def dist(self, x, m):
        logits = self.net(x).masked_fill(~m, -1e9)
        return torch.distributions.Categorical(logits=logits)


def train_rl(cache, metric, lam, big_cost, budget, seed, epochs=150, batch=128, lr=3e-3, ent=0.05):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pol = Policy()
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    claims = list(cache.values())
    for _ in range(epochs):
        order = rng.permutation(len(claims))
        for b in range(0, len(order), batch):
            logps, ents, rewards = [], [], []
            for j in order[b:b + batch]:
                rows, taken, lp, en = claims[j], [], 0.0, 0.0
                for step in range(budget + 1):
                    m = torch.from_numpy(mask(taken, step, budget))
                    if step >= budget:
                        break
                    d = pol.dist(torch.from_numpy(features(rows, taken, step, budget)), m)
                    a = d.sample()
                    lp, en = lp + d.log_prob(a), en + d.entropy()
                    if a.item() == STOP:
                        break
                    taken.append(EXTRA[a.item() - 1])
                logps.append(lp)
                ents.append(en)
                rewards.append(episode_reward(rows, taken, metric, lam, big_cost))
            r = torch.tensor(rewards)
            adv = (r - r.mean()) / (r.std() + 1e-6)
            loss = -(adv * torch.stack(logps)).mean() - ent * torch.stack(ents).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def choose(rows, taken, step):
        if step >= budget:
            return STOP
        m = torch.from_numpy(mask(taken, step, budget))
        logits = pol.net(torch.from_numpy(features(rows, taken, step, budget))).masked_fill(~m, -1e9)
        return int(logits.argmax())
    return choose


def _fit_q(xs, ys, ms, seed, epochs=300, lr=3e-3, wd=1e-4):
    """Multi-output MLP regression of action values; loss only on legal actions."""
    torch.manual_seed(seed)
    net = Policy().net
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    x, y, m = torch.from_numpy(np.stack(xs)), torch.tensor(np.stack(ys), dtype=torch.float32), torch.from_numpy(np.stack(ms))
    for _ in range(epochs):
        loss = (((net(x) - y) ** 2) * m).sum() / m.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def train_fqi(cache, metric, lam, big_cost, budget, seed):
    """Fitted Q-iteration, backwards from the last decision. Every action's
    outcome is in the cache, so each state gets a target for every legal
    action; earlier steps bootstrap from the fitted later-step network, not
    from the true best continuation."""
    nets = {}
    for step in range(budget, -1, -1):
        xs, ys, ms = [], [], []
        for rows in cache.values():
            for taken in (s for s in sequences(budget) if len(s) == step):
                m = mask(list(taken), step, budget)
                y = np.zeros(N_ACTIONS, dtype=np.float32)
                y[STOP] = episode_reward(rows, list(taken), metric, lam, big_cost)
                for i, a in enumerate(EXTRA):
                    if m[i + 1]:
                        nxt = list(taken) + [a]
                        if step + 1 > budget or step + 1 not in nets:
                            y[i + 1] = episode_reward(rows, nxt, metric, lam, big_cost)
                        else:
                            with torch.no_grad():
                                q = nets[step + 1](torch.from_numpy(features(rows, nxt, step + 1, budget)))
                            q = q.numpy()
                            q[~mask(nxt, step + 1, budget)] = -1e9
                            y[i + 1] = q.max()
                xs.append(features(rows, list(taken), step, budget))
                ys.append(y)
                ms.append(m)
        nets[step] = _fit_q(xs, ys, ms, seed + 17 * step)

    @torch.no_grad()
    def choose(rows, taken, step):
        if step >= budget:
            return STOP
        q = nets[step](torch.from_numpy(features(rows, taken, step, budget))).numpy()
        q[~mask(taken, step, budget)] = -1e9
        return int(q.argmax())
    return choose


def fmt(name, res):
    rate = " ".join(f"{a}={v:.0%}" for a, v in res["action_rate"].items() if v > 0.005)
    return (f"  {name:16s} reward {res['reward']:.3f}  acc {res['accuracy']:.3f}  FEVER {res['fever_score']:.3f}"
            f"  calls {res['cost']:.2f}  big {res['big_calls']:.0%}  [{rate}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--metric", choices=("correct", "fever_score"), default="fever_score")
    ap.add_argument("--lam", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05])
    ap.add_argument("--big-cost", type=float, default=2.0, help="cost of one big-model call, in local verify calls")
    ap.add_argument("--budget", type=int, default=2, help="extra actions after the first search")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--output", help="JSON results")
    a = ap.parse_args()
    train, dev = load_cache(a.train), load_cache(a.dev)
    seqs = sequences(a.budget)
    print(f"train {len(train)} claims, dev {len(dev)} claims, {len(seqs)} sequences, metric {a.metric}, "
          f"big call = {a.big_cost} local calls")
    results = []
    for lam in a.lam:
        args = (a.metric, lam, a.big_cost)
        print(f"\n== lambda {lam}")
        rows = {}
        rows["fixed"] = (seq_policy(()), {})
        s = best_sequence(train, seqs, *args)
        rows["best-sequence"] = (seq_policy(s), {"sequence": list(s)})
        rows["rule-by-verdict"] = rule_by_verdict(train, seqs, *args)
        rows["rule-threshold"] = rule_threshold(train, seqs, *args)
        out = {"lam": lam}
        for name, (choose, info) in rows.items():
            res = evaluate_policy(dev, choose, a.metric, lam, a.big_cost, a.budget)
            out[name] = {**res, "fitted": info}
            print(fmt(name, res) + (f"  {info}" if info else ""))
        rl = []
        for seed in range(a.seeds):
            choose = train_rl(train, a.metric, lam, a.big_cost, a.budget, seed)
            res = evaluate_policy(dev, choose, a.metric, lam, a.big_cost, a.budget)
            rl.append(res)
            print(fmt(f"rl seed {seed}", res))
        out["rl"] = rl
        fqi = []
        for seed in range(a.seeds):
            res = evaluate_policy(dev, train_fqi(train, a.metric, lam, a.big_cost, a.budget, seed),
                                  a.metric, lam, a.big_cost, a.budget)
            fqi.append(res)
            print(fmt(f"fqi seed {seed}", res))
        out["fqi"] = fqi
        res = evaluate_policy(dev, oracle(dev, seqs, *args), a.metric, lam, a.big_cost, a.budget)
        out["oracle"] = res
        print(fmt("oracle (dev)", res))
        results.append(out)
    if a.output:
        os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump({"args": vars(a), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
