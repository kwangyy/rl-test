"""Payoff of a two-model cascade from saved predictions: answer with the 2B,
hand the claim to the big model when the 2B's verdict-token probability is
below a threshold. No new model calls.

Thresholds are fitted on a calibration file and applied to a test file, so
the reported numbers are not tuned on the claims they are scored on. The
oracle router (hand off exactly the claims the big model fixes) is the upper
bound a learned controller could reach with these two models.

    python -m verifier.evaluation.cascade_payoff \
        --small-cal verifier/evaluation/results/sft-direct-cap160/cascade-cal900/predictions.jsonl \
        --big-cal   verifier/evaluation/results/big-direct-cap160/cascade-cal900/predictions.jsonl \
        --small     verifier/evaluation/results/sft-direct-cap160/cascade-dev900/predictions.jsonl \
        --big       verifier/evaluation/results/big-direct-cap160/cascade-dev900/predictions.jsonl
"""

import argparse
import json

import numpy as np

from verifier.model import BIG_MODEL, LABELS

PRICES = {BIG_MODEL: (0.15, 0.47)}  # $/M tokens in/out, OpenRouter 2026-09-19
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {str(r["id"]): r for r in rows}


def pair(small, big):
    ids = sorted(set(small) & set(big))
    if len(ids) != len(small) or len(ids) != len(big):
        raise SystemExit(f"claim IDs differ: {len(small)} small, {len(big)} big, {len(ids)} shared")
    return ids


def conf(r):
    return float(r["token_conf"]) if r.get("token_conf") is not None else 0.0


def cost(r):
    cin, cout = PRICES.get(r["model"], PRICES[BIG_MODEL])
    return (r["in_tokens"] * cin + r["out_tokens"] * cout) / 1e6


def outcome(small, big, ids, routed, metric):
    n = len(ids)
    return {
        "score": sum(bool((big if i in routed else small)[i][metric]) for i in ids) / n,
        "routed": len(routed) / n,
        "cost_usd": sum(cost(big[i]) for i in routed),
    }


def threshold_for(small, ids, frac):
    """Smallest probability such that routing conf < t sends about frac of claims."""
    cs = sorted(conf(small[i]) for i in ids)
    k = round(frac * len(cs))
    return cs[k] if k < len(cs) else float("inf")


def route(small, ids, t):
    return {i for i in ids if conf(small[i]) < t}


def features(r):
    lp = r.get("label_probs") or {}
    return ([lp.get(k, 0.0) for k in LABELS] + [float(r["pred"] == k) for k in LABELS]
            + [r["n_shown"] / 50, len(r["claim"]) / 100, len(r.get("cited") or []) / 3, 1.0])


def logreg(x, y, l2=1e-2, steps=3000, lr=0.5):
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-x @ w))
        w -= lr * (x.T @ (p - y) / len(y) + l2 * w)
    return lambda z: 1 / (1 + np.exp(-z @ w))


def learned_router(small_cal, big_cal, cal_ids, small, big, ids, metric):
    """Two logistic regressions on the 2B's own output (label probabilities,
    verdict, context size, claim length, citations) predict whether each model
    is right; route when P(big right) - P(2B right) exceeds a margin fitted on
    the calibration claims."""
    xc = np.array([features(small_cal[i]) for i in cal_ids])
    xt = np.array([features(small[i]) for i in ids])
    p_big = logreg(xc, np.array([float(bool(big_cal[i][metric])) for i in cal_ids]))
    p_small = logreg(xc, np.array([float(bool(small_cal[i][metric])) for i in cal_ids]))
    gain_c, gain_t = p_big(xc) - p_small(xc), p_big(xt) - p_small(xt)
    margin = max(sorted(set(np.round(gain_c, 4))), key=lambda t: outcome(
        small_cal, big_cal, cal_ids, {i for i, g in zip(cal_ids, gain_c) if g > t}, metric)["score"])
    return {"margin": float(margin), **outcome(small, big, ids, {i for i, g in zip(ids, gain_t) if g > margin}, metric)}


def evaluate(small_cal, big_cal, small, big, metrics):
    cal_ids, ids = pair(small_cal, big_cal), pair(small, big)
    out = {"n_cal": len(cal_ids), "n": len(ids), "metrics": {}}
    for m in metrics:
        res = {
            "small": outcome(small, big, ids, set(), m),
            "big": outcome(small, big, ids, set(ids), m),
            "oracle": outcome(small, big, ids, {i for i in ids if big[i][m] and not small[i][m]}, m),
            "route_small_nei": outcome(small, big, ids, {i for i in ids if small[i]["pred"] == "NOT_ENOUGH_INFO"}, m),
            "route_small_non_nei": outcome(small, big, ids, {i for i in ids if small[i]["pred"] != "NOT_ENOUGH_INFO"}, m),
            "learned": learned_router(small_cal, big_cal, cal_ids, small, big, ids, m),
            "fractions": [],
        }
        for f in FRACTIONS:
            t = threshold_for(small_cal, cal_ids, f)
            res["fractions"].append({"target": f, "threshold": t, **outcome(small, big, ids, route(small, ids, t), m)})
        # threshold with the best calibration score
        cands = sorted({conf(small_cal[i]) for i in cal_ids} | {float("inf")})
        best = max(cands, key=lambda t: (outcome(small_cal, big_cal, cal_ids, route(small_cal, cal_ids, t), m)["score"], -t))
        res["cal_best"] = {"threshold": best, **outcome(small, big, ids, route(small, ids, best), m)}
        out["metrics"][m] = res
    return out


def report(name, res):
    print(f"\n== {name}  (threshold fitted on {res['n_cal']} calibration claims, scored on {res['n']})")
    for m, r in res["metrics"].items():
        s, b, o = r["small"]["score"], r["big"]["score"], r["oracle"]["score"]
        print(f"  [{m}] 2B {s:.3f}   big {b:.3f} (${r['big']['cost_usd']:.3f})   "
              f"oracle router {o:.3f} at {r['oracle']['routed']:.0%} routed")
        for row in r["fractions"] + [{"target": "best", **r["cal_best"]}]:
            tgt = row["target"] if isinstance(row["target"], str) else f"{row['target']:.0%}"
            print(f"    target {tgt:>4}  t={row['threshold']:.3f}  routed {row['routed']:4.0%}  "
                  f"{row['score']:.3f}  ${row['cost_usd']:.3f}")
        for key, label in [("route_small_nei", "route when 2B says NEI"),
                           ("route_small_non_nei", "route when 2B says SUP/REF"),
                           ("learned", "learned router (logreg)")]:
            row = r[key]
            print(f"    {label:28s} routed {row['routed']:4.0%}  {row['score']:.3f}  ${row['cost_usd']:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--small-cal", required=True)
    p.add_argument("--big-cal", required=True)
    p.add_argument("--small", required=True)
    p.add_argument("--big", required=True)
    p.add_argument("--metrics", default="correct,fever_score",
                   help="comma-separated boolean prediction fields (AVeriTeC has only 'correct')")
    p.add_argument("--name", default="cascade")
    p.add_argument("--output", help="write the full result as JSON")
    args = p.parse_args()
    res = evaluate(load(args.small_cal), load(args.big_cal), load(args.small), load(args.big),
                   args.metrics.split(","))
    res["inputs"] = {k: getattr(args, k) for k in ("small_cal", "big_cal", "small", "big")}
    report(args.name, res)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
