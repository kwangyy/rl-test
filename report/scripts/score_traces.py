"""Summarise controller traces (schema 3) committed in git.

python -I score_traces.py <repo> <branch> <path> [<path> ...]
Prints FEVER, accuracy, mean harm, NEI P/R, Brier/ECE of final confidence,
ops / searches / verifies per claim, seconds and tokens per claim, abstain rate.
"""
import gzip, json, subprocess, sys

sys.stdout.reconfigure(encoding="utf-8")
REPO, BR = sys.argv[1], sys.argv[2]
L = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
H = {("SUPPORTED", "REFUTED"): 1, ("SUPPORTED", "NOT_ENOUGH_INFO"): 1,
     ("REFUTED", "SUPPORTED"): 1.5, ("REFUTED", "NOT_ENOUGH_INFO"): 1,
     ("NOT_ENOUGH_INFO", "SUPPORTED"): 1.75, ("NOT_ENOUGH_INFO", "REFUTED"): 1}


def load(path):
    raw = subprocess.run(["git", "-C", REPO, "show", f"{BR}:{path}"], capture_output=True,
                         check=True).stdout
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    return [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]


def fever_ok(r):
    f = r["final"]
    if f.get("verdict") != r["gold"]:
        return False
    if r["gold"] == "NOT_ENOUGH_INFO":
        return True
    cited = set(f.get("evidence") or [])
    return any(set(g) <= cited for g in r.get("gold_evidence") or [])


print("| trace | n | FEVER | Acc | Harm | R→S | N→S | NEI P | NEI R | Brier | ECE | ops | search | follow | verify | s/claim | tok in | abstain |")
print("|" + "---|" * 18)
for p in sys.argv[3:]:
    rows = load(p)
    n = len(rows)
    acc = fev = harm = rs = ns = 0
    pn = tn = 0
    cr = []
    agg = dict(n_ops=0, n_search=0, n_follow=0, n_verify=0, seconds=0, tokens_in=0)
    ab = 0
    for r in rows:
        g, f = r["gold"], r["final"]
        p_ = f.get("verdict")
        ok = p_ == g
        acc += ok; fev += fever_ok(r)
        harm += H.get((g, p_), 0) if p_ in L else 1
        rs += (g, p_) == ("REFUTED", "SUPPORTED"); ns += (g, p_) == ("NOT_ENOUGH_INFO", "SUPPORTED")
        pn += p_ == "NOT_ENOUGH_INFO"; tn += ok and g == "NOT_ENOUGH_INFO"
        if f.get("confidence") is not None:
            cr.append((float(f["confidence"]), 1.0 if ok else 0.0))
        for k in agg:
            agg[k] += (r.get("cost") or {}).get(k) or 0
        ab += f.get("ended_by") == "abstain"
    brier = sum((c - y) ** 2 for c, y in cr) / len(cr) if cr else float("nan")
    ece = 0.0
    for b in range(10):
        bi = [(c, y) for c, y in cr if b / 10 <= c < (b + 1) / 10 or (b == 9 and c >= 1.0)]
        if bi:
            ece += len(bi) / len(cr) * abs(sum(c for c, _ in bi) / len(bi) - sum(y for _, y in bi) / len(bi))
    nn = sum(r["gold"] == "NOT_ENOUGH_INFO" for r in rows)
    name = p.split("/")[-1].replace(".jsonl.gz", "").replace(".jsonl", "")
    print(f"| {name} | {n} | {fev/n:.3f} | {acc/n:.3f} | {harm/n:.3f} | {rs} | {ns} | {tn/pn if pn else 0:.3f} | "
          f"{tn/nn:.3f} | {brier:.3f} | {ece:.3f} | {agg['n_ops']/n:.2f} | {agg['n_search']/n:.2f} | "
          f"{agg['n_follow']/n:.2f} | {agg['n_verify']/n:.2f} | {agg['seconds']/n:.2f} | {agg['tokens_in']/n:.0f} | {ab/n:.3f} |")
