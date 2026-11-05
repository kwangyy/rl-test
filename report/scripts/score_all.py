"""Score committed verifier predictions.jsonl files from git branches.

Usage: python -I score_all.py <repo>  (prints a markdown table)
Harm matrix (report Eq. 3): rows gold, cols pred.  Invalid predictions get harm 1
and count as wrong.  Token Brier/ECE use token_conf vs correctness, 10 equal bins,
on rows that have a token_conf.
"""
import json, subprocess, sys

sys.stdout.reconfigure(encoding="utf-8")
REPO = sys.argv[1]
H = {("SUPPORTED", "REFUTED"): 1, ("SUPPORTED", "NOT_ENOUGH_INFO"): 1,
     ("REFUTED", "SUPPORTED"): 1.5, ("REFUTED", "NOT_ENOUGH_INFO"): 1,
     ("NOT_ENOUGH_INFO", "SUPPORTED"): 1.75, ("NOT_ENOUGH_INFO", "REFUTED"): 1}
L = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def files(branch):
    out = git("ls-tree", "-r", "--name-only", branch, "verifier/evaluation/results/")
    return [f for f in out.splitlines() if f.endswith(".jsonl") and
            ("predictions" in f or "/archive/" in f)]


def score(rows):
    n = len(rows)
    acc = sum(bool(r.get("correct")) for r in rows) / n
    fev = sum(bool(r.get("fever_score")) for r in rows) / n
    harm, rs, ns, inval = 0.0, 0, 0, 0
    conf = {g: {p: 0 for p in L + ["INVALID"]} for g in L}
    for r in rows:
        g, p = r["gold"], r.get("pred")
        if p not in L:
            inval += 1; harm += 1; conf[g]["INVALID"] += 1; continue
        conf[g][p] += 1
        harm += H.get((g, p), 0)
        rs += (g, p) == ("REFUTED", "SUPPORTED")
        ns += (g, p) == ("NOT_ENOUGH_INFO", "SUPPORTED")
    pn = sum(conf[g]["NOT_ENOUGH_INFO"] for g in L)
    nei_p = conf["NOT_ENOUGH_INFO"]["NOT_ENOUGH_INFO"] / pn if pn else float("nan")
    nei_r = conf["NOT_ENOUGH_INFO"]["NOT_ENOUGH_INFO"] / sum(conf["NOT_ENOUGH_INFO"].values())
    cr = [(float(r["token_conf"]), 1.0 if r.get("correct") else 0.0) for r in rows
          if r.get("token_conf") is not None and r.get("pred") in L]
    brier = sum((c - y) ** 2 for c, y in cr) / len(cr) if cr else float("nan")
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        bin_ = [(c, y) for c, y in cr if (lo <= c < hi) or (b == 9 and c == 1.0)]
        if bin_:
            ece += len(bin_) / len(cr) * abs(sum(c for c, _ in bin_) / len(bin_) -
                                             sum(y for _, y in bin_) / len(bin_))
    sc = [float(r["stated_conf"]) for r in rows if r.get("stated_conf") is not None]
    out = sum(r.get("out_tokens") or 0 for r in rows) / n
    return dict(n=n, acc=acc, fever=fev, harm=harm / n, rs=rs, ns=ns, nei_p=nei_p,
                nei_r=nei_r, brier=brier, ece=ece if cr else float("nan"), inval=inval,
                stated=len(sc), out=out, conf=conf,
                retrieval=rows[0].get("retrieval"), style=rows[0].get("prompt_style"))


seen = set()
res = []
for br in sys.argv[2:]:
    for f in files(br):
        if f in seen:
            continue
        seen.add(f)
        rows = [json.loads(l) for l in git("show", f"{br}:{f}").splitlines() if l.strip()]
        if len(rows) < 100 or "gold" not in rows[0]:
            continue
        s = score(rows); s["file"] = f; s["branch"] = br
        res.append(s)

print("| file | branch | ret | style | n | Acc | FEVER | Harm | R→S | N→S | NEI P | NEI R | Brier tok | ECE tok | Invalid | out tok |")
print("|" + "---|" * 16)
for s in res:
    name = s["file"].replace("verifier/evaluation/results/", "").replace("/predictions.jsonl", "")
    print(f"| {name} | {s['branch']} | {s['retrieval']} | {s['style']} | {s['n']} | {s['acc']:.3f} | {s['fever']:.3f} | "
          f"{s['harm']:.3f} | {s['rs']} | {s['ns']} | {s['nei_p']:.3f} | {s['nei_r']:.3f} | {s['brier']:.3f} | "
          f"{s['ece']:.3f} | {s['inval']} | {s['out']:.0f} |")
json.dump(res, open(sys.argv[0].replace("score_all.py", "scores.json"), "w"), indent=1)
