"""Pick representative fixed-vs-imitation cases from dev 900 traces.

python -I cases.py <repo> <branch> <fixed trace> <controller trace>
"""
import gzip, json, subprocess, sys

sys.stdout.reconfigure(encoding="utf-8")
REPO, BR, FX, CT = sys.argv[1:5]


def load(path):
    raw = subprocess.run(["git", "-C", REPO, "show", f"{BR}:{path}"], capture_output=True,
                         check=True).stdout
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    return {r["id"]: r for r in (json.loads(l) for l in raw.decode().splitlines() if l.strip())}


def fev(r):
    f = r["final"]
    if f.get("verdict") != r["gold"]:
        return False
    return r["gold"] == "NOT_ENOUGH_INFO" or any(
        set(g) <= set(f.get("evidence") or []) for g in r.get("gold_evidence") or [])


def desc(r):
    out = []
    for s in r["steps"]:
        a, args, res = s["action"], s.get("args") or {}, s.get("result") or {}
        if a == "search":
            out.append(f'search("{args.get("query")}")')
        elif a == "follow":
            out.append(f'follow({json.dumps(args)[:60]})')
        elif a == "decompose":
            out.append(f'decompose -> {json.dumps(res)[:160]}')
        elif a == "verify":
            c = res.get("confidence")
            out.append(f'verify[{args.get("claim", "")[:50]}] -> {res.get("verdict")} '
                       f'({c:.2f}) cites {res.get("evidence")}' if c is not None else f"verify -> {res}")
        else:
            out.append(a)
    return out


fx, ct = load(FX), load(CT)
cats = {"recovery (fixed FEVER-wrong, controller right, >1 search)": [],
        "regression (fixed right, controller wrong)": [],
        "decomposition used": [],
        "failed retry (>=2 verifies, still FEVER-wrong)": []}
for i, r in ct.items():
    f = fx.get(i)
    if not f:
        continue
    acts = [s["action"] for s in r["steps"]]
    if not fev(f) and fev(r) and acts.count("search") > 1:
        cats["recovery (fixed FEVER-wrong, controller right, >1 search)"].append(i)
    if fev(f) and not fev(r):
        cats["regression (fixed right, controller wrong)"].append(i)
    if "decompose" in acts:
        cats["decomposition used"].append(i)
    if acts.count("verify") >= 2 and not fev(r):
        cats["failed retry (>=2 verifies, still FEVER-wrong)"].append(i)
for k, ids in cats.items():
    print(f"\n### {k}: {len(ids)} claims")
    for i in ids[:2]:
        r, f = ct[i], fx[i]
        print(f"\n- id {i}, gold {r['gold']}, claim: \"{r['claim']}\"")
        print(f"  - gold evidence sets: {r.get('gold_evidence')}")
        print(f"  - fixed loop: {f['final'].get('verdict')} cites {f['final'].get('evidence')} (FEVER {int(fev(f))})")
        print(f"  - controller: " + " → ".join(desc(r)))
        print(f"  - controller final: {r['final'].get('verdict')} cites {r['final'].get('evidence')} "
              f"ended_by {r['final'].get('ended_by')} (FEVER {int(fev(r))})")
