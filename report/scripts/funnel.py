"""Stage-wise evidence funnel on non-NEI claims of controller traces.

python -I funnel.py <repo> <branch> <path> [...]
Stages (complete gold set = every sentence of at least one annotated set):
 pages  : all pages of some gold set appeared among search/follow hits
 hits   : a complete gold set appeared among search/follow hits (union over steps)
 union  : a complete gold set is inside the union of all verify inputs
 single : a complete gold set is inside one verify input
 label  : final verdict correct
 fever  : verdict correct and a complete gold set cited
Error split: correct label but incomplete citations; wrong label despite gold in one input.
"""
import gzip, json, subprocess, sys

sys.stdout.reconfigure(encoding="utf-8")
REPO, BR = sys.argv[1], sys.argv[2]


def load(path):
    raw = subprocess.run(["git", "-C", REPO, "show", f"{BR}:{path}"], capture_output=True,
                         check=True).stdout
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    return [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]


def complete(sets, have):
    return any(set(g) <= have for g in sets)


print("| trace | n | gold pages found | gold set retrieved | in union of verifier inputs | in one verifier input | label correct | FEVER | correct label, incomplete cites | wrong label, gold in input |")
print("|" + "---|" * 10)
for p in sys.argv[3:]:
    rows = [r for r in load(p) if r["gold"] != "NOT_ENOUGH_INFO"]
    c = dict(pages=0, hits=0, union=0, single=0, label=0, fever=0, inc=0, wrong_in=0)
    for r in rows:
        gs = r.get("gold_evidence") or []
        pages, hits, union, single = set(), set(), set(), False
        for s in r["steps"]:
            res = s.get("result") or {}
            if isinstance(res, dict) and "hits" in res:
                for h in res["hits"]:
                    pages.add(h["page"]); hits.add(f"{h['page']}:{h['line']}")
            if s["action"] == "verify":
                ev = set((s.get("args") or {}).get("evidence") or [])
                union |= ev
                single = single or complete(gs, ev)
        pg = any({x.rsplit(":", 1)[0] for x in g} <= pages for g in gs)
        f = r["final"]
        ok = f.get("verdict") == r["gold"]
        cited_ok = complete(gs, set(f.get("evidence") or []))
        c["pages"] += pg; c["hits"] += complete(gs, hits); c["union"] += complete(gs, union)
        c["single"] += single; c["label"] += ok; c["fever"] += ok and cited_ok
        c["inc"] += ok and not cited_ok; c["wrong_in"] += (not ok) and single
    n = len(rows)
    name = p.split("/")[-1].replace(".jsonl.gz", "").replace(".jsonl", "")
    pct = lambda k: f"{c[k]} ({c[k]/n:.1%})"
    print(f"| {name} | {n} | {pct('pages')} | {pct('hits')} | {pct('union')} | {pct('single')} | {pct('label')} | {pct('fever')} | {c['inc']} | {c['wrong_in']} |")
