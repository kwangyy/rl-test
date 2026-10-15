"""Does decomposing help? (AG-2) Fixed policies on the claims the decomposer splits.

On every claim the cached decomposer split into 2+ sub-claims, compare fixed
policies with the same verifier:

    full             search the claim, verify the claim (the fixed policy, read from its trace)
    each             search each sub-claim, verify each sub-claim on its own search, combine
    pool             search each sub-claim, verify the full claim once on the union of those
                     searches; tests the retrieval gain only
    each-pooled      search the claim and each sub-claim, verify each sub-claim on the union,
                     so a sub-claim whose own search missed can use the others' pages
    each-pooled-ctx  as each-pooled, and the verifier also sees the original claim next to
                     the sub-claim, so it is not judged out of context

A union is 5 pages (as `full`), taken round-robin across the searches, the
claim's own search first. `each*` combine the sub-claim verdicts with a hard
rule (any REFUTED refutes, all SUPPORTED supports, else NEI); the report also
scores a soft combine from the verifier's label probabilities, P(SUPPORTED) =
prod P_S, P(REFUTED) = 1 - prod (1 - P_R), NEI the rest, argmax. Traces go to
verifier/data/traces/ (schema 2, policy decompose-<name>) for RL-3; a policy
whose trace is complete is not rerun.

    python -m agent.decompose --n 900     # the cache, first
    python -m agent.eval_decompose --adapter verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints \\
        --full verifier/data/traces/fever_dev_fixed-Qwen_Qwen3.5-2B_n6000_n900.jsonl
"""

import argparse
import json
import os
import time

from agent.agent_loop import TRACE_DIR, FeverSearch, LocalVerify, reward_version
from agent.decompose import Decomposer, cache_path
from agent.prompted import NEI, Episode, to_row
from verifier.model import (BIG_MODEL, K_PAGES, LABELS, LOCAL_MODEL, MAX_SENTS_PER_PAGE, SEED, adapter_label,
                            sample_claims)

K = K_PAGES * MAX_SENTS_PER_PAGE
POLICIES = ["each", "pool", "each-pooled", "each-pooled-ctx"]


def round_robin_pages(hit_lists):
    """Union of several searches' hits: whole pages, best rank first across searches, K_PAGES of them."""
    pages, by_page = [], {}
    for hs in hit_lists:
        for h in hs:
            by_page.setdefault(h["page"], []).append(h)
    ranked = [list(dict.fromkeys(h["page"] for h in hs)) for hs in hit_lists]
    for rank in range(max(map(len, ranked), default=0)):
        for r in ranked:
            if rank < len(r) and r[rank] not in pages:
                pages.append(r[rank])
    seen, out = set(), []
    for p in pages[:K_PAGES]:
        for h in by_page[p]:
            if (h["page"], h["line"]) not in seen:
                seen.add((h["page"], h["line"]))
                out.append(h)
    return out[:K]


def with_context(sub, claim):
    return f'{sub} (This is one part of the claim: "{claim}")'


def ids_of(hits):
    return [f"{h['page']}:{h['line']}" for h in hits]


def run_chunk(claims, decomp, search, verify, infos):
    """One chunk of claims under every policy in infos (name -> run_info). Returns name -> rows."""
    subs = [decomp.run(c["claim"]) for c in claims]
    pooled = any(p.startswith("each-pooled") for p in infos)
    queries = [(i, j, s) for i, d in enumerate(subs) for j, s in enumerate(d["subclaims"])]
    if pooled:
        queries += [(i, None, c["claim"]) for i, c in enumerate(claims)]
    t0 = time.time()
    hits = dict(zip([(i, j) for i, j, _ in queries], search.batch([q for _, _, q in queries], K)))
    s_search = (time.time() - t0) / len(queries)
    sub_hits = [[hits[(i, j)] for j in range(len(d["subclaims"]))] for i, d in enumerate(subs)]

    out = {}
    for name, info in infos.items():
        eps = [Episode(c) for c in claims]
        for i, (ep, d) in enumerate(zip(eps, subs)):
            ep.subclaims = d["subclaims"]
            ep.ctl_in, ep.ctl_out = d["tokens_in"], d["tokens_out"]
            ep.log("decompose", {"claim": ep.c["claim"]}, {"subclaims": d["subclaims"]}, 0.0)
            if name.startswith("each-pooled"):
                ep.log("search", {"query": ep.c["claim"], "k": K}, {"hits": hits[(i, None)]}, s_search)
            for j, s in enumerate(d["subclaims"]):
                ep.log("search", {"query": s, "k": K}, {"hits": hits[(i, j)]}, s_search)

        jobs = []  # (episode, subclaim index or None, verifier text, evidence ids)
        for i, ep in enumerate(eps):
            if name == "each":
                jobs += [(ep, j, s, ids_of(sub_hits[i][j])) for j, s in enumerate(ep.subclaims)]
            elif name == "pool":
                jobs.append((ep, None, ep.c["claim"], ids_of(round_robin_pages(sub_hits[i]))))
            else:
                ev = ids_of(round_robin_pages([hits[(i, None)]] + sub_hits[i]))
                ctx = name.endswith("-ctx")
                jobs += [(ep, j, with_context(s, ep.c["claim"]) if ctx else s, ev)
                         for j, s in enumerate(ep.subclaims)]
        t0 = time.time()
        res = verify.batch([t for _, _, t, _ in jobs], [ev for _, _, _, ev in jobs])
        s_verify = (time.time() - t0) / len(jobs)
        for (ep, j, text, ev), r in zip(jobs, res):
            tin, tout = r.pop("_tokens")
            ep.v_in, ep.v_out = getattr(ep, "v_in", 0) + tin, getattr(ep, "v_out", 0) + tout
            if j is None:
                ep.claim_verdict = r
            else:
                ep.sub_verdicts[j] = r
            ep.log("verify", {"claim": text, "evidence": ev, "subclaim": j}, r, s_verify)
        for ep in eps:
            ep.log("commit", {}, {}, 0.0)
            ep.end = "commit"
        out[name] = [to_row(ep, info) for ep in eps]
    return out


def soft_verdict(row):
    """Soft combine of a decompose-each* row's sub-claim verdicts; the row's own verdict if it has none."""
    probs = [s["result"].get("token_prob") for s in row["steps"]
             if s["action"] == "verify" and s["args"].get("subclaim") is not None]
    if not probs or not all(probs):
        return row["final"]["verdict"]
    p_s, p_not_r = 1.0, 1.0
    for p in probs:
        p_s *= p.get("SUPPORTED", 0)
        p_not_r *= 1 - p.get("REFUTED", 0)
    dist = {"SUPPORTED": p_s, "REFUTED": 1 - p_not_r, NEI: max(0.0, p_not_r - p_s)}
    return max(dist, key=dist.get)


def load(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def compare(runs, ids):
    """runs: name -> {id: trace row}. Prints metrics on the shared ids."""
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")

    def fever_ok(r):
        if r["final"]["verdict"] != r["gold"]:
            return False
        return r["gold"] == NEI or any(set(s) <= set(r["final"]["evidence"]) for s in r["gold_evidence"])

    names = list(runs)
    rows = {n: [runs[n][i] for i in ids] for n in names}
    print(f"\n== {len(ids)} claims split by the decomposer ==")
    print(f"  {'':18s}" + "".join(f"{n:>17s}" for n in names))
    lines = [("label accuracy", lambda rs: mean([r["final"]["verdict"] == r["gold"] for r in rs])),
             ("  soft combine", lambda rs: mean([soft_verdict(r) == r["gold"] for r in rs])),
             ("FEVER score", lambda rs: mean([fever_ok(r) for r in rs]))]
    lines += [(f"recall {L[:9]}", lambda rs, L=L: mean([r["final"]["verdict"] == L for r in rs if r["gold"] == L]))
              for L in LABELS]
    lines += [("searches / claim", lambda rs: mean([r["cost"]["n_search"] for r in rs])),
              ("verifies / claim", lambda rs: mean([r["cost"]["n_verify"] for r in rs])),
              ("tokens / claim", lambda rs: mean([r["cost"]["tokens_in"] + r["cost"]["tokens_out"] for r in rs]))]
    for label, f in lines:
        print(f"  {label:18s}" + "".join(f"{f(rows[n]):17.3f}" for n in names))
    base = names[0]
    ok = lambda n, i: runs[n][i]["final"]["verdict"] == runs[n][i]["gold"]
    for n in names[1:]:
        print(f"  {n} vs {base}: fixes {sum(ok(n, i) and not ok(base, i) for i in ids)}, "
              f"breaks {sum(ok(base, i) and not ok(n, i) for i in ids)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LOCAL_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--decomposer", default=BIG_MODEL)
    ap.add_argument("--full", required=True, help="trace of the fixed policy with the same verifier, n=--n")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cache = cache_path(a.decomposer, a.n)
    if not os.path.exists(cache):
        raise SystemExit(f"no decomposer cache at {cache}; run python -m agent.decompose --n {a.n} first")
    decomp = Decomposer(cache=cache)
    claims = [c for c in sample_claims(a.n, SEED) if len(decomp(c["claim"])) > 1]
    if any(not d["cached"] for d in map(decomp.run, (c["claim"] for c in claims))):
        raise SystemExit("cache is incomplete; rerun python -m agent.decompose")

    policies = a.policies.split(",")
    suffix = (adapter_label(a.adapter) if a.adapter else "base") + (f"_{a.tag}" if a.tag else "")
    paths = {p: os.path.join(TRACE_DIR, f"fever_dev_decompose-{p}-{suffix}_n{a.n}.jsonl") for p in policies}
    want = {c["id"] for c in claims}
    todo_p = [p for p in policies if not want <= set(load(paths[p]))]
    print(f"{len(claims)} split claims; to run: {', '.join(todo_p) or 'nothing'}")
    if todo_p:
        search = FeverSearch()
        verify = LocalVerify(search, a.model, a.adapter)
        base = {"controller": None, "decomposer": decomp.name, "verifier": verify.name,
                "reward_version": reward_version(), "seed": SEED}
        for p in todo_p:
            done = set(load(paths[p]))
            todo = [c for c in claims if c["id"] not in done]
            t0 = time.time()
            for i in range(0, len(todo), a.chunk):
                rows = run_chunk(todo[i:i + a.chunk], decomp, search, verify,
                                 {p: {**base, "policy": f"decompose-{p}"}})[p]
                with open(paths[p], "a", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"  {p}: {i + len(rows)}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)

    runs = {"full": load(a.full), **{p: load(paths[p]) for p in policies}}
    ids = [c["id"] for c in claims if all(c["id"] in r for r in runs.values())]
    compare(runs, ids)
