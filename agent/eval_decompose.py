"""Does decomposing help? (AG-2) Fixed policies on the claims the decomposer splits.

On every claim the cached decomposer split into 2+ sub-claims, compare three
fixed policies with the same verifier:

    full   search the claim, verify the claim (the fixed policy, read from its trace)
    each   decompose, search each sub-claim, verify each sub-claim, combine
           (any REFUTED refutes, all SUPPORTED supports, else NOT_ENOUGH_INFO)
    pool   decompose, search each sub-claim, verify the full claim on the union of
           those searches (same 5 pages as `full`, taken round-robin across searches)

`each` tests the decomposition itself; `pool` tests only the retrieval gain, at
one verify. Traces go to verifier/data/traces/ (schema 2, policy
decompose-each / decompose-pool) for RL-3.

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
from agent.prompted import Episode, to_row
from verifier.model import (BIG_MODEL, K_PAGES, LABELS, LOCAL_MODEL, MAX_SENTS_PER_PAGE, SEED, adapter_label,
                            sample_claims)

K = K_PAGES * MAX_SENTS_PER_PAGE


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


def run_chunk(claims, decomp, search, verify, info_each, info_pool):
    eps_each, eps_pool = [Episode(c) for c in claims], [Episode(c) for c in claims]
    subs = [decomp.run(c["claim"]) for c in claims]
    for ep_e, ep_p, d in zip(eps_each, eps_pool, subs):
        for ep in (ep_e, ep_p):
            ep.subclaims = d["subclaims"]
            ep.ctl_in, ep.ctl_out = d["tokens_in"], d["tokens_out"]
            ep.log("decompose", {"claim": ep.c["claim"]}, {"subclaims": d["subclaims"]}, 0.0)

    pairs = [(i, j, s) for i, d in enumerate(subs) for j, s in enumerate(d["subclaims"])]
    t0 = time.time()
    hits = search.batch([s for _, _, s in pairs], K)
    s_search = (time.time() - t0) / len(pairs)
    per_claim = [[] for _ in claims]
    for (i, j, s), hs in zip(pairs, hits):
        per_claim[i].append(hs)
        for ep in (eps_each[i], eps_pool[i]):
            ep.log("search", {"query": s, "k": K}, {"hits": hs}, s_search)

    # each: verify every sub-claim against its own search
    ids = [[f"{h['page']}:{h['line']}" for h in hs] for hs in hits]
    t0 = time.time()
    res = verify.batch([s for _, _, s in pairs], ids)
    s_verify = (time.time() - t0) / len(pairs)
    for (i, j, s), ev, r in zip(pairs, ids, res):
        ep = eps_each[i]
        tin, tout = r.pop("_tokens")
        ep.v_in, ep.v_out = getattr(ep, "v_in", 0) + tin, getattr(ep, "v_out", 0) + tout
        ep.sub_verdicts[j] = r
        ep.log("verify", {"claim": s, "evidence": ev, "subclaim": j}, r, s_verify)

    # pool: verify the full claim once on the union of the sub-claim searches
    pooled = [[f"{h['page']}:{h['line']}" for h in round_robin_pages(hl)] for hl in per_claim]
    t0 = time.time()
    res = verify.batch([c["claim"] for c in claims], pooled)
    s_verify = (time.time() - t0) / len(claims)
    for ep, ev, r in zip(eps_pool, pooled, res):
        tin, tout = r.pop("_tokens")
        ep.v_in, ep.v_out = tin, tout
        ep.claim_verdict = r
        ep.log("verify", {"claim": ep.c["claim"], "evidence": ev, "subclaim": None}, r, s_verify)

    for ep in eps_each + eps_pool:
        ep.log("commit", {}, {}, 0.0)
        ep.end = "commit"
    return [to_row(ep, info_each) for ep in eps_each], [to_row(ep, info_pool) for ep in eps_pool]


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def compare(runs, ids):
    """runs: name -> {id: trace row}. Prints metrics on the shared ids."""
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")

    def fever_ok(r):
        if r["final"]["verdict"] != r["gold"]:
            return False
        return r["gold"] == "NOT_ENOUGH_INFO" or any(set(s) <= set(r["final"]["evidence"])
                                                    for s in r["gold_evidence"])

    names = list(runs)
    print(f"\n== {len(ids)} claims split by the decomposer ==")
    print(f"  {'':22s}" + "".join(f"{n:>10s}" for n in names))
    rows = {n: [runs[n][i] for i in ids] for n in names}
    lines = [("label accuracy", lambda rs: mean([r["final"]["verdict"] == r["gold"] for r in rs])),
             ("FEVER score", lambda rs: mean([fever_ok(r) for r in rs]))]
    lines += [(f"recall {L}", lambda rs, L=L: mean([r["final"]["verdict"] == L for r in rs if r["gold"] == L]))
              for L in LABELS]
    lines += [("searches / claim", lambda rs: mean([r["cost"]["n_search"] for r in rs])),
              ("verifies / claim", lambda rs: mean([r["cost"]["n_verify"] for r in rs])),
              ("tokens / claim", lambda rs: mean([r["cost"]["tokens_in"] + r["cost"]["tokens_out"] for r in rs]))]
    for name, f in lines:
        print(f"  {name:22s}" + "".join(f"{f(rows[n]):10.3f}" for n in names))
    base = names[0]
    for n in names[1:]:
        win = sum(runs[n][i]["final"]["verdict"] == runs[n][i]["gold"] != runs[base][i]["final"]["verdict"] for i in ids)
        loss = sum(runs[base][i]["final"]["verdict"] == runs[base][i]["gold"] != runs[n][i]["final"]["verdict"] for i in ids)
        print(f"  {n} vs {base}: fixes {win}, breaks {loss}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LOCAL_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--decomposer", default=BIG_MODEL)
    ap.add_argument("--full", required=True, help="trace of the fixed policy with the same verifier, n=--n")
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

    suffix = (adapter_label(a.adapter) if a.adapter else "base") + (f"_{a.tag}" if a.tag else "")
    paths = {p: os.path.join(TRACE_DIR, f"fever_dev_decompose-{p}-{suffix}_n{a.n}.jsonl") for p in ("each", "pool")}
    done = set(load(paths["each"])) & set(load(paths["pool"])) if all(map(os.path.exists, paths.values())) else set()
    todo = [c for c in claims if c["id"] not in done]
    print(f"{len(claims)} split claims, {len(todo)} to run")
    if todo:
        search = FeverSearch()
        verify = LocalVerify(search, a.model, a.adapter)
        base = {"controller": None, "decomposer": decomp.name, "verifier": verify.name,
                "reward_version": reward_version(), "seed": SEED}
        infos = {p: {**base, "policy": f"decompose-{p}"} for p in ("each", "pool")}
        t0 = time.time()
        for i in range(0, len(todo), a.chunk):
            each, pool = run_chunk(todo[i:i + a.chunk], decomp, search, verify, infos["each"], infos["pool"])
            for p, rows in (("each", each), ("pool", pool)):
                with open(paths[p], "a", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {i + len(each)}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)

    runs = {"full": load(a.full), "each": load(paths["each"]), "pool": load(paths["pool"])}
    ids = [c["id"] for c in claims if all(c["id"] in r for r in runs.values())]
    compare(runs, ids)
