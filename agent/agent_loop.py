"""Agent loop skeleton (AG-2): search -> verify -> commit/abstain, one trace row per claim.

Fixed policy, no learning: search the claim once, verify against everything
returned, abstain if the verdict is NOT_ENOUGH_INFO, else commit. With the
default search size this shows the verifier exactly what eval_verifier.py's "titles"
retrieval shows it, in the same order and batches, so the loop must reproduce
untrained baseline before anything smarter is built on it:

    python -m agent.agent_loop                                              # base 2B, untrained baseline: acc 0.459
    python -m agent.agent_loop --adapter verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints   # SFT: acc 0.759
    python -m agent.agent_loop --n 30 --tag smoke
    python -m agent.agent_loop --method bm25 --search-k 5
    python -m agent.agent_loop --method hybrid --search-k 5

Writes verifier/data/traces/fever_dev_<run>.jsonl in docs/trace-format.md (schema 1),
then prints accuracy and per-claim agreement with the matching untrained baseline file.
A rerun skips claims already in the trace file.

search() wraps FeverRetriever until AG-1's sentence-level search lands; verify()
wraps untrained baseline's LocalModel. Both follow core/interfaces.py, so either can be
swapped without touching the loop.
"""

import argparse
from copy import copy
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from core import reward as R
from core.interfaces import Hit
from retrieval.build_bm25_index import DATA
from retrieval.fever_retrieval import FeverRetriever
from verifier.model import (FEVER_LABEL, K_PAGES, LABELS, LOCAL_MODEL, MAX_CITED, MAX_SENTS_PER_PAGE, SEED,
                    adapter_label, gold_sets, label_probs, parse, render, sample_claims)

TRACE_DIR = os.path.join(DATA, "..", "traces")
ARCHIVED_EVAL_DIR = os.path.join(DATA, "..", "..", "evaluation", "results", "archive")
SCHEMA = 1
NEI = "NOT_ENOUGH_INFO"
REWARD_LABEL = {"SUPPORTED": R.SUPPORTED, "REFUTED": R.REFUTED, NEI: R.NEI}


class FeverSearch:
    """interfaces.Search over the page-level FEVER index: the sentences of the
    top K_PAGES title-match + BM25 pages, page by page, at most k of them.
    Score is page-level (K_PAGES - page rank), so every sentence of a page ties.

    Usage:
        search = FeverSearch(method="bm25")  # or "hybrid"
        hits = search("Ada Lovelace wrote an algorithm", k=5)
    """

    def __init__(self, *, method="bm25", candidate_pages=K_PAGES, retriever=None):
        """Create a retriever, or copy and configure a supplied retriever."""
        if retriever is None:
            self.r = FeverRetriever(method=method, candidate_pages=candidate_pages)
        else:
            if method not in ("bm25", "hybrid") or candidate_pages < 1:
                raise ValueError("choose bm25/hybrid and a positive page count")
            self.r = copy(retriever)
            self.r.method = method
            self.r.candidate_pages = candidate_pages

    def __call__(self, query: str, k: int = K_PAGES * MAX_SENTS_PER_PAGE) -> list[Hit]:
        """Search Protocol: up to k ranked sentences, not k pages."""
        return self.batch([query], k)[0]

    def batch(self, queries: list[str], k: int = K_PAGES * MAX_SENTS_PER_PAGE) -> list[list[Hit]]:
        return self.r.search(queries, k)


class LocalVerify:
    """interfaces.Verify with untrained baseline's prompt and model. Evidence ids are
    numbered E1..En in the order given, the model's cited E-ids are mapped back
    to "page:line", and confidence is the verdict-token probability."""

    def __init__(self, search, model=LOCAL_MODEL, adapter=None):
        from verifier.model import LocalModel
        self.m = LocalModel(model, adapter=adapter)
        self.name = self.m.name
        self.r = search.r

    def _text(self, eid):
        page, line = eid.rsplit(":", 1)
        return dict(self.r.sentences(page)).get(int(line), "")

    def __call__(self, claim, evidence):
        return self.batch([claim], [evidence])[0]

    def batch(self, claims, evidences):
        shown = [[(f"E{i + 1}", e.rsplit(":", 1)[0], int(e.rsplit(":", 1)[1]), self._text(e))
                  for i, e in enumerate(ev)] for ev in evidences]
        gens = self.m.run([render(c, s) for c, s in zip(claims, shown)])
        out = []
        for s, g in zip(shown, gens):
            obj, ok = parse(g["text"])
            by_eid = {eid: f"{p}:{line}" for eid, p, line, _ in s}
            verdict = obj.get("verdict") if ok["verdict"] else None
            cited = [by_eid[e] for e in (obj.get("evidence") or []) if ok["evidence"] and e in by_eid][:MAX_CITED]
            probs = label_probs(g["tokens"]) or {}
            out.append({"verdict": verdict, "evidence": cited,
                        "confidence": probs.get(verdict) if verdict else None,
                        "token_prob": probs,
                        "_tokens": (g["in_tokens"], g["out_tokens"])})  # loop-internal, not part of Verify
        return out


def reward_version():
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h", "--", "reward.py"], capture_output=True,
                              text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
    except OSError:
        return None


def run_chunk(claims, search, verify, run_info, search_k=None):
    """The fixed policy over a batch of claims: one search, one verify, then commit or abstain."""
    t0 = time.time()
    hits = search.batch([c["claim"] for c in claims], K_PAGES * MAX_SENTS_PER_PAGE if search_k is None else search_k)
    t1 = time.time()
    ids = [[f"{h['page']}:{h['line']}" for h in hs] for hs in hits]
    results = verify.batch([c["claim"] for c in claims], ids)
    t2 = time.time()
    s_search, s_verify = (t1 - t0) / len(claims), (t2 - t1) / len(claims)
    rows = []
    for c, hs, ev, res in zip(claims, hits, ids, results):
        tin, tout = res.pop("_tokens")
        end = "abstain" if res["verdict"] == NEI else "commit"
        gold = FEVER_LABEL[c["label"]]
        gs = gold_sets(c)
        gold_ids = sorted({f"{p}:{l}" for s in gs for p, l in s})
        n_ops = 2
        reward = None
        if res["verdict"] is not None and res["confidence"] is not None:
            r, parts = R.total(REWARD_LABEL[gold], REWARD_LABEL[res["verdict"]], res["confidence"],
                               gold_ids, res["evidence"], n_ops)
            reward = {"total": r, "parts": parts}
        rows.append({
            "schema": SCHEMA, "id": c["id"], "dataset": "fever", "split": "dev",
            "claim": c["claim"], "gold": gold,
            "gold_evidence": [sorted(f"{p}:{l}" for p, l in s) for s in gs],
            "run": run_info,
            "steps": [
                {"t": 0, "action": "search", "args": {"query": c["claim"], "k": len(hs) if search_k is None else search_k},
                 "result": {"hits": hs}, "seconds": round(s_search, 3)},
                {"t": 1, "action": "verify", "args": {"claim": c["claim"], "evidence": ev},
                 "result": res, "seconds": round(s_verify, 3)},
                {"t": 2, "action": end, "args": {}, "result": {}, "seconds": 0.0},
            ],
            "final": {"verdict": res["verdict"], "evidence": res["evidence"],
                      "confidence": res["confidence"], "ended_by": end},
            "cost": {"n_search": 1, "n_decompose": 0, "n_verify": 1, "n_ops": n_ops,
                     "tokens_in": tin, "tokens_out": tout, "seconds": round(s_search + s_verify, 3)},
            "reward": reward,
        })
    return rows


def report(path, archived_eval_path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    n = len(rows)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    acc = mean([r["final"]["verdict"] == r["gold"] for r in rows])

    def fever_ok(r):
        if r["final"]["verdict"] != r["gold"]:
            return False
        return r["gold"] == NEI or any(set(s) <= set(r["final"]["evidence"]) for s in r["gold_evidence"])

    print(f"\n== {os.path.basename(path)}  ({n} claims) ==")
    print(f"  label accuracy   {acc:.3f}")
    print(f"  FEVER score      {mean([fever_ok(r) for r in rows]):.3f}")
    print(f"  valid verdict    {mean([r['final']['verdict'] is not None for r in rows]):.3f}")
    for L in LABELS:
        sub = [r for r in rows if r["gold"] == L]
        print(f"  recall {L:16s} {mean([r['final']['verdict'] == L for r in sub]):.3f}  (n={len(sub)})")
    print(f"  seconds/claim    {mean([r['cost']['seconds'] for r in rows]):.2f}")
    if archived_eval_path and os.path.exists(archived_eval_path):
        with open(archived_eval_path, encoding="utf-8") as f:
            s0 = {r["id"]: r for r in map(json.loads, f)}
        both = [r for r in rows if r["id"] in s0]
        same = mean([r["final"]["verdict"] == s0[r["id"]]["pred"] for r in both])
        s0_acc = mean([s0[r["id"]]["correct"] for r in both])
        print(f"  vs archived verifier run ({os.path.basename(archived_eval_path)}): accuracy {s0_acc:.3f} on the same "
              f"{len(both)} claims, same verdict on {same:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LOCAL_MODEL)
    ap.add_argument("--adapter", default=None, help="LoRA dir(s), as eval_verifier.py --adapter")
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--method", choices=("bm25", "hybrid"), default="bm25")
    ap.add_argument("--search-k", type=int, default=K_PAGES * MAX_SENTS_PER_PAGE)
    ap.add_argument("--candidate-pages", type=int, default=K_PAGES)
    a = ap.parse_args()
    if a.search_k < 1 or a.candidate_pages < 1:
        ap.error("search-k and candidate-pages must be positive")

    run = "fixed-" + (adapter_label(a.adapter) if a.adapter else "base")
    run += f"_{a.tag}" if a.tag else ""
    search_tag = f"{a.method}-p{a.candidate_pages}-k{a.search_k}"
    run += f"_{search_tag}"
    os.makedirs(TRACE_DIR, exist_ok=True)
    path = os.path.join(TRACE_DIR, f"fever_dev_{run}_n{a.n}.jsonl")
    eval_tag = "local" + (f"+{adapter_label(a.adapter)}" if a.adapter else "")
    archived_eval_path = os.path.join(ARCHIVED_EVAL_DIR, f"{eval_tag}_n{a.n}_titles.jsonl")

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    claims = [c for c in sample_claims(a.n, SEED) if c["id"] not in done]
    print(f"{len(done)} done, {len(claims)} to go -> {path}")
    if claims:
        search = FeverSearch(method=a.method, candidate_pages=a.candidate_pages)
        verify = LocalVerify(search, a.model, a.adapter)
        run_info = {"policy": "fixed", "verifier": verify.name, "reward_version": reward_version(), "seed": SEED}
        run_info["policy"] = f"fixed-{search_tag}"
        t0 = time.time()
        for i in range(0, len(claims), a.chunk):
            rows = run_chunk(claims[i:i + a.chunk], search, verify, run_info, search_k=a.search_k)
            with open(path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {i + len(rows)}/{len(claims)}  {time.time() - t0:.0f}s", flush=True)
    report(path, archived_eval_path)
