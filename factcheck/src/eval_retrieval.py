"""Page recall of plain BM25 vs BM25 + title matching.

Tuned on FEVER *train* claims so the dev claims used by stage 0 stay unseen.
A claim counts as a hit at k when some complete gold evidence set has all
its pages in the top k. NEI claims are excluded (no gold evidence).

    python eval_retrieval.py [--split train|dev] [--n 1000]
"""

import argparse
import json
import os
import random
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from build_bm25_index import DATA
from fever_retrieval import FeverRetriever

KS = (1, 3, 5, 10)


def recall(claims, results):
    out = {}
    for k in KS:
        hit = 0
        for c, pages in zip(claims, results):
            top = set(pages[:k])
            hit += any(all(e[2] in top for e in s) for s in c["evidence"])
        out[k] = hit / len(claims)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    fname = {"train": "train.jsonl", "dev": "shared_task_dev.jsonl"}[a.split]
    with open(os.path.join(DATA, fname), encoding="utf-8") as f:
        claims = [c for c in map(json.loads, f) if c["label"] != "NOT ENOUGH INFO"]
    claims = random.Random(1).sample(claims, a.n)
    texts = [c["claim"] for c in claims]

    r = FeverRetriever()
    t = time.time()
    r.title_matches("warm up")  # builds the title index
    print(f"title index: {len(r.exact):,} titles, {time.time() - t:.0f}s")

    kmax = max(KS)
    runs = {"bm25": lambda: r.search(texts, k=kmax)}
    for mt in (2, 3, 5):
        for lo in (True, False):
            runs[f"titles(max {mt}{', longest' if lo else ', all'})"] = \
                lambda mt=mt, lo=lo: r.search_with_titles(texts, k=kmax, max_title=mt, longest_only=lo)
    print(f"\n{a.split}, {a.n} non-NEI claims")
    print(f"{'':24s}" + "".join(f"  @{k:<5d}" for k in KS) + "  ms/claim")
    for name, fn in runs.items():
        t = time.time()
        res = fn()
        ms = (time.time() - t) / len(texts) * 1000
        rc = recall(claims, res)
        print(f"{name:24s}" + "".join(f"  {rc[k]:.3f} " for k in KS) + f"  {ms:.0f}")
