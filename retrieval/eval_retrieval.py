"""Page recall of plain BM25 vs BM25 + title matching.

Tuned on FEVER *train* claims so the dev claims used by untrained baseline stay unseen.
A claim counts as a hit at k when some complete gold evidence set has all
its pages in the top k. NEI claims are excluded (no gold evidence).

    python -m retrieval.eval_retrieval [--split train|dev] [--n 1000]
"""

import argparse
import json
import os
import random
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from retrieval.build_bm25_index import DATA
from retrieval.fever_retrieval import FeverRetriever
from utils.fever_eval import assert_not_reserved, load_manifest_claims

KS = (1, 3, 5, 10)


def select_claims(split="train", n=None, ids_file=None):
    fname = {"train": "train.jsonl", "dev": "shared_task_dev.jsonl"}[split]
    source = os.path.join(DATA, fname)
    if ids_file:
        if split != "dev":
            raise ValueError("--claim-ids requires the dev split")
        return [row for row in load_manifest_claims(source, ids_file)
                if row["label"] != "NOT ENOUGH INFO"]
    with open(source, encoding="utf-8") as stream:
        pool = [row for row in map(json.loads, stream) if row["label"] != "NOT ENOUGH INFO"]
    claims = random.Random(1).sample(pool, n if n is not None else 1000)
    assert_not_reserved(claims, "implicit retrieval development")
    return claims


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
    ap.add_argument("--split", choices=["train", "dev"], help="default train, or dev with --claim-ids")
    ap.add_argument("--n", type=int, help="default 1000 non-NEI; ignored when --claim-ids is supplied")
    ap.add_argument("--claim-ids", help="explicit frozen dev set; NEI claims are omitted from page recall")
    a = ap.parse_args()
    a.split = a.split or ("dev" if a.claim_ids else "train")
    claims = select_claims(a.split, a.n, a.claim_ids)
    a.n = len(claims)
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
