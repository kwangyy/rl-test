"""Sentence recall: BM25 versus optional BM25+dense hybrid.

Usage:
    python -m retrieval.eval_sentence_retrieval --split train --n 1000 --hybrid

k counts exact (page, line) hits. Sampling and complete-set scoring are shared
with eval_retrieval; candidate-page selection is identical for both methods.
"""

import argparse
from copy import copy
from pathlib import Path
import time

from retrieval.build_bm25_index import DATA
from retrieval.eval_retrieval import load_claims, recall, save_results
from retrieval.fever_retrieval import FeverRetriever

KS = (1, 3, 5, 10, 20, 30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ks", type=int, nargs="+", default=KS)
    parser.add_argument("--candidate-pages", type=int, default=5)
    parser.add_argument("--hybrid", action="store_true", help="add the hybrid comparison row")
    parser.add_argument("--data-dir", type=Path, default=Path(DATA))
    parser.add_argument("--ids", type=Path, help="optional frozen claim-id JSON")
    parser.add_argument("--output", type=Path, help="optional new .json summary and .jsonl predictions")
    args = parser.parse_args()
    ks = sorted(set(args.ks))
    if min(args.n, args.candidate_pages, *ks) < 1:
        parser.error("counts must be positive")
    if args.output and (args.output.suffix != ".json" or args.output.exists() or args.output.with_suffix(".jsonl").exists()):
        parser.error("output must be a NEW .json path")
    try:
        claims = load_claims(args.data_dir, args.split, args.n, args.seed, args.ids)
        if not claims:
            raise ValueError("no non-NEI claims to evaluate")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    texts = [claim["claim"] for claim in claims]
    r = FeverRetriever(args.data_dir, candidate_pages=args.candidate_pages)
    try:
        t = time.perf_counter()
        r.title_matches("warm up")
        print(f"title index: {len(r.exact):,} titles, {time.perf_counter() - t:.0f}s")
        runs = {"sentence_bm25": lambda: r.search(texts, max(ks))}
        if args.hybrid:
            import torch
            torch.set_num_threads(min(4, torch.get_num_threads()))
            hybrid = copy(r)  # Share corpus resources and pool settings; change only ranking.
            hybrid.method = "hybrid"
            runs["hybrid"] = lambda: hybrid.search(texts, max(ks))
        print(f"\n{args.split}, {len(claims)} non-NEI claims; @k counts sentences; candidate pages={args.candidate_pages}")
        print(f"{'':24s}" + "".join(f"  @{k:<5d}" for k in ks) + "  ms/claim")
        metrics, predictions = {}, {}
        for name, fn in runs.items():
            t = time.perf_counter()
            results = fn()
            ms = (time.perf_counter() - t) * 1000 / len(claims)
            scores = recall(claims, results, ks, level="sentences")
            metrics[name] = {"recall": scores, "ms_per_claim": ms}
            if args.output:
                predictions[name] = results
            print(f"{name:24s}" + "".join(f"  {scores[k]:.3f} " for k in ks) + f"  {ms:.0f}")
        if args.output:
            save_results(args.output, claims, metrics, predictions, split=args.split,
                         seed=args.seed, ks=ks, level="sentences", retriever=r,
                         candidate_pages=args.candidate_pages)
    finally:
        r.db.close()


if __name__ == "__main__":
    main()
