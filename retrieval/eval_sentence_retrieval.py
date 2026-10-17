"""Sentence recall: BM25 versus optional BM25+dense hybrid.

Usage:
    python -m retrieval.eval_sentence_retrieval --split train --n 1000 --hybrid

k counts exact (page, line) hits. Uses the page evaluator's sampling and
complete-set scoring rules; candidate pages are identical for both methods.
"""

import argparse
import hashlib
import json
import random
from copy import copy
from pathlib import Path
import time

from retrieval.build_bm25_index import DATA
from utils.fever_eval import assert_not_reserved, load_manifest_claims, manifest_metadata
from retrieval.fever_retrieval import FeverRetriever

KS = (1, 3, 5, 10, 20, 30)


def load_claims(data_dir, split, n=1000, seed=1, ids_path=None):
    """Original train sample, or explicit frozen dev IDs, with leakage guards."""
    filename = "train.jsonl" if split == "train" else "shared_task_dev.jsonl"
    source = Path(data_dir) / filename
    if ids_path:
        if split != "dev":
            raise ValueError("--claim-ids requires the dev split")
        selected = load_manifest_claims(source, ids_path)
        return [row for row in selected if row["label"] != "NOT ENOUGH INFO"]
    with source.open(encoding="utf-8") as handle:
        pool = [row for row in map(json.loads, handle) if row["label"] != "NOT ENOUGH INFO"]
    selected = random.Random(seed).sample(pool, n)
    assert_not_reserved(selected, f"implicit retrieval {split} development")
    return selected


def recall(claims, results, ks=(1, 3, 5, 10), level="pages"):
    """Original complete-set recall, with pages or exact (page, line) as the unit.

    A claim succeeds if ANY alternative gold set is FULLY present in top-k.
    Sentence mode changes only the matching unit, not the averaging rule.
    """
    if not claims or len(claims) != len(results):
        raise ValueError("need nonempty, aligned claims and results")
    out = {}
    for k in ks:
        hits = 0
        for claim, retrieved in zip(claims, results):
            top = (set(retrieved[:k]) if level == "pages" else
                   {(h["page"], h["line"]) for h in retrieved[:k]})
            gold = [{e[2] if level == "pages" else (e[2], e[3]) for e in group}
                    for group in claim["evidence"]]
            hits += any(group <= top for group in gold)
        out[k] = hits / len(claims)
    return out


def save_results(path, claims, metrics, predictions, *, split, seed, ks, level, retriever, candidate_pages=None, ids_file=None):
    """Save a new summary and per-claim predictions; never overwrite a run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"split": split, "seed": seed, "level": level, "ks": ks,
               "n_non_nei": len(claims), "claim_ids": [c["id"] for c in claims],
               "tokenization": {"shared_page_sentence_policy": True, "stemming": "english",
                   "lower": retriever.page_tokenizer.lower,
                   "token_pattern": retriever.page_tokenizer.splitter.__self__.pattern,
                   "stopwords": sorted(retriever.page_tokenizer.stopwords)},
               "retriever_sha256": hashlib.sha256(Path(__file__).with_name("fever_retrieval.py").read_bytes()).hexdigest(),
               **manifest_metadata(ids_file), "metrics": metrics}
    if candidate_pages is not None:
        summary["candidate_pages"] = candidate_pages
    with path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with path.with_suffix(".jsonl").open("x", encoding="utf-8") as handle:
        for i, claim in enumerate(claims):
            handle.write(json.dumps({"id": claim["id"], "claim": claim["claim"],
                "gold_evidence": claim["evidence"],
                "methods": {name: rows[i] for name, rows in predictions.items()}}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "dev"), help="default train, or dev with --claim-ids")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ks", type=int, nargs="+", default=KS)
    parser.add_argument("--candidate-pages", type=int, default=5)
    parser.add_argument("--hybrid", action="store_true", help="add the hybrid comparison row")
    parser.add_argument("--data-dir", type=Path, default=Path(DATA))
    parser.add_argument("--claim-ids", "--ids", dest="claim_ids", type=Path,
                        help="frozen dev manifest; uses all non-NEI IDs in order, ignoring --n")
    parser.add_argument("--output", type=Path, help="optional new .json summary and .jsonl predictions")
    args = parser.parse_args()
    args.split = args.split or ("dev" if args.claim_ids else "train")
    ks = sorted(set(args.ks))
    if min(args.n if not args.claim_ids else 1, args.candidate_pages, *ks) < 1:
        parser.error("counts must be positive")
    if args.output and (args.output.suffix != ".json" or args.output.exists() or args.output.with_suffix(".jsonl").exists()):
        parser.error("output must be a NEW .json path")
    try:
        claims = load_claims(args.data_dir, args.split, args.n, args.seed, args.claim_ids)
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
                         candidate_pages=args.candidate_pages, ids_file=args.claim_ids)
    finally:
        r.db.close()


if __name__ == "__main__":
    main()
