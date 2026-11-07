"""Controller action cache (AG-2): run every evidence/verify action once per
claim and save its outcome, so a controller can be trained and compared offline
without further model calls.

Each action gathers a context and asks one verifier for a verdict:

    bm25          title match + page BM25, 5 pages, top 100 sentences (the fixed policy)
    bm25-k20      same 5 pages, top 20 BM25 sentences
    hybrid-k20    same 5 pages, top 20 by BM25 + MiniLM rank fusion
    pages10-k20   10 candidate pages, top 20 BM25 sentences (search wider)
    follow-k20    5 pages plus up to 5 pages linked from the top 10 sentences,
                  top 20 BM25 sentences (retrieval/follow.py, origin/retrieval)
    flash-bm25    the bm25 context, verified by the big model (OpenRouter)

Claim sets:
    --claim-ids <manifest>        a frozen dev manifest (evaluation)
    --train-n N --seed S          N train claims balanced by label, excluding every
                                  train ID in the usage audit (SFT/GRPO data) and
                                  every reserved calibration/holdout ID

    python -m agent.action_cache --train-n 1500 --seed 1 --actions bm25,bm25-k20,hybrid-k20,pages10-k20,follow-k20 \
        --adapter verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints
    python -m agent.action_cache --train-n 1500 --seed 1 --actions flash-bm25

Writes verifier/data/controller/<set>/<action>.jsonl, one row per claim, and
resumes by claim id. Calibration and holdout IDs are refused: the controller is
trained on train claims, chosen on dev 900 and validated on holdout only once.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import json
import os
import random
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from retrieval.build_bm25_index import DATA
from retrieval.fever_retrieval import FeverRetriever
from retrieval.follow import FeverFollow, outgoing_links
from utils.fever_eval import FROZEN_ROOT, assert_not_reserved, manifest_tag
from verifier.model import (BIG_MODEL, FEVER_LABEL, LOCAL_MODEL, SEED, adapter_label, gold_sets, render,
                            sample_claims, score)

OUT_ROOT = os.path.join(DATA, "..", "controller")
LOCAL_ACTIONS = ("bm25", "bm25-k20", "hybrid-k20", "pages10-k20", "follow-k20")
API_ACTIONS = ("flash-bm25",)


def train_claims(n, seed):
    """n unused FEVER train claims, balanced by label."""
    with open(FROZEN_ROOT / "usage-audit.json", encoding="utf-8") as f:
        used = set(map(str, json.load(f)["used_train_ids"]))
    with open(os.path.join(DATA, "train.jsonl"), encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if str(r["id"]) not in used]
    rng = random.Random(seed)
    out = []
    for i, label in enumerate(FEVER_LABEL):
        pool = [r for r in rows if r["label"] == label]
        out += rng.sample(pool, n // 3 + (1 if i < n % 3 else 0))
    rng.shuffle(out)
    assert_not_reserved(out, "controller training claims")
    return out


def select_follow_links(retriever, links, query, limit, excluded=()):
    """Anchor/query word overlap, canonical dedup (origin/retrieval pipeline.py)."""
    terms = lambda s: set(re.findall(r"\w+", s.casefold()))
    ordered = sorted(links, key=lambda link: (-len(terms(query) & terms(link["anchor"] + " " + link["target"])),
                                              link["id"]))
    seen, chosen = set(excluded), []
    for link in ordered:
        canonical = retriever.resolve_page(link["target"]) or link["target"]
        if canonical not in seen:
            chosen.append(link)
            seen.add(canonical)
        if len(chosen) >= limit:
            break
    return chosen


def with_retry(model, prompt, tries=5):
    for k in range(tries):
        try:
            return model.run([prompt])[0]
        except Exception as e:  # transient API/provider errors
            if k == tries - 1:  # record a failed call as an empty, invalid answer
                print(f"  giving up: {type(e).__name__}: {e}", flush=True)
                return {"text": "", "tokens": [], "provider": None, "in_tokens": 0, "out_tokens": 0}
            print(f"  retry {k + 1}: {type(e).__name__}: {e}", flush=True)
            time.sleep(5 * (k + 1))


class Contexts:
    def __init__(self):
        self.bm25 = FeverRetriever(method="bm25", candidate_pages=5)
        self.wide = copy(self.bm25)
        self.wide.candidate_pages = 10
        self._hybrid = None

    @property
    def hybrid(self):
        if self._hybrid is None:
            self._hybrid = FeverRetriever(method="hybrid", candidate_pages=5)
        return self._hybrid

    def follow(self, query, k=20):
        r = self.bm25
        pages = r.search_with_titles([query], k=5)[0]
        initial = r.rank_sentences(query, r.collect_sentences(pages), 10)
        tool = FeverFollow(r, max_actions=5)
        tool.observe(initial, pages)
        new = []
        for link in select_follow_links(r, outgoing_links(r, initial), query, 5, pages):
            new += tool.follow(link["id"], query, 10)["pages"]
        union = list(dict.fromkeys(pages + new))
        return r.rank_sentences(query, r.collect_sentences(union), k), {"followed_pages": new}

    def __call__(self, action, queries):
        if action in ("bm25", "flash-bm25"):
            return [(h, {}) for h in self.bm25.search(queries, 100)]
        if action == "bm25-k20":
            return [(h, {}) for h in self.bm25.search(queries, 20)]
        if action == "hybrid-k20":
            return [(h, {}) for h in self.hybrid.search(queries, 20)]
        if action == "pages10-k20":
            return [(h, {}) for h in self.wide.search(queries, 20)]
        if action == "follow-k20":
            return [self.follow(q) for q in queries]
        raise ValueError(action)


def row(claim, action, hits, extra, gen, seconds):
    shown = [(f"E{i + 1}", h["page"], h["line"], h["text"]) for i, h in enumerate(hits)]
    gs = gold_sets(claim)
    keys = {(p, l) for _, p, l, _ in shown}
    pages = list(dict.fromkeys(h["page"] for h in hits))
    ctx = {"shown": shown,
           "gold_page_retrieved": any({p for p, _ in s} <= set(pages) for s in gs) if gs else None,
           "gold_in_context": any(s <= keys for s in gs) if gs else None}
    out = score(claim, ctx, gen)
    for k in ("reasoning", "reasoning_present", "reasoning_cited_ids", "reasoning_citations_valid"):
        out.pop(k, None)
    out.update(action=action, pages=pages, hit_scores=[round(h.get("score", 0.0), 4) for h in hits[:10]],
               seconds=round(seconds, 3), **extra)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actions", required=True, help="comma-separated: " + ",".join(LOCAL_ACTIONS + API_ACTIONS))
    ap.add_argument("--claim-ids", help="frozen dev manifest")
    ap.add_argument("--train-n", type=int, help="number of unused train claims")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default=LOCAL_MODEL)
    ap.add_argument("--adapter")
    ap.add_argument("--chunk", type=int, default=20)
    a = ap.parse_args()
    actions = a.actions.split(",")
    if bad := set(actions) - set(LOCAL_ACTIONS + API_ACTIONS):
        ap.error(f"unknown actions {sorted(bad)}")
    if bool(a.claim_ids) == bool(a.train_n):
        ap.error("give exactly one of --claim-ids or --train-n")
    if a.claim_ids:
        claims = sample_claims(None, ids_file=a.claim_ids)
        assert_not_reserved(claims, "controller action cache")
        set_name = f"dev_{manifest_tag(a.claim_ids)}"
    else:
        claims = train_claims(a.train_n, a.seed)
        set_name = f"train_unused-n{a.train_n}-seed{a.seed}"
    out_dir = os.path.join(OUT_ROOT, set_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "claim_ids.json"), "w", encoding="utf-8") as f:
        json.dump({"split": "train" if a.train_n else "shared_task_dev", "seed": a.seed,
                   "ids": [c["id"] for c in claims]}, f)

    ctx = Contexts()
    local = api = None
    for action in actions:
        path = os.path.join(out_dir, f"{action}.jsonl")
        verifier = BIG_MODEL if action in API_ACTIONS else (a.model + (f"+{adapter_label(a.adapter)}" if a.adapter else ""))
        done = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                prev = [json.loads(l) for l in f]
            if prev and prev[0]["verifier"] != verifier:
                raise SystemExit(f"{path} was made with {prev[0]['verifier']}, not {verifier}; use a new set")
            done = {r["id"] for r in prev}
        todo = [c for c in claims if c["id"] not in done]
        print(f"[{action}] {len(done)} done, {len(todo)} to go -> {path}", flush=True)
        if not todo:
            continue
        if action in API_ACTIONS:
            if api is None:
                from verifier.model import OpenRouterModel
                api = OpenRouterModel(BIG_MODEL)
            model = api
        else:
            if local is None:
                from verifier.model import LocalModel
                local = LocalModel(a.model, adapter=a.adapter)
            model = local
        t0 = time.time()
        for i in range(0, len(todo), a.chunk):
            batch = todo[i:i + a.chunk]
            t = time.time()
            got = ctx(action, [c["claim"] for c in batch])
            prompts = [render(c["claim"], [(f"E{j + 1}", h["page"], h["line"], h["text"]) for j, h in enumerate(hits)])
                       for c, (hits, _) in zip(batch, got)]
            if action in API_ACTIONS:  # one request per prompt, in parallel
                with ThreadPoolExecutor(len(prompts)) as pool:
                    gens = list(pool.map(lambda p: with_retry(model, p), prompts))
            else:
                gens = model.run(prompts)
            per = (time.time() - t) / len(batch)
            with open(path, "a", encoding="utf-8") as f:
                for c, (hits, extra), g in zip(batch, got, gens):
                    r = row(c, action, hits, extra, g, per)
                    r["verifier"] = verifier
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  [{action}] {i + len(batch)}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
