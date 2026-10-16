"""Shared GRPO prompt loading, reward and adapter paths."""

import argparse
import json
import os
import random
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from core import reward as R
from retrieval.build_bm25_index import DATA
from .experiment_layout import artifacts_dir
from .fever_eval import assert_not_reserved
from verifier.model import LOCAL_MODEL, MAX_CITED, SEED, SYSTEM, parse

SFT_DIR = os.path.join(DATA, "..", "train", "sft")
OUT_DIR = os.path.join(DATA, "..", "train", "grpo")
DEFAULT_SFT_DATA = os.path.join(SFT_DIR, "fever-oracle-full")
DEFAULT_SFT = str(artifacts_dir("sft", "Qwen_Qwen3.5-2B_n6000",
                                config_name="sft-direct-from-base-qwen3p5-2b-5895-samples") / "checkpoints")
VERDICT = {"SUPPORTED": R.SUPPORTED, "REFUTED": R.REFUTED, "NOT_ENOUGH_INFO": R.NEI}
FORMAT_FAIL = -2.0  # below the worst well-formed answer (about -1.5), so breaking format never pays


def load_rows(sft_dir, n, seed):
    """Prompts and gold labels/evidence ids from a prepared dataset."""
    with open(os.path.join(sft_dir, "labeled_prompts.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert_not_reserved(rows, "GRPO training")
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        target = json.loads(r["completion"])
        out.append({
            "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["prompt"]}],
            "gold": VERDICT[r["gold"]],
            "gold_ids": target["evidence"],
        })
    return out


def score_one(text, gold, gold_ids, cfg, rng):
    """(reward, parts) for one completion; parts is None on a format failure."""
    obj, ok = parse(text)
    if not all(ok.values()):
        if cfg.mode == "format_only":
            return 0.0, None
        if cfg.mode == "random":
            return rng.random(), None
        return FORMAT_FAIL, None
    cited = obj["evidence"][:MAX_CITED]
    return R.total(gold, VERDICT[obj["verdict"]], float(obj["confidence"]), gold_ids, cited,
                   n_ops=0, format_ok=True, cfg=cfg, rng=rng)


def make_reward_funcs(cfg, seed):
    """The training reward, plus zero-weight copies of its parts so TRL logs them."""
    rng = random.Random(seed)
    cache = {}

    def scored(completions, gold, gold_ids):
        key = id(completions)
        if key not in cache:
            cache.clear()
            cache[key] = [score_one(c[-1]["content"], g, ids, cfg, rng)
                          for c, g, ids in zip(completions, gold, gold_ids)]
        return cache[key]

    def verifier_reward(completions, gold, gold_ids, **_):
        return [r for r, _ in scored(completions, gold, gold_ids)]

    def part(name):
        def f(completions, gold, gold_ids, **_):
            return [p.get(name, 0.0) if p else 0.0 for _, p in scored(completions, gold, gold_ids)]
        f.__name__ = name
        return f

    def format_ok(completions, gold, gold_ids, **_):
        return [0.0 if p is None else 1.0 for _, p in scored(completions, gold, gold_ids)]

    funcs = [verifier_reward, part("correctness"), part("calibration"), part("evidence"), format_ok]
    return funcs, [1.0, 0.0, 0.0, 0.0, 0.0]


def spread_path(data, n):
    """disagreement.py's output for this prompt set (named after the data dir)."""
    return os.path.join(OUT_DIR, f"disagreement_{os.path.basename(os.path.normpath(data))}_n{n}.jsonl")


def filter_spread(rows, data, n):
    """Keep claims whose 8 SFT samples scored differently (disagreement.py)."""
    path = spread_path(data, n)
    with open(path, encoding="utf-8") as f:
        spread = {r["i"] for r in map(json.loads, f) if r["reward_std"] > 1e-9}
    return [r for i, r in enumerate(rows) if i in spread]
