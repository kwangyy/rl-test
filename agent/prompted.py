"""Prompted controller (AG-2, stage 0b): the big model picks the actions, the 2B model verifies.

The agentic baseline the learned controller (RL-3) has to beat. Each step the
controller (BIG_MODEL on OpenRouter, prompted, no training) sees the claim, what
its searches found and what the verifier said, and picks one action:

    search    {"query": str}          retrieve with a query of its choosing
    decompose {}                      split the claim into sub-claims (big model)
    verify    {"subclaim": int|null}  2B verifier on the claim (or a sub-claim)
    commit    {}                      answer with the latest verdict on the claim
    abstain   {}                      answer NOT_ENOUGH_INFO

Claims in a chunk run in lockstep so searches and verifies are batched on the
GPU while controller calls run in parallel threads.

    python -m agent.prompted --n 30 --tag smoke
    python -m agent.prompted --adapter verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints

Writes verifier/data/traces/fever_dev_prompted-<verifier>_n<N>.jsonl in
docs/trace-format.md, schema 2: schema 1 plus the controller's own calls and
tokens. Reruns skip claims already in the file.
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from agent.agent_loop import (NEI, REWARD_LABEL, TRACE_DIR, FeverSearch, LocalVerify, report,
                              reward_version)
from core import reward as R
from verifier.model import (BIG_MODEL, FEVER_LABEL, K_PAGES, LOCAL_MODEL, MAX_CITED, MAX_SENTS_PER_PAGE, SEED,
                            adapter_label, gold_sets, sample_claims)

SCHEMA = 2
K = K_PAGES * MAX_SENTS_PER_PAGE  # sentences per search, as the fixed policy
MAX_POOL = K  # sentences the verifier sees at once, as the fixed policy
MAX_STEPS = 8  # actions before the episode is cut off (ended_by "budget")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CONTROLLER_SYSTEM = """You control a fact-checking agent. Your job is to reach a verdict on a claim
(SUPPORTED, REFUTED or NOT_ENOUGH_INFO) using Wikipedia, at the lowest cost.
You cannot read evidence closely yourself; a separate verifier model does that.

Actions (reply with exactly one JSON object, nothing else):
  {"action": "search", "query": "<text>"}   find Wikipedia sentences for a query
  {"action": "decompose"}                   split the claim into simpler sub-claims
  {"action": "verify", "subclaim": null}    verifier judges the claim against the
                                            latest search results plus evidence it
                                            cited before; use an index to verify a
                                            sub-claim instead
  {"action": "commit"}                      answer with the verifier's latest verdict
                                            on the full claim
  {"action": "abstain"}                     answer NOT_ENOUGH_INFO

Every action costs. Search, then verify. Commit when the verifier is confident and
its cited evidence is on topic. Search again with a better query when the pages
found are about the wrong entity or the verifier cites nothing. Abstain when
repeated searches cannot find relevant pages."""

DECOMPOSE_SYSTEM = """Split a factual claim into the smallest set of simpler sub-claims that are
all true exactly when the claim is true. A claim that is already simple stays as one.
Reply with a JSON list of strings and nothing else."""


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except ImportError:
        pass


class BigModel:
    """Text-in, text-out calls to the controller model on OpenRouter."""

    def __init__(self, name=BIG_MODEL, max_tokens=200):
        from openai import OpenAI
        load_env()
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=os.environ["OPENROUTER_API_KEY"], max_retries=3)
        self.name = name
        self.max_tokens = max_tokens

    def __call__(self, system, user):
        r = self.client.chat.completions.create(
            model=self.name, temperature=0, seed=SEED, max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            extra_body={"reasoning": {"enabled": False}},
        )
        return {"text": r.choices[0].message.content or "",
                "in_tokens": r.usage.prompt_tokens, "out_tokens": r.usage.completion_tokens}


def first_json(text, kind):
    m = re.search(r"\{.*\}" if kind is dict else r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, kind) else None


class Episode:
    """One claim's state while the controller acts on it."""

    def __init__(self, claim):
        self.c = claim
        self.steps = []
        self.subclaims = []
        self.last_hits = []
        self.cited = []  # evidence ids the verifier has cited so far, oldest first
        self.claim_verdict = None  # latest verify result on the full claim
        self.sub_verdicts = {}  # subclaim index -> latest verify result
        self.end = None
        self.ctl_calls = self.ctl_in = self.ctl_out = 0

    def log(self, action, args, result, seconds, controller=None):
        step = {"t": len(self.steps), "action": action, "args": args, "result": result,
                "seconds": round(seconds, 3)}
        if controller is not None:
            step["controller"] = controller
        self.steps.append(step)

    def pool(self):
        """Verifier input: evidence cited earlier, then the latest search, deduplicated."""
        ids = self.cited + [f"{h['page']}:{h['line']}" for h in self.last_hits]
        return list(dict.fromkeys(ids))[:MAX_POOL]

    def observation(self):
        out = [f"Claim: {self.c['claim']}"]
        if self.subclaims:
            out.append("Sub-claims: " + "; ".join(f"[{i}] {s}" for i, s in enumerate(self.subclaims)))
        for s in self.steps:
            a, r = s["action"], s["result"]
            if a == "search":
                pages = list(dict.fromkeys(h["page"].replace("_", " ") for h in r["hits"]))
                out.append(f"search {s['args']['query']!r} -> pages: " + (", ".join(pages) or "none"))
            elif a == "decompose":
                out.append(f"decompose -> {len(r['subclaims'])} sub-claims")
            elif a == "verify":
                target = (f"sub-claim [{s['args']['subclaim']}]" if s["args"].get("subclaim") is not None
                          else "claim")
                conf = r["confidence"]
                out.append(f"verify {target} -> {r['verdict'] or 'invalid output'}"
                           + (f" (confidence {conf:.2f})" if conf is not None else "")
                           + (f", cites: {'; '.join(self.texts.get(e, e) for e in r['evidence'])}"
                              if r["evidence"] else ", cites nothing"))
        out.append(f"Actions left: {MAX_STEPS - len(self.steps)}. Next action?")
        return "\n".join(out)


def decide(ep, big):
    """Ask the controller for the next action; fall back to a safe default on bad output."""
    t0 = time.time()
    g = big(CONTROLLER_SYSTEM, ep.observation())
    ep.ctl_calls += 1
    ep.ctl_in += g["in_tokens"]
    ep.ctl_out += g["out_tokens"]
    obj = first_json(g["text"], dict) or {}
    action = obj.get("action")
    valid = action in ("search", "decompose", "verify", "commit", "abstain")
    if action == "search" and not str(obj.get("query") or "").strip():
        valid = False
    if action == "verify" and obj.get("subclaim") is not None and not (
            isinstance(obj["subclaim"], int) and 0 <= obj["subclaim"] < len(ep.subclaims)):
        obj["subclaim"] = None
    if action == "verify" and not ep.pool():
        valid = False  # nothing to verify against yet
    if not valid:  # search the claim first, verify next, then commit
        obj = ({"action": "search", "query": ep.c["claim"]} if not ep.last_hits
               else {"action": "verify", "subclaim": None} if ep.claim_verdict is None
               else {"action": "commit"})
    ctl = {"text": g["text"], "valid": valid, "tokens_in": g["in_tokens"], "tokens_out": g["out_tokens"],
           "seconds": round(time.time() - t0, 3)}
    return obj, ctl


def run_chunk(claims, search, verify, big, run_info, workers=8):
    eps = [Episode(c) for c in claims]
    texts = {}  # "page:line" -> sentence, shared so observations can quote cited evidence
    for ep in eps:
        ep.texts = texts
    pool = ThreadPoolExecutor(workers)
    while True:
        live = [ep for ep in eps if ep.end is None]
        if not live:
            break
        for ep in live:
            if len(ep.steps) >= MAX_STEPS:
                ep.end = "budget"
        live = [ep for ep in live if ep.end is None]
        if not live:
            break
        decisions = list(pool.map(lambda ep: decide(ep, big), live))

        searches = [(ep, a, c) for ep, (a, c) in zip(live, decisions) if a["action"] == "search"]
        if searches:
            t0 = time.time()
            hits = search.batch([a["query"] for _, a, _ in searches], K)
            s = (time.time() - t0) / len(searches)
            for (ep, a, c), hs in zip(searches, hits):
                ep.last_hits = hs
                texts.update({f"{h['page']}:{h['line']}": h["text"] for h in hs})
                ep.log("search", {"query": a["query"], "k": K}, {"hits": hs}, s, c)

        decomps = [(ep, a, c) for ep, (a, c) in zip(live, decisions) if a["action"] == "decompose"]
        if decomps:
            def split(ep):
                t0 = time.time()
                g = big(DECOMPOSE_SYSTEM, ep.c["claim"])
                subs = [s for s in (first_json(g["text"], list) or []) if isinstance(s, str) and s.strip()]
                return subs or [ep.c["claim"]], g, time.time() - t0
            for (ep, a, c), (subs, g, s) in zip(decomps, pool.map(lambda x: split(x[0]), decomps)):
                ep.subclaims = subs
                ep.ctl_in += g["in_tokens"]
                ep.ctl_out += g["out_tokens"]
                ep.log("decompose", {"claim": ep.c["claim"]}, {"subclaims": subs}, s,
                       {**c, "decompose_tokens_in": g["in_tokens"], "decompose_tokens_out": g["out_tokens"]})

        verifies = [(ep, a, c) for ep, (a, c) in zip(live, decisions) if a["action"] == "verify"]
        if verifies:
            t0 = time.time()
            targets = [ep.subclaims[a["subclaim"]] if a.get("subclaim") is not None else ep.c["claim"]
                       for ep, a, _ in verifies]
            evs = [ep.pool() for ep, _, _ in verifies]
            results = verify.batch(targets, evs)
            s = (time.time() - t0) / len(verifies)
            for (ep, a, c), target, ev, res in zip(verifies, targets, evs, results):
                ep.v_in = getattr(ep, "v_in", 0) + res["_tokens"][0]
                ep.v_out = getattr(ep, "v_out", 0) + res["_tokens"][1]
                res = {k: v for k, v in res.items() if k != "_tokens"}
                ep.cited = list(dict.fromkeys(ep.cited + res["evidence"]))
                if a.get("subclaim") is None:
                    ep.claim_verdict = res
                else:
                    ep.sub_verdicts[a["subclaim"]] = res
                ep.log("verify", {"claim": target, "evidence": ev, "subclaim": a.get("subclaim")}, res, s, c)

        for ep, (a, c) in zip(live, decisions):
            if a["action"] in ("commit", "abstain"):
                ep.log(a["action"], {}, {}, 0.0, c)
                ep.end = a["action"]
    pool.shutdown()
    return [to_row(ep, run_info) for ep in eps]


def combine(sub_verdicts, n):
    """Verdict on the claim from its sub-claims: any refuted refutes, all supported supports."""
    vs = [sub_verdicts.get(i) for i in range(n)]
    got = [v for v in vs if v and v["verdict"]]
    refuted = [v for v in got if v["verdict"] == "REFUTED"]
    if refuted:
        best = max(refuted, key=lambda v: v["confidence"] or 0)
        return {"verdict": "REFUTED", "evidence": best["evidence"], "confidence": best["confidence"]}
    if len(got) == n and all(v["verdict"] == "SUPPORTED" for v in got):
        ev = list(dict.fromkeys(e for v in got for e in v["evidence"]))[:MAX_CITED]
        conf = 1.0
        for v in got:
            conf *= v["confidence"] or 0
        return {"verdict": "SUPPORTED", "evidence": ev, "confidence": conf}
    return None


def final_answer(ep):
    """commit and budget answer with the latest verdict on the claim, else the sub-claims';
    abstain answers NOT_ENOUGH_INFO with the verifier's probability of that label."""
    v = ep.claim_verdict
    if ep.end == "abstain":
        conf = v["token_prob"].get(NEI) if v and v.get("token_prob") else None
        return {"verdict": NEI, "evidence": [], "confidence": conf}
    if v and v["verdict"]:
        return {"verdict": v["verdict"], "evidence": v["evidence"], "confidence": v["confidence"]}
    if ep.subclaims:
        combined = combine(ep.sub_verdicts, len(ep.subclaims))
        if combined:
            return combined
    return {"verdict": None, "evidence": [], "confidence": None}


def to_row(ep, run_info):
    c = ep.c
    fin = final_answer(ep)
    gold = FEVER_LABEL[c["label"]]
    gs = gold_sets(c)
    gold_ids = sorted({f"{p}:{l}" for s in gs for p, l in s})
    count = lambda name: sum(s["action"] == name for s in ep.steps)
    n_search, n_decompose, n_verify = count("search"), count("decompose"), count("verify")
    n_ops = n_search + n_decompose + n_verify
    reward = None
    if fin["verdict"] is not None and fin["confidence"] is not None:
        r, parts = R.total(REWARD_LABEL[gold], REWARD_LABEL[fin["verdict"]], fin["confidence"],
                           gold_ids, fin["evidence"], n_ops)
        reward = {"total": r, "parts": parts}
    v_in, v_out = getattr(ep, "v_in", 0), getattr(ep, "v_out", 0)
    return {
        "schema": SCHEMA, "id": c["id"], "dataset": "fever", "split": "dev",
        "claim": c["claim"], "gold": gold,
        "gold_evidence": [sorted(f"{p}:{l}" for p, l in s) for s in gs],
        "run": run_info,
        "steps": ep.steps,
        "final": {**fin, "ended_by": ep.end},
        "cost": {"n_search": n_search, "n_decompose": n_decompose, "n_verify": n_verify, "n_ops": n_ops,
                 "tokens_in": v_in + ep.ctl_in, "tokens_out": v_out + ep.ctl_out,
                 "seconds": round(sum(s["seconds"] + s.get("controller", {}).get("seconds", 0)
                                      for s in ep.steps), 3),
                 "n_controller": ep.ctl_calls, "controller_tokens_in": ep.ctl_in,
                 "controller_tokens_out": ep.ctl_out},
        "reward": reward,
    }


def summarize_actions(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"  searches/claim   {mean([r['cost']['n_search'] for r in rows]):.2f}")
    print(f"  decomposes/claim {mean([r['cost']['n_decompose'] for r in rows]):.2f}")
    print(f"  verifies/claim   {mean([r['cost']['n_verify'] for r in rows]):.2f}")
    for end in ("commit", "abstain", "budget"):
        print(f"  ended by {end:8s} {mean([r['final']['ended_by'] == end for r in rows]):.3f}")
    ctl = [s["controller"]["valid"] for r in rows for s in r["steps"] if "controller" in s]
    print(f"  valid controller {mean(ctl):.3f}  ({len(ctl)} calls)")
    print(f"  controller tok   {mean([r['cost']['controller_tokens_in'] for r in rows]):.0f} in, "
          f"{mean([r['cost']['controller_tokens_out'] for r in rows]):.0f} out per claim")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LOCAL_MODEL, help="verifier base model")
    ap.add_argument("--adapter", default=None, help="verifier LoRA dir(s), as eval_verifier.py --adapter")
    ap.add_argument("--controller", default=BIG_MODEL)
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    run = "prompted-" + (adapter_label(a.adapter) if a.adapter else "base")
    run += f"_{a.tag}" if a.tag else ""
    os.makedirs(TRACE_DIR, exist_ok=True)
    path = os.path.join(TRACE_DIR, f"fever_dev_{run}_n{a.n}.jsonl")

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    claims = [c for c in sample_claims(a.n, SEED) if c["id"] not in done]
    print(f"{len(done)} done, {len(claims)} to go -> {path}")
    if claims:
        big = BigModel(a.controller)
        search = FeverSearch()
        verify = LocalVerify(search, a.model, a.adapter)
        run_info = {"policy": "prompted", "controller": big.name, "verifier": verify.name,
                    "reward_version": reward_version(), "seed": SEED, "max_steps": MAX_STEPS}
        t0 = time.time()
        for i in range(0, len(claims), a.chunk):
            rows = run_chunk(claims[i:i + a.chunk], search, verify, big, run_info)
            with open(path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {i + len(rows)}/{len(claims)}  {time.time() - t0:.0f}s", flush=True)
    report(path, None)
    summarize_actions(path)
