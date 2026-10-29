"""Prompted controller (AG-2, stage 0b): the big model picks the actions, the 2B model verifies.

The agentic baseline the learned controller (RL-3) has to beat. Each step the
controller (BIG_MODEL on OpenRouter, prompted, no training) sees the claim, what
its searches found and what the verifier said, and picks one action:

    search    {"query": str}          retrieve with a query of its choosing
    decompose {}                      split the claim into sub-claims (big model)
    verify    {"subclaim": int|null}  2B verifier on the claim (or a sub-claim)
    commit    {}                      answer with the latest verdict on the claim
    abstain   {}                      answer NOT_ENOUGH_INFO

`--tools` changes the action set; the prompt rules are otherwise v2 and the trace
name gains the mode as a tag:
    full      the actions above (default)
    nosearch  the claim is searched once up front, as in the fixed loop; the controller
              can then only decompose, verify, commit or abstain
    follow    full plus follow {"link": "L<n>"}: open a page linked from a retrieved or
              cited sentence and put its sentences first in the verifier's input
    pages     follow plus open_page {"title": str} and find_pages {"query": str}
    web       pages plus web_pages {"query": str}: Exa search of en.wikipedia.org mapped to
              the FEVER snapshot; replays --exa-cache, live requests need --exa-network
The retrieval tools are in agent/tools.py.

Claims in a chunk run in lockstep so searches and verifies are batched on the
GPU while controller calls run in parallel threads.

    python -m agent.prompted --n 30 --tag smoke
    python -m agent.prompted --adapter verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints
    python -m agent.prompted --tools follow --adapter ...
    python -m agent.prompted --tools web --exa-network --exa-max-requests 100 --adapter ...
    python -m agent.prompted --tools follow --train-n 6000 --shard 0/3 --chunk 64 --workers 32 --adapter ...

--train-n samples unused train claims as agent/action_cache.py does (no SFT/GRPO or
reserved IDs) for controller imitation data; --shard splits one set across processes.

Writes verifier/data/traces/fever_<split>_prompted-<verifier>_n<N>.jsonl in
docs/trace-format.md, schema 2: schema 1 plus the controller's own calls and
tokens. Reruns skip claims already in the file.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from agent.decompose import Decomposer, big_model, cache_path, first_json
from agent.agent_loop import (NEI, REWARD_LABEL, TRACE_DIR, FeverSearch, LocalVerify, report,
                              reward_version)
from agent.tools import FIRST, ToolKit, describe
from core import reward as R
from verifier.model import (BIG_MODEL, FEVER_LABEL, K_PAGES, LOCAL_MODEL, MAX_CITED, MAX_SENTS_PER_PAGE, SEED,
                            adapter_label, gold_sets, sample_claims)

SCHEMA = 2
K = K_PAGES * MAX_SENTS_PER_PAGE  # sentences per search, as the fixed policy
MAX_POOL = K  # sentences the verifier sees at once, as the fixed policy
MAX_STEPS = 8  # actions before the episode is cut off (ended_by "budget")

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

Every action costs. Search, then verify. Commit when the verifier's confidence is
0.70 or higher and its cited evidence is on topic. Search again with a better query
when the pages found are about the wrong entity, the verifier cites nothing, or its
confidence is below 0.70; after one such retry, commit the latest verdict.
A NOT_ENOUGH_INFO verdict is not a failure to search past: if the verifier said
NOT_ENOUGH_INFO earlier, a later SUPPORTED or REFUTED only counts when its
confidence is higher than that earlier one's; otherwise abstain. Abstain when
repeated searches cannot find relevant pages."""
CONTROLLER_PROMPT = "v2"  # v1: no confidence threshold; v2: retry below 0.70, NEI not overridden by weaker verdicts

# --tools nosearch: v2 without the search action; the claim has already been searched once.
NOSEARCH_SYSTEM = (CONTROLLER_SYSTEM
    .replace('  {"action": "search", "query": "<text>"}   find Wikipedia sentences for a query\n', "")
    .replace("latest search results plus evidence it", "claim's search results plus evidence it")
    .replace("""Every action costs. Search, then verify. Commit when the verifier's confidence is
0.70 or higher and its cited evidence is on topic. Search again with a better query
when the pages found are about the wrong entity, the verifier cites nothing, or its
confidence is below 0.70; after one such retry, commit the latest verdict.""",
"""The claim has been searched once and you cannot search again. Every action costs.
Verify, then commit when the verifier's confidence is 0.70 or higher and its cited
evidence is on topic. When the verifier cites nothing or its confidence is below
0.70, you may decompose the claim and verify sub-claims; then commit the latest verdict.""")
    .replace("""Abstain when
repeated searches cannot find relevant pages.""", """Abstain when the
pages found are not relevant."""))

# --tools follow: v2 plus the follow action.
FOLLOW_SYSTEM = (CONTROLLER_SYSTEM
    .replace('  {"action": "decompose"}', '''  {"action": "follow", "link": "L3"}        open the page a listed link points to; its
                                            sentences go first in the verifier's input
  {"action": "decompose"}''')
    .replace("""Abstain when
repeated searches cannot find relevant pages.""", """Follow a link when the
claim depends on a second entity (a film's director, a person's birthplace) that a
found sentence links to but no search has opened. Abstain when repeated searches
and links cannot find relevant pages."""))

# --tools pages: follow plus opening pages by title and listing candidate titles.
PAGES_SYSTEM = (FOLLOW_SYSTEM
    .replace('  {"action": "decompose"}', '''  {"action": "open_page", "title": "<title>"} open a Wikipedia page by its title; its
                                            sentences go first in the verifier's input
  {"action": "find_pages", "query": "<text>"} list Wikipedia page titles for a query;
                                            the verifier sees nothing new
  {"action": "decompose"}''')
    .replace("""Abstain when repeated searches
and links cannot find relevant pages.""", """Use find_pages when
the claim's name is ambiguous (several films, people or places share it) or
searches keep returning the wrong pages, then open_page the right title. Open a
page directly when you know its title. Abstain when repeated searches, links and
pages cannot find relevant pages."""))

# --tools web: pages plus a web search engine restricted to Wikipedia.
WEB_SYSTEM = (PAGES_SYSTEM
    .replace('  {"action": "decompose"}', '''  {"action": "web_pages", "query": "<text>"} web search engine over Wikipedia; the
                                            sentences of the pages it finds replace
                                            the latest search results
  {"action": "decompose"}''')
    .replace("""Open a
page directly when you know its title.""", """Open a
page directly when you know its title. Try web_pages when search finds no
relevant page."""))

SYSTEMS = {"full": CONTROLLER_SYSTEM, "nosearch": NOSEARCH_SYSTEM, "follow": FOLLOW_SYSTEM,
           "pages": PAGES_SYSTEM, "web": WEB_SYSTEM}
TOOLSETS = {"full": (), "nosearch": (), "follow": ("follow",), "pages": ("follow", "open_page", "find_pages"),
            "web": ("follow", "open_page", "find_pages", "web_pages")}  # agent/tools.py tools per preset
ACTIONS = {name: (() if name == "nosearch" else ("search",)) + TOOLSETS[name]
           + ("decompose", "verify", "commit", "abstain") for name in SYSTEMS}

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
        self.tools = "full"
        self.kit = None  # agent.tools.ToolKit for the preset's retrieval tools

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
            elif a in TOOLSETS["web"]:
                out.append(describe(s))
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
        if "follow" in TOOLSETS[self.tools]:
            shown = self.kit.open_links()
            out.append("Links: " + ("; ".join(f"{lid} {l['target']} (from {l['source'].rsplit(':', 1)[0].replace('_', ' ')})"
                                              for lid, l in shown) or "none"))
        out.append(f"Actions left: {MAX_STEPS - len(self.steps)}. Next action?")
        return "\n".join(out)


def decide(ep, big, g=None):
    """Ask the controller for the next action; fall back to a safe default on bad output.
    g is a reply already generated in a batch (a local controller), with its share of the seconds."""
    t0 = time.time()
    if g is None:
        g = big(SYSTEMS[ep.tools], ep.observation())
    ep.ctl_calls += 1
    ep.ctl_in += g["in_tokens"]
    ep.ctl_out += g["out_tokens"]
    obj = first_json(g["text"], dict) or {}
    action = obj.get("action")
    valid = action in ACTIONS[ep.tools]
    if action in TOOLSETS[ep.tools] and not ep.kit.valid(action, obj):
        valid = False
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
           "seconds": round(g.get("seconds", time.time() - t0), 3)}
    return obj, ctl


def run_chunk(claims, search, verify, big, decomposer, run_info, workers=8, search_k=K, tools="full",
              exa=None):
    eps = [Episode(c) for c in claims]
    texts = {}  # "page:line" -> sentence, shared so observations can quote cited evidence
    for ep in eps:
        ep.texts = texts
        ep.tools = tools
        ep.kit = ToolKit(search.r, TOOLSETS[tools], exa)
    if tools == "nosearch":  # the fixed loop's one search of the claim, before any controller call
        act(eps, [({"action": "search", "query": ep.c["claim"]}, None) for ep in eps],
            search, verify, decomposer, None, search_k)
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
        if hasattr(big, "batch"):  # local controller: one generate call for the chunk
            t0 = time.time()
            gs = big.batch([(SYSTEMS[ep.tools], ep.observation()) for ep in live])
            s = (time.time() - t0) / len(live)
            decisions = [decide(ep, big, {**g, "seconds": s}) for ep, g in zip(live, gs)]
        else:
            decisions = list(pool.map(lambda ep: decide(ep, big), live))
        act(live, decisions, search, verify, decomposer, pool, search_k)
    pool.shutdown()
    return [to_row(ep, run_info) for ep in eps]


def act(live, decisions, search, verify, decomposer, pool, search_k=K):
    """Carry out one round of decisions, one per live episode, batching searches and verifies."""
    searches = [(ep, a, c) for ep, (a, c) in zip(live, decisions) if a["action"] == "search"]
    if searches:
        t0 = time.time()
        hits = search.batch([a["query"] for _, a, _ in searches], search_k)
        s = (time.time() - t0) / len(searches)
        for (ep, a, c), hs in zip(searches, hits):
            ep.last_hits = hs
            ep.texts.update({f"{h['page']}:{h['line']}": h["text"] for h in hs})
            ep.kit.observe(hs)
            ep.log("search", {"query": a["query"], "k": search_k}, {"hits": hs}, s, c)

    for ep, (a, c) in zip(live, decisions):  # agent/tools.py tools, one call at a time
        if a["action"] not in TOOLSETS[ep.tools]:
            continue
        args, res, s = ep.kit.run(a["action"], a, ep.c["claim"])
        hs = res["hits"]
        if hs and a["action"] in FIRST:  # the opened page's sentences first, then the rest of the latest results
            pages = {h["page"] for h in hs}
            ep.last_hits = hs + [h for h in ep.last_hits if h["page"] not in pages]
        elif hs:  # web_pages: a new set of results, as search
            ep.last_hits = hs
        ep.texts.update({f"{h['page']}:{h['line']}": h["text"] for h in hs})
        ep.log(a["action"], args, res, s, c)

    decomps = [(ep, a, c) for ep, (a, c) in zip(live, decisions) if a["action"] == "decompose"]
    if decomps:
        def split(ep):
            t0 = time.time()
            return decomposer.run(ep.c["claim"]), time.time() - t0
        for (ep, a, c), (g, s) in zip(decomps, pool.map(lambda x: split(x[0]), decomps)):
            ep.subclaims = g["subclaims"]
            ep.ctl_in += g["tokens_in"]
            ep.ctl_out += g["tokens_out"]
            ep.log("decompose", {"claim": ep.c["claim"]}, {"subclaims": g["subclaims"]}, s,
                   {**c, "decompose_tokens_in": g["tokens_in"], "decompose_tokens_out": g["tokens_out"]})

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
            ep.kit.observe_cited(res["evidence"])
            if a.get("subclaim") is None:
                ep.claim_verdict = res
            else:
                ep.sub_verdicts[a["subclaim"]] = res
            ep.log("verify", {"claim": target, "evidence": ev, "subclaim": a.get("subclaim")}, res, s, c)

    for ep, (a, c) in zip(live, decisions):
        if a["action"] in ("commit", "abstain"):
            ep.log(a["action"], {}, {}, 0.0, c)
            ep.end = a["action"]


def combine(sub_verdicts, n):
    """Verdict on the claim from its sub-claims: any refuted refutes, all supported supports,
    anything else (a sub-claim unverified, NOT_ENOUGH_INFO or unparsable) is NOT_ENOUGH_INFO.
    None only when no sub-claim was verified at all."""
    vs = [sub_verdicts.get(i) for i in range(n)]
    if not any(vs):
        return None
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
    nei = [v["token_prob"].get(NEI) for v in vs if v and v.get("token_prob")]
    return {"verdict": NEI, "evidence": [], "confidence": max(nei) if nei else None}


FINAL_RULE = "v3"  # v1: any full-claim verdict beat the sub-claims; v2, v3: see final_answer


def firmest_nei(ep):
    """The most confident NOT_ENOUGH_INFO verdict from any verify of the full claim, or None."""
    neis = [s["result"] for s in ep.steps if s["action"] == "verify"
            and s["args"].get("subclaim") is None and s["result"].get("verdict") == NEI]
    return max(neis, key=lambda r: r["confidence"] or 0) if neis else None


def final_answer(ep):
    """commit and budget answer with the latest verdict on the claim or the combined
    sub-claim verdicts; abstain answers NOT_ENOUGH_INFO with the verifier's probability of that label.

    When both exist, the combined sub-claim answer wins only when it is decisive
    (SUPPORTED/REFUTED) and more confident than the full-claim answer, so it never
    overrides a firmer NEI (as the v2 controller prompt asks). v1 always took the full-claim verdict, which threw
    away a correct REFUTED from the sub-claims (docs/agent-loop-notes.md section 4). `source`
    records which side answered.

    v3 enforces the prompt's NEI rule across repeated verifies of the full claim: a
    decisive answer stands only if it is more confident than every NEI the verifier gave
    the claim; otherwise the answer is that NEI, with `source` "nei_rule". v2 left this to
    the controller, which committed REFUTED at 0.52 after an NEI at 0.64 (notes section 3)."""
    ans = _answer(ep)
    nei = firmest_nei(ep) if FINAL_RULE == "v3" and ep.end != "abstain" else None
    if nei and ans["verdict"] not in (None, NEI) and (ans["confidence"] or 0) <= (nei["confidence"] or 0):
        return {"verdict": NEI, "evidence": [], "confidence": nei["confidence"], "source": "nei_rule"}
    return ans


def _answer(ep):
    """final_answer under rule v2."""
    v = ep.claim_verdict
    if ep.end == "abstain":
        conf = v["token_prob"].get(NEI) if v and v.get("token_prob") else None
        return {"verdict": NEI, "evidence": [], "confidence": conf, "source": "abstain"}
    full = ({"verdict": v["verdict"], "evidence": v["evidence"], "confidence": v["confidence"]}
            if v and v["verdict"] else None)
    parts = combine(ep.sub_verdicts, len(ep.subclaims)) if ep.subclaims else None
    if full and parts:
        if parts["verdict"] != NEI and (parts["confidence"] or 0) > (full["confidence"] or 0):
            return {**parts, "source": "subclaims"}
        return {**full, "source": "claim"}
    if full:
        return {**full, "source": "claim"}
    if parts:
        return {**parts, "source": "subclaims"}
    return {"verdict": None, "evidence": [], "confidence": None, "source": None}


def to_row(ep, run_info):
    c = ep.c
    fin = final_answer(ep)
    gold = FEVER_LABEL[c["label"]]
    gs = gold_sets(c)
    gold_ids = sorted({f"{p}:{l}" for s in gs for p, l in s})
    count = lambda name: sum(s["action"] == name for s in ep.steps)
    n_search = count("search") + sum(count(t) for t in TOOLSETS["web"])
    n_decompose, n_verify = count("decompose"), count("verify")
    n_ops = n_search + n_decompose + n_verify  # each agent/tools.py call costs as a search
    reward = None
    if fin["verdict"] is not None and fin["confidence"] is not None:
        r, parts = R.total(REWARD_LABEL[gold], REWARD_LABEL[fin["verdict"]], fin["confidence"],
                           gold_ids, fin["evidence"], n_ops)
        reward = {"total": r, "parts": parts}
    v_in, v_out = getattr(ep, "v_in", 0), getattr(ep, "v_out", 0)
    return {
        "schema": 3 if "claim_ids_manifest" in run_info else SCHEMA,
        "id": c["id"], "dataset": "fever", "split": run_info.get("split", "dev"),
        "claim": c["claim"], "gold": gold,
        "gold_evidence": [sorted(f"{p}:{l}" for p, l in s) for s in gs],
        "run": run_info,
        "steps": ep.steps,
        "final": {**fin, "ended_by": ep.end},
        "cost": {"n_search": n_search, "n_follow": count("follow"),
                 **{f"n_{t}": count(t) for t in TOOLSETS["web"][1:] if t in TOOLSETS[ep.tools]},
                 "n_decompose": n_decompose,
                 "n_verify": n_verify, "n_ops": n_ops,
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
    print(f"  searches/claim   {mean([r['cost']['n_search'] for r in rows]):.2f} (tool calls included)")
    print(f"  follows/claim    {mean([r['cost'].get('n_follow', 0) for r in rows]):.2f}")
    for t in TOOLSETS["web"][1:]:
        if any(f"n_{t}" in r["cost"] for r in rows):
            print(f"  {t + ' /claim':16s} {mean([r['cost'].get(f'n_{t}', 0) for r in rows]):.2f}")
    web = [s["result"] for r in rows for s in r["steps"] if s["action"] == "web_pages"]
    if web:
        print(f"  web_pages        {sum(not w.get('cache_hit') for w in web)} live of {len(web)}, "
              f"${sum(w.get('cost_dollars') or 0 for w in web if not w.get('cache_hit')):.2f} recorded")
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
    ap.add_argument("--controller", default=BIG_MODEL,
                    help="OpenRouter model, claude-* (Anthropic API), or local:<LoRA dir> for a trained controller")
    ap.add_argument("--n", type=int, help="default 900; ignored when --claim-ids is supplied")
    ap.add_argument("--claim-ids", help="frozen FEVER dev ID manifest")
    ap.add_argument("--train-n", type=int,
                    help="N unused train claims (agent.action_cache.train_claims) instead of dev")
    ap.add_argument("--train-seed", type=int, default=2, help="--train-n sample; seed 1 is the controller set")
    ap.add_argument("--shard", help="I/N: run every Nth claim of the set from I, into its own trace")
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8, help="parallel controller calls per chunk")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tools", choices=tuple(SYSTEMS), default="full",
                    help="action set; nosearch and follow are added to the trace tag")
    ap.add_argument("--exa-cache", default=os.path.join("verifier", "data", "exa-cache"),
                    help="--tools web: Exa response cache (git-ignored)")
    ap.add_argument("--exa-network", action="store_true",
                    help="--tools web: allow live Exa requests (EXA_API_KEY in .env); otherwise cache only")
    ap.add_argument("--exa-max-requests", type=int, default=100, help="--tools web: live request budget")
    ap.add_argument("--method", choices=("bm25", "hybrid"), default="bm25")
    ap.add_argument("--search-k", type=int, default=K)
    ap.add_argument("--candidate-pages", type=int, default=K_PAGES)
    a = ap.parse_args()
    if a.search_k < 1 or a.candidate_pages < 1:
        ap.error("search-k and candidate-pages must be positive")
    if a.train_n and (a.claim_ids or a.n):
        ap.error("--train-n replaces --n and --claim-ids")
    from utils.fever_eval import manifest_metadata, manifest_tag, validate_resume_ids
    if a.train_n:
        from agent.action_cache import train_claims
        selected, split = train_claims(a.train_n, a.train_seed), "train"
    else:
        selected = sample_claims(a.n if a.n is not None or a.claim_ids else 900, SEED,
                                 ids_file=a.claim_ids)
        split = "dev"
    a.n = len(selected)
    shard = ""
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        if not 0 <= i < n:
            ap.error("--shard I/N needs 0 <= I < N")
        selected, shard = selected[i::n], f"_shard{i}of{n}"

    run = "prompted-" + (adapter_label(a.adapter) if a.adapter else "base")
    # a non-default controller is named in the trace, so it never resumes another controller's file
    ctl_tag = ("" if a.controller == BIG_MODEL
               else "ctl-local-" + adapter_label(a.controller[len("local:"):]) if a.controller.startswith("local:")
               else "ctl-" + a.controller.split("/")[-1].replace(".", "p"))
    tag = "_".join(t for t in (a.tag, ctl_tag, "" if a.tools == "full" else a.tools) if t)
    run += f"_{tag}" if tag else ""
    search_tag = f"{a.method}-p{a.candidate_pages}-k{a.search_k}"
    run += f"_{search_tag}"
    run += f"_{manifest_tag(a.claim_ids)}" if a.claim_ids else ""
    run += f"_unused-seed{a.train_seed}" if a.train_n else ""
    os.makedirs(TRACE_DIR, exist_ok=True)
    path = os.path.join(TRACE_DIR, f"fever_{split}_{run}_n{a.n}{shard}.jsonl")

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done_ids = [json.loads(line)["id"] for line in f]
            validate_resume_ids(done_ids, [c["id"] for c in selected])
            done = set(done_ids)
    claims = [c for c in selected if c["id"] not in done]
    print(f"{len(done)} done, {len(claims)} to go -> {path}")
    if claims:
        big = big_model(a.controller)
        # a local controller still decomposes with BIG_MODEL, so its sub-claims match flash's cache
        dec_big = None if a.controller.startswith("local:") else big
        decomposer = Decomposer(dec_big, cache_path(dec_big.name if dec_big else BIG_MODEL, a.n, split=split,
                                                    claim_ids=a.claim_ids))
        search = FeverSearch(method=a.method, candidate_pages=a.candidate_pages)
        verify = LocalVerify(search, a.model, a.adapter)
        exa = None
        if a.tools == "web":
            from agent.decompose import load_env
            from retrieval.exa import ExaClient
            load_env()
            exa = ExaClient(a.exa_cache, allow_network=a.exa_network, max_requests=a.exa_max_requests)
        run_info = {"policy": "prompted", "controller": big.name, "verifier": verify.name,
                    "reward_version": reward_version(), "seed": SEED, "max_steps": MAX_STEPS,
                    "controller_prompt": CONTROLLER_PROMPT + ("" if a.tools == "full" else f"-{a.tools}"),
                    "tools": a.tools, "tool_names": list(TOOLSETS[a.tools]),
                    **({"exa": {"cache": a.exa_cache, "network": a.exa_network,
                                "max_requests": a.exa_max_requests,
                                "include_domains": ["en.wikipedia.org"]}} if exa else {}),
                    "decomposer_prompt": decomposer.prompt, "final_rule": FINAL_RULE,
                    "split": split,
                    **({"train_claims": {"n": a.n, "seed": a.train_seed, "shard": a.shard}} if a.train_n else {}),
                    **manifest_metadata(a.claim_ids)}
        run_info["policy"] = f"prompted-{search_tag}"
        t0 = time.time()
        for i in range(0, len(claims), a.chunk):
            rows = run_chunk(claims[i:i + a.chunk], search, verify, big, decomposer, run_info,
                             workers=a.workers, search_k=a.search_k, tools=a.tools, exa=exa)
            with open(path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {i + len(rows)}/{len(claims)}  {time.time() - t0:.0f}s", flush=True)
    report(path, None)
    summarize_actions(path)
