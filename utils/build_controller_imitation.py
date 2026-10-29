"""Controller imitation data (AG-2): replay prompted-controller traces into
(observation, action) pairs for behavioural cloning of the controller.

Each trace step the controller chose becomes one example: the prompt is the
observation agent/prompted.py showed the controller before that step, rebuilt
by replaying the earlier steps through Episode and ToolKit (link ids L<n>
depend on the order links were seen, so the tools' state is replayed too); the
completion is the action that ran, as one JSON object. Steps whose controller
output was invalid are skipped (the loop ran a fallback the controller did not
choose), but their effects are still replayed.

Only episodes with a correct FEVER score are kept by default. Claims are split
into train and val by a seeded shuffle, so all steps of a claim share a split.

    python -m utils.build_controller_imitation \
        --traces verifier/data/controller/imitation-flash-follow-train-unused-n6000-seed2

Writes verifier/data/train/controller/<name>/{train,val}.jsonl and manifest.json.
"""

import argparse
import glob
import gzip
import json
import os
import random
from collections import Counter
from pathlib import Path

from agent.prompted import SYSTEMS, TOOLSETS, Episode
from agent.tools import ToolKit
from retrieval.build_bm25_index import DATA

OUT_ROOT = os.path.join(DATA, "..", "train", "controller")
NEI = "NOT_ENOUGH_INFO"


def fever_ok(r):
    f = r["final"]
    if f["verdict"] != r["gold"]:
        return False
    return r["gold"] == NEI or any(set(g) <= set(f["evidence"] or []) for g in r["gold_evidence"])


def action_json(step):
    """The action a step ran, in the controller's reply format."""
    a, args = step["action"], step["args"]
    obj = {"action": a}
    if a == "search":
        obj["query"] = args["query"]
    elif a == "follow":
        obj["link"] = args["link"]
    elif a == "verify":
        obj["subclaim"] = args.get("subclaim")
    elif a not in ("decompose", "commit", "abstain"):
        raise ValueError(f"no imitation target for action {a}")
    return json.dumps(obj, ensure_ascii=False)


def replay(row, retriever):
    """Yield (step, observation) for each controller step, applying every step's effects as run_chunk did."""
    tools = row["run"]["tools"]
    if set(TOOLSETS[tools]) - {"follow"}:
        raise ValueError(f"replay supports the full and follow presets, not {tools}")
    ep = Episode({"id": row["id"], "claim": row["claim"]})
    ep.tools, ep.texts = tools, {}
    ep.kit = ToolKit(retriever, TOOLSETS[tools])
    for s in row["steps"]:
        if "controller" in s:
            yield s, ep.observation()
        a, args, res = s["action"], s["args"], s["result"]
        if a == "search":
            ep.last_hits = res["hits"]
            ep.kit.observe(res["hits"])
        elif a == "follow":
            link = ep.kit.links.get(args["link"])
            if link is None or link["target"] != args["target"]:
                raise ValueError(f"claim {row['id']}: replayed link {args['link']} does not match the trace")
            ep.kit.followed.add(args["link"])
            hs = res["hits"]
            if hs:
                ep.kit.observe(hs)
                pages = {h["page"] for h in hs}
                ep.last_hits = hs + [h for h in ep.last_hits if h["page"] not in pages]
        elif a == "decompose":
            ep.subclaims = res["subclaims"]
        elif a == "verify":
            ep.cited = list(dict.fromkeys(ep.cited + res["evidence"]))
            ep.kit.observe_cited(res["evidence"])
            if args.get("subclaim") is None:
                ep.claim_verdict = res
            else:
                ep.sub_verdicts[args["subclaim"]] = res
        ep.texts.update({f"{h['page']}:{h['line']}": h["text"] for h in res.get("hits", [])})
        ep.steps.append(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="directory of traces-*.jsonl.gz, or one .jsonl/.jsonl.gz")
    ap.add_argument("--name", default="flash-follow-imitation")
    ap.add_argument("--all-episodes", action="store_true", help="keep episodes with a FEVER score of 0 too")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.traces, "*.jsonl*"))) if os.path.isdir(a.traces) else [a.traces]
    opener = lambda f: gzip.open(f, "rt", encoding="utf-8") if f.endswith(".gz") else open(f, encoding="utf-8")
    rows = [json.loads(line) for f in files for line in opener(f)]
    tools = {r["run"]["tools"] for r in rows}
    if len(tools) != 1:
        raise SystemExit(f"traces mix presets {sorted(tools)}")
    tools = tools.pop()
    kept = rows if a.all_episodes else [r for r in rows if fever_ok(r)]

    from retrieval.fever_retrieval import FeverRetriever
    retriever = FeverRetriever()
    ids = sorted({r["id"] for r in kept})
    random.Random(a.seed).shuffle(ids)
    val_ids = set(ids[:round(len(ids) * a.val_frac)])

    out = {"train": [], "val": []}
    skipped = Counter()
    for i, r in enumerate(kept):
        for s, obs in replay(r, retriever):
            if not s["controller"]["valid"]:
                skipped[s["action"]] += 1
                continue
            out["val" if r["id"] in val_ids else "train"].append({
                "id": r["id"], "t": s["t"], "gold": r["gold"], "action": s["action"],
                "prompt": obs, "completion": action_json(s)})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(kept)} episodes", flush=True)

    d = os.path.join(OUT_ROOT, f"{a.name}-{len(out['train'])}-samples")
    os.makedirs(d, exist_ok=True)
    for split, xs in out.items():
        with open(os.path.join(d, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for x in xs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    manifest = {
        "traces": [Path(os.path.relpath(f)).as_posix() for f in files], "tools": tools, "system_prompt": SYSTEMS[tools],
        "episodes": len(rows), "episodes_kept": len(kept), "filter": "all" if a.all_episodes else "fever_score == 1",
        "claims": {"train": len(ids) - len(val_ids), "val": len(val_ids)}, "split_seed": a.seed,
        "examples": {k: len(v) for k, v in out.items()},
        "actions": {k: dict(Counter(x["action"] for x in v)) for k, v in out.items()},
        "skipped_invalid_controller_steps": dict(skipped),
    }
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "system_prompt"}, indent=1))
    print(f"-> {d}")
