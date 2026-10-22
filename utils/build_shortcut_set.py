"""Freeze development claims and shuffled context donors for the shortcut diagnostic."""

import argparse
import json
from pathlib import Path
import random

from utils.fever_eval import LEGACY_IDS, assert_not_reserved, read_id_manifest, sha256
from verifier.model import FEVER_LABEL, LABELS, SEED

DEFAULT_TRACE = "verifier/data/traces/fever_dev_fixed-base_n900.jsonl"
DEFAULT_SET = "verifier/data/eval/fever-shortcut-dev-300-ids.json"
CONDITIONS = ("normal", "empty", "shuffled")
NEI = "NOT_ENOUGH_INFO"


def load_examples(path):
    """Keep the exact evidence order from a fixed policy's single verification."""
    allowed = set(read_id_manifest(LEGACY_IDS)["ids"])
    examples = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row["id"] not in allowed or row["id"] in examples
                or row["dataset"] != "fever" or row["split"] != "dev"
                or row["run"]["policy"] != "fixed"):
            raise ValueError("source must contain unique frozen development IDs and fixed-policy traces")
        searches = [s for s in row["steps"] if s["action"] == "search"]
        verifies = [s for s in row["steps"] if s["action"] == "verify"]
        if len(searches) != 1 or len(verifies) != 1 or verifies[0]["args"]["claim"] != row["claim"]:
            raise ValueError("expected one search and one full-claim verification per trace")
        hits = {f"{h['page']}:{h['line']}": h for h in searches[0]["result"]["hits"]}
        shown = [(f"E{i + 1}", hits[e]["page"], hits[e]["line"], hits[e]["text"])
                 for i, e in enumerate(verifies[0]["args"]["evidence"])]
        evidence = [[[None, None, e.rsplit(":", 1)[0], int(e.rsplit(":", 1)[1])]
                     for e in group] for group in row["gold_evidence"]]
        claim = {"id": row["id"], "claim": row["claim"],
                 "label": {v: k for k, v in FEVER_LABEL.items()}[row["gold"]], "evidence": evidence}
        examples[row["id"]] = {"claim": claim, "gold": row["gold"], "shown": shown,
                               "gold_sets": [set(group) for group in row["gold_evidence"]]}
    assert_not_reserved([e["claim"] for e in examples.values()], "shortcut development set")
    return examples


def context(example, shown):
    keys = {f"{page}:{line}" for _, page, line, _ in shown}
    pages = {page for _, page, _, _ in shown}
    gold = example["gold_sets"]
    return {"shown": shown,
            "gold_in_context": any(group <= keys for group in gold) if gold else None,
            "gold_page_retrieved": any(all(e.rsplit(":", 1)[0] in pages for e in group)
                                       for group in gold) if gold else None}


def prepare(trace, output, n=300, seed=SEED):
    if n < 3 or n % 3:
        raise ValueError("n must be a positive multiple of three")
    examples = load_examples(trace)
    rng = random.Random(seed)
    ids = []
    for label in LABELS:
        pool = sorted(cid for cid, e in examples.items() if e["gold"] == label
                      and e["shown"] and (label == NEI or context(e, e["shown"])["gold_in_context"]))
        if len(pool) < n // 3:
            raise ValueError(f"not enough eligible {label} development claims")
        ids.extend(rng.sample(pool, n // 3))
    rng.shuffle(ids)
    pages = {cid: {p for _, p, _, _ in examples[cid]["shown"]} for cid in ids}
    gold_pages = {cid: {e.rsplit(":", 1)[0] for group in examples[cid]["gold_sets"] for e in group}
                  for cid in ids}
    donors = ids.copy()
    for _ in range(1000):
        rng.shuffle(donors)
        if all(cid != donor and not pages[donor] & (pages[cid] | gold_pages[cid])
               for cid, donor in zip(ids, donors)):
            break
    else:
        raise ValueError("could not shuffle contexts without shared pages; try another seed")
    record = {"dataset": "FEVER", "split": "shared_task_dev", "purpose": "shortcut-development",
              "count": n, "seed": seed, "source_trace": str(trace), "source_trace_sha256": sha256(trace),
              "development_ids_sha256": sha256(LEGACY_IDS), "conditions": list(CONDITIONS),
              "selection": "balanced labels; complete gold evidence in normal context for non-NEI",
              "ids": ids, "shuffled_donor_ids": donors}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=DEFAULT_TRACE)
    parser.add_argument("--output", default=DEFAULT_SET)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    spec = prepare(args.trace, args.output, args.n, args.seed)
    print(f"Prepared {spec['count']} claims x 3 conditions -> {args.output}")


if __name__ == "__main__":
    main()
