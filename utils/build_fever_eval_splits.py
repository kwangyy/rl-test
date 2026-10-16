"""Freeze FEVER calibration/holdout IDs from recorded-unused shared-task dev claims.

Build once: python -m utils.build_fever_eval_splits
Verify:     python -m utils.build_fever_eval_splits --check
Before use: python -m utils.build_fever_eval_splits --check-unused
Guard data: python -m utils.build_fever_eval_splits --check-training path/to/rows.jsonl
"""

import argparse
from collections import Counter
import hashlib
import json
import random
import subprocess
import unicodedata
from pathlib import Path

from utils.fever_eval import (FROZEN_ROOT, LEGACY_IDS, ROOT, assert_not_reserved,
                             claim_id, sha256)

TRAIN = ROOT / "verifier/data/fever/train.jsonl"
DEV = ROOT / "verifier/data/fever/shared_task_dev.jsonl"
LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
SCAN_ROOTS = ("verifier/data/train", "verifier/data/traces", "verifier/data/decompose",
              "verifier/evaluation/results", "verifier/experiments")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def normalize_claim(text):
    """Compare duplicate text during split construction."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def claim_lookup(rows):
    by_text = {}
    for row in rows:
        by_text.setdefault(normalize_claim(row["claim"]), set()).add(str(row["id"]))
    return by_text


def identities(payload, by_text):
    ids = set()
    count = 0
    for line in payload.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("expected JSONL objects in claim usage artifacts")
        cid = claim_id(row)
        if cid is not None:
            ids.add(cid)
        elif isinstance(row.get("claim"), str):
            # Some old rollouts contain only claim text. Resolve those to FEVER IDs.
            ids.update(by_text.get(normalize_claim(row["claim"]), ()))
        count += 1
    return ids, count


def current_artifacts():
    return sorted({path for folder in SCAN_ROOTS for path in (ROOT / folder).rglob("*.jsonl")})


def audit_usage(by_text=None):
    """Include locally available ignored artifacts and all reachable JSONL Git blobs."""
    used_ids, sources, parsed = set(), [], {}
    by_text = by_text or {}

    def consume(payload, source):
        digest = hashlib.sha256(payload).hexdigest()
        if digest not in parsed:
            parsed[digest] = identities(payload, by_text)
        ids, count = parsed[digest]
        used_ids.update(ids)
        sources.append({**source, "sha256": digest, "rows": count,
                        "unique_ids": len(ids)})

    for path in current_artifacts():
        consume(path.read_bytes(), {"kind": "working_tree", "path": str(path.relative_to(ROOT))})
    objects = git("rev-list", "--objects", "--all").decode().splitlines()
    for line in sorted(objects):
        oid, _, name = line.partition(" ")
        if not name.endswith(".jsonl"):
            continue
        # A raw corpus is a source pool, not evidence of running every claim.
        if "/fever/" in name or "/wiki-pages/" in name:
            continue
        consume(git("cat-file", "blob", oid), {"kind": "git_blob", "path": name, "oid": oid})
    legacy = json.loads(LEGACY_IDS.read_text(encoding="utf-8"))
    used_ids.update(map(str, legacy["ids"]))
    return {"revision": git("rev-parse", "HEAD").decode().strip(),
            "scope": "Current claim JSONL artifacts (including ignored files), all reachable Git JSONL blobs, and the legacy 900 IDs. External/unrecorded runs and model pretraining are not audited.",
            "used_ids": sorted(used_ids, key=int),
            "sources": sources}


def source_rows(train_path, dev_path):
    train, dev = read_rows(train_path), read_rows(dev_path)
    for name, rows in (("train", train), ("dev", dev)):
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate IDs in FEVER {name} source")
        if any(row.get("label") not in LABELS or not row.get("claim") for row in rows):
            raise ValueError(f"invalid FEVER {name} source rows")
    if {row["id"] for row in train} & {row["id"] for row in dev}:
        raise ValueError("official train/dev sources overlap by ID")
    return train, dev


def select_sets(train, dev, audit, seed=0, calibration_size=900, holdout_size=3000):
    if min(calibration_size, holdout_size) < 3 or calibration_size % 3 or holdout_size % 3:
        raise ValueError("set sizes must be positive multiples of three")
    seen_ids = set(audit["used_ids"])
    seen_texts = {normalize_claim(row["claim"]) for row in train}
    # Resolve IDs even when historical records retained only labels/predictions.
    seen_texts.update(normalize_claim(row["claim"]) for row in dev if str(row["id"]) in seen_ids)
    eligible, duplicate_texts = [], set()
    excluded = Counter()
    for row in sorted(dev, key=lambda r: r["id"]):
        text = normalize_claim(row["claim"])
        if str(row["id"]) in seen_ids:
            excluded["previously_used_id"] += 1
        elif text in seen_texts:
            excluded["training_or_used_text_match"] += 1
        elif text in duplicate_texts:
            excluded["duplicate_candidate_text"] += 1
        else:
            eligible.append(row)
            duplicate_texts.add(text)
    rng = random.Random(seed)
    calibration, holdout = [], []
    for label in LABELS:
        pool = [row for row in eligible if row["label"] == label]
        rng.shuffle(pool)
        cal_n, test_n = calibration_size // 3, holdout_size // 3
        if len(pool) < cal_n + test_n:
            raise ValueError(f"not enough unused {label} claims")
        calibration.extend(pool[:cal_n])
        holdout.extend(pool[cal_n:cal_n + test_n])
    rng.shuffle(calibration)
    rng.shuffle(holdout)
    return calibration, holdout, {"eligible": len(eligible), "excluded": dict(excluded)}


def write_json(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def id_record(rows, role, source_hash=None):
    record = {"dataset": "FEVER", "split": "shared_task_dev", "role": role,
              "count": len(rows), "ids": [row["id"] for row in rows]}
    if source_hash:
        record.update(source_sha256=source_hash,
                      label_counts=dict(Counter(row["label"] for row in rows)))
    return record


def build(directory=FROZEN_ROOT, train_path=TRAIN, dev_path=DEV):
    directory = Path(directory)
    if directory.exists():
        raise SystemExit(f"{directory} already exists; use --check. Frozen sets must not be overwritten.")
    train, dev = source_rows(train_path, dev_path)
    if len(train) != 145449 or len(dev) != 19998:
        raise SystemExit("expected official FEVER train (145449) and shared-task dev (19998)")
    audit = audit_usage(claim_lookup(train + dev))
    calibration, holdout, selection = select_sets(train, dev, audit)
    used = set(audit["used_ids"])
    used_train = sorted(int(row["id"]) for row in train if str(row["id"]) in used)
    used_dev = [row for row in dev if str(row["id"]) in used]
    audit["counts"] = {"recorded_used_train_ids": len(used_train),
                       "recorded_used_dev_ids": len(used_dev),
                       "working_tree_artifacts": sum(s["kind"] == "working_tree" for s in audit["sources"]),
                       "historical_git_blobs": sum(s["kind"] == "git_blob" for s in audit["sources"])}
    audit["used_train_ids"] = used_train
    directory.mkdir(parents=True)
    write_json(directory / "usage-audit.json", audit)
    records = {"development-used-ids.json": id_record(sorted(used_dev, key=lambda r: r["id"]), "development"),
               "calibration-900-ids.json": id_record(calibration, "calibration", sha256(dev_path)),
               "holdout-3000-ids.json": id_record(holdout, "holdout", sha256(dev_path))}
    for name, record in records.items():
        write_json(directory / name, record)
    manifest = {"schema": 1, "name": "fever-evaluation-v1", "seed": 0,
                "sampling": "Sort dev by ID; exclude all recorded usage and normalized text matches to the full official train pool or used claims; deduplicate candidates; shuffle each label with Random(0); allocate 300 calibration then 1000 holdout per label; shuffle each set.",
                "normalization": "Unicode NFKC, casefold and whitespace collapse, compared directly during selection only. No per-claim hashes are stored. Near-duplicates and entity/page-disjointness are not claimed.",
                "sources": {"train": {"path": str(Path(train_path).relative_to(ROOT)), "count": len(train), "sha256": sha256(train_path)},
                            "dev": {"path": str(Path(dev_path).relative_to(ROOT)), "count": len(dev), "sha256": sha256(dev_path)}},
                "legacy_development": {"path": str(LEGACY_IDS.relative_to(ROOT)), "count": 900, "sha256": sha256(LEGACY_IDS)},
                "audit": {"path": "usage-audit.json", "sha256": sha256(directory / "usage-audit.json")},
                "selection": selection,
                "sets": {name: {"role": record["role"], "count": record["count"], "sha256": sha256(directory / name)}
                         for name, record in records.items()},
                "policy": {"training": "Keep official train available for verifier and controller training; never train on calibration or holdout IDs. Runtime leakage guards compare FEVER IDs.",
                           "development": "Keep the historical 900 unchanged. Previously inspected dev claims remain development data, including smoke-test extras.",
                           "calibration": "Fit confidence adjustments only; do not use for verifier/controller weights or policy/prompt selection.",
                           "holdout": "Evaluate finalized configurations only; no training, teacher-target generation, prompt tuning, or failure-driven policy changes. If used for development, retire/version this holdout.",
                           "scope": "Unused in available project records at freeze time; no guarantee about unrecorded external runs or pretrained-model exposure.",
                           "labels": "Class-balanced evaluation; report this sampling scheme, since precision and calibration depend on class prevalence."}}
    write_json(directory / "manifest.json", manifest)
    return audit["counts"]


def check(directory=FROZEN_ROOT, unused=False):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for source in (*manifest["sources"].values(), manifest["legacy_development"]):
        if sha256(ROOT / source["path"]) != source["sha256"]:
            raise ValueError(f"frozen source changed: {source['path']}")
    audit_path = directory / manifest["audit"]["path"]
    if sha256(audit_path) != manifest["audit"]["sha256"]:
        raise ValueError("frozen usage audit changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    train, dev = source_rows(ROOT / manifest["sources"]["train"]["path"],
                             ROOT / manifest["sources"]["dev"]["path"])
    calibration, holdout, selection = select_sets(train, dev, audit, manifest["seed"])
    if selection != manifest["selection"]:
        raise ValueError("frozen candidate selection changed")
    expected = {"calibration-900-ids.json": calibration, "holdout-3000-ids.json": holdout}
    seen_ids = set()
    for name, details in manifest["sets"].items():
        path = directory / name
        if sha256(path) != details["sha256"]:
            raise ValueError(f"frozen set changed: {name}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if name in expected and record != id_record(expected[name], details["role"], manifest["sources"]["dev"]["sha256"]):
            raise ValueError(f"frozen set is not reproducible: {name}")
        ids = set(record["ids"])
        if len(ids) != details["count"] or ids & seen_ids:
            raise ValueError("frozen sets overlap or have incorrect counts")
        seen_ids.update(ids)
    # Legitimate post-freeze evaluations may contain reserved IDs. Training may not.
    for path in sorted((ROOT / "verifier/data/train").rglob("*.jsonl")):
        assert_not_reserved(read_rows(path), f"training file {path.relative_to(ROOT)}", directory)
    if unused:
        current = audit_usage(claim_lookup(train + dev))
        protected_ids = {str(row["id"]) for rows in (calibration, holdout) for row in rows}
        if protected_ids & set(current["used_ids"]):
            raise ValueError("calibration/holdout has appeared in project artifacts since freezing")
    return audit["counts"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify source/set hashes, deterministic selection and current training leakage")
    parser.add_argument("--check-unused", action="store_true", help="also require no recorded evaluation exposure since freezing")
    parser.add_argument("--check-training", nargs="+", type=Path, help="reject reserved IDs in proposed training/controller JSONL files")
    args = parser.parse_args()
    if args.check_training:
        for path in args.check_training:
            rows = read_rows(path)
            if any(claim_id(row) is None for row in rows):
                raise SystemExit("proposed training rows must include a FEVER id or claim_id")
            assert_not_reserved(rows, f"proposed training {path}")
        print("Proposed training data has no reserved calibration/holdout IDs")
        return
    counts = check(unused=args.check_unused) if args.check or args.check_unused else build()
    print(json.dumps({"frozen_sets_verified": True, "calibration": 900, "holdout": 3000, **counts}, indent=2))


if __name__ == "__main__":
    main()
