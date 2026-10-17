"""Lightweight loading and leakage guards for frozen FEVER evaluation IDs."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "verifier/data/eval"
FROZEN_ROOT = EVAL_ROOT / "fever-evaluation-v1"
LEGACY_IDS = EVAL_ROOT / "fever-dev-900-ids.json"
RESERVED_FILES = ("calibration-900-ids.json", "holdout-3000-ids.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def claim_id(row):
    """Read prediction, trace, teacher-request and prepared-training identities."""
    value = row.get("id", row.get("claim_id"))
    return str(value) if value is not None else None


def read_id_manifest(path):
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    ids = record.get("ids")
    if (not isinstance(ids, list) or not ids
            or any(type(cid) is not int for cid in ids) or len(set(ids)) != len(ids)):
        raise ValueError(f"{path}: expected nonempty unique integer claim IDs")
    if record.get("dataset", "FEVER").upper() != "FEVER":
        raise ValueError(f"{path}: not a FEVER manifest")
    if record.get("split") != "shared_task_dev":
        raise ValueError(f"{path}: evaluation IDs must come from shared_task_dev")
    if record.get("count", len(ids)) != len(ids):
        raise ValueError(f"{path}: manifest count disagrees with IDs")
    return record


def load_manifest_claims(source, manifest):
    record = read_id_manifest(manifest)
    if record.get("source_sha256") and sha256(source) != record["source_sha256"]:
        raise ValueError("FEVER dev source differs from the frozen source hash")
    wanted = set(record["ids"])
    with Path(source).open(encoding="utf-8") as stream:
        by_id = {row["id"]: row for row in map(json.loads, stream) if row["id"] in wanted}
    missing = set(record["ids"]) - by_id.keys()
    if missing:
        raise ValueError(f"FEVER source is missing {len(missing)} frozen claim IDs")
    return [by_id[cid] for cid in record["ids"]]


def manifest_tag(path):
    return f"{Path(path).stem}-{sha256(path)[:8]}"


def manifest_metadata(path):
    return {"claim_ids_manifest": str(path), "claim_ids_sha256": sha256(path)} if path else {}


def reserved_ids(directory=FROZEN_ROOT):
    ids = set()
    for name in RESERVED_FILES:
        path = Path(directory) / name
        if Path(directory).exists() and not path.is_file():
            raise ValueError(f"reserved evaluation manifest missing: {path}")
        if path.exists():
            record = read_id_manifest(path)
            ids.update(map(str, record["ids"]))
    return ids


def assert_not_reserved(rows, purpose="training/development", directory=FROZEN_ROOT):
    ids = reserved_ids(directory)
    for row in rows:
        if claim_id(row) in ids:
            raise ValueError(f"reserved calibration/holdout claim in {purpose}; "
                             "use a development set, or an explicit --claim-ids evaluation")


def validate_resume_ids(actual, expected):
    actual = list(actual)
    if len(actual) != len(set(actual)) or not set(actual) <= set(expected):
        raise ValueError("existing output contains duplicate IDs or claims outside the selected set")
