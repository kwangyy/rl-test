"""Score saved verifier prediction JSONL without a GPU or raw FEVER corpus.

    python -m verifier.evaluation.score_predictions verifier/evaluation/results/archive/local_n900_titles.jsonl

The binary Brier score here measures whether the *selected verdict* is
correct. It is not the multiclass Brier score over all three labels.
"""

import argparse
import json
import math
from pathlib import Path

from core import reward as R

LABELS = ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO")
CONFIDENCES = ("stated_conf", "token_conf")


def mean(values):
    return sum(values) / len(values) if values else None


def verdict_harm(gold, pred):
    gold = R.NEI if gold == "NOT_ENOUGH_INFO" else gold
    pred = R.NEI if pred == "NOT_ENOUGH_INFO" else pred
    return 1.0 - R.correctness(gold, pred)


def read_rows(path):
    rows, ids = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            claim_id = row.get("id")
            if claim_id is None or claim_id in ids:
                raise ValueError(f"{path}:{line_number}: missing or duplicate claim id {claim_id!r}")
            if row.get("gold") not in LABELS or row.get("pred") not in (*LABELS, None):
                raise ValueError(f"{path}:{line_number}: invalid gold or predicted verdict")
            if row.get("correct") is not (row["gold"] == row["pred"]):
                raise ValueError(f"{path}:{line_number}: correctness disagrees with labels")
            for key in CONFIDENCES:
                value = row.get(key)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                          or not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f"{path}:{line_number}: invalid {key}={value!r}")
            ids.add(claim_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no evaluation rows")
    return rows


def calibration(rows, key, bins=10):
    available = [r for r in rows if r.get(key) is not None]
    result = {"n": len(available), "coverage": len(available) / len(rows),
              "brier": None, "ece": None, "risk_at_coverage": None}
    if not available:
        return result
    result["brier"] = mean([(r[key] - int(r["correct"])) ** 2 for r in available])
    ece = 0.0
    for index in range(bins):
        group = [r for r in available if min(int(r[key] * bins), bins - 1) == index]
        if group:
            ece += len(group) / len(available) * abs(mean([r[key] for r in group])
                                                   - mean([int(r["correct"]) for r in group]))
    result["ece"] = ece

    # Include all rows with a tied score at the boundary. This avoids an
    # arbitrary ordering of the many SFT predictions with confidence 1.0.
    scores = sorted({r[key] for r in available}, reverse=True)
    risk = {}
    for target in (0.5, 0.8, 1.0):
        threshold = next(score for score in scores
                         if sum(r[key] >= score for r in available) / len(available) >= target)
        selected = [r for r in available if r[key] >= threshold]
        risk[str(target)] = {"coverage": len(selected) / len(rows),
                             "risk": 1 - mean([int(r["correct"]) for r in selected]),
                             "threshold": threshold}
    result["risk_at_coverage"] = risk
    return result


def summarize(rows):
    non_nei = [r for r in rows if r["gold"] != "NOT_ENOUGH_INFO"]
    strata = {
        "gold_in_context": [r for r in non_nei if r.get("gold_in_context") is True],
        "gold_not_in_context": [r for r in non_nei if r.get("gold_in_context") is False],
        "gold_nei": [r for r in rows if r["gold"] == "NOT_ENOUGH_INFO"],
    }

    def metrics(part):
        format_rows = [r for r in part if r.get("format_ok") is not None]
        return {
            "n": len(part),
            "accuracy": mean([int(r["correct"]) for r in part]),
            "format_rate": mean([int(r["format_ok"]) for r in format_rows]),
            "fever_score": mean([int(r["fever_score"]) for r in part if r.get("fever_score") is not None]),
            "evidence_f1_non_nei": mean([r["evidence_f1"] for r in part
                                         if r.get("evidence_f1") is not None]),
            "recall": {label: mean([int(r["pred"] == label) for r in part if r["gold"] == label])
                       for label in LABELS},
            "harm_weighted_error": mean([verdict_harm(r["gold"], r["pred"]) for r in part]),
            "nei_precision": mean([int(r["gold"] == "NOT_ENOUGH_INFO") for r in part
                                   if r["pred"] == "NOT_ENOUGH_INFO"]),
            "calibration": {key: calibration(part, key) for key in CONFIDENCES} if part else None,
        }

    return {"model": rows[0].get("model"), "retrieval": rows[0].get("retrieval"),
            "prompt_style": rows[0].get("prompt_style", "direct"),
            "all": metrics(rows), "strata": {name: metrics(part) for name, part in strata.items()},
            "mean_tokens_in": mean([r["in_tokens"] for r in rows if r.get("in_tokens") is not None]),
            "mean_tokens_out": mean([r["out_tokens"] for r in rows if r.get("out_tokens") is not None]),
            "reasoning_present_rate": mean([int(bool(r.get("reasoning_present"))) for r in rows])
            if rows[0].get("prompt_style") in ("reasoning", "cot_chain", "cot_brief", "cot_sft") else None,
            "reasoning_valid_citations_rate": mean([
                int(r["reasoning_citations_valid"]) for r in rows
                if r.get("reasoning_citations_valid") is not None])
            if rows[0].get("prompt_style") in ("reasoning", "cot_chain", "cot_brief", "cot_sft") else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="eval_verifier JSONL files; each is evaluated separately")
    args = parser.parse_args()
    print(json.dumps({path: summarize(read_rows(path)) for path in args.runs},
                     ensure_ascii=False, indent=2))
