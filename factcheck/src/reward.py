"""Reward terms, kept separate from the env so the weights are one file to
argue about.

All four terms are rule-based -- no LLM judge on the critical training path,
which is the cost/reproducibility claim the proposal rests on.

    R = correctness + L1 * evidence_f1 + L2 * calibration - L3 * cost

The asymmetry is the part that does not exist in the math version of this
project: a wrong SUPPORTED on a false claim (a lie waved through) is
penalised harder than a wrong REFUTED on a true claim (a truth wrongly
flagged), and both are penalised harder when asserted with high confidence
than with low. That cost matrix IS the Responsible-AI contribution -- if it
gets flattened, the project reduces to "we fine-tuned a small model on FEVER".
"""

from __future__ import annotations

from toy_corpus import NEI, REFUTED, SUPPORTED

L1_EVIDENCE = 0.5
L2_CALIBRATION = 0.5
L3_COST = 0.02  # per retrieval call / decomposition / cross-check

# (gold, predicted) -> harm multiplier on the confident-and-wrong penalty.
# Waving a false claim through is worse than over-flagging a true one.
HARM = {
    (REFUTED, SUPPORTED): 1.5,
    (SUPPORTED, REFUTED): 1.0,
    (NEI, SUPPORTED): 1.25,
    (NEI, REFUTED): 1.0,
}


def correctness(gold: str, pred: str) -> float:
    return 1.0 if gold == pred else 0.0


def evidence_f1(gold_ids: list[str], cited_ids: list[str]) -> float:
    """F1 over evidence sentence ids. Stops the model getting the right
    verdict for the wrong reason -- the historical FEVER shortcut where label
    is guessable from claim phrasing alone.

    Gold NEI claims have no gold evidence, so F1 is undefined; scored 0 and
    excluded from the shaped term by the env rather than silently counted as
    a perfect score.
    """
    if not gold_ids or not cited_ids:
        return 0.0
    g, c = set(gold_ids), set(cited_ids)
    tp = len(g & c)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(c), tp / len(g)
    return 2 * precision * recall / (precision + recall)


def calibration(gold: str, pred: str, confidence: float) -> float:
    """Reward being right *and* confident; punish being wrong *and* confident,
    scaled by the asymmetric harm of that particular error.

    Correct abstention (predicting NEI when the gold label really is NEI) is
    rewarded here rather than merely not-penalised, so abstention is a
    first-class action and not a fallback the policy learns to avoid.
    """
    if gold == pred:
        return confidence if pred != NEI else 1.0
    return -HARM.get((gold, pred), 1.0) * confidence


def total(
    gold: str,
    pred: str,
    confidence: float,
    gold_ids: list[str],
    cited_ids: list[str],
    n_ops: int,
) -> tuple[float, dict]:
    parts = {
        "correctness": correctness(gold, pred),
        "evidence_f1": evidence_f1(gold_ids, cited_ids) if gold_ids else 0.0,
        "calibration": calibration(gold, pred, confidence),
        "cost": float(n_ops),
    }
    r = (
        parts["correctness"]
        + L1_EVIDENCE * parts["evidence_f1"]
        + L2_CALIBRATION * parts["calibration"]
        - L3_COST * parts["cost"]
    )
    return r, parts
