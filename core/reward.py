"""Reward terms, kept separate from the env so the weights are one file to
argue about.

All terms are rule-based -- no LLM judge on the critical training path,
which is the cost/reproducibility claim the proposal rests on.

    R = correctness + L1 * evidence + L2 * calibration - L3 * cost

The asymmetry is the part that does not exist in the math version of this
project: a wrong SUPPORTED on a false claim (a lie waved through) is
penalised harder than a wrong REFUTED on a true claim (a truth wrongly
flagged). That cost matrix IS the Responsible-AI contribution -- if it gets
flattened, the project reduces to "we fine-tuned a small model on FEVER".

Harm sits on the verdict, not on the confidence: the calibration term is a
Brier score, a proper scoring rule, and scaling it per error type would
break propriety (the optimum confidence would no longer equal the chance of
being right). Confidence is scored for honesty; the verdict is scored for harm.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from agent.toy_corpus import NEI, REFUTED, SUPPORTED

L1_EVIDENCE = 0.5
L2_CALIBRATION = 0.5
L3_COST = 0.02  # per retrieval call / decomposition / cross-check

# (gold, predicted) -> harm of that wrong verdict. Unlisted errors cost 1.0.
# Waving a false claim through is worse than over-flagging a true one.
# A wrong verdict scores 1 - harm, so harm 1.0 is plain 0/1 accuracy and the
# A harm-ratio sweep (for example 1, 1.5, 3, 5) changes this table.
HARM = {
    (REFUTED, SUPPORTED): 1.5,
    (SUPPORTED, REFUTED): 1.0,
    (NEI, SUPPORTED): 1.75,
    (NEI, REFUTED): 1.0,
}

# Gold-NEI claims have no gold evidence. Citing nothing is correct; citing
# anything asserts that insufficient evidence settles the claim.
NEI_EMPTY_REWARD = 1.0
NEI_CITED_PENALTY = 0.5


@dataclass(frozen=True)
class RewardConfig:
    """Switches for reward ablations (see docs/plan.md and the root TODO.md).

    mode: "full" (the shaped reward), "format_only" (1 if the output parsed,
    else 0), or "random" (uniform in [0, 1], ignores the episode).

    evidence_gate: a correct SUPPORTED/REFUTED earns correctness credit only when it
    cites at least one gold sentence (FEVER's evidence requirement); gold NEI is
    unchanged. Off everywhere except the GRPO ablation that names it.
    """

    mode: str = "full"
    calibration: bool = True
    harm: bool = True
    evidence: bool = True
    evidence_gate: bool = False


FULL = RewardConfig()
ABLATIONS = {
    "full": FULL,
    "no_calibration": RewardConfig(calibration=False),
    # GRPO from base learned to guess REFUTED without using evidence
    # (docs/multiturn-grpo-results.md); this ablation withholds credit for unsupported verdicts.
    "no_calibration_evidence_gated": RewardConfig(calibration=False, evidence_gate=True),
    "no_harm": RewardConfig(harm=False),
    "format_only": RewardConfig(mode="format_only"),
    "random": RewardConfig(mode="random"),
}


def correctness(gold: str, pred: str, harm: bool = True) -> float:
    """1 for the right verdict, 1 - harm for a wrong one (0 with harm off)."""
    if gold == pred:
        return 1.0
    if not harm:
        return 0.0
    return 1.0 - HARM.get((gold, pred), 1.0)


def evidence_f1(gold_ids: list[str], cited_ids: list[str]) -> float:
    """F1 over evidence sentence ids. Stops the model getting the right
    verdict for the wrong reason -- the historical FEVER shortcut where label
    is guessable from claim phrasing alone.

    Undefined when gold_ids is empty (gold NEI); returns 0 there. Use
    `evidence_score` for the reward, which handles NEI explicitly.
    """
    if not gold_ids or not cited_ids:
        return 0.0
    g, c = set(gold_ids), set(cited_ids)
    tp = len(g & c)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(c), tp / len(g)
    return 2 * precision * recall / (precision + recall)


def evidence_score(gold: str, pred: str, gold_ids: list[str], cited_ids: list[str]) -> float:
    """Evidence F1 on SUPPORTED/REFUTED; the empty-list rule on gold NEI.

    Credit only comes with the right verdict, as in the FEVER score: otherwise
    citing the right sentence would refund the harm penalty on a lie waved
    through, and SUPPORTED-with-no-evidence on an NEI claim would collect the
    abstention reward. Citing anything on a gold-NEI claim is penalised
    whatever the verdict.
    """
    if pred != gold:
        return -NEI_CITED_PENALTY if gold == NEI and cited_ids else 0.0
    if gold == NEI:
        return NEI_EMPTY_REWARD if not cited_ids else -NEI_CITED_PENALTY
    return evidence_f1(gold_ids, cited_ids)


def calibration(gold: str, pred: str, confidence: float) -> float:
    """Negative Brier score of the stated confidence that `pred` is right.

    In [-1, 0]. Expected value over outcomes is maximised exactly when
    confidence equals the true chance of being right, so it rewards graded
    confidence instead of pushing it to 0 or 1. Abstention (pred NEI) is
    scored the same way: a correct NEI earns its reward through correctness.
    """
    outcome = 1.0 if gold == pred else 0.0
    return -((confidence - outcome) ** 2)


def total(
    gold: str,
    pred: str,
    confidence: float,
    gold_ids: list[str],
    cited_ids: list[str],
    n_ops: int,
    format_ok: bool = True,
    cfg: RewardConfig = FULL,
    rng: random.Random | None = None,
) -> tuple[float, dict]:
    if cfg.mode == "random":
        r = (rng or random).random()
        return r, {"random": r}
    if cfg.mode == "format_only":
        r = 1.0 if format_ok else 0.0
        return r, {"format": r}
    if cfg.mode != "full":
        raise ValueError(f"unknown reward mode {cfg.mode!r}")

    right = correctness(gold, pred, harm=cfg.harm)
    if cfg.evidence_gate and gold == pred and gold != NEI and evidence_f1(gold_ids, cited_ids) == 0.0:
        right = 0.0  # a right verdict with no gold sentence cited is scored like a harmless miss
    parts = {
        "correctness": right,
        "evidence": evidence_score(gold, pred, gold_ids, cited_ids) if cfg.evidence else 0.0,
        "calibration": calibration(gold, pred, confidence) if cfg.calibration else 0.0,
        "cost": float(n_ops),
        "evidence_f1": evidence_f1(gold_ids, cited_ids),  # metric only, not in r
    }
    r = (
        parts["correctness"]
        + L1_EVIDENCE * parts["evidence"]
        + L2_CALIBRATION * parts["calibration"]
        - L3_COST * parts["cost"]
    )
    return r, parts
