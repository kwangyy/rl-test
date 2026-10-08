"""Tests that encode what the reward is for, not just what it returns.

    cd factcheck/src && python -m pytest test_reward.py -q
"""

import random

import numpy as np
import pytest

import reward as R
from toy_corpus import NEI, REFUTED, SUPPORTED


def expected_calibration(p_right: float, confidence: float) -> float:
    return p_right * R.calibration(SUPPORTED, SUPPORTED, confidence) + (1 - p_right) * R.calibration(
        REFUTED, SUPPORTED, confidence
    )


@pytest.mark.parametrize("p_right", [0.0, 0.2, 0.5, 0.73, 1.0])
def test_brier_is_proper(p_right):
    """Optimum stated confidence equals the true chance of being right."""
    grid = np.linspace(0, 1, 101)
    best = grid[np.argmax([expected_calibration(p_right, c) for c in grid])]
    assert best == pytest.approx(p_right, abs=0.01)


def test_brier_does_not_push_to_extremes():
    """The old term rewarded confidence 1 whenever accuracy > 0.5."""
    assert expected_calibration(0.7, 0.7) > expected_calibration(0.7, 1.0)
    assert expected_calibration(0.7, 0.7) > expected_calibration(0.7, 0.0)


def test_calibration_range():
    assert R.calibration(SUPPORTED, SUPPORTED, 1.0) == 0.0
    assert R.calibration(SUPPORTED, REFUTED, 1.0) == -1.0
    assert R.calibration(NEI, NEI, 0.5) == -0.25  # abstention scored like any verdict


def test_harm_asymmetry():
    """Waving a false claim through costs more than wrongly flagging a true one."""
    lie_through = R.correctness(REFUTED, SUPPORTED)
    truth_flagged = R.correctness(SUPPORTED, REFUTED)
    assert lie_through < truth_flagged
    assert R.correctness(NEI, SUPPORTED) < R.correctness(NEI, REFUTED)


def test_harm_one_is_plain_accuracy():
    assert R.correctness(SUPPORTED, SUPPORTED) == 1.0
    assert R.correctness(SUPPORTED, REFUTED) == 0.0  # harm 1.0
    assert R.correctness(SUPPORTED, NEI) == 0.0  # unlisted -> 1.0


def test_harm_independent_of_confidence():
    """Harm on the verdict, not the confidence: the asymmetry survives at any confidence."""
    for c in (0.1, 0.5, 0.9):
        lie, _ = R.total(REFUTED, SUPPORTED, c, ["a"], [], 0)
        flag, _ = R.total(SUPPORTED, REFUTED, c, ["a"], [], 0)
        assert lie < flag


def test_nei_evidence_rule():
    assert R.evidence_score(NEI, [], []) == R.NEI_EMPTY_REWARD
    assert R.evidence_score(NEI, [], ["x:1"]) == -R.NEI_CITED_PENALTY
    assert R.evidence_score(SUPPORTED, ["a", "b"], ["a"]) == pytest.approx(2 / 3)
    assert R.evidence_score(SUPPORTED, ["a"], []) == 0.0


def test_nei_citing_nothing_beats_citing_something():
    quiet, _ = R.total(NEI, NEI, 0.8, [], [], 1)
    cites, _ = R.total(NEI, NEI, 0.8, [], ["x:1"], 1)
    assert quiet > cites


def test_full_reward_composition():
    r, parts = R.total(SUPPORTED, SUPPORTED, 0.9, ["a"], ["a"], 3)
    expected = 1.0 + R.L1_EVIDENCE * 1.0 + R.L2_CALIBRATION * -(0.1**2) - R.L3_COST * 3
    assert r == pytest.approx(expected)
    assert parts["evidence_f1"] == 1.0


def test_ablation_no_calibration():
    r, parts = R.total(SUPPORTED, REFUTED, 1.0, ["a"], [], 0, cfg=R.ABLATIONS["no_calibration"])
    assert parts["calibration"] == 0.0
    assert r == 0.0


def test_ablation_no_harm():
    cfg = R.ABLATIONS["no_harm"]
    lie, _ = R.total(REFUTED, SUPPORTED, 0.5, ["a"], [], 0, cfg=cfg)
    flag, _ = R.total(SUPPORTED, REFUTED, 0.5, ["a"], [], 0, cfg=cfg)
    assert lie == flag


def test_ablation_format_only():
    cfg = R.ABLATIONS["format_only"]
    assert R.total(REFUTED, SUPPORTED, 1.0, ["a"], [], 5, format_ok=True, cfg=cfg)[0] == 1.0
    assert R.total(SUPPORTED, SUPPORTED, 1.0, ["a"], ["a"], 0, format_ok=False, cfg=cfg)[0] == 0.0


def test_ablation_random_ignores_episode():
    cfg = R.ABLATIONS["random"]
    a = R.total(SUPPORTED, SUPPORTED, 1.0, ["a"], ["a"], 0, cfg=cfg, rng=random.Random(0))[0]
    b = R.total(REFUTED, SUPPORTED, 1.0, ["a"], [], 8, cfg=cfg, rng=random.Random(0))[0]
    assert a == b and 0.0 <= a <= 1.0


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        R.total(SUPPORTED, SUPPORTED, 1.0, [], [], 0, cfg=R.RewardConfig(mode="nope"))
