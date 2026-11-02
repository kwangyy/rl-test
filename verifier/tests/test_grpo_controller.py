"""Reward, advantage and spread logic of the controller GRPO trainer (no GPU)."""

from verifier.experiments._shared.grpo_controller import (fever_ok, fever_spread, group_advantages,
                                                         rollout_reward, spread_summary)


def row(gold, verdict, evidence=(), gold_evidence=(("A:1",),), n_ops=3, valid=(True, True, True)):
    return {"gold": gold, "gold_evidence": [list(s) for s in gold_evidence],
            "final": {"verdict": verdict, "evidence": list(evidence)},
            "cost": {"n_ops": n_ops},
            "steps": [{"action": "x", "controller": {"valid": v}} for v in valid]}


def test_fever_needs_label_and_a_full_gold_set():
    assert fever_ok(row("SUPPORTED", "SUPPORTED", ["A:1", "B:2"]))
    assert not fever_ok(row("SUPPORTED", "SUPPORTED", ["B:2"]))
    assert not fever_ok(row("SUPPORTED", "REFUTED", ["A:1"]))
    assert fever_ok(row("NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO", [], gold_evidence=()))
    assert not fever_ok(row("SUPPORTED", None, ["A:1"]))
    two = row("REFUTED", "REFUTED", ["A:1"], gold_evidence=(("A:1", "C:4"),))
    assert not fever_ok(two)


def test_reward_charges_ops_and_invalid_replies():
    r = row("SUPPORTED", "SUPPORTED", ["A:1"], n_ops=4, valid=(True, False, True))
    assert abs(rollout_reward(r, 0.01, 0.5) - (1 - 0.04 - 0.5)) < 1e-9
    wrong = row("SUPPORTED", "REFUTED", ["A:1"], n_ops=2)
    assert abs(rollout_reward(wrong, 0.01, 0.5) + 0.02) < 1e-9


def test_group_advantages_zero_for_flat_groups_and_centred_otherwise():
    adv = group_advantages([1.0, 1.0, 0.0, 1.0], 2)
    assert adv[:2] == [0.0, 0.0]
    assert abs(adv[2] + adv[3]) < 1e-9 and adv[3] > 0
    raw = group_advantages([0.0, 1.0], 2, scale_std=False)
    assert raw == [-0.5, 0.5]


def test_spread_counts():
    assert spread_summary([1, 1, 0, 1, 0.5, 0.5], 2)["claims_with_spread"] == 1
    rows = [row("SUPPORTED", "SUPPORTED", ["A:1"]), row("SUPPORTED", "REFUTED"),
            row("SUPPORTED", "SUPPORTED", ["A:1"]), row("SUPPORTED", "SUPPORTED", ["A:1"])]
    assert fever_spread(rows, 2) == 1
