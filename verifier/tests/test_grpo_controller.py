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


def test_spread_claim_ids_keeps_mixed_groups_once(tmp_path):
    import json
    from verifier.experiments._shared.grpo_controller import spread_claim_ids
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    rows_a = [{"id": 1, "group": 0, "fever": True}, {"id": 1, "group": 0, "fever": False},
              {"id": 2, "group": 1, "fever": True}, {"id": 2, "group": 1, "fever": True}]
    rows_b = [{"id": 3, "group": 0, "fever": False}, {"id": 3, "group": 0, "fever": True},
              {"id": 1, "group": 1, "fever": True}, {"id": 1, "group": 1, "fever": False}]
    a.write_text("".join(json.dumps(r) + "\n" for r in rows_a))
    b.write_text("".join(json.dumps(r) + "\n" for r in rows_b))
    assert spread_claim_ids([a, b]) == [1, 3]


def test_evidence_bonus_only_when_gold_reached_the_verifier():
    from verifier.experiments._shared.grpo_controller import delivered_gold
    r = row("SUPPORTED", "REFUTED", ["A:1"], n_ops=2)
    r["steps"].append({"action": "verify", "args": {"evidence": ["B:2", "A:1"]}})
    assert delivered_gold(r)
    assert abs(rollout_reward(r, 0.01, 0.5, 0.3) - (0.3 - 0.02)) < 1e-9
    assert abs(rollout_reward(r, 0.01, 0.5) + 0.02) < 1e-9  # default: FEVER-only, unchanged
    miss = row("SUPPORTED", "SUPPORTED", ["A:1"], gold_evidence=(("A:1", "C:4"),))
    miss["steps"].append({"action": "verify", "args": {"evidence": ["A:1"]}})
    assert not delivered_gold(miss)
    nei = row("NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO", gold_evidence=())
    nei["steps"].append({"action": "verify", "args": {"evidence": ["A:1"]}})
    assert not delivered_gold(nei)
