"""final_answer rule v3: a decisive answer must beat every NEI the verifier gave the full claim.

No index, GPU or API key is needed:

    python -m pytest verifier/tests/test_prompted_final.py -q
"""

import pytest

from agent import prompted
from agent.prompted import NEI, Episode, final_answer

CLAIM = {"id": 1, "claim": "The Mighty Ducks was produced by Disney.", "label": "REFUTES", "evidence": []}


def verdict(v, conf, evidence=()):
    return {"verdict": v, "evidence": list(evidence), "confidence": conf,
            "token_prob": {"SUPPORTED": 0.1, "REFUTED": 0.1, NEI: 0.8}}


def episode(*verifies, subclaim_verifies=(), subclaims=(), end="commit"):
    """An episode whose full claim was verified in order with `verifies` ((verdict, conf) pairs)."""
    ep = Episode(CLAIM)
    ep.subclaims = list(subclaims)
    for v, conf in verifies:
        res = verdict(v, conf, ["Page:0"] if v != NEI else [])
        ep.claim_verdict = res
        ep.log("verify", {"claim": CLAIM["claim"], "evidence": [], "subclaim": None}, res, 0.0)
    for j, (v, conf) in subclaim_verifies:
        res = verdict(v, conf, ["Page:1"])
        ep.sub_verdicts[j] = res
        ep.log("verify", {"claim": ep.subclaims[j], "evidence": [], "subclaim": j}, res, 0.0)
    ep.log(end, {}, {}, 0.0)
    ep.end = end
    return ep


def test_weaker_verdict_after_nei_returns_the_nei():
    fin = final_answer(episode((NEI, 0.64), ("REFUTED", 0.52)))
    assert fin == {"verdict": NEI, "evidence": [], "confidence": 0.64, "source": "nei_rule"}


def test_tie_with_earlier_nei_returns_the_nei():
    assert final_answer(episode((NEI, 0.7), ("SUPPORTED", 0.7)))["source"] == "nei_rule"


def test_firmer_verdict_after_nei_stands():
    fin = final_answer(episode((NEI, 0.6), ("SUPPORTED", 0.85)))
    assert (fin["verdict"], fin["source"], fin["evidence"]) == ("SUPPORTED", "claim", ["Page:0"])


def test_firmest_of_several_neis_is_the_bar():
    fin = final_answer(episode((NEI, 0.55), (NEI, 0.75), ("REFUTED", 0.7)))
    assert (fin["verdict"], fin["confidence"]) == (NEI, 0.75)


def test_no_earlier_nei_keeps_v2_answer():
    assert final_answer(episode(("REFUTED", 0.4)))["source"] == "claim"


def test_subclaim_answer_is_held_to_the_same_bar():
    ep = episode((NEI, 0.8), subclaims=["a", "b"], subclaim_verifies=[(0, ("REFUTED", 0.7))])
    assert final_answer(ep)["source"] == "claim"  # v2 already keeps the firmer full-claim NEI
    ep = episode((NEI, 0.6), ("SUPPORTED", 0.5), subclaims=["a", "b"],
                 subclaim_verifies=[(0, ("REFUTED", 0.55))])
    assert final_answer(ep)["source"] == "nei_rule"  # sub-claims beat 0.5 but not the 0.6 NEI


def test_subclaim_nei_does_not_count():
    ep = episode(("SUPPORTED", 0.6), subclaims=["a", "b"], subclaim_verifies=[(0, (NEI, 0.9))])
    assert final_answer(ep)["source"] == "claim"


def test_abstain_is_unchanged():
    assert final_answer(episode(("SUPPORTED", 0.9), end="abstain"))["source"] == "abstain"


def test_v2_rule_still_available(monkeypatch):
    monkeypatch.setattr(prompted, "FINAL_RULE", "v2")
    fin = final_answer(episode((NEI, 0.64), ("REFUTED", 0.52)))
    assert (fin["verdict"], fin["source"]) == ("REFUTED", "claim")


def test_row_records_the_rule():
    row = prompted.to_row(episode((NEI, 0.64), ("REFUTED", 0.52)), {"final_rule": prompted.FINAL_RULE})
    assert row["final"]["verdict"] == NEI and row["final"]["ended_by"] == "commit"
    assert row["run"]["final_rule"] == "v3"
