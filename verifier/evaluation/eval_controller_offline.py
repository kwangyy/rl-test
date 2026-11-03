"""Run agent.prompted without API calls: a decompose of a claim missing from the cache keeps the claim whole.

The local controller still decomposes with flash's cached splits (agent.prompted), and a claim
missing from the cache would call OpenRouter, which needs a key. SoC has none, and the frozen
holdout has no cached splits. The imitation controller decomposed 0 of 900 dev claims, so this
rarely matters; every fallback is counted and printed at the end.

    python -m verifier.evaluation.eval_controller_offline --tools follow --chunk 64 \
        --controller local:<imitation>,<grpo> --adapter <verifier> --claim-ids <ids.json>
"""

import runpy

import agent.decompose as decompose

_Decomposer = decompose.Decomposer
missing = []


class OfflineDecomposer(_Decomposer):
    def run(self, claim):
        if claim in self.cache:
            return {**self.cache[claim], "cached": True}
        missing.append(claim)
        return {"subclaims": [claim], "tokens_in": 0, "tokens_out": 0, "cached": False}


decompose.Decomposer = OfflineDecomposer  # agent.prompted imports it by name when it runs

if __name__ == "__main__":
    try:
        runpy.run_module("agent.prompted", run_name="__main__", alter_sys=True)
    finally:
        print(f"  decompose without a cached split (claim kept whole): {len(missing)}")
