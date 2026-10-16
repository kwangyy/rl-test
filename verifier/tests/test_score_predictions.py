"""Metric contract for evaluating the selected verdict's correctness."""

import json
import tempfile
import unittest
from pathlib import Path

from verifier.evaluation.score_predictions import calibration, read_rows, summarize, verdict_harm


def row(claim_id, correct, confidence, gold="SUPPORTED", in_context=True):
    return {"id": claim_id, "model": "test", "retrieval": "oracle", "gold": gold,
            "pred": gold if correct else "REFUTED", "correct": correct,
            "fever_score": correct, "evidence_f1": 1.0 if correct else 0.0,
            "stated_conf": confidence, "token_conf": confidence,
            "gold_in_context": in_context, "in_tokens": 10, "out_tokens": 2}


class SavedRunMetricsTest(unittest.TestCase):
    def test_verdict_harm_cost_matrix_and_unparseable_verdicts(self):
        # Columns: predicted SUPPORTED, REFUTED, NOT_ENOUGH_INFO, invalid.
        costs = {
            "SUPPORTED": (0.0, 1.0, 1.0, 1.0),
            "REFUTED": (1.5, 0.0, 1.0, 1.0),
            "NOT_ENOUGH_INFO": (1.75, 1.0, 0.0, 1.0),
        }
        predictions = ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO", None)
        for gold, expected in costs.items():
            for pred, cost in zip(predictions, expected):
                with self.subTest(gold=gold, pred=pred):
                    self.assertEqual(verdict_harm(gold, pred), cost)

    def test_brier_ece_and_tied_risk_coverage(self):
        rows = [row(1, True, 1.0), row(2, False, 1.0),
                row(3, True, 0.5), row(4, False, 0.5)]
        result = calibration(rows, "stated_conf", bins=2)
        self.assertAlmostEqual(result["brier"], 0.375)
        self.assertAlmostEqual(result["ece"], 0.25)
        self.assertEqual(result["risk_at_coverage"]["0.5"], {
            "coverage": 0.5, "risk": 0.5, "threshold": 1.0})
        self.assertEqual(result["risk_at_coverage"]["0.8"]["coverage"], 1.0)

    def test_nei_precision_uses_predicted_nei_as_denominator(self):
        cases = [("NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO"),
                 ("SUPPORTED", "NOT_ENOUGH_INFO"), ("REFUTED", "NOT_ENOUGH_INFO"),
                 ("NOT_ENOUGH_INFO", "REFUTED"), ("SUPPORTED", None)]
        rows = [{**row(i, gold == pred, None, gold=gold), "pred": pred}
                for i, (gold, pred) in enumerate(cases)]
        result = summarize(rows)
        self.assertEqual(result["all"]["nei_precision"], 1 / 3)
        self.assertEqual(result["all"]["recall"]["NOT_ENOUGH_INFO"], 0.5)
        self.assertEqual(result["strata"]["gold_in_context"]["nei_precision"], 0.0)
        self.assertEqual(result["strata"]["gold_nei"]["nei_precision"], 1.0)

    def test_missing_confidence_and_context_strata(self):
        rows = [row(1, True, 0.9), row(2, False, None, in_context=False),
                row(3, True, 0.7, gold="NOT_ENOUGH_INFO", in_context=None)]
        rows[1].update(gold="REFUTED", pred="SUPPORTED")
        result = summarize(rows)
        self.assertAlmostEqual(result["all"]["calibration"]["stated_conf"]["coverage"], 2 / 3)
        self.assertEqual(result["strata"]["gold_in_context"]["n"], 1)
        self.assertEqual(result["strata"]["gold_not_in_context"]["n"], 1)
        self.assertEqual(result["strata"]["gold_nei"]["n"], 1)
        self.assertEqual(result["all"]["harm_weighted_error"], 0.5)
        self.assertEqual(result["strata"]["gold_not_in_context"]["harm_weighted_error"], 1.5)

    def test_format_rate_counts_invalid_json(self):
        rows = [row(1, True, 0.9), row(2, False, None)]
        rows[0]["format_ok"] = True
        rows[1].update(pred=None, format_ok=False)
        result = summarize(rows)
        self.assertEqual(result["all"]["format_rate"], 0.5)
        self.assertEqual(result["all"]["harm_weighted_error"], 0.5)
        self.assertIsNone(result["strata"]["gold_nei"]["harm_weighted_error"])
        self.assertIsNone(result["all"]["nei_precision"])
        self.assertIsNone(result["strata"]["gold_nei"]["nei_precision"])

    def test_cot_sft_reports_reasoning_without_stated_confidence(self):
        item = row(1, True, None)
        item.update(prompt_style="cot_sft", reasoning_present=True,
                    reasoning_citations_valid=True, format_ok=True)
        result = summarize([item])
        self.assertEqual(result["reasoning_present_rate"], 1.0)
        self.assertEqual(result["reasoning_valid_citations_rate"], 1.0)
        self.assertEqual(result["all"]["calibration"]["stated_conf"]["coverage"], 0.0)

    def test_cot_brief_reports_reasoning_metrics(self):
        item = row(1, True, 0.9)
        item.update(prompt_style="cot_brief", reasoning_present=True,
                    reasoning_citations_valid=True, format_ok=True)
        result = summarize([item])
        self.assertEqual(result["reasoning_present_rate"], 1.0)
        self.assertEqual(result["reasoning_valid_citations_rate"], 1.0)

    def test_duplicate_ids_rejected(self):
        item = row(1, True, 0.9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicates.jsonl"
            path.write_text(json.dumps(item) + "\n" + json.dumps(item) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate claim id"):
                read_rows(path)


if __name__ == "__main__":
    unittest.main()
