"""Shortcut controls must change context without changing claims or FEVER labels."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.build_shortcut_set import (CONDITIONS, DEFAULT_SET, DEFAULT_TRACE, NEI, context,
                                      load_examples, prepare)
from utils.fever_eval import reserved_ids
from verifier.evaluation.shortcut_test import run


class ShortcutTest(unittest.TestCase):
    def test_frozen_selection_and_donors(self):
        examples = load_examples(DEFAULT_TRACE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "set.json"
            spec = prepare(DEFAULT_TRACE, path)
            other = json.loads(Path(DEFAULT_SET).read_text(encoding="utf-8"))
            self.assertEqual(spec, other)
            self.assertEqual(len(spec["ids"]), 300)
            self.assertFalse(set(map(str, spec["ids"])) & reserved_ids())
            self.assertEqual(sorted(spec["ids"]), sorted(spec["shuffled_donor_ids"]))
            for gold in ("SUPPORTED", "REFUTED", NEI):
                self.assertEqual(sum(examples[cid]["gold"] == gold for cid in spec["ids"]), 100)
            for cid, donor in zip(spec["ids"], spec["shuffled_donor_ids"]):
                e = examples[cid]
                self.assertNotEqual(cid, donor)
                normal_pages = {p for _, p, _, _ in e["shown"]}
                donor_pages = {p for _, p, _, _ in examples[donor]["shown"]}
                self.assertFalse(normal_pages & donor_pages)
                if e["gold"] != NEI:
                    self.assertTrue(context(e, e["shown"])["gold_in_context"])
                    self.assertFalse(context(e, examples[donor]["shown"])["gold_page_retrieved"])
            with self.assertRaises(FileExistsError):
                prepare(DEFAULT_TRACE, path)

    def test_paired_run_preserves_labels_and_scores_abstention_separately(self):
        examples = load_examples(DEFAULT_TRACE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "set.json"
            spec = prepare(DEFAULT_TRACE, path, n=30)
            output = Path(directory) / "run1"
            with patch("verifier.evaluation.shortcut_test.LocalModel") as factory:
                model = factory.return_value
                model.name, model.batch_size = "test", 4
                model.torch.cuda.get_device_name.return_value = "mock"
                generations = []
                for condition in CONDITIONS:
                    batch = []
                    for cid in spec["ids"]:
                        pred = examples[cid]["gold"] if condition == "normal" else NEI
                        batch.append({"text": json.dumps({"verdict": pred,
                                      "evidence": [] if pred == NEI else ["E1"],
                                      "confidence": 0.9 if condition == "normal" else 0.8}),
                                      "tokens": [], "provider": "mock", "in_tokens": 10, "out_tokens": 10})
                    generations.append(batch)
                model.run.side_effect = generations
                metrics = run(path, output)
                prompts = [call.args[0] for call in model.run.call_args_list]
            for condition, condition_prompts in zip(CONDITIONS, prompts):
                rows = [json.loads(line) for line in (output / f"{condition}.jsonl").read_text().splitlines()]
                self.assertEqual([r["id"] for r in rows], spec["ids"])
                for row, prompt in zip(rows, condition_prompts):
                    e = examples[row["id"]]
                    self.assertEqual(row["claim"], e["claim"]["claim"])
                    self.assertEqual(row["gold"], e["gold"])
                    if condition == "empty":
                        self.assertEqual(prompt, f"Claim: {row['claim']}\n\nEvidence:\n")
                        self.assertEqual(row["n_shown"], 0)
                    if condition != "normal" and row["gold"] != NEI:
                        self.assertFalse(row["gold_in_context"])
            for condition in CONDITIONS[1:]:
                paired = metrics["paired"][condition]
                self.assertEqual(paired["normal_correct_non_nei_n"], 20)
                self.assertEqual(paired["normal_correct_non_nei_to_nei_rate"], 1.0)
                self.assertAlmostEqual(paired["accuracy_delta"], -2 / 3)
                self.assertAlmostEqual(paired["confidence_delta"]["stated_conf"], -0.1)
            with patch("verifier.evaluation.shortcut_test.LocalModel") as factory:
                with self.assertRaises(FileExistsError):
                    run(path, output)
                factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
