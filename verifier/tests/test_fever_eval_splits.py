"""Protect frozen evaluation membership, reproducibility and leakage boundaries."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.build_fever_eval_splits import audit_usage, build, normalize_claim, select_sets
from utils.fever_eval import (FROZEN_ROOT, LEGACY_IDS, assert_not_reserved,
                             load_manifest_claims, read_id_manifest, sha256,
                             validate_resume_ids)
from verifier.model import sample_claims


def dump(path, record):
    path.write_text(json.dumps(record), encoding="utf-8")


def reserved_fixture(directory):
    for name, cid in (("calibration-900-ids.json", 101), ("holdout-3000-ids.json", 102)):
        dump(directory / name, {"dataset": "FEVER", "split": "shared_task_dev",
                               "ids": [cid]})


class FrozenEvaluationTest(unittest.TestCase):
    def test_selection_excludes_prior_ids_text_duplicates_and_full_train_pool(self):
        train = [{"id": 1, "claim": "  Known TRAIN claim ", "label": "SUPPORTS"}]
        dev = [{"id": 10 + i, "claim": f"Unused claim {i}", "label": label}
               for i, label in enumerate(("SUPPORTS", "REFUTES", "NOT ENOUGH INFO") * 8)]
        dev.extend([
            {"id": 90, "claim": "known train CLAIM", "label": "SUPPORTS"},
            {"id": 91, "claim": "Previously inspected", "label": "REFUTES"},
            {"id": 92, "claim": " previously INSPECTED ", "label": "REFUTES"},
            {"id": 93, "claim": "Rolled out without an ID", "label": "NOT ENOUGH INFO"},
            {"id": 94, "claim": "unused claim 0", "label": "SUPPORTS"},
        ])
        audit = {"used_ids": ["91", "93"]}
        cal, holdout, stats = select_sets(train, dev, audit, calibration_size=3, holdout_size=6)
        chosen = cal + holdout
        self.assertFalse({r["id"] for r in chosen} & {90, 91, 92, 93, 94})
        self.assertEqual(len({normalize_claim(r["claim"]) for r in chosen}), 9)
        self.assertEqual(stats["excluded"], {"training_or_used_text_match": 2,
                                             "previously_used_id": 2,
                                             "duplicate_candidate_text": 1})
        self.assertEqual((cal, holdout, stats), select_sets(
            train, list(reversed(dev)), audit, calibration_size=3, holdout_size=6))
        self.assertEqual([r["label"] for r in cal].count("SUPPORTS"), 1)
        self.assertEqual([r["label"] for r in holdout].count("NOT ENOUGH INFO"), 2)

    def test_insufficient_unused_pool_fails(self):
        with self.assertRaisesRegex(ValueError, "not enough unused"):
            select_sets([], [], {"used_ids": []},
                        calibration_size=3, holdout_size=3)

    def test_reserved_ids_block_training_and_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            reserved_fixture(directory)
            for row in ({"id": 101}, {"id": "102"}, {"claim_id": 101},
                        {"id": 102, "final": {"verdict": "SUPPORTED"}}):
                with self.subTest(row=row), self.assertRaisesRegex(ValueError, "reserved"):
                    assert_not_reserved([row], directory=directory)
            assert_not_reserved([{"id": 999, "claim": "Another claim"}], directory=directory)

    def test_manifest_loader_preserves_ids_order_and_checks_raw_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, manifest = directory / "dev.jsonl", directory / "ids.json"
            rows = [{"id": 1, "claim": "One"}, {"id": 2, "claim": "Two"}]
            source.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            dump(manifest, {"split": "shared_task_dev", "ids": [2, 1],
                            "source_sha256": sha256(source)})
            self.assertEqual([r["id"] for r in load_manifest_claims(source, manifest)], [2, 1])
            source.write_text(source.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source hash"):
                load_manifest_claims(source, manifest)

    def test_invalid_and_missing_manifest_ids_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, manifest = directory / "dev.jsonl", directory / "ids.json"
            source.write_text('{"id":1,"claim":"One"}\n', encoding="utf-8")
            for ids in ([1, 1], [True], ["1"], []):
                dump(manifest, {"split": "shared_task_dev", "ids": ids})
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    read_id_manifest(manifest)
            dump(manifest, {"split": "shared_task_dev", "ids": [2]})
            with self.assertRaisesRegex(ValueError, "missing"):
                load_manifest_claims(source, manifest)

    def test_resume_rejects_mixed_sets_and_duplicate_predictions(self):
        validate_resume_ids([1, 2], [1, 2, 3])
        for actual in ([1, 4], [1, 1]):
            with self.subTest(actual=actual), self.assertRaisesRegex(ValueError, "existing output"):
                validate_resume_ids(actual, [1, 2, 3])

    def test_proposed_training_requires_claim_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "controller.jsonl"
            path.write_text('{"claim":"Claim without an ID"}\n')
            project = Path(__file__).resolve().parents[2]
            result = subprocess.run([sys.executable, "-m", "utils.build_fever_eval_splits",
                                     "--check-training", str(path)], cwd=project,
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must include a FEVER id", result.stderr)

    def test_existing_frozen_directory_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SystemExit, "must not be overwritten"):
                build(Path(tmp))

    def test_model_loader_uses_historical_ids_and_explicit_full_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = [{"id": 300000 + i, "claim": f"Claim {i}", "label": "SUPPORTS"} for i in range(900)]
            source = directory / "shared_task_dev.jsonl"
            source.write_text("\n".join(map(json.dumps, reversed(rows))), encoding="utf-8")
            legacy = directory / "legacy.json"
            dump(legacy, {"split": "shared_task_dev", "ids": [r["id"] for r in rows]})
            with patch("verifier.model.DATA", str(directory)), patch("verifier.model.LEGACY_IDS", legacy):
                self.assertEqual(sample_claims(900), rows)
                self.assertEqual(sample_claims(None, ids_file=legacy), rows)
                for n in (30, 0, -1):
                    self.assertEqual(sample_claims(n, ids_file=legacy), rows)
                with self.assertRaisesRegex(ValueError, "dev split"):
                    sample_claims(None, split="train", ids_file=legacy)

    def test_implicit_model_sampling_refuses_reserved_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = [{"id": i, "claim": str(i), "label": label}
                    for i, label in enumerate(("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"))]
            (directory / "shared_task_dev.jsonl").write_text("\n".join(map(json.dumps, rows)))
            with patch("verifier.model.DATA", str(directory)), \
                    patch("utils.fever_eval.reserved_ids", return_value={"1"}):
                with self.assertRaisesRegex(ValueError, "implicit FEVER dev sampling"):
                    sample_claims(3)

    def test_slurm_launcher_uses_explicit_size_and_records_manifest(self):
        # Fake srun/index files exercise the launcher without a model, GPU or cluster.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in (".venv/bin", "bin", "verifier/data/fever/bm25_index"):
                (root / relative).mkdir(parents=True)
            for relative in ("README.md", "verifier/data/fever/shared_task_dev.jsonl",
                             "verifier/data/fever/pages.sqlite", "verifier/data/fever/bm25_index/doc_ids.json"):
                (root / relative).touch()
            binary = root / "bin"
            (binary / "python").symlink_to(sys.executable)
            (root / ".venv/bin/activate").write_text(f'export PATH={shlex.quote(str(binary))}:"$PATH"\n')
            git_stub = binary / "git"
            git_stub.write_text("#!/bin/sh\necho fixture-revision\n")
            git_stub.chmod(0o755)
            srun = binary / "srun"
            srun.write_text(f"#!{sys.executable}\n" + '''import json, sys
from pathlib import Path
args = sys.argv
manifest = Path(args[args.index('--claim-ids') + 1])
output = Path(args[args.index('--output') + 1])
ids = json.loads(manifest.read_text())['ids']
rows = [{'id':i,'model':'fixture','gold':'SUPPORTED','pred':'SUPPORTED','correct':True} for i in ids]
output.write_text(''.join(json.dumps(row)+'\\n' for row in rows))
''')
            srun.chmod(0o755)
            ids_file = root / "ids.json"
            dump(ids_file, {"split": "shared_task_dev", "ids": [300001, 300002, 300003]})
            project = Path(__file__).resolve().parents[2]
            env = {k: v for k, v in os.environ.items() if k not in {
                "N_CLAIMS", "RESULTS_ROOT", "RESULT_DIR", "CHECKPOINTS", "MAX_NEW_TOKENS",
                "EVAL_CONDITION", "PROMPT_STYLE", "THINKING", "PYTHONPATH"}}
            env.update(SLURM_SUBMIT_DIR=str(root), SLURM_JOB_ID="1", RUN_NAME="fixture",
                       CLAIM_IDS=str(ids_file), PYTHONPATH=str(project), TMPDIR=str(root / "tmp"))
            script = project / "verifier/evaluation/run_eval.sh"
            result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((root / "verifier/evaluation/results/baseline-direct-cap160/fixture/manifest.json").read_text())
            self.assertEqual(manifest["n_claims"], 3)
            self.assertEqual(manifest["claim_ids_sha256"], sha256(ids_file))
            for index, n in enumerate(("900", "invalid")):
                env.update(N_CLAIMS=n, RUN_NAME=f"fixture-ignore-n-{index}")
                result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                run_manifest = root / f"verifier/evaluation/results/baseline-direct-cap160/fixture-ignore-n-{index}/manifest.json"
                self.assertEqual(json.loads(run_manifest.read_text())["n_claims"], 3)

    def test_audit_includes_deleted_history_and_claims_without_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = Path(tmp) / "current.jsonl"
            legacy = Path(tmp) / "legacy.json"
            current.write_text('{"id":7,"claim":"Current"}\n', encoding="utf-8")
            dump(legacy, {"ids": [9]})

            def fake_git(*args):
                if args[0] == "rev-list":
                    return b"abc old/deleted-predictions.jsonl\n"
                if args[0] == "cat-file":
                    return b'{"id":8,"claim":"Deleted"}\n{"claim":"No ID"}\n'
                return b"revision\n"

            # Path provenance does not depend on the fixture's location.
            with patch("utils.build_fever_eval_splits.ROOT", Path(tmp)), \
                    patch("utils.build_fever_eval_splits.LEGACY_IDS", legacy), \
                    patch("utils.build_fever_eval_splits.current_artifacts", return_value=[current]), \
                    patch("utils.build_fever_eval_splits.git", side_effect=fake_git):
                audit = audit_usage({normalize_claim("No ID"): {"10"}})
            self.assertEqual(audit["used_ids"], ["7", "8", "9", "10"])
            self.assertNotIn("claim_text_sha256", audit)
            self.assertEqual({s["kind"] for s in audit["sources"]}, {"working_tree", "git_blob"})

    def test_committed_sets_are_disjoint_and_preserve_historical_benchmark(self):
        manifest = json.loads((FROZEN_ROOT / "manifest.json").read_text())
        legacy = read_id_manifest(LEGACY_IDS)
        self.assertEqual(sha256(LEGACY_IDS), manifest["legacy_development"]["sha256"])
        dev = read_id_manifest(FROZEN_ROOT / "development-used-ids.json")
        cal = read_id_manifest(FROZEN_ROOT / "calibration-900-ids.json")
        holdout = read_id_manifest(FROZEN_ROOT / "holdout-3000-ids.json")
        self.assertTrue(set(legacy["ids"]) <= set(dev["ids"]))
        self.assertFalse(set(dev["ids"]) & (set(cal["ids"]) | set(holdout["ids"])))
        self.assertFalse(set(cal["ids"]) & set(holdout["ids"]))
        self.assertNotIn("claim_text_sha256", cal)
        self.assertNotIn("claim_text_sha256", holdout)
        self.assertEqual(set(cal["label_counts"].values()), {300})
        self.assertEqual(set(holdout["label_counts"].values()), {1000})


if __name__ == "__main__":
    unittest.main()
