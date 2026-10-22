"""Run and score the paired retrieval-use diagnostic from a prepared shortcut set."""

import argparse
import json
from pathlib import Path
import subprocess

from utils.build_shortcut_set import CONDITIONS, DEFAULT_SET, NEI, context, load_examples
from utils.fever_eval import LEGACY_IDS, read_id_manifest, sha256
from verifier.evaluation.score_predictions import CONFIDENCES, mean, summarize
from verifier.model import LOCAL_MODEL, SEED, LocalModel, render, score


def paired_metrics(normal, altered):
    valid = [(a, b) for a, b in zip(normal, altered) if a["pred"] and b["pred"]]
    decisive = [(a, b) for a, b in zip(normal, altered) if a["correct"] and a["gold"] != NEI]
    return {"n": len(normal), "valid_pairs": len(valid),
            "verdict_change_rate_valid_pairs": mean([int(a["pred"] != b["pred"]) for a, b in valid]),
            "normal_correct_non_nei_n": len(decisive),
            "normal_correct_non_nei_to_nei_rate": mean([int(b["pred"] == NEI) for _, b in decisive]),
            "accuracy_delta": mean([int(b["correct"]) - int(a["correct"])
                                    for a, b in zip(normal, altered)]),
            "confidence_delta": {key: mean([b[key] - a[key] for a, b in zip(normal, altered)
                                            if a[key] is not None and b[key] is not None])
                                 for key in CONFIDENCES}}


def run(dataset, output_dir, model_name=LOCAL_MODEL, adapter=None, max_new_tokens=1024):
    if max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    spec = read_id_manifest(dataset)
    if (spec.get("purpose") != "shortcut-development"
            or sha256(spec["source_trace"]) != spec["source_trace_sha256"]
            or sha256(LEGACY_IDS) != spec["development_ids_sha256"]):
        raise ValueError("shortcut source trace differs from the prepared set")
    examples = load_examples(spec["source_trace"])
    ids, donors = spec["ids"], spec["shuffled_donor_ids"]
    if len(donors) != len(ids) or set(donors) != set(ids) or any(a == b for a, b in zip(ids, donors)):
        raise ValueError("shuffled donors must be a permutation with no self matches")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"choose a new output directory: {output_dir}")
    model = LocalModel(model_name, adapter=adapter, max_new_tokens=max_new_tokens)
    output_dir.mkdir(parents=True)
    (output_dir / "logs").mkdir()
    manifest = {"dataset": str(dataset), "dataset_sha256": sha256(dataset), "seed": SEED,
                "model": model.name, "adapter": adapter, "prompt_style": "direct",
                "max_new_tokens": max_new_tokens, "batch_size": model.batch_size,
                "gpu": model.torch.cuda.get_device_name(),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    runs = {}
    for condition in CONDITIONS:
        showns = [examples[donor if condition == "shuffled" else cid]["shown"]
                  if condition != "empty" else [] for cid, donor in zip(ids, donors)]
        gens = model.run([render(examples[cid]["claim"]["claim"], shown) for cid, shown in zip(ids, showns)])
        if len(gens) != len(ids):
            raise ValueError("model returned an incomplete condition")
        rows = []
        for cid, donor, shown, gen in zip(ids, donors, showns, gens):
            e = examples[cid]
            rows.append({"model": model.name, "retrieval": "archived-fixed", "prompt_style": "direct",
                         "shortcut_condition": condition, "context_source_id": donor if condition == "shuffled" else cid,
                         **score(e["claim"], context(e, shown), gen)})
        with (output_dir / f"{condition}.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        runs[condition] = rows
        message = f"{condition}: {len(rows)} predictions saved"
        print(message, flush=True)
        with (output_dir / "logs/run.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
    metrics = {"conditions": {name: summarize(rows) for name, rows in runs.items()},
               "paired": {name: paired_metrics(runs["normal"], runs[name]) for name in CONDITIONS[1:]}}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_SET)
    parser.add_argument("--output-dir", required=True, help="new results/<condition>/<run> directory")
    parser.add_argument("--model", default=LOCAL_MODEL)
    parser.add_argument("--adapter", help="LoRA checkpoint dir(s); omitted for the base model")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    run(args.dataset, args.output_dir, args.model, args.adapter, args.max_new_tokens)


if __name__ == "__main__":
    main()
