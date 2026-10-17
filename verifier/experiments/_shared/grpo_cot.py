"""Native-thinking GRPO from SFT, using verdict/evidence reward without calibration."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.experiment_layout import artifacts_dir
from utils.fever_eval import assert_not_reserved
from core import reward as R
from verifier.model import LABELS, LOCAL_MODEL, MAX_CITED, NATIVE_THINKING_SYSTEM, SEED, parse

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "verifier/data/train/sft/fever-oracle-matched-grpo-1200"


def completion_text(completion):
    return completion[-1]["content"] if isinstance(completion, list) else completion


def score_one(completion, gold, gold_ids):
    """Old shaped reward on final JSON, excluding only confidence calibration."""
    obj, ok = parse(completion_text(completion))
    if not all(ok[key] for key in ("json", "verdict", "evidence")):
        return -2.0, None
    return R.total(gold, obj["verdict"], 0.0, gold_ids, obj["evidence"][:MAX_CITED],
                   n_ops=0, cfg=R.RewardConfig(calibration=False))


def make_reward_funcs():
    """Only the first reward is optimized; the others are logged diagnostics."""
    cache = {}

    def scored(completions, gold, gold_ids):
        key = id(completions)
        if key not in cache:
            cache.clear()
            cache[key] = [score_one(c, label, ids)
                          for c, label, ids in zip(completions, gold, gold_ids)]
        return cache[key]

    def verifier_reward(completions, gold, gold_ids, **_):
        return [reward for reward, _ in scored(completions, gold, gold_ids)]

    def correctness(completions, gold, gold_ids, **_):
        return [parts["correctness"] if parts else 0.0 for _, parts in scored(completions, gold, gold_ids)]

    def evidence(completions, gold, gold_ids, **_):
        return [parts["evidence"] if parts else 0.0 for _, parts in scored(completions, gold, gold_ids)]

    def final_present(completions, gold, gold_ids, **_):
        return [float(parts is not None) for _, parts in scored(completions, gold, gold_ids)]

    return [verifier_reward, correctness, evidence, final_present, thinking_present]


def thinking_present(completions, **_):
    """Zero-weight diagnostic: a nonempty generated thinking section closed."""
    out = []
    for c in completions:
        text = completion_text(c)
        prefix, separator, _ = text.partition("</think>") if isinstance(text, str) else ("", "", "")
        out.append(float(bool(separator and prefix.replace("<think>", "").strip())))
    return out


def load_train_rows(data_dir, n_train):
    path = Path(data_dir) / "labeled_prompts.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert_not_reserved(rows, "native-thinking GRPO training")
    ids = [str(row["id"]) for row in rows]
    dev_ids = set(map(str, json.loads((ROOT / "verifier/data/eval/fever-dev-900-ids.json").read_text())["ids"]))
    if len(rows) != 1200 or len(set(ids)) != 1200 or set(ids) & dev_ids:
        raise ValueError("expected the fixed 1,200 unique train claims with no frozen dev overlap")
    if n_train < 1 or n_train > len(rows) or any(row["gold"] not in LABELS for row in rows):
        raise ValueError("invalid n_train or gold label")
    random.Random(SEED).shuffle(rows)
    return [{"prompt": [{"role": "system", "content": NATIVE_THINKING_SYSTEM},
                        {"role": "user", "content": row["prompt"]}],
             "gold": row["gold"],
             "gold_ids": json.loads(row["completion"])["evidence"]} for row in rows[:n_train]]


def main(config_name, n_train, default_steps):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--sft", required=True, help="completed Qwen3.5-2B SFT adapter")
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root")
    parser.add_argument("--max-steps", type=int, default=default_steps)
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps must be nonnegative")
    checkpoint = Path(args.sft)
    if not (checkpoint / "adapter_config.json").is_file() or not any(
        (checkpoint / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        parser.error(f"incomplete SFT adapter: {checkpoint}")
    rows = load_train_rows(args.data, n_train)
    artifacts = artifacts_dir("grpo", config_name, args.seed, args.run_name,
                              args.output_root, config_name=config_name)
    if artifacts.exists():
        raise SystemExit(f"run already exists: {artifacts}; choose a new run name")

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    base = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, str(checkpoint), is_trainable=True)
    model.print_trainable_parameters()
    out = artifacts / "checkpoints"
    out.mkdir(parents=True)
    config = GRPOConfig(
        output_dir=str(out), seed=args.seed, num_generations=8,
        per_device_train_batch_size=8, gradient_accumulation_steps=2,
        max_completion_length=1024, temperature=1.0, beta=0.0,
        chat_template_kwargs={"enable_thinking": True},
        reward_weights=[1.0, 0.0, 0.0, 0.0, 0.0],
        learning_rate=2e-5, lr_scheduler_type="constant_with_warmup", warmup_steps=10,
        num_train_epochs=1, max_steps=args.max_steps or -1,
        bf16=True, gradient_checkpointing=True,
        logging_steps=1, log_completions=False,
        save_strategy="steps", save_steps=50, save_total_limit=4,
        report_to="none",
    )
    trainer = GRPOTrainer(model=model, args=config, train_dataset=Dataset.from_list(rows),
                          reward_funcs=make_reward_funcs(),
                          processing_class=tokenizer)
    started = time.time()
    trainer.train()
    seconds = time.time() - started
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    logs = artifacts / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1) + "\n")
    meta = {
        "experiment": config_name, "model": LOCAL_MODEL, "sft": str(checkpoint),
        "data": str(Path(args.data)), "n_train": n_train, "seed": args.seed,
        "reward": "verdict_harm_plus_evidence_no_calibration", "thinking_mode": True,
        "diagnostics": ["correctness", "evidence", "final_present", "thinking_present"],
        "num_generations": 8, "max_completion_length": 1024, "beta": 0.0,
        "max_steps": args.max_steps, "steps": trainer.state.global_step,
        "train_seconds": round(seconds),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    (artifacts / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
