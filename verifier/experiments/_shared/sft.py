"""SFT capacity probe: can the small model verify a claim when the gold
evidence is on the page in front of it?

Decides 2B vs 4B before any GPU time goes to GRPO. LoRA SFT on FEVER
train, with the pages of one gold evidence set guaranteed in context and
title-match pages as distractors (the verifier evaluator's "oracle" retrieval). Target is
the gold verdict plus the gold evidence ids, in the exact JSON format
eval_verifier asks for. Evaluate with eval_verifier.py --adapter on the same 900 dev
claims, so every number lands in the same table as the untrained model.

    python verifier/experiments/_shared/sft.py --model Qwen/Qwen3.5-2B --n-train 6000 --run-name run2
    python -m verifier.evaluation.eval_verifier --model local --adapter verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run2/artifacts/checkpoints --n 900 --retrieval titles

The stated confidence is 1.0 in every target, so after SFT it carries no
information by construction. Read accuracy, FEVER score, evidence F1 and
token-probability ECE. Calibration is stage 1b's job, not this probe's.
See docs/sft-probe.md for the decision rule.
"""

import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from retrieval.build_bm25_index import DATA
from utils.experiment_layout import artifacts_dir
from verifier.model import (FEVER_LABEL, LOCAL_MODEL, SEED, SYSTEM, build_context, gold_sets,
                    render, sample_claims)

OUT_DIR = os.path.join(DATA, "..", "train", "sft")
NEI = "NOT_ENOUGH_INFO"
MAX_LEN = 4096  # prompt + completion; ~13% of untrained baseline prompts exceed 2048 tokens


def build_examples(n, seed, chunk=200):
    """Prompt/completion pairs from FEVER train with gold pages in context.
    Non-NEI claims whose gold line falls past the verifier evaluator's per-page sentence cut
    are dropped rather than trained with a wrong evidence list."""
    from retrieval.fever_retrieval import FeverRetriever
    retriever = FeverRetriever()
    claims = sample_claims(n, seed=seed, split="train")
    rows, dropped = [], 0
    t0 = time.time()
    for i in range(0, len(claims), chunk):
        batch = claims[i:i + chunk]
        for c, x in zip(batch, build_context(retriever, batch, "oracle")):
            gold = FEVER_LABEL[c["label"]]
            eids = []
            if gold != NEI:
                if not x["gold_in_context"]:
                    dropped += 1
                    continue
                by_key = {(p, line): eid for eid, p, line, _ in x["shown"]}
                best = next(s for s in gold_sets(c) if s <= set(by_key))
                eids = sorted((by_key[k] for k in best), key=lambda e: int(e[1:]))
            target = json.dumps({"verdict": gold, "evidence": eids, "confidence": 1.0})
            rows.append({"id": c["id"], "gold": gold, "prompt": render(c["claim"], x["shown"]), "completion": target})
        print(f"  {i + len(batch)}/{len(claims)} claims  {time.time() - t0:.0f}s", flush=True)
    print(f"{len(rows)} examples, {dropped} dropped (gold line not shown)")
    return rows


def train(model_name, rows, out, qlora, max_steps, lr, rank, seed, system_prompt=SYSTEM,
          init_adapter=None):
    from utils.fever_eval import assert_not_reserved
    assert_not_reserved(rows, "SFT training")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(model_name)
    # Same template call as eval_verifier.LocalModel, so training matches inference.
    def to_text(r):
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return {"prompt": prompt, "completion": r["completion"] + tok.eos_token}
    ds = Dataset.from_list([to_text(r) for r in rows])
    n_long = sum(len(tok(r["prompt"] + r["completion"]).input_ids) > MAX_LEN for r in ds)
    print(f"{n_long}/{len(ds)} examples over {MAX_LEN} tokens (completion truncated)")

    kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
    if qlora:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    if init_adapter:
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)

    # transformers 5 folded warmup_ratio into warmup_steps (a float < 1 is a ratio)
    warmup = ({"warmup_ratio": 0.03} if "warmup_ratio" in SFTConfig.__dataclass_fields__
              else {"warmup_steps": 0.03})
    cfg = SFTConfig(
        output_dir=out, seed=seed,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=lr, lr_scheduler_type="cosine", **warmup,
        num_train_epochs=1, max_steps=max_steps if max_steps else -1,
        max_length=MAX_LEN, bf16=True, gradient_checkpointing=True,
        completion_only_loss=True, logging_steps=10,
        save_strategy="no", report_to="none", dataset_num_proc=1,
    )
    peft_args = {} if init_adapter else {"peft_config": LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM")}
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, **peft_args)
    trainer.model.print_trainable_parameters()
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    trainer.save_model(out)
    tok.save_pretrained(out)
    log_dir = os.path.join(os.path.dirname(out), "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    return secs, torch.cuda.max_memory_allocated() / 2**30


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LOCAL_MODEL)
    ap.add_argument("--n-train", type=int, default=6000, help="claims sampled from FEVER train, balanced by label")
    ap.add_argument("--qlora", action="store_true", help="4-bit base weights (use for 4B on 16GB)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early; for smoke tests")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--run-name", default=None, help="repeat label under verifier/experiments/<configuration>/runs/ (default run1 for seed 0)")
    ap.add_argument("--output-root", default=None, help="experiments directory for run artifacts (default: verifier/experiments)")
    ap.add_argument("--subset-of", default=None,
                    help="SFT data dir: train on the first n-train of its examples after a seeded shuffle, "
                         "the same claims grpo_verifier.load_rows picks (a matched SFT-vs-GRPO run)")
    ap.add_argument("--data", default=None,
                    help="prepared SFT data dir containing exactly n-train labeled_prompts.jsonl rows")
    ap.add_argument("--init-adapter", default=None,
                    help="resume SFT from an existing LoRA adapter (continue training its weights)")
    ap.add_argument("--experiment-name", default=None,
                    help="preserve a historical experiment configuration name when data storage is renamed")
    ap.add_argument("--config-name", default=None, help="directory name for this training configuration")
    a = ap.parse_args()
    if a.data and a.subset_of:
        ap.error("--data and --subset-of cannot be combined")
    if a.init_adapter:
        adapter = Path(a.init_adapter)
        if not (adapter / "adapter_config.json").is_file() or not any(
            (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            ap.error(f"missing adapter config or weights in {adapter}")
        if a.qlora:
            ap.error("--init-adapter with --qlora is not supported")

    tag = f"_from_{os.path.basename(os.path.normpath(a.subset_of))}" if a.subset_of else ""
    name = a.experiment_name or f"{a.model.replace('/', '_')}_n{a.n_train}{'_qlora' if a.qlora else ''}{tag}"
    if a.data:
        data_path = os.path.join(a.data, "labeled_prompts.jsonl")
        if not os.path.isfile(data_path):
            ap.error(f"missing prepared SFT data: {data_path}")
    else:
        if a.subset_of:
            dataset_name = f"fever-oracle-matched-grpo-{a.n_train}"
        elif a.n_train == 60:
            dataset_name = "fever-oracle-smoke-60"
        elif a.n_train == 6000:
            dataset_name = "fever-oracle-full"
        else:
            dataset_name = f"fever-oracle-{a.n_train}"
        data_out = os.path.join(OUT_DIR, dataset_name)
        os.makedirs(data_out, exist_ok=True)
        data_path = os.path.join(data_out, "labeled_prompts.jsonl")
    if a.subset_of and not os.path.exists(data_path):
        import random
        with open(os.path.join(a.subset_of, "labeled_prompts.jsonl"), encoding="utf-8") as f:
            src = [json.loads(line) for line in f]
        random.Random(a.seed).shuffle(src)
        with open(data_path, "w", encoding="utf-8") as f:
            for r in src[:a.n_train]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if os.path.exists(data_path):
        with open(data_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        print(f"reusing {len(rows)} examples from {data_path}")
        if a.data and (len(rows) != a.n_train or len({str(r['id']) for r in rows}) != len(rows)):
            ap.error(f"prepared SFT data must contain {a.n_train} unique claim IDs")
    else:
        rows = build_examples(a.n_train, a.seed)
        with open(data_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    artifacts = artifacts_dir("sft", name, a.seed, a.run_name, a.output_root, a.config_name)
    if artifacts.exists():
        raise SystemExit(f"run already exists: {artifacts}; use --run-name for a new run")
    checkpoint_dir = artifacts / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    secs, peak_gb = train(a.model, rows, str(checkpoint_dir), a.qlora, a.max_steps, a.lr, a.rank,
                          a.seed, init_adapter=a.init_adapter)
    meta = {"model": a.model, "n_train": len(rows), "qlora": a.qlora, "max_steps": a.max_steps,
            "lr": a.lr, "rank": a.rank, "seed": a.seed, "subset_of": a.subset_of, "train_seconds": round(secs),
            "data": a.data, "init_adapter": a.init_adapter, "peak_gpu_gb": round(peak_gb, 2)}
    with open(artifacts / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    print(f"\nadapter saved to {checkpoint_dir}\nnext: python -m verifier.evaluation.eval_verifier --model local --adapter {checkpoint_dir} --n 900 --retrieval oracle")
