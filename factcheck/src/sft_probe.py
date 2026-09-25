"""SFT capacity probe: can the small model verify a claim when the gold
evidence is on the page in front of it?

Decides 2B vs 4B before any GPU time goes to GRPO. LoRA SFT on FEVER
train, with the pages of one gold evidence set guaranteed in context and
title-match pages as distractors (stage0's "oracle" retrieval). Target is
the gold verdict plus the gold evidence ids, in the exact JSON format
stage0 asks for. Evaluate with stage0.py --adapter on the same 900 dev
claims, so every number lands in the same table as the untrained model.

    python sft_probe.py --model Qwen/Qwen3.5-2B --n-train 6000
    python sft_probe.py --model Qwen/Qwen3.5-4B --n-train 6000 --qlora
    python stage0.py --model local --adapter ../data/sft_probe/Qwen_Qwen3.5-2B_n6000 --n 900 --retrieval oracle
    python stage0.py --model local --adapter ../data/sft_probe/Qwen_Qwen3.5-2B_n6000 --n 900 --retrieval titles

Smoke test (a few minutes):  python sft_probe.py --n-train 60 --max-steps 5

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

from build_bm25_index import DATA
from stage0 import (FEVER_LABEL, LOCAL_MODEL, SEED, SYSTEM, build_context, gold_sets,
                    render, sample_claims)

OUT_DIR = os.path.join(DATA, "..", "sft_probe")
NEI = "NOT_ENOUGH_INFO"


def build_examples(n, seed, chunk=200):
    """Prompt/completion pairs from FEVER train with gold pages in context.
    Non-NEI claims whose gold line falls past stage0's per-page sentence cut
    are dropped rather than trained with a wrong evidence list."""
    from fever_retrieval import FeverRetriever
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


def train(model_name, rows, out, qlora, max_steps, lr, rank):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(model_name)
    # Same template call as stage0.LocalModel, so training matches inference.
    def to_text(r):
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return {"prompt": prompt, "completion": r["completion"] + tok.eos_token}
    ds = Dataset.from_list([to_text(r) for r in rows])

    kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
    if qlora:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)

    cfg = SFTConfig(
        output_dir=out, seed=SEED,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        num_train_epochs=1, max_steps=max_steps if max_steps else -1,
        max_length=2048, bf16=True, gradient_checkpointing=True,
        completion_only_loss=True, logging_steps=10,
        save_strategy="no", report_to="none", dataset_num_proc=1,
    )
    lora = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
    trainer.model.print_trainable_parameters()
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    trainer.save_model(out)
    tok.save_pretrained(out)
    with open(os.path.join(out, "train_log.json"), "w", encoding="utf-8") as f:
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
    a = ap.parse_args()

    out = os.path.join(OUT_DIR, f"{a.model.replace('/', '_')}_n{a.n_train}{'_qlora' if a.qlora else ''}")
    os.makedirs(out, exist_ok=True)
    data_path = os.path.join(out, "train.jsonl")
    if os.path.exists(data_path):
        with open(data_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        print(f"reusing {len(rows)} examples from {data_path}")
    else:
        rows = build_examples(a.n_train, a.seed)
        with open(data_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    secs, peak_gb = train(a.model, rows, out, a.qlora, a.max_steps, a.lr, a.rank)
    meta = {"model": a.model, "n_train": len(rows), "qlora": a.qlora, "max_steps": a.max_steps,
            "lr": a.lr, "rank": a.rank, "seed": a.seed, "train_seconds": round(secs),
            "peak_gpu_gb": round(peak_gb, 2)}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    print(f"\nadapter saved to {out}\nnext: python stage0.py --model local --adapter {out} --n 900 --retrieval oracle")
