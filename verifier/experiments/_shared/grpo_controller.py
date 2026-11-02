"""Multi-turn GRPO of the agent loop's controller, with the verifier frozen.

Handover: verifier/docs/controller-grpo-handover.md, section 6. The policy is the
imitation controller (a Qwen3.5-2B LoRA trained on flash's follow-preset decisions);
the verifier (`sft-direct-from-base-qwen3p5-2b-5895-samples/run1`) stays greedy and
frozen. Each claim is played G times in the prompted loop (agent/prompted.py, follow
preset, at most 8 actions, final_answer rule v3) with sampled controller decisions.

Reward per rollout, scored against the gold label only:
    FEVER score (0/1) - ops_cost * n_ops - invalid_penalty * (invalid controller replies)
Advantage: reward minus the mean of the claim's G rollouts, divided by their std
(groups whose G rewards are all equal carry no signal and are skipped). Every
controller step of a rollout gets its rollout's advantage; the loss is on the
step's completion tokens only, averaged over all completion tokens in the batch.
One policy update per batch of rollouts (on-policy, so the GRPO ratio is 1).
KL (k3 estimator) is to the imitation controller: the imitation adapter is merged
into the base and a fresh LoRA of the same shape is trained, so disabling the LoRA
gives the reference policy.

--rollouts-only is the handover's pilot: sample G rollouts per claim, score them and
report how many claims have rewards that differ, without any update.

Evaluate a trained run with the merged pair:
    python -m agent.prompted --tools follow --chunk 64 \
        --controller local:<imitation checkpoints>,<run checkpoints> \
        --adapter <verifier checkpoints> --claim-ids verifier/data/eval/fever-dev-900-ids.json
"""

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

NEI = "NOT_ENOUGH_INFO"
IMITATION = ("verifier/experiments/sft-controller-from-base-qwen3p5-2b-flash-follow-18291-samples/"
             "runs/run1/artifacts/checkpoints")
VERIFIER = "verifier/experiments/sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run1/artifacts/checkpoints"


# ---------------------------------------------------------------- reward and advantage (no GPU)

def fever_ok(row):
    """FEVER score of one trace row, as agent.agent_loop.report computes it."""
    fin = row["final"]
    if fin["verdict"] != row["gold"]:
        return False
    return row["gold"] == NEI or any(set(s) <= set(fin["evidence"] or []) for s in row["gold_evidence"])


def n_invalid(row):
    return sum(not s["controller"]["valid"] for s in row["steps"] if "controller" in s)


def rollout_reward(row, ops_cost, invalid_penalty):
    """FEVER (0/1) minus the per-operation cost and the invalid-output penalty."""
    return (float(fever_ok(row)) - ops_cost * row["cost"]["n_ops"]
            - invalid_penalty * n_invalid(row))


def group_advantages(rewards, group_size, scale_std=True, eps=1e-4):
    """Rewards in claim-major order (G per claim) -> advantages; zero-spread groups get 0."""
    out = []
    for i in range(0, len(rewards), group_size):
        g = rewards[i:i + group_size]
        mean = sum(g) / len(g)
        std = math.sqrt(sum((r - mean) ** 2 for r in g) / len(g))
        if std < 1e-8:
            out += [0.0] * len(g)
        else:
            out += [(r - mean) / ((std + eps) if scale_std else 1.0) for r in g]
    return out


def spread_summary(rewards, group_size):
    groups = [rewards[i:i + group_size] for i in range(0, len(rewards), group_size)]
    differ = sum(max(g) - min(g) > 1e-8 for g in groups)
    return {"claims": len(groups), "claims_with_spread": differ,
            "frac_with_spread": differ / len(groups) if groups else float("nan")}


def fever_spread(rows, group_size):
    """Claims whose G rollouts disagree on FEVER (some right, some wrong): spread from the
    outcome itself rather than only from differing operation counts."""
    ok = [fever_ok(r) for r in rows]
    groups = [ok[i:i + group_size] for i in range(0, len(ok), group_size)]
    return sum(0 < sum(g) < len(g) for g in groups)


def compact(row, reward, group, sample):
    """A trace row without retrieved hits, for rollouts.jsonl."""
    steps = []
    for s in row["steps"]:
        st = {"action": s["action"]}
        if s["action"] == "search":
            st["query"] = s["args"].get("query")
        elif s["action"] == "follow":
            st["link"] = s["args"].get("link") or s["args"].get("target")
        elif s["action"] == "verify":
            st["verdict"] = s["result"].get("verdict")
            st["confidence"] = s["result"].get("confidence")
        if "controller" in s:
            st["text"] = s["controller"]["text"]
            st["valid"] = s["controller"]["valid"]
        steps.append(st)
    return {"id": row["id"], "group": group, "sample": sample, "gold": row["gold"],
            "final": row["final"], "fever": fever_ok(row), "n_ops": row["cost"]["n_ops"],
            "n_invalid": n_invalid(row), "reward": reward, "steps": steps}


# ---------------------------------------------------------------- policy

class SampledController:
    """The trainable controller: base + merged imitation LoRA + a fresh LoRA.

    sample() decodes with temperature and records the token ids of every prompt and
    completion so the update can recompute their log-probs."""

    def __init__(self, base, imitation, temperature, max_new_tokens=64, gen_batch=32, seed=0):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(base)
        self.tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(model, imitation).merge_and_unload()
        with open(os.path.join(imitation, "adapter_config.json"), encoding="utf-8") as f:
            sc = json.load(f)
        cfg = LoraConfig(r=sc["r"], lora_alpha=sc["lora_alpha"], lora_dropout=0.0,
                         target_modules=sc["target_modules"], task_type="CAUSAL_LM")
        self.model = get_peft_model(model, cfg)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.gen_batch = gen_batch
        self.seed = seed
        self.calls = 0
        eos = self.model.generation_config.eos_token_id
        self.stop = set(eos if isinstance(eos, list) else [eos]) | {self.tok.eos_token_id}
        self.name = f"{base}+{imitation}+grpo"

    def prompt_text(self, system, user):
        return self.tok.apply_chat_template([{"role": "system", "content": system},
                                             {"role": "user", "content": user}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def sample(self, system, users):
        torch = self.torch
        self.model.eval()
        out = []
        for i in range(0, len(users), self.gen_batch):
            texts = [self.prompt_text(system, u) for u in users[i:i + self.gen_batch]]
            inp = self.tok(texts, return_tensors="pt", padding=True).to("cuda")
            # LocalModel reseeds the global RNG on every verifier call, so seed each round here
            torch.manual_seed(self.seed * 1_000_003 + self.calls)
            self.calls += 1
            with torch.no_grad():
                seq = self.model.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=True,
                                          temperature=self.temperature, top_p=1.0, top_k=0,
                                          repetition_penalty=1.0, pad_token_id=self.tok.pad_token_id)
            n_in = inp["input_ids"].shape[1]
            for b in range(len(texts)):
                mask = inp["attention_mask"][b].bool()
                prompt_ids = inp["input_ids"][b][mask].tolist()
                comp = []
                for t in seq[b, n_in:].tolist():
                    comp.append(t)
                    if t in self.stop:
                        break
                text = self.tok.decode([t for t in comp if t not in self.stop], skip_special_tokens=True)
                out.append({"text": text, "in_tokens": len(prompt_ids), "out_tokens": len(comp),
                            "prompt_ids": prompt_ids, "completion_ids": comp})
        return out

    def logps(self, batch, use_reference=False):
        """Per-token log-probs of each completion given its prompt (temperature-scaled, as sampled)."""
        torch = self.torch
        pad = self.tok.pad_token_id
        seqs = [p + c for p, c in batch]
        width = max(map(len, seqs))
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):  # right padding for the forward pass
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        ids, att = ids.cuda(), att.cuda()
        if use_reference:
            with torch.no_grad(), self.model.disable_adapter():
                logits = self.model(input_ids=ids, attention_mask=att).logits
        else:
            logits = self.model(input_ids=ids, attention_mask=att).logits
        res = []
        for i, (p, c) in enumerate(batch):
            lg = logits[i, len(p) - 1:len(p) + len(c) - 1].float() / self.temperature
            tgt = torch.tensor(c, device=lg.device)
            res.append(torch.log_softmax(lg, -1).gather(-1, tgt[:, None]).squeeze(-1))
        return res


class FallbackDecomposer:
    """agent.decompose.Decomposer from the flash cache; a claim with no cached split is not
    split (no API key on the cluster). Decomposition was ruled out on FEVER and is rare."""

    def __init__(self, cache):
        from agent.decompose import Decomposer
        self.d = Decomposer(None, cache)
        self.prompt = self.d.prompt
        self.missing = 0

    def run(self, claim):
        if claim in self.d.cache:
            return {**self.d.cache[claim], "cached": True}
        self.missing += 1
        return {"subclaims": [claim], "tokens_in": 0, "tokens_out": 0, "cached": False}


# ---------------------------------------------------------------- rollouts

def rollout(claims, group_size, policy, search, verify, decomposer, run_info, pool, tools="follow"):
    """Play every claim group_size times in the prompted loop; claim-major order.
    Returns trace rows and, per rollout, the (prompt_ids, completion_ids) of each controller step."""
    from agent.prompted import K, MAX_STEPS, SYSTEMS, TOOLSETS, Episode, act, decide, to_row
    from agent.tools import ToolKit
    eps, texts = [], {}
    for c in claims:
        for _ in range(group_size):
            ep = Episode(c)
            ep.texts, ep.tools = texts, tools
            ep.kit = ToolKit(search.r, TOOLSETS[tools], None)
            ep.records = []
            eps.append(ep)
    while True:
        for ep in eps:
            if ep.end is None and len(ep.steps) >= MAX_STEPS:
                ep.end = "budget"
        live = [ep for ep in eps if ep.end is None]
        if not live:
            break
        t0 = time.time()
        gens = policy.sample(SYSTEMS[tools], [ep.observation() for ep in live])
        s = (time.time() - t0) / len(live)
        decisions = []
        for ep, g in zip(live, gens):
            decisions.append(decide(ep, None, {**g, "seconds": s}))
            ep.records.append((g["prompt_ids"], g["completion_ids"]))
        act(live, decisions, search, verify, decomposer, pool, K)
    return [to_row(ep, run_info) for ep in eps], [ep.records for ep in eps]


# ---------------------------------------------------------------- update

def update(policy, optimizer, records, advantages, beta, micro_batch, max_grad_norm=1.0):
    """One policy-gradient step over every controller step with a non-zero advantage."""
    torch = policy.torch
    items = [(p, c, adv) for recs, adv in zip(records, advantages) if adv != 0.0 for p, c in recs if c]
    if not items:
        return {"loss": 0.0, "kl": 0.0, "grad_norm": 0.0, "n_seqs": 0, "n_tokens": 0}
    n_tokens = sum(len(c) for _, c, _ in items)
    policy.model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = total_kl = 0.0
    random.Random(len(items)).shuffle(items)
    for i in range(0, len(items), micro_batch):
        mb = items[i:i + micro_batch]
        pairs = [(p, c) for p, c, _ in mb]
        lps = policy.logps(pairs)
        refs = policy.logps(pairs, use_reference=True) if beta > 0 else None
        loss = 0.0
        for j, (lp, (_, _, adv)) in enumerate(zip(lps, mb)):
            loss = loss - adv * lp.sum()
            if refs is not None:
                d = refs[j] - lp
                kl = (torch.exp(d) - d - 1).sum()
                loss = loss + beta * kl
                total_kl += kl.item()
        loss = loss / n_tokens
        loss.backward()
        total_loss += loss.item()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in policy.model.parameters() if p.requires_grad], max_grad_norm).item()
    optimizer.step()
    return {"loss": total_loss, "kl": total_kl / n_tokens, "grad_norm": grad_norm,
            "n_seqs": len(items), "n_tokens": n_tokens}


# ---------------------------------------------------------------- main

def main(defaults=None):
    from agent.action_cache import train_claims
    from agent.agent_loop import FeverSearch, LocalVerify, reward_version
    from agent.decompose import cache_path
    from agent.prompted import CONTROLLER_PROMPT, FINAL_RULE, MAX_STEPS
    from utils.experiment_layout import artifacts_dir
    from verifier.model import BIG_MODEL, LOCAL_MODEL

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=LOCAL_MODEL)
    ap.add_argument("--imitation", default=IMITATION, help="imitation controller LoRA (the start and KL reference)")
    ap.add_argument("--verifier", default=VERIFIER, help="frozen verifier LoRA")
    ap.add_argument("--n-train", type=int, default=64, help="unused train claims (agent.action_cache.train_claims)")
    ap.add_argument("--train-seed", type=int, default=3, help="claim sample; seed 2 is the imitation set")
    ap.add_argument("--group-size", type=int, default=8, help="G rollouts per claim")
    ap.add_argument("--claims-per-step", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ops-cost", type=float, default=0.01, help="lambda per search/follow/decompose/verify")
    ap.add_argument("--invalid-penalty", type=float, default=0.5, help="per invalid controller reply")
    ap.add_argument("--no-std-scale", action="store_true", help="advantage = reward - group mean only")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04, help="KL to the imitation controller")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--verify-batch", type=int, default=16)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--rollouts-only", action="store_true", help="pilot: sample and score, no update")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--config-name", default=None)
    ap.add_argument("--output-root", default=None)
    if defaults:
        ap.set_defaults(**defaults)
    a = ap.parse_args()

    artifacts = artifacts_dir("grpo", "", a.seed, a.run_name, a.output_root, a.config_name)
    if artifacts.exists():
        raise SystemExit(f"run already exists: {artifacts}; use a new --run-name")
    (artifacts / "logs").mkdir(parents=True)

    import torch
    claims = train_claims(a.n_train, a.train_seed)
    print(f"{len(claims)} claims x {a.group_size} rollouts, {'pilot (no update)' if a.rollouts_only else 'training'}")
    policy = SampledController(a.model, a.imitation, a.temperature, gen_batch=a.gen_batch, seed=a.seed)
    search = FeverSearch(method="bm25")
    verify = LocalVerify(search, a.model, a.verifier)
    verify.m.batch_size = a.verify_batch
    decomposer = FallbackDecomposer(cache_path(BIG_MODEL, a.n_train, split="train"))
    optimizer = None if a.rollouts_only else torch.optim.AdamW(
        [p for p in policy.model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    run_info = {"policy": "prompted-bm25-p3-k15", "controller": policy.name, "verifier": verify.name,
                "reward_version": reward_version(), "max_steps": MAX_STEPS, "split": "train",
                "controller_prompt": CONTROLLER_PROMPT + "-follow", "tools": "follow", "final_rule": FINAL_RULE,
                "train_claims": {"n": a.n_train, "seed": a.train_seed}}

    log, all_rewards, fever_differ = [], [], 0
    pool = ThreadPoolExecutor(4)
    t_start = time.time()
    n_steps = math.ceil(len(claims) / a.claims_per_step)
    roll_path = artifacts / "rollouts.jsonl"
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    for step in range(1, n_steps + 1):
        t0 = time.time()
        batch = claims[(step - 1) * a.claims_per_step:step * a.claims_per_step]
        rows, records = rollout(batch, a.group_size, policy, search, verify, decomposer, run_info, pool)
        t_roll = time.time() - t0
        rewards = [rollout_reward(r, a.ops_cost, a.invalid_penalty) for r in rows]
        adv = group_advantages(rewards, a.group_size, scale_std=not a.no_std_scale)
        stats = {"loss": 0.0, "kl": 0.0, "grad_norm": 0.0, "n_seqs": 0, "n_tokens": 0}
        if not a.rollouts_only:
            stats = update(policy, optimizer, records, adv, a.beta, a.micro_batch)
        all_rewards += rewards
        fever_differ += fever_spread(rows, a.group_size)
        with open(roll_path, "a", encoding="utf-8") as f:
            for k, (r, rw) in enumerate(zip(rows, rewards)):
                f.write(json.dumps(compact(r, rw, (step - 1) * a.claims_per_step + k // a.group_size,
                                           k % a.group_size), ensure_ascii=False) + "\n")
        ctl = [s["controller"]["valid"] for r in rows for s in r["steps"] if "controller" in s]
        sp = spread_summary(rewards, a.group_size)
        entry = {"step": step, "reward": mean(rewards),
                 "reward_std": math.sqrt(mean([(x - mean(rewards)) ** 2 for x in rewards])),
                 "frac_reward_zero_std": 1 - sp["frac_with_spread"],
                 "frac_fever_spread": fever_spread(rows, a.group_size) / len(batch),
                 "fever": mean([fever_ok(r) for r in rows]),
                 "accuracy": mean([r["final"]["verdict"] == r["gold"] for r in rows]),
                 "n_ops": mean([r["cost"]["n_ops"] for r in rows]),
                 "n_search": mean([r["cost"]["n_search"] for r in rows]),
                 "n_follow": mean([r["cost"].get("n_follow", 0) for r in rows]),
                 "n_verify": mean([r["cost"]["n_verify"] for r in rows]),
                 "invalid_rate": 1 - mean(ctl),
                 "ended_budget": mean([r["final"]["ended_by"] == "budget" for r in rows]),
                 "ended_abstain": mean([r["final"]["ended_by"] == "abstain" for r in rows]),
                 **stats, "rollout_seconds": round(t_roll, 1), "step_seconds": round(time.time() - t0, 1)}
        log.append(entry)
        with open(artifacts / "logs" / "train_log.json", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items()}), flush=True)
        if not a.rollouts_only and a.save_steps and step % a.save_steps == 0:
            policy.model.save_pretrained(artifacts / "checkpoints" / f"checkpoint-{step}")

    secs = time.time() - t_start
    if not a.rollouts_only:
        policy.model.save_pretrained(artifacts / "checkpoints")
        policy.tok.save_pretrained(artifacts / "checkpoints")
    meta = {**vars(a), "steps": n_steps, "n_claims": len(claims),
            "claim_ids": [c["id"] for c in claims],
            "spread": {**spread_summary(all_rewards, a.group_size), "claims_with_fever_spread": fever_differ},
            "mean_reward": mean(all_rewards), "decompose_uncached": decomposer.missing,
            "train_seconds": round(secs), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "reward": "fever - ops_cost*n_ops - invalid_penalty*n_invalid"}
    with open(artifacts / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "claim_ids"}, indent=1))
    pool.shutdown()


if __name__ == "__main__":
    main()
