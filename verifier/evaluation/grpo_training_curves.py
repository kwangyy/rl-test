"""Summarise GRPO training curves from committed run artifacts.

Reads each run's `artifacts/logs/train_log.json` (TRL log history) and, when present,
`artifacts/rollouts.jsonl` (multi-turn runs), then prints a Markdown table per run with
one row per window of steps and optionally writes a figure.

Usage (repository root):
    python -m verifier.evaluation.grpo_training_curves \
        --runs verifier/experiments/<config>/runs/<run> [...] \
        --window 50 --figure verifier/docs/figures/<name>.png
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

LABELS = ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO")


def load_steps(run: Path) -> list[dict]:
    log = json.loads((run / "artifacts" / "logs" / "train_log.json").read_text())
    return [row for row in log if "reward" in row and "step" in row]


def predicted_label(messages: list[dict]) -> str | None:
    """Verdict in the final assistant JSON message, or None if it does not parse."""
    for message in reversed(messages):
        if message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        content = message.get("content") or ""
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            verdict = json.loads(content[start:end + 1]).get("verdict")
        except (json.JSONDecodeError, AttributeError):
            return None
        return verdict if verdict in LABELS else None
    return None


def load_rollout_labels(run: Path, per_step: int) -> dict[int, Counter]:
    """Predicted-label counts per training step (rollouts are written in step order)."""
    path = run / "artifacts" / "rollouts.jsonl"
    if not path.exists():
        return {}
    by_step: dict[int, Counter] = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            step = index // per_step + 1
            label = predicted_label(json.loads(line).get("messages", []))
            by_step.setdefault(step, Counter())[label or "invalid"] += 1
    return by_step


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarise(run: Path, window: int) -> tuple[str, list[dict]]:
    meta = json.loads((run / "artifacts" / "meta.json").read_text())
    steps = load_steps(run)
    per_step = int(meta.get("num_generations", 8)) * int(meta.get("prompts_per_step", 2))
    labels = load_rollout_labels(run, per_step)
    multiturn = any("rewards/searches/mean" in row for row in steps)

    rows = []
    for start in range(1, (steps[-1]["step"] if steps else 0) + 1, window):
        chunk = [row for row in steps if start <= row["step"] < start + window]
        if not chunk:
            continue
        row = {
            "steps": f"{start}-{chunk[-1]['step']}",
            "reward": mean([r["reward"] for r in chunk]),
            "correctness": mean([r.get("rewards/correctness/mean", float("nan")) for r in chunk]),
            "evidence": mean([r.get("rewards/evidence/mean", float("nan")) for r in chunk]),
            "format_ok": mean([r.get("rewards/format_ok/mean", float("nan")) for r in chunk]),
        }
        if multiturn:
            row["searches"] = mean([r["rewards/searches/mean"] for r in chunk])
        counts = Counter()
        for step in range(start, start + window):
            counts.update(labels.get(step, Counter()))
        total = sum(counts.values())
        if total:
            for label in LABELS:
                row[label] = counts[label] / total
        rows.append(row)

    columns = list(rows[0]) if rows else []
    header = {"SUPPORTED": "pred S", "REFUTED": "pred R", "NOT_ENOUGH_INFO": "pred NEI",
              "format_ok": "valid format"}
    lines = [f"`{meta.get('config_name')}` / `{meta.get('run_name')}` "
             f"(reward `{meta.get('reward')}`, {window}-step windows)", "",
             "| " + " | ".join(header.get(c, c) for c in columns) + " |",
             "|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|"]
    for row in rows:
        cells = [row["steps"]] + [f"{row[c]:.2f}" for c in columns[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines), steps


def plot(runs: list[Path], all_steps: list[list[dict]], figure: Path, smooth: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rolling(values: list[float]) -> list[float]:
        return [mean(values[max(0, i - smooth + 1):i + 1]) for i in range(len(values))]

    panels = [("reward", "Mean reward"), ("rewards/correctness/mean", "Correctness term"),
              ("rewards/evidence/mean", "Evidence term"), ("rewards/searches/mean", "Searches per rollout")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for ax, (key, title) in zip(axes.flat, panels):
        for run, steps in zip(runs, all_steps):
            if not steps or key not in steps[0]:
                continue
            meta = json.loads((run / "artifacts" / "meta.json").read_text())
            name = meta.get("config_name", run.parent.parent.name)
            label = name.replace("grpo-", "").replace("-qwen3p5-2b", "").replace("-1000-samples", "")
            ax.plot([r["step"] for r in steps], rolling([r[key] for r in steps]), label=label, linewidth=1.4)
        ax.set_title(f"{title} ({smooth}-step rolling mean)", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("training step")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=110)
    print(f"\nwrote {figure}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--smooth", type=int, default=25)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()

    all_steps = []
    for run in args.runs:
        table, steps = summarise(run, args.window)
        all_steps.append(steps)
        print(table + "\n")
    if args.figure:
        plot(args.runs, all_steps, args.figure, args.smooth)


if __name__ == "__main__":
    main()
