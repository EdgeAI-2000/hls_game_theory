"""Wait for a fixed training budget, then evaluate and compare with frozen P4."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
SEEDS = list(range(2026082001, 2026082021))
METRICS = ["avg_net", "ron_game_rate", "dealin_game_rate", "cvar10", "hu_game_rate"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def summarize(rows, seed, legacy=False):
    if len(rows) != 1000 or {r["seed"] for r in rows} != {seed * 1000000 + i for i in range(1000)}:
        raise ValueError(f"Incomplete or duplicate seeds: {seed}")
    if legacy:
        if any(not r.get("settlement_reconciled") or r["end_reason"] == "max_steps" for r in rows):
            raise ValueError("P4 settlement audit failed")
        score, ron, dr, hu = "agent_net", "agent_hu_by_discard", "agent_dealin", "agent_hu"
    else:
        if any(r["max_steps_or_unsettled"] for r in rows):
            raise ValueError("Mix-PPO unsettled game")
        score, ron, dr, hu = "net", "ron", "dealt_in", "hu"
    nets = sorted(float(r[score]) for r in rows)
    result = dict(avg_net=sum(nets) / 1000, cvar10=sum(nets[:100]) / 100,
                  ron_game_rate=sum(r[ron] for r in rows) / 1000,
                  dealin_game_rate=sum(r[dr] for r in rows) / 1000,
                  hu_game_rate=sum(r[hu] for r in rows) / 1000)
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite metrics")
    return result


def comparison(output):
    series = {"Mix-PPO-adapted": []}
    hashes = {}
    checkpoint_hashes = set()
    for seed in SEEDS:
        folder = output / f"seed_{seed}"
        metadata = json.loads((folder / "summary.json").read_text())
        checkpoint_hashes.add(metadata["checkpoint_sha256"])
        if metadata["split"] != "holdout" or metadata["training_updates"] != 100:
            raise ValueError("Unexpected evaluation protocol")
        series["Mix-PPO-adapted"].append(summarize(read_rows(folder / "games.jsonl"), seed))
        hashes[str(folder / "games.jsonl")] = digest(folder / "games.jsonl")
    if len(checkpoint_hashes) != 1:
        raise ValueError("Checkpoints differ between seeds")
    base = ROOT / "Formal_E2_E3_P4_STRICT_FFR_20260823/formal_e3"
    for name, variant in [("L0", "L0"), ("L1-z", "L1"), ("no-deceive", "L2_no_deceive"),
                          ("no-course", "L2_no_course"), ("MASK-full", "L2_full")]:
        series[name] = []
        for seed in SEEDS:
            path = base / variant / f"seed_{seed}/gate1_games.jsonl"
            series[name].append(summarize(read_rows(path), seed, legacy=True))
            hashes[str(path)] = digest(path)
    means = {name: {m: float(np.mean([s[m] for s in values])) for m in METRICS}
             for name, values in series.items()}
    paired = {}
    for name in series:
        if name == "Mix-PPO-adapted":
            continue
        paired[name] = {}
        for metric in METRICS:
            d = np.array([s[metric] for s in series["Mix-PPO-adapted"]]) - np.array([s[metric] for s in series[name]])
            mean, se = float(d.mean()), float(stats.sem(d))
            p = float(stats.ttest_1samp(d, 0).pvalue) if se else (1.0 if mean == 0 else 0.0)
            half = float(stats.t.ppf(.975, 19)) * se
            paired[name][metric] = {"difference": mean, "ci95": [mean-half, mean+half], "p_nominal": p}
    lines = ["# Mix-PPO-adapted holdout comparison", "",
             "20 paired outer seeds, 1,000 games each. Fixed update-100 checkpoint; no holdout selection.", "",
             "| System | Mean score | Ron rate | DR | CVaR-10% | Hu rate |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, row in means.items():
        lines.append(f"| {name} | {row['avg_net']:.4f} | {row['ron_game_rate']:.3%} | {row['dealin_game_rate']:.3%} | {row['cvar10']:.4f} | {row['hu_game_rate']:.3%} |")
    lines += ["", "Differences below are Mix-PPO-adapted minus MASK-full. Rates use percentage points.", "",
              "| Metric | Difference | 95% CI | Nominal paired t-test p |", "|---|---:|---|---:|"]
    for metric, result in paired["MASK-full"].items():
        scale = 100 if metric.endswith("rate") else 1
        lo, hi = result["ci95"]
        lines.append(f"| {metric} | {result['difference']*scale:+.4f} | [{lo*scale:.4f}, {hi*scale:.4f}] | {result['p_nominal']:.6g} |")
    lines += ["", "P-values are unadjusted and exploratory. CVaR is the mean of seed-level tail means.",
              "This comparison uses one training seed and a fixed budget; it does not establish RL convergence or an RL performance ceiling.",
              "Strict FFR is not measured by this evaluator. A ron-rate difference alone does not identify deception as its cause."]
    (output / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    (output / "comparison.json").write_text(json.dumps({"outer_seeds": SEEDS, "by_seed": series,
        "means": means, "paired_mix_minus_baseline": paired, "source_hashes": hashes,
        "checkpoint_sha256": checkpoint_hashes.pop()}, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "Mix_PPO_Sichuan_20260916")
    args = parser.parse_args()
    train = args.run / "formal_train"
    output = args.run / "formal_holdout_eval"
    status = args.run / "pipeline_status.json"

    def update(state, **details):
        status.write_text(json.dumps({"state": state, "updated_at": time.time(), **details}, indent=2) + "\n")

    try:
        update("waiting_for_training", training_pid=args.training_pid, workers=8, outer_seeds=SEEDS, games_per_seed=1000)
        while not (train / "run_complete.json").exists():
            os.kill(args.training_pid, 0)
            time.sleep(20)
        completed = json.loads((train / "run_complete.json").read_text())
        if completed["completed_updates"] != 100 or digest(train / "latest.pt") != completed["checkpoint_sha256"]:
            raise ValueError("Training completion/checkpoint audit failed")
        metrics = read_rows(train / "metrics.jsonl")
        games = read_rows(train / "games.jsonl")
        if [m["update"] for m in metrics] != list(range(1, 101)) or any(g["max_steps_or_unsettled"] for g in games):
            raise ValueError("Training update/settlement audit failed")
        if len(games) != completed["total_games"] or len({g["seed"] for g in games}) != len(games):
            raise ValueError("Training game audit failed")
        output.mkdir(exist_ok=False)
        update("evaluating", workers=8, checkpoint_sha256=completed["checkpoint_sha256"])

        def worker(seed):
            with (output / f"seed_{seed}.log").open("w") as log:
                subprocess.run([sys.executable, str(ROOT / "eval_mix_ppo_adapted.py"),
                    "--checkpoint", str(train / "latest.pt"), "--output", str(output / f"seed_{seed}"),
                    "--split", "holdout", "--outer-seeds", str(seed), "--games", "1000",
                    "--device", f"cuda:{(seed-SEEDS[0]) % 2}"], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            return seed

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, SEEDS))
        update("analyzing")
        comparison(output)
        update("complete", games=20000, report=str(output / "COMPARISON.md"))
    except Exception as error:
        update("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
