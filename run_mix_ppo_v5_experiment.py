"""Frozen-budget V5 training and process-parallel evaluation, run under tmux."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "Mix_PPO_Sichuan_20260918_MixPPO_V5"
STAGES = (50, 200, 500, 1000)
DEV_SEEDS = list(range(2026092001, 2026092009))
HOLDOUT_SEEDS = list(range(2026092101, 2026092121))


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if RUN.exists() and not args.resume:
        raise FileExistsError(RUN)
    if args.resume and not (RUN / "protocol.json").exists():
        raise FileNotFoundError("No frozen protocol to resume")
    RUN.mkdir(exist_ok=True)

    def status(state, **extra):
        write_json(RUN / "status.json", dict(state=state, updated_at=time.time(), **extra))

    def execute(command, log):
        with log.open("a") as output:
            subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)

    def evaluate(checkpoint, name, split, seeds, games):
        target = RUN / name
        if (target / "summary.json").exists():
            report = json.loads((target / "summary.json").read_text())
            if report["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
                raise ValueError("Completed evaluation checkpoint mismatch")
            return
        # Preserve failed/incomplete attempts, then restart this bounded stage.
        if target.exists():
            target.rename(RUN / f"{name}_incomplete_{time.time_ns()}")
        execute([sys.executable, str(ROOT / "eval_mix_ppo_parallel.py"),
                 "--checkpoint", str(checkpoint), "--output", str(target),
                 "--split", split, "--outer-seeds", *map(str, seeds),
                 "--games", str(games), "--workers", "8", "--devices", "cuda:0", "cuda:1"],
                RUN / "evaluation.log")

    try:
        if not args.resume:
            snapshot = RUN / "source_snapshot"
            snapshot.mkdir()
            files = ["train_mix_ppo_adapted.py", "mix_ppo_baseline.py", "mix_ppo_rollout.py",
                     "mix_ppo_parallel.py", "mix_ppo_search.py", "mix_ppo_selfplay.py",
                     "eval_mix_ppo_adapted.py", "eval_mix_ppo_parallel.py", Path(__file__).name,
                     "game.py", "tiles.py", "prompt_builder.py", "rule_engine.py",
                     "run_gate1_experiments.py", "opponent_curriculum.py"]
            for filename in files:
                shutil.copy2(ROOT / filename, snapshot / filename)
            write_json(RUN / "protocol.json", {
                "method": "Mix-PPO-adapted", "final_update": 1000,
                "minimum_decision_budget": 2048000, "paper_budget": 100000000,
                "budget_scope": "Initial fixed-budget adaptation; not paper-budget reproduction",
                "stages": STAGES, "train_seed": 2026091805,
                "privileged_until_steps": 100000, "train_workers": 8, "eval_workers": 8,
                "rollout_steps": 2048, "epochs": 8, "minibatch": 256,
                "dev_seeds": DEV_SEEDS, "dev_games_per_seed": 25,
                "holdout_seeds": HOLDOUT_SEEDS, "holdout_games_per_seed": 1000,
                "selection_rule": "Final update 1000 only; no holdout-based selection",
                "failure_rule": "Stop on training/evaluation failure; no automatic method changes",
                "paired_with_legacy_P4": False,
                "comparison_note": "Fresh holdout seeds require matched comparator reruns before paired claims",
                "sources": {f: hashlib.sha256((snapshot / f).read_bytes()).hexdigest() for f in files}})
        protocol = json.loads((RUN / "protocol.json").read_text())
        for filename, digest in protocol["sources"].items():
            if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Frozen source changed: {filename}")
        for stage in STAGES:
            checkpoint = RUN / f"update_{stage}.pt"
            if not checkpoint.exists():
                status("training", target_update=stage, final_budget=1000, workers=8)
                command = [sys.executable, str(ROOT / "train_mix_ppo_adapted.py"),
                    "--output", str(RUN / "train"), "--context-inputs", "--search-inputs",
                    "--reward-mode", "paper_adapted", "--opponents", "random", "--goal-selfplay",
                    "--privileged-until-steps", "100000", "--seed", "2026091805",
                    "--workers", "8", "--envs", "8", "--updates", str(stage),
                    "--rollout-steps", "2048", "--minibatch", "256", "--epochs", "8",
                    "--device", "cuda:0"]
                if (RUN / "train/latest.pt").exists():
                    command.append("--resume")
                execute(command, RUN / "training.log")
                temporary = checkpoint.with_suffix(".tmp")
                shutil.copy2(RUN / "train/latest.pt", temporary)
                temporary.replace(checkpoint)
            status("development_evaluation", update=stage, games=200, workers=8)
            evaluate(checkpoint, f"dev_{stage}", "train", DEV_SEEDS, 25)
        status("holdout_evaluation", update=1000, games=20000, workers=8)
        evaluate(RUN / "update_1000.pt", "holdout_20k", "holdout", HOLDOUT_SEEDS, 1000)
        report = json.loads((RUN / "holdout_20k/summary.json").read_text())
        rows = report["by_seed"]
        write_json(RUN / "RESULTS.json", {
            "method": "Mix-PPO-adapted", "outer_seeds": len(rows),
            "games": sum(row["games"] for row in rows),
            "seed_mean_metrics": {key: sum(row[key] for row in rows)/len(rows)
                                  for key in ("avg_net", "ron_game_rate", "dealin_game_rate", "hu_game_rate", "cvar10")},
            "max_steps_or_unsettled": sum(row["max_steps_or_unsettled"] for row in rows),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "paired_comparison_available": False,
            "note": "Standalone frozen evaluation; no significance claim against unmatched legacy P4 seeds"})
        status("complete", holdout_games=20000, paired_comparison_available=False)
    except Exception as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
