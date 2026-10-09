"""Three-seed, fixed-budget development pilot. No holdout or auto long run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "Mix_PPO_Sichuan_20260919_V6_MultiSeed"
SEEDS = (2026091911, 2026091912, 2026091913)
STAGES = (10, 25, 50)
DEV = tuple(range(2026092401, 2026092409))


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if RUN.exists() and not args.resume:
        raise FileExistsError(RUN)
    if args.resume and not (RUN / "protocol.json").exists():
        raise FileNotFoundError("Missing frozen protocol")
    RUN.mkdir(exist_ok=True)

    def status(state, **extra):
        save(RUN / "status.json", dict(state=state, updated_at=time.time(), **extra))

    def execute(command, log):
        with log.open("a") as output:
            subprocess.run([sys.executable, *command], cwd=ROOT, stdout=output,
                           stderr=subprocess.STDOUT, check=True)

    try:
        if not args.resume:
            snapshot = RUN / "source_snapshot"
            snapshot.mkdir()
            files = ["train_mix_ppo_adapted.py", "mix_ppo_baseline.py", "mix_ppo_rollout.py",
                     "mix_ppo_search.py", "mix_ppo_parallel.py", "mix_ppo_selfplay.py",
                     "eval_mix_ppo_adapted.py", "eval_mix_ppo_parallel.py", Path(__file__).name,
                     "game.py", "tiles.py", "rule_engine.py", "opponent_curriculum.py",
                     "run_gate1_experiments.py", "prompt_builder.py"]
            for filename in files:
                shutil.copy2(ROOT / filename, snapshot / filename)
            save(RUN / "protocol.json", dict(
                method="Mix-PPO-adapted V6 development pilot", train_seeds=SEEDS,
                updates_per_seed=50, stages=STAGES, rollout_steps=2048,
                minimum_decisions_per_seed=102400, epochs=8, minibatch=256,
                reward="balance difference / 10, no shaping", privileged_until_steps=10000,
                promotion_interval=25, promotion_games_minimum=200,
                promotion_rule="one-sided 95% t lower bound of independent mean net > 0",
                promotion_seed_families=[2026092501+i for i in range(3)],
                development_seeds=DEV, development_games_per_seed=25,
                train_workers=8, eval_workers=8, devices=["cuda:0"],
                scheduling="stage-major sequential training seeds; parallel environments/evaluation",
                holdout=False, auto_long_training=False,
                interpretation="Learning diagnostics only; no selected winning seed or significance claim",
                sources={f: hashlib.sha256((snapshot/f).read_bytes()).hexdigest() for f in files}))
        protocol = json.loads((RUN / "protocol.json").read_text())
        for filename, expected in protocol["sources"].items():
            if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen source changed: {filename}")
        reports = []
        for stage in STAGES:
            for i, seed in enumerate(SEEDS):
                folder = RUN / f"seed_{seed}"
                folder.mkdir(exist_ok=True)
                checkpoint = folder / f"update_{stage}.pt"
                if not checkpoint.exists():
                    status("training", seed=seed, target_update=stage, final_update=50,
                           completed_evaluations=len(reports), workers=8)
                    command = ["train_mix_ppo_adapted.py", "--output", str(folder/"train"),
                        "--context-inputs", "--search-inputs", "--reward-mode", "net",
                        "--reward-scale", "10", "--opponents", "random", "--goal-selfplay",
                        "--promotion-interval", "25", "--promotion-seed", str(2026092501+i),
                        "--privileged-until-steps", "10000", "--seed", str(seed),
                        "--workers", "8", "--envs", "8", "--updates", str(stage),
                        "--rollout-steps", "2048", "--minibatch", "256", "--epochs", "8",
                        "--device", "cuda:0"]
                    if (folder / "train/latest.pt").exists():
                        command.append("--resume")
                    execute(command, folder/"training.log")
                    temporary = checkpoint.with_suffix(".tmp")
                    shutil.copy2(folder/"train/latest.pt", temporary)
                    temporary.replace(checkpoint)
                target = folder / f"dev_{stage}"
                digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                if not (target/"summary.json").exists():
                    if target.exists():
                        target.rename(folder/f"dev_{stage}_incomplete_{time.time_ns()}")
                    status("development_evaluation", seed=seed, update=stage, games=200, workers=8)
                    execute(["eval_mix_ppo_parallel.py", "--checkpoint", str(checkpoint),
                        "--output", str(target), "--split", "train", "--outer-seeds", *map(str, DEV),
                        "--games", "25", "--workers", "8", "--devices", "cuda:0"], folder/"evaluation.log")
                report = json.loads((target/"summary.json").read_text())
                rows = report["by_seed"]
                if report["checkpoint_sha256"] != digest or {r["outer_seed"] for r in rows} != set(DEV):
                    raise ValueError("Evaluation identity mismatch")
                if any(r["games"] != 25 or r["max_steps_or_unsettled"] for r in rows):
                    raise ValueError("Evaluation completeness failure")
                reports.append(dict(train_seed=seed, update=stage, games=200,
                    **{k: sum(r[k] for r in rows)/len(rows) for k in
                       ("avg_net", "hu_game_rate", "ron_game_rate", "dealin_game_rate", "cvar10")},
                    hu_opportunities=sum(r["hu_opportunities"] for r in rows),
                    missed_hu=sum(r["missed_hu"] for r in rows)))
                save(RUN/"development_summary.json", reports)
        status("pilot_complete", completed_evaluations=len(reports), holdout=False,
               auto_long_training=False, requires_learning_review=True)
    except Exception as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
