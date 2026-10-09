"""Fixed train-only development schedule for the context-encoding correction."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "Mix_PPO_Sichuan_20260917_ContextV3"


def main():
    RUN.mkdir(exist_ok=False)
    status = RUN / "status.json"

    def write(state, **extra):
        status.write_text(json.dumps({"state": state, "updated_at": time.time(), **extra}, indent=2) + "\n")

    try:
        for stage in [50, 100, 200, 300]:
            write("training", target_update=stage, final_budget=300)
            command = [sys.executable, str(ROOT / "train_mix_ppo_adapted.py"),
                       "--output", str(RUN / "train"), "--context-inputs", "--opponents", "curriculum",
                       "--seed", "2026091704", "--updates", str(stage), "--envs", "8", "--device", "cuda:0"]
            if stage != 50:
                command.append("--resume")
            subprocess.run(command, cwd=ROOT, check=True)
            checkpoint = RUN / f"update_{stage}.pt"
            shutil.copy2(RUN / "train/latest.pt", checkpoint)
            write("train_development_evaluation", update=stage, games=200)
            subprocess.run([sys.executable, str(ROOT / "eval_mix_ppo_adapted.py"),
                            "--checkpoint", str(checkpoint), "--output", str(RUN / f"dev_{stage}"),
                            "--split", "train", "--outer-seeds", "2026091801", "2026091802",
                            "--games", "100", "--device", "cuda:1"], cwd=ROOT, check=True)
        reports = []
        for stage in [50, 100, 200, 300]:
            data = json.loads((RUN / f"dev_{stage}/summary.json").read_text())
            summaries = data["by_seed"]
            reports.append({"update": stage, **{key: sum(r[key] for r in summaries) / len(summaries)
                             for key in ["avg_net", "hu_game_rate", "ron_game_rate", "dealin_game_rate"]},
                            "hu_opportunities": sum(r["hu_opportunities"] for r in summaries),
                            "missed_hu": sum(r["missed_hu"] for r in summaries)})
        (RUN / "development_summary.json").write_text(json.dumps(reports, indent=2) + "\n")
        write("development_complete", formal_holdout=False, requires_strength_review=True)
    except Exception as error:
        write("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
