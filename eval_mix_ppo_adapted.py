"""Evaluate a frozen Mix-PPO-adapted checkpoint; default is train-only development."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch

from mix_ppo_baseline import (SichuanResNet101, FEATURE_VERSION, CONTEXT_FEATURE_VERSION,
                              PROSPECTIVE_FEATURE_VERSION, SEARCH_FEATURE_VERSION, ACTIONS)
from mix_ppo_rollout import episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "holdout"], default="train")
    parser.add_argument("--outer-seeds", type=int, nargs="+", default=[2026091701])
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--manifest", default="Opponent_Curriculum_formal/screened.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.games <= 0 or args.games >= 1000000 or len(set(args.outer_seeds)) != len(args.outer_seeds):
        parser.error("Require 0 < games < 1000000 and unique outer seeds")
    if args.output.exists():
        parser.error("Refusing to overwrite evaluation output")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    manifest_hash = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    if manifest_hash != saved["config"]["manifest_sha256"]:
        raise ValueError("Curriculum manifest changed since training")
    for filename, expected in saved["config"]["source_sha256"].items():
        if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Training source changed: {filename}")
    context_inputs = saved["config"].get("context_inputs", False)
    prospective_inputs = saved["config"].get("prospective_inputs", False)
    search_inputs = saved["config"].get("search_inputs", False)
    privileged_inputs = saved["config"].get("privileged_until_steps", 0) > 0
    if args.split == "holdout" and saved["total_steps"] < saved["config"].get("privileged_until_steps", 0):
        raise ValueError("Cannot publish holdout evaluation before privileged annealing completes")
    expected_version = SEARCH_FEATURE_VERSION if search_inputs else (PROSPECTIVE_FEATURE_VERSION if prospective_inputs else (CONTEXT_FEATURE_VERSION if context_inputs else FEATURE_VERSION))
    if saved["config"]["feature_version"] != expected_version or saved["config"]["actions"] != list(ACTIONS):
        raise ValueError("Checkpoint feature/action schema mismatch")
    training_seed = saved["config"]["seed"]
    training_games = range(training_seed * 1000000, training_seed * 1000000 + saved["game_index"])
    for seed in args.outer_seeds:
        if any(seed * 1000000 + i in training_games for i in range(args.games)):
            raise ValueError("Evaluation seeds overlap with checkpoint training")
    model = SichuanResNet101(context_inputs=context_inputs,
                             prospective_inputs=prospective_inputs, search_inputs=search_inputs,
                             privileged_inputs=privileged_inputs).to(args.device).eval()
    model.load_state_dict(saved["model"])
    args.output.mkdir(parents=True)
    summaries = []
    with (args.output / "games.jsonl").open("w") as out:
        for outer_seed in args.outer_seeds:
            rows = []
            for index in range(args.games):
                stream = episode(outer_seed * 1000000 + index, "curriculum", args.manifest, args.split,
                                 context_inputs=context_inputs,
                                 prospective_inputs=prospective_inputs, search_inputs=search_inputs,
                                 privileged_inputs=privileged_inputs, privileged_scale=0.)
                try:
                    obs, mask, _ = next(stream)
                    while True:
                        with torch.no_grad():
                            dist, _ = model(obs.unsqueeze(0).to(args.device), mask.unsqueeze(0).to(args.device))
                        obs, mask, _ = stream.send(int(dist.probs.argmax()))
                except StopIteration as ended:
                    row = {"outer_seed": outer_seed, "game_index": index, **ended.value}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    rows.append(row)
            nets = sorted(row["net"] for row in rows)
            tail = max(1, math.ceil(len(nets) * 0.1))
            result = {"outer_seed": outer_seed, "games": len(rows),
                      "avg_net": sum(nets) / len(nets), "cvar10": sum(nets[:tail]) / tail,
                      "ron_game_rate": sum(r["ron"] for r in rows) / len(rows),
                      "dealin_game_rate": sum(r["dealt_in"] for r in rows) / len(rows),
                      "hu_game_rate": sum(r["hu"] for r in rows) / len(rows),
                      "max_steps_or_unsettled": sum(r["max_steps_or_unsettled"] for r in rows)}
            summaries.append(result)
            counts = Counter()
            for row in rows:
                counts.update(row["behavior"])
            result["behavior"] = dict(counts)
            result["hu_opportunities"] = sum(counts[k + "_hu_available"] for k in ("turn", "response"))
            result["hu_taken"] = sum(counts[k + "_hu_taken"] for k in ("turn", "response"))
            result["missed_hu"] = result["hu_opportunities"] - result["hu_taken"]
            shanten = [r["final_shanten"] for r in rows if r["final_shanten"] is not None]
            result["mean_unwon_final_shanten"] = sum(shanten) / len(shanten) if shanten else None
            print(json.dumps(result), flush=True)
    report = {"method": "Mix-PPO-adapted", "action_selection": "argmax",
              "privileged_scale": 0.,
              "split": args.split, "games_per_seed": args.games,
              "training_updates": saved["update"], "training_steps": saved["total_steps"],
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "manifest_sha256": manifest_hash,
              "source_sha256": {**saved["config"]["source_sha256"],
                                "eval_mix_ppo_adapted.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "by_seed": summaries}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
