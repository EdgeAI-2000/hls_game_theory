"""Collect, train and evaluate the shared formal MASK E2/E3 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import List


REPO_DIR = Path(__file__).resolve().parent


def run_command(command: List[str], log_path: Path, gpu: str, force: bool, marker: Path) -> None:
    signature_path = marker.with_name(marker.name + ".command.sha256")
    input_hashes = {}
    for token in command:
        candidate = Path(token)
        if candidate.is_file():
            input_hashes[str(candidate.resolve())] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    signature = hashlib.sha256(
        json.dumps(
            {"command": command, "gpu": gpu, "input_hashes": input_hashes},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if (
        marker.is_file()
        and signature_path.is_file()
        and signature_path.read_text(encoding="utf-8").strip() == signature
        and not force
    ):
        print(f"[formal-e3] reuse {marker}", flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    env["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"[formal-e3] run: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO_DIR, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not marker.is_file():
        raise RuntimeError(f"stage completed without expected artifact: {marker}")
    signature_path.write_text(signature + "\n", encoding="utf-8")


def collect_command(args: argparse.Namespace, output: Path, style: str) -> List[str]:
    command = [
        str(args.python), "collect_gate_mode_values.py",
        "--output-dir", str(output),
        "--model-path", str(args.base_model),
        "--opponent-style", style,
        "--target-states", str(args.target_states),
        "--target-deceive-states", str(args.target_deceive_states),
        "--max-games", str(args.max_collection_games),
        "--sample-stride", str(args.sample_stride),
        "--crn-seeds", str(args.crn_seeds),
        "--mask-oracle-samples", str(args.mask_oracle_samples),
        "--mask-oracle-beta", str(args.mask_oracle_beta),
        "--mask-danger-threshold", str(args.mask_danger_threshold),
        "--auto-hu", args.auto_hu,
        "--require-target",
        "--seed", str(args.collection_seed + (0 if style == "curriculum" else 100000)),
    ]
    if style == "curriculum":
        command.extend([
            "--curriculum-manifest", str(args.curriculum_manifest),
            "--curriculum-split", "train",
        ])
    if args.belief_calibration_summary:
        command.extend([
            "--belief-calibration-summary", str(args.belief_calibration_summary),
            "--belief-public-model", str(args.belief_public_model),
        ])
    return command


def train_command(
    args: argparse.Namespace,
    data: Path,
    output: Path,
    feature_ablation: str = "none",
    allowed_modes: tuple[str, ...] = ("exploit", "safe", "deceive"),
) -> List[str]:
    return [
        str(args.python), "train_gate_value_policy.py",
        "--input", str(data),
        "--output-dir", str(output),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--min-examples", str(args.min_examples),
        "--group-split",
        "--feature-ablation", feature_ablation,
        "--allowed-modes", *allowed_modes,
        "--belief-balance-strength", str(args.belief_balance_strength),
        "--device", args.train_device,
        "--seed", str(args.training_seed),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("py10/bin/python3"))
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--curriculum-manifest", type=Path, required=True)
    parser.add_argument("--belief-calibration-summary", type=Path, required=True)
    parser.add_argument("--belief-public-model", type=Path, required=True)
    parser.add_argument("--mask-oracle-samples", type=int, default=60)
    parser.add_argument("--mask-oracle-beta", type=float, default=2.0)
    parser.add_argument("--mask-danger-threshold", type=int, default=0)
    parser.add_argument("--target-states", type=int, default=900)
    parser.add_argument("--target-deceive-states", type=int, default=240)
    parser.add_argument("--max-collection-games", type=int, default=7000)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--crn-seeds", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-examples", type=int, default=500)
    parser.add_argument("--belief-balance-strength", type=float, default=0.5)
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--games-per-seed", type=int, default=1000)
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--auto-hu", choices=["common", "off"], default="off")
    parser.add_argument("--collection-seed", type=int, default=2026081201)
    parser.add_argument("--training-seed", type=int, default=3431)
    parser.add_argument("--train-device", default="cpu")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    course_data_dir = args.output_dir / "data_course"
    manual_data_dir = args.output_dir / "data_no_course"
    run_command(
        collect_command(args, course_data_dir, "curriculum"),
        course_data_dir / "collect.log", args.gpu, args.force,
        course_data_dir / "gate_mode_value_collection_summary.json",
    )
    run_command(
        collect_command(args, manual_data_dir, "mixed"),
        manual_data_dir / "collect.log", args.gpu, args.force,
        manual_data_dir / "gate_mode_value_collection_summary.json",
    )

    models = {
        "full": (course_data_dir / "gate_mode_values.jsonl", "none", ("exploit", "safe", "deceive")),
        "no_z": (course_data_dir / "gate_mode_values.jsonl", "no_z", ("exploit", "safe", "deceive")),
        "no_belief": (course_data_dir / "gate_mode_values.jsonl", "no_belief", ("exploit", "safe", "deceive")),
        "no_deceive": (course_data_dir / "gate_mode_values.jsonl", "none", ("exploit", "safe")),
        "no_course": (manual_data_dir / "gate_mode_values.jsonl", "none", ("exploit", "safe", "deceive")),
    }
    for name, (data, feature_ablation, allowed_modes) in models.items():
        output = args.output_dir / f"model_{name}"
        run_command(
            train_command(args, data, output, feature_ablation, allowed_modes),
            output / "train.log", args.gpu, args.force,
            output / "gate_value_policy.pth",
        )

    eval_dir = args.output_dir / "formal_e3"
    command = [
        str(args.python), "run_e3_gate_value_ablation.py",
        "--output-dir", str(eval_dir),
        "--python", str(args.python),
        "--base-model", str(args.base_model),
        "--opponent-style", "curriculum",
        "--curriculum-manifest", str(args.curriculum_manifest),
        "--curriculum-split", "holdout",
        "--belief-calibration-summary", str(args.belief_calibration_summary),
        "--belief-public-model", str(args.belief_public_model),
        "--mask-oracle-samples", str(args.mask_oracle_samples),
        "--mask-oracle-beta", str(args.mask_oracle_beta),
        "--mask-danger-threshold", str(args.mask_danger_threshold),
        "--full-model", str(args.output_dir / "model_full" / "gate_value_policy.pth"),
        "--no-z-model", str(args.output_dir / "model_no_z" / "gate_value_policy.pth"),
        "--no-belief-model", str(args.output_dir / "model_no_belief" / "gate_value_policy.pth"),
        "--no-deceive-model", str(args.output_dir / "model_no_deceive" / "gate_value_policy.pth"),
        "--no-course-model", str(args.output_dir / "model_no_course" / "gate_value_policy.pth"),
        "--variants", "L0", "L1", "L2_no_deceive", "L2_full", "L2_no_z",
        "L2_no_belief", "L2_no_risk_gate", "L2_no_course",
        "--seeds", *[str(seed) for seed in args.eval_seeds],
        "--games", str(args.games_per_seed),
        "--parallel-workers", str(args.parallel_workers),
        "--gpu", args.gpu,
        "--auto-hu", args.auto_hu,
    ]
    # If this aggregate stage is stale, every per-variant result must be
    # regenerated too; the inner E3 runner otherwise reuses summaries by path.
    command.append("--force")
    run_command(
        command,
        eval_dir / "formal_e3.log", args.gpu, args.force,
        eval_dir / "e3_ablation_summary.json",
    )
    summary = json.loads((eval_dir / "e3_ablation_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "e2_ladder_table": summary["e2_ladder_table"],
        "e3_ablation_table": summary["e3_ablation_table"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
