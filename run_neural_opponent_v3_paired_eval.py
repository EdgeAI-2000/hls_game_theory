"""Run full/no-deceive pilots against predicted-risk and risk-blind V3 opponents."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from experiment_trace import write_json


REPO_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_freeze_manifest(
    args: argparse.Namespace,
    training_manifest_path: Path,
) -> dict[str, Any] | None:
    if args.freeze_manifest is None:
        return None

    freeze_path = args.freeze_manifest.resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    evaluation = freeze["evaluation"]
    expected = {
        "outer_seeds": args.seeds,
        "games_per_seed": args.games,
        "risk_modes": args.risk_modes,
        "parallel_workers": args.parallel_workers,
    }
    for key, actual in expected.items():
        if evaluation.get(key) != actual:
            raise ValueError(
                f"Freeze mismatch for evaluation.{key}: "
                f"expected {evaluation.get(key)!r}, got {actual!r}"
            )

    expected_training_sha = freeze["training_manifest_sha256"]
    actual_training_sha = sha256(training_manifest_path)
    if actual_training_sha != expected_training_sha:
        raise ValueError(
            "Training manifest SHA mismatch: "
            f"expected {expected_training_sha}, got {actual_training_sha}"
        )

    for item in freeze["artifacts"]:
        path = (REPO_DIR / item["path"]).resolve()
        actual_sha = sha256(path)
        if actual_sha != item["sha256"]:
            raise ValueError(
                f"Artifact SHA mismatch for {path}: "
                f"expected {item['sha256']}, got {actual_sha}"
            )
    return {
        "path": str(freeze_path),
        "sha256": sha256(freeze_path),
        "status": "validated",
    }


def command_for(
    args: argparse.Namespace,
    checkpoint: Path,
    risk_mode: str,
    output_dir: Path,
) -> list[str]:
    return [
        str(args.python),
        "run_e3_gate_value_ablation.py",
        "--output-dir", str(output_dir),
        "--base-model", str(args.base_model),
        "--opponent-style", "neural",
        "--opponent-model", str(checkpoint),
        "--neural-opponent-risk-mode", risk_mode,
        "--belief-calibration-summary", str(args.belief_calibration_summary),
        "--belief-public-model", str(args.belief_public_model),
        "--full-model", str(args.full_model),
        "--no-belief-model", str(args.no_belief_model),
        "--no-deceive-model", str(args.no_deceive_model),
        "--variants", "L2_full", "L2_no_deceive",
        "--seeds", *(str(seed) for seed in args.seeds),
        "--games", str(args.games),
        "--parallel-workers", str(args.parallel_workers),
        "--gpu", args.gpu,
        "--auto-hu", "off",
    ]


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary["aggregate"]
    return {
        "aggregate": {
            variant: {
                key: aggregate[variant].get(key)
                for key in (
                    "games",
                    "avg_net",
                    "hu_rate",
                    "ron_game_rate",
                    "dealin_rate",
                    "cvar_10pct",
                    "FFR",
                    "FFR_counts",
                    "conditional_defensive_switch_rate",
                    "DIR_totals",
                    "mode_counts",
                )
            }
            for variant in ("L2_full", "L2_no_deceive")
        },
        "paired_net": summary.get("paired_net"),
        "paired_behavior": summary.get("paired_behavior"),
        "excess_DIR": summary.get("excess_DIR"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("py10/bin/python3"))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--risk-modes",
        nargs="+",
        choices=["predicted", "blind"],
        default=["predicted", "blind"],
    )
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument(
        "--allow-shared-gpu-workers",
        action="store_true",
        help=(
            "Explicitly allow multiple local-Qwen subprocesses on one GPU. "
            "Use only after a concurrent loading smoke test on the target GPU."
        ),
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        help="Validate frozen artifacts and evaluation settings before launching.",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--base-model", type=Path, default=Path("models/Qwen-Mahjong-V3-Merged"))
    parser.add_argument(
        "--belief-calibration-summary",
        type=Path,
        default=Path("H1_belief_results_v11_public_model_clustered_final/h1_summary.json"),
    )
    parser.add_argument(
        "--belief-public-model",
        type=Path,
        default=Path("Public_Belief_Model_v4_multifamily/public_belief_model.joblib"),
    )
    parser.add_argument(
        "--full-model",
        type=Path,
        default=Path("Formal_E2_E3_P3R_20260817/model_full/gate_value_policy.pth"),
    )
    parser.add_argument(
        "--no-belief-model",
        type=Path,
        default=Path("Formal_E2_E3_P3R_20260817/model_no_belief/gate_value_policy.pth"),
    )
    parser.add_argument(
        "--no-deceive-model",
        type=Path,
        default=Path("Formal_E2_E3_P3R_20260817/model_no_deceive/gate_value_policy.pth"),
    )
    args = parser.parse_args()

    if args.parallel_workers < 1:
        raise ValueError("parallel-workers must be positive")
    if args.parallel_workers != 1 and not args.allow_shared_gpu_workers:
        raise ValueError(
            "Multiple workers load one Qwen instance each on --gpu. Pass "
            "--allow-shared-gpu-workers only after validating GPU capacity."
        )

    args.training_dir = args.training_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_manifest_path = args.training_dir / "training_manifest.json"
    freeze_validation = validate_freeze_manifest(args, training_manifest_path)
    training_manifest = json.loads(
        training_manifest_path.read_text(encoding="utf-8")
    )
    if training_manifest.get("failures"):
        raise RuntimeError("Cannot evaluate a training manifest with failures")

    results = []
    for profile in training_manifest["profiles"]:
        name = str(profile["profile"])
        checkpoint = Path(profile["checkpoint"])
        actual_sha = sha256(checkpoint)
        if actual_sha != profile["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint SHA mismatch for {name}")
        for risk_mode in args.risk_modes:
            output = args.output_dir / risk_mode / name
            print(
                f"[neural-v3-eval] start profile={name} risk_mode={risk_mode}",
                flush=True,
            )
            subprocess.run(
                command_for(args, checkpoint, risk_mode, output),
                cwd=REPO_DIR,
                check=True,
            )
            summary_path = output / "e3_ablation_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            results.append(
                {
                    "profile": name,
                    "risk_mode": risk_mode,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": actual_sha,
                    "summary_path": str(summary_path),
                    "results": compact_summary(summary),
                }
            )
            print(
                f"[neural-v3-eval] done profile={name} risk_mode={risk_mode}",
                flush=True,
            )

    write_json(
        args.output_dir / "paired_eval_manifest.json",
        {
            "status": "complete",
            "interpretation": (
                "Supplementary mechanism pilot. predicted vs blind changes only the V3 "
                "opponent action head's risk condition."
            ),
            "seeds": args.seeds,
            "risk_modes": args.risk_modes,
            "games_per_seed_variant_profile_risk_mode": args.games,
            "parallel_workers": args.parallel_workers,
            "shared_gpu_workers_explicitly_allowed": args.allow_shared_gpu_workers,
            "freeze_manifest": freeze_validation,
            "profiles": results,
        },
    )


if __name__ == "__main__":
    main()
