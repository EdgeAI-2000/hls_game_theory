"""Generate and train risk-aware neural opponents without engineered tell inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from experiment_trace import write_json
from neural_opponent_policy import FEATURE_VERSION_V3


REPO_DIR = Path(__file__).resolve().parent
PROFILES = {
    "aggressive": {
        "threat_fold_threshold": 0.75,
        "ffr_hand_shanten": 0,
        "p0_mask_rate": 0.35,
        "response_loss_weight": 1.00,
        "data_seed": 2026082911,
        "train_seed": 2026083011,
    },
    "balanced": {
        "threat_fold_threshold": 0.50,
        "ffr_hand_shanten": 1,
        "p0_mask_rate": 0.50,
        "response_loss_weight": 1.00,
        "data_seed": 2026082921,
        "train_seed": 2026083021,
    },
    "cautious": {
        "threat_fold_threshold": 0.30,
        "ffr_hand_shanten": 1,
        "p0_mask_rate": 0.65,
        "response_loss_weight": 1.50,
        "data_seed": 2026082931,
        "train_seed": 2026083031,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=REPO_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def train_profile(
    args: argparse.Namespace, name: str, profile: dict[str, Any]
) -> dict[str, Any]:
    profile_dir = args.output_dir / name
    data_path = profile_dir / "teacher_data.jsonl"
    model_dir = profile_dir / "model"
    profile_dir.mkdir(parents=True, exist_ok=True)

    collect_command = [
        str(args.python),
        "generate_neural_opponent_data.py",
        "--games", str(args.games),
        "--seed", str(profile["data_seed"]),
        "--max-steps", "300",
        "--output-file", str(data_path),
        "--feature-version", FEATURE_VERSION_V3,
        "--risk-target-max-shanten", str(args.risk_target_max_shanten),
        "--threat-fold-threshold", str(profile["threat_fold_threshold"]),
        "--ffr-hand-shanten", str(profile["ffr_hand_shanten"]),
        "--defender-tell-weight", "0",
        "--p0-mask-rate", str(profile["p0_mask_rate"]),
        "--log-every", str(max(25, args.games // 10)),
    ]
    run_logged(collect_command, profile_dir / "collect.log")

    train_command = [
        str(args.python),
        "train_neural_opponent.py",
        "--data-file", str(data_path),
        "--output-dir", str(model_dir),
        "--architecture", "risk_aware_multitask",
        "--profile", name,
        "--risk-loss-weight", str(args.risk_loss_weight),
        "--response-loss-weight", str(profile["response_loss_weight"]),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--seed", str(profile["train_seed"]),
    ]
    run_logged(train_command, profile_dir / "train.log")

    checkpoint = model_dir / "neural_opponent_policy.pth"
    data_summary = json.loads(
        data_path.with_suffix(".summary.json").read_text(encoding="utf-8")
    )
    training_summary = json.loads(
        (model_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    return {
        "profile": name,
        "profile_config": profile,
        "data_summary": data_summary,
        "training_summary": training_summary,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("py10/bin/python3"))
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parallel-workers", type=int, default=3)
    parser.add_argument("--risk-target-max-shanten", type=int, default=0)
    parser.add_argument("--risk-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "protocol.json",
        {
            "feature_version": FEATURE_VERSION_V3,
            "architecture": "risk_aware_multitask",
            "engineered_p0_tell_features": False,
            "runtime_hidden_p0_hand": False,
            "training_only_risk_label": (
                f"p0_oracle_shanten<={args.risk_target_max_shanten}"
            ),
            "action_condition": "predicted_public_p0_risk_probability",
            "risk_blind_ablation": "replace action condition with training prevalence",
            "teacher_threat_model": "learned",
            "teacher_tell_weight": 0.0,
            "teacher_independence_limitation": (
                "Action targets remain distilled from ResponsiveDefender; this V3 validates "
                "the risk-prediction/response mechanism but is not independent self-play RL."
            ),
            "games_per_profile": args.games,
            "epochs": args.epochs,
            "risk_loss_weight": args.risk_loss_weight,
            "profiles": PROFILES,
        },
    )

    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel_workers)) as pool:
        futures = {
            pool.submit(train_profile, args, name, profile): name
            for name, profile in PROFILES.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                results.append(result)
                best = next(
                    item
                    for item in result["training_summary"]["history"]
                    if int(item["epoch"])
                    == int(result["training_summary"]["best_epoch"])
                )
                print(
                    f"[neural-v3] trained {name}: rows={result['data_summary']['rows']} "
                    f"action_acc={best['legal_masked_acc']:.4f} "
                    f"risk_auc={best['risk_auc']:.4f}",
                    flush=True,
                )
            except Exception as exc:
                failures.append({"profile": name, "error": repr(exc)})
                print(f"[neural-v3] failed {name}: {exc}", flush=True)

    summary = {
        "status": "complete" if not failures else "failed",
        "profiles": sorted(results, key=lambda item: item["profile"]),
        "failures": failures,
    }
    write_json(args.output_dir / "training_manifest.json", summary)
    if failures:
        raise RuntimeError(f"{len(failures)} neural opponent profile(s) failed")


if __name__ == "__main__":
    main()
