"""Run the formal E5 gate-aggressiveness and deception-resistance matrix."""

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
from rerun_continuous_gate_neural_llm import aggregate, step_funnel
from run_e3_gate_value_ablation import (
    audit_game_seed_schedule,
    belief_shift_summary,
    completed_run,
    deployed_belief_summary,
    enrich_aggregate,
)


REPO_DIR = Path(__file__).resolve().parent
METHOD = "llm_mask"
SETTINGS = {
    "no_deceive": {
        "checkpoint": "no_deceive",
        "safe_margin": 0.05,
        "deceive_margin": 0.05,
        "risk_ablation": "off",
    },
    "tight": {
        "checkpoint": "full",
        "safe_margin": 0.05,
        "deceive_margin": 0.15,
        "risk_ablation": "off",
    },
    "default": {
        "checkpoint": "full",
        "safe_margin": 0.05,
        "deceive_margin": 0.05,
        "risk_ablation": "off",
    },
    "loose": {
        "checkpoint": "full",
        "safe_margin": 0.05,
        "deceive_margin": 0.0,
        "risk_ablation": "off",
    },
    "constant_deceive": {
        "checkpoint": "full",
        "safe_margin": 0.05,
        "deceive_margin": 0.05,
        "risk_ablation": "constant_deceive",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_for(args: argparse.Namespace, suite: str, setting: str, seed: int, output: Path) -> list[str]:
    config = SETTINGS[setting]
    checkpoint = args.no_deceive_model if config["checkpoint"] == "no_deceive" else args.full_model
    command = [
        str(args.python),
        "run_gate1_experiments.py",
        "--methods", METHOD,
        "--games", str(args.games),
        "--seed", str(seed),
        "--opponent-style", "curriculum",
        "--curriculum-manifest", str(args.curriculum_manifest),
        "--curriculum-split", "holdout",
        "--danger-threshold", "1",
        "--ffr-hand-shanten", "1",
        "--sample-every", "0",
        "--auto-hu", args.auto_hu,
        "--snapshot-oracle-samples", "1",
        "--snapshot-crn-seeds", "1",
        "--backend", "local_qwen",
        "--model-path", str(args.base_model),
        "--max-new-tokens", "64",
        "--temperature", "0.0",
        "--mask-oracle-samples", str(args.mask_oracle_samples),
        "--mask-oracle-beta", str(args.mask_oracle_beta),
        "--mask-danger-threshold", str(args.mask_danger_threshold),
        "--belief-calibration-summary", str(args.belief_calibration_summary),
        "--belief-public-model", str(args.belief_public_model),
        "--mask-gate-policy", "neural_value_gate",
        "--gate-value-model-path", str(checkpoint),
        "--gate-value-safe-margin", str(config["safe_margin"]),
        "--gate-value-deceive-margin", str(config["deceive_margin"]),
        "--neural-gate-device", "cpu",
        "--mask-deceive-style", "threat",
        "--mask-threat-max-result-shanten", "0",
        "--mask-threat-max-shanten-regret", "0",
        "--mask-threat-min-ukeire-ratio", "1.0",
        "--defender-threat-model", "blend",
        "--output-dir", str(output),
    ]
    if suite == "anti_deception":
        command.extend(["--curriculum-spec-id", args.anti_deception_spec])
    if config["risk_ablation"] != "off":
        command.extend(["--mask-risk-gate-ablation", str(config["risk_ablation"])])
    return command


def run_one(args: argparse.Namespace, suite: str, setting: str, seed: int) -> tuple[str, str, int, dict[str, Any]]:
    output = args.output_dir / suite / setting / f"seed_{seed}"
    if args.force or not completed_run(output, METHOD, args.games):
        output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.setdefault("PYTHONHASHSEED", "0")
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"[E5] start suite={suite} setting={setting} seed={seed}", flush=True)
        with (output / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command_for(args, suite, setting, seed, output),
                cwd=REPO_DIR,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not completed_run(output, METHOD, args.games):
            raise RuntimeError(f"incomplete E5 output: {suite}/{setting}/seed_{seed}")
    summary = json.loads((output / "gate1_summary.json").read_text(encoding="utf-8"))
    row = summary["E2_ladder"][METHOD]
    row["_funnel"] = step_funnel(output)
    row["_belief"] = deployed_belief_summary(output)
    row["_belief_shift"] = belief_shift_summary(output)
    print(
        f"[E5] done suite={suite} setting={setting} seed={seed} net={row['avg_net']:.3f}",
        flush=True,
    )
    return suite, setting, seed, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("py10/bin/python3"))
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--curriculum-manifest", type=Path, required=True)
    parser.add_argument("--belief-calibration-summary", type=Path, required=True)
    parser.add_argument("--belief-public-model", type=Path, required=True)
    parser.add_argument("--full-model", type=Path, required=True)
    parser.add_argument("--no-deceive-model", type=Path, required=True)
    parser.add_argument("--anti-deception-spec", default="beta_b2_NN_4")
    parser.add_argument("--suites", nargs="+", choices=["holdout", "anti_deception"], required=True)
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--mask-oracle-samples", type=int, default=60)
    parser.add_argument("--mask-oracle-beta", type=float, default=2.0)
    parser.add_argument("--mask-danger-threshold", type=int, default=0)
    parser.add_argument("--auto-hu", choices=["common", "off"], default="off")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_game_seed_schedule(args.seeds, args.games)
    write_json(args.output_dir / "game_seed_audit.json", audit)
    manifest = json.loads(args.curriculum_manifest.read_text(encoding="utf-8"))
    anti_specs = [
        item
        for item in manifest.get("opponents", [])
        if item.get("id") == args.anti_deception_spec
    ]
    if "anti_deception" in args.suites:
        if len(anti_specs) != 1:
            raise ValueError(
                f"expected one anti-deception spec {args.anti_deception_spec!r}, "
                f"found {len(anti_specs)}"
            )
        if anti_specs[0].get("split") != "holdout":
            raise ValueError("anti-deception stress spec must remain in the frozen holdout split")
    inputs = {
        "base_model": str(args.base_model.resolve()),
        "curriculum_manifest": str(args.curriculum_manifest.resolve()),
        "curriculum_sha256": sha256(args.curriculum_manifest),
        "belief_calibration_summary": str(args.belief_calibration_summary.resolve()),
        "belief_calibration_sha256": sha256(args.belief_calibration_summary),
        "belief_public_model": str(args.belief_public_model.resolve()),
        "belief_public_model_sha256": sha256(args.belief_public_model),
        "full_model": str(args.full_model.resolve()),
        "full_model_sha256": sha256(args.full_model),
        "no_deceive_model": str(args.no_deceive_model.resolve()),
        "no_deceive_model_sha256": sha256(args.no_deceive_model),
    }
    protocol = {
        "suites": args.suites,
        "settings": {name: SETTINGS[name] for name in args.settings},
        "anti_deception_spec": args.anti_deception_spec,
        "anti_deception_spec_snapshot": anti_specs[0] if anti_specs else None,
        "seeds": args.seeds,
        "games_per_cell": args.games,
        "parallel_workers": args.parallel_workers,
        "game_seed_audit": audit,
        "inputs": inputs,
        "interpretation": (
            "This is an empirical GateValue intervention-margin sweep. It changes deployment-time "
            "deceive selectivity, not a theoretical exploitability or calibrated rho guarantee."
        ),
    }
    write_json(args.output_dir / "protocol.json", protocol)

    jobs = [
        (suite, setting, seed)
        for suite in args.suites
        for setting in args.settings
        for seed in args.seeds
    ]
    completed: list[tuple[str, str, int, dict[str, Any]]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel_workers)) as pool:
        futures = {
            pool.submit(run_one, args, suite, setting, seed): (suite, setting, seed)
            for suite, setting, seed in jobs
        }
        for future in as_completed(futures):
            suite, setting, seed = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append(f"{suite}/{setting}/seed_{seed}: {exc}")
                print(f"[E5] failed {failures[-1]}", flush=True)
    if failures:
        write_json(args.output_dir / "failures.json", failures)
        raise RuntimeError(f"{len(failures)} E5 cells failed")

    rows = {(suite, setting, seed): row for suite, setting, seed, row in completed}
    result = {
        "protocol": protocol,
        "aggregate": {
            suite: {
                setting: enrich_aggregate(
                    aggregate([rows[(suite, setting, seed)] for seed in args.seeds]),
                    [rows[(suite, setting, seed)] for seed in args.seeds],
                )
                for setting in args.settings
            }
            for suite in args.suites
        },
        "per_seed": {
            suite: {
                setting: [rows[(suite, setting, seed)] for seed in args.seeds]
                for setting in args.settings
            }
            for suite in args.suites
        },
    }
    write_json(args.output_dir / "e5_run_summary.json", result)
    print(f"[E5] complete: {args.output_dir / 'e5_run_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
