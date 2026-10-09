"""Evaluate causal GateValue ablations on paired seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

from experiment_trace import summarize_belief_samples, write_json
from rerun_continuous_gate_neural_llm import aggregate, compact_row, step_funnel
from rerun_v2_e2_ladder_3seeds import paired_stats
from run_gate1_experiments import GAME_SEED_STRIDE, derive_game_seed


REPO_DIR = Path(__file__).resolve().parent
METHOD = {
    "L0": "llm_base",
    "L1": "llm_reactive_z",
    "L1_z_bphi": "llm_reactive_z_bphi",
    "L2_full": "llm_mask",
    "L2_no_z": "llm_mask",
    "L2_no_belief": "llm_mask",
    "L2_no_deceive": "llm_mask",
    "L2_no_risk_gate": "llm_mask",
    "L2_no_course": "llm_mask",
    "L2_candidate": "llm_mask",
    "L2_candidate_no_belief": "llm_mask",
    "L2_candidate_no_deceive": "llm_mask",
    "L2_candidate_no_risk_gate": "llm_mask",
    "L2_candidate_unweighted": "llm_mask",
    "L2_candidate_no_belief_matched": "llm_mask",
}
DEFAULT_VARIANTS = [
    "L1", "L2_full", "L2_no_belief", "L2_no_deceive", "L2_no_risk_gate",
]


def audit_game_seed_schedule(outer_seeds: list[int], games: int) -> Dict[str, Any]:
    if games <= 0:
        raise ValueError(f"games must be positive, got {games}")
    if games > GAME_SEED_STRIDE:
        raise ValueError(
            f"games={games} exceeds collision-free stride {GAME_SEED_STRIDE}"
        )
    if len(set(outer_seeds)) != len(outer_seeds):
        raise ValueError("outer seeds must be unique")

    generated = [
        derive_game_seed(outer_seed, game_index)
        for outer_seed in outer_seeds
        for game_index in range(games)
    ]
    unique_count = len(set(generated))
    if unique_count != len(generated):
        raise ValueError(
            f"game seed collision: {unique_count} unique of {len(generated)}"
        )
    return {
        "scheme": "outer_seed * 1000000 + game_index",
        "outer_seeds": outer_seeds,
        "outer_seed_count": len(outer_seeds),
        "games_per_outer_seed": games,
        "total_game_seeds": len(generated),
        "unique_game_seeds": unique_count,
        "collision_count": 0,
        "ranges": [
            {
                "outer_seed": outer_seed,
                "first": derive_game_seed(outer_seed, 0),
                "last": derive_game_seed(outer_seed, games - 1),
            }
            for outer_seed in outer_seeds
        ],
    }


def gate_arguments(args: argparse.Namespace, variant: str) -> list[str]:
    if variant in {"L2_no_risk_gate", "L2_candidate_no_risk_gate"}:
        model_path = (
            args.full_model
            if variant == "L2_no_risk_gate"
            else args.candidate_model
        )
        return [
            "--mask-gate-policy", "neural_value_gate",
            "--gate-value-model-path", str(model_path),
            "--gate-value-safe-margin", str(args.safe_margin),
            "--gate-value-deceive-margin", str(args.deceive_margin),
            "--neural-gate-device", "cpu",
            "--mask-risk-gate-ablation", "constant_deceive",
            "--mask-deceive-style", "threat",
            "--mask-threat-max-result-shanten", "0",
            "--mask-threat-max-shanten-regret", "0",
            "--mask-threat-min-ukeire-ratio", "1.0",
        ]
    model_paths = {
        "L2_full": args.full_model,
        "L2_no_z": args.no_z_model,
        "L2_no_belief": args.no_belief_model,
        "L2_no_deceive": args.no_deceive_model,
        "L2_no_course": args.no_course_model,
        "L2_candidate": args.candidate_model,
        "L2_candidate_no_belief": args.candidate_no_belief_model,
        "L2_candidate_no_deceive": args.candidate_no_deceive_model,
        "L2_candidate_unweighted": args.candidate_unweighted_model,
        "L2_candidate_no_belief_matched": args.candidate_no_belief_matched_model,
    }
    if variant not in model_paths:
        return []
    return [
        "--mask-gate-policy", "neural_value_gate",
        "--gate-value-model-path", str(model_paths[variant]),
        "--gate-value-safe-margin", str(args.safe_margin),
        "--gate-value-deceive-margin", str(args.deceive_margin),
        "--neural-gate-device", "cpu",
        "--mask-deceive-style", "threat",
        "--mask-threat-max-result-shanten", "0",
        "--mask-threat-max-shanten-regret", "0",
        "--mask-threat-min-ukeire-ratio", "1.0",
    ]


def command_for(
    args: argparse.Namespace, variant: str, seed: int, output: Path
) -> list[str]:
    command = [
        str(args.python),
        "run_gate1_experiments.py",
        "--methods", METHOD[variant],
        "--games", str(args.games),
        "--seed", str(seed),
        "--opponent-style", args.opponent_style,
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
        "--belief-public-model", str(args.belief_public_model),
        "--output-dir", str(output),
    ]
    if args.belief_calibration_summary:
        command.extend([
            "--belief-calibration-summary", str(args.belief_calibration_summary),
        ])
    if args.opponent_style == "neural":
        command.extend([
            "--neural-opponent-model-path", str(args.opponent_model),
            "--neural-opponent-device", "cpu",
            "--neural-opponent-risk-mode", args.neural_opponent_risk_mode,
        ])
    elif args.opponent_style == "curriculum":
        command.extend([
            "--curriculum-manifest", str(args.curriculum_manifest),
            "--curriculum-split", args.curriculum_split,
            "--defender-threat-model", "blend",
        ])
    command.extend(gate_arguments(args, variant))
    return command


def deployed_belief_summary(output: Path) -> Dict[str, Any]:
    """Score the B_phi values actually consumed by the gate at decision time."""
    path = output / "gate1_steps.jsonl"
    samples = []
    raw_samples = []
    mc_samples = []
    if not path.is_file():
        return {"samples": 0, "note": "gate1_steps.jsonl missing"}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = int(bool(row.get("true_tenpai")))
            for opponent_id, belief in (row.get("belief_json") or {}).items():
                if not isinstance(belief, dict) or "tenpai_confidence" not in belief:
                    continue
                score = float(belief["tenpai_confidence"])
                samples.append({
                    "true_tenpai": label,
                    "tenpai_confidence": score,
                    "think_i_am_tenpai": "yes" if score >= 0.5 else "no",
                    "backend": "deployed_gate_bphi",
                    "target_opponent": opponent_id,
                })
                raw_value = belief.get(
                    "uncalibrated_tenpai_confidence",
                    belief.get("raw_tenpai_confidence"),
                )
                if raw_value is not None:
                    raw_score = float(raw_value)
                    raw_samples.append({
                        "true_tenpai": label,
                        "tenpai_confidence": raw_score,
                        "think_i_am_tenpai": "yes" if raw_score >= 0.5 else "no",
                        "backend": "deployed_gate_bphi_raw",
                        "target_opponent": opponent_id,
                    })
                if "raw_tenpai_confidence" in belief:
                    mc_score = float(belief["raw_tenpai_confidence"])
                    mc_samples.append({
                        "true_tenpai": label,
                        "tenpai_confidence": mc_score,
                        "think_i_am_tenpai": "yes" if mc_score >= 0.5 else "no",
                        "backend": "deployed_public_mc_diagnostic",
                        "target_opponent": opponent_id,
                    })
    result = summarize_belief_samples(samples)
    result["label_scope"] = "P0 true tenpai; scores are the deployed public-only B_phi gate inputs"
    result["raw"] = summarize_belief_samples(raw_samples)
    result["mc_diagnostic"] = summarize_belief_samples(mc_samples)
    return result


def belief_shift_summary(output: Path) -> Dict[str, Any]:
    path = output / "gate1_steps.jsonl"
    by_mode: Dict[str, list[float]] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                values = [float(value) for value in (row.get("threat_delta") or {}).values()]
                if values:
                    by_mode.setdefault(str(row.get("mode", "none")), []).append(sum(values) / len(values))
    return {
        mode: {
            "samples": len(values),
            "mean_delta": sum(values) / len(values) if values else None,
            "positive_rate": sum(value > 0 for value in values) / len(values) if values else None,
        }
        for mode, values in sorted(by_mode.items())
    }


def completed_run(output: Path, method: str, expected_games: int) -> bool:
    """Require both structured outputs and the post-write success marker."""
    summary_path = output / "gate1_summary.json"
    games_path = output / "gate1_games.jsonl"
    log_path = output / "run.log"
    if not all(path.is_file() for path in (summary_path, games_path, log_path)):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        reported_games = int(summary["E2_ladder"][method]["games"])
        with games_path.open("rb") as handle:
            actual_games = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 20), b""))
        with log_path.open("rb") as handle:
            handle.seek(max(0, log_path.stat().st_size - 8192))
            success_marker = b"Saved Gate1 outputs under:" in handle.read()
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return reported_games == expected_games == actual_games and success_marker


def enrich_aggregate(base: Dict[str, Any], rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    belief_rows = [row.get("_belief") or {} for row in rows]
    total_samples = sum(int(item.get("samples", 0)) for item in belief_rows)

    def weighted(field: str) -> float | None:
        values = [
            (int(item.get("samples", 0)), item.get(field))
            for item in belief_rows
            if int(item.get("samples", 0)) > 0 and item.get(field) is not None
        ]
        denominator = sum(count for count, _ in values)
        return (
            sum(count * float(value) for count, value in values) / denominator
            if denominator else None
        )

    shift: Dict[str, Dict[str, Any]] = {}
    modes = {
        mode
        for row in rows
        for mode in (row.get("_belief_shift") or {}).keys()
    }
    for mode in sorted(modes):
        values = [
            (row.get("_belief_shift") or {}).get(mode) or {}
            for row in rows
        ]
        count = sum(int(item.get("samples", 0)) for item in values)
        shift[mode] = {
            "samples": count,
            "mean_delta": (
                sum(int(item.get("samples", 0)) * float(item.get("mean_delta", 0.0)) for item in values) / count
                if count else None
            ),
            "positive_rate": (
                sum(int(item.get("samples", 0)) * float(item.get("positive_rate", 0.0)) for item in values) / count
                if count else None
            ),
        }
    base["B_phi_calibration"] = {
        "samples": total_samples,
        "auc": weighted("auc"),
        "brier": weighted("brier"),
        "bse_abs": weighted("bse_abs"),
        "ece": weighted("ece"),
    }
    raw_beliefs = [item.get("raw") or {} for item in belief_rows]

    def weighted_raw(field: str) -> float | None:
        values = [
            (int(item.get("samples", 0)), item.get(field))
            for item in raw_beliefs
            if int(item.get("samples", 0)) > 0 and item.get(field) is not None
        ]
        denominator = sum(count for count, _ in values)
        return (
            sum(count * float(value) for count, value in values) / denominator
            if denominator else None
        )

    base["B_phi_raw"] = {
        "samples": sum(int(item.get("samples", 0)) for item in raw_beliefs),
        "auc": weighted_raw("auc"),
        "brier": weighted_raw("brier"),
        "bse_abs": weighted_raw("bse_abs"),
        "ece": weighted_raw("ece"),
    }
    base["belief_shift_by_mode"] = shift
    base["DIR_window_diagnostic"] = (
        base["DIR_totals"]["induced_dealin"] / base["DIR_totals"]["deceive_windows"]
        if base["DIR_totals"]["deceive_windows"] else 0.0
    )
    return base


def run_one(
    args: argparse.Namespace, variant: str, seed: int
) -> tuple[str, int, Dict[str, Any]]:
    reuse_root = args.reuse_variant_roots.get(variant)
    output = (reuse_root if reuse_root is not None else args.output_dir / variant) / f"seed_{seed}"
    summary_path = output / "gate1_summary.json"
    run_complete = completed_run(output, METHOD[variant], args.games)
    if reuse_root is not None and not run_complete:
        raise FileNotFoundError(
            f"Missing or incomplete reused result for {variant} seed {seed}: {output}"
        )
    if reuse_root is None and (args.force or not run_complete):
        output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.setdefault("PYTHONHASHSEED", "0")
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"[E3] start variant={variant} seed={seed}", flush=True)
        with (output / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command_for(args, variant, seed, output),
                cwd=REPO_DIR,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not completed_run(output, METHOD[variant], args.games):
            raise RuntimeError(f"Run exited without complete outputs: {variant} seed {seed}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = summary["E2_ladder"][METHOD[variant]]
    row["_funnel"] = step_funnel(output)
    row["_belief"] = deployed_belief_summary(output)
    row["_belief_shift"] = belief_shift_summary(output)
    print(f"[E3] done variant={variant} seed={seed} net={row['avg_net']:.3f}", flush=True)
    return variant, seed, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("py10/bin/python3"))
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--opponent-model", type=Path, default=Path("Neural_opponent_model/neural_opponent_policy.pth"))
    parser.add_argument("--opponent-style", choices=["neural", "responsive", "curriculum"], default="curriculum")
    parser.add_argument(
        "--neural-opponent-risk-mode",
        choices=["predicted", "blind"],
        default="predicted",
    )
    parser.add_argument("--curriculum-manifest", type=Path)
    parser.add_argument("--curriculum-split", choices=["train", "holdout", "all"], default="holdout")
    parser.add_argument("--belief-calibration-summary", type=Path)
    parser.add_argument("--belief-public-model", type=Path, required=True)
    parser.add_argument("--mask-oracle-samples", type=int, default=60)
    parser.add_argument("--mask-oracle-beta", type=float, default=2.0)
    parser.add_argument("--mask-danger-threshold", type=int, default=0)
    parser.add_argument("--full-model", type=Path, required=True)
    parser.add_argument("--no-z-model", type=Path)
    parser.add_argument("--no-belief-model", type=Path, required=True)
    parser.add_argument("--no-deceive-model", type=Path, required=True)
    parser.add_argument("--no-course-model", type=Path)
    parser.add_argument("--candidate-model", type=Path)
    parser.add_argument("--candidate-no-belief-model", type=Path)
    parser.add_argument("--candidate-no-deceive-model", type=Path)
    parser.add_argument("--candidate-unweighted-model", type=Path)
    parser.add_argument("--candidate-no-belief-matched-model", type=Path)
    parser.add_argument("--variants", nargs="+", choices=list(METHOD), default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--auto-hu", choices=["common", "off"], default="off")
    parser.add_argument("--safe-margin", type=float, default=0.05)
    parser.add_argument("--deceive-margin", type=float, default=0.05)
    parser.add_argument(
        "--reuse-variant-root",
        action="append",
        default=[],
        metavar="VARIANT=PATH",
        help="Read an existing variant's seed_* results without rerunning it.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    game_seed_audit = audit_game_seed_schedule(args.seeds, args.games)
    write_json(args.output_dir / "game_seed_audit.json", game_seed_audit)
    if "L2_no_z" in args.variants and args.no_z_model is None:
        parser.error("L2_no_z requires --no-z-model")
    if "L2_candidate" in args.variants and args.candidate_model is None:
        parser.error("L2_candidate requires --candidate-model")
    if "L2_candidate_no_risk_gate" in args.variants and args.candidate_model is None:
        parser.error("L2_candidate_no_risk_gate requires --candidate-model")
    if "L2_candidate_no_belief" in args.variants and args.candidate_no_belief_model is None:
        parser.error("L2_candidate_no_belief requires --candidate-no-belief-model")
    if "L2_candidate_no_deceive" in args.variants and args.candidate_no_deceive_model is None:
        parser.error("L2_candidate_no_deceive requires --candidate-no-deceive-model")
    if "L2_candidate_unweighted" in args.variants and args.candidate_unweighted_model is None:
        parser.error("L2_candidate_unweighted requires --candidate-unweighted-model")
    if "L2_candidate_no_belief_matched" in args.variants and args.candidate_no_belief_matched_model is None:
        parser.error("L2_candidate_no_belief_matched requires --candidate-no-belief-matched-model")
    if "L2_no_course" in args.variants and args.no_course_model is None:
        parser.error("L2_no_course requires --no-course-model")
    if args.opponent_style == "curriculum" and args.curriculum_manifest is None:
        parser.error("--opponent-style curriculum requires --curriculum-manifest")
    args.reuse_variant_roots = {}
    for item in args.reuse_variant_root:
        if "=" not in item:
            parser.error(f"--reuse-variant-root must be VARIANT=PATH, got: {item}")
        variant, raw_path = item.split("=", 1)
        if variant not in METHOD:
            parser.error(f"Unknown reused variant: {variant}")
        if variant in args.reuse_variant_roots:
            parser.error(f"Duplicate reused variant: {variant}")
        args.reuse_variant_roots[variant] = Path(raw_path).resolve()
    jobs = [(variant, seed) for variant in args.variants for seed in args.seeds]
    completed = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel_workers)) as pool:
        futures = [pool.submit(run_one, args, variant, seed) for variant, seed in jobs]
        for future in as_completed(futures):
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append(exc)
                print(f"[E3] worker failed: {exc}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} E3 worker(s) failed; see per-seed run.log files")

    rows = {(variant, seed): row for variant, seed, row in completed}
    per_variant = {
        variant: [rows[(variant, seed)] for seed in args.seeds]
        for variant in args.variants
    }
    result = {
        "config": {
            "seeds": args.seeds,
            "games_per_seed_per_variant": args.games,
            "game_seed_audit": game_seed_audit,
            "variants": args.variants,
            "auto_hu": args.auto_hu,
            "base_model": str(args.base_model),
            "opponent_model": str(args.opponent_model),
            "belief_calibration_summary": str(args.belief_calibration_summary or ""),
            "belief_public_model": str(args.belief_public_model),
            "belief_estimator_config": {
                "oracle_samples": args.mask_oracle_samples,
                "oracle_beta": args.mask_oracle_beta,
                "danger_threshold": args.mask_danger_threshold,
            },
            "models": {
                "L2_full": str(args.full_model),
                "L2_no_z": str(args.no_z_model or ""),
                "L2_no_belief": str(args.no_belief_model),
                "L2_no_deceive": str(args.no_deceive_model),
                "L2_no_course": str(args.no_course_model or ""),
                "L2_candidate": str(args.candidate_model or ""),
                "L2_candidate_no_belief": str(args.candidate_no_belief_model or ""),
                "L2_candidate_no_deceive": str(args.candidate_no_deceive_model or ""),
                "L2_candidate_unweighted": str(args.candidate_unweighted_model or ""),
                "L2_candidate_no_belief_matched": str(args.candidate_no_belief_matched_model or ""),
            },
            "ablation_definitions": {
                "L2_no_z": "Same course data and training settings; z_j feature block and runtime tracker input removed.",
                "L2_no_belief": "Same course data and training settings; B_phi feature block removed.",
                "L2_no_deceive": "Same course data and training settings; deceive mode removed.",
                "L2_no_course": "Same architecture and training settings; trained on hand-authored mixed bots instead of C_LLM train split.",
                "L2_candidate_no_risk_gate": (
                    "Keep the L2_candidate B_phi, candidate generator, legality, "
                    "shanten and ukeire guards; select deceive whenever it is an "
                    "available behaviorally distinct mode, otherwise exploit; "
                    "never use the public-risk safe fallback."
                ),
            },
            "reused_variant_roots": {
                variant: str(path)
                for variant, path in args.reuse_variant_roots.items()
            },
        },
        "aggregate": {
            variant: enrich_aggregate(aggregate(variant_rows), variant_rows)
            for variant, variant_rows in per_variant.items()
        },
        "per_seed": {
            variant: [
                compact_row(seed, row)
                for seed, row in zip(args.seeds, variant_rows)
            ]
            for variant, variant_rows in per_variant.items()
        },
    }
    if "L2_full" in per_variant:
        result["paired_net"] = {
            f"L2_full_vs_{variant}": paired_stats(
                [float(row["avg_net"]) for row in per_variant["L2_full"]],
                [float(row["avg_net"]) for row in per_variant[variant]],
            )
            for variant in args.variants
            if variant != "L2_full"
        }
        result["paired_behavior"] = {
            f"L2_full_vs_{variant}": {
                "dealin_rate": paired_stats(
                    [float(row["dealin_rate"]) for row in per_variant["L2_full"]],
                    [float(row["dealin_rate"]) for row in per_variant[variant]],
                ),
                "discard_hu_rate": paired_stats(
                    [float(row.get("discard_hu_rate", 0.0)) for row in per_variant["L2_full"]],
                    [float(row.get("discard_hu_rate", 0.0)) for row in per_variant[variant]],
                ),
                "ron_game_rate": paired_stats(
                    [float(row.get("ron_game_rate", row.get("discard_hu_rate", 0.0))) for row in per_variant["L2_full"]],
                    [float(row.get("ron_game_rate", row.get("discard_hu_rate", 0.0))) for row in per_variant[variant]],
                ),
                "FFR": paired_stats(
                    [float(row.get("FFR", 0.0)) for row in per_variant["L2_full"]],
                    [float(row.get("FFR", 0.0)) for row in per_variant[variant]],
                ),
                "conditional_defensive_switch_rate": paired_stats(
                    [float(row.get("conditional_defensive_switch_rate", 0.0)) for row in per_variant["L2_full"]],
                    [float(row.get("conditional_defensive_switch_rate", 0.0)) for row in per_variant[variant]],
                ),
                "cvar_10pct": paired_stats(
                    [float(row.get("cvar_10pct", 0.0)) for row in per_variant["L2_full"]],
                    [float(row.get("cvar_10pct", 0.0)) for row in per_variant[variant]],
                ),
            }
            for variant in args.variants
            if variant != "L2_full"
        }
        if "L2_no_deceive" in per_variant:
            result["excess_DIR"] = {
                "definition": (
                    "Paired excess ron-game rate: MASK-full minus no-deceive on "
                    "identical seeds and holdout opponents. This is the formal baseline-adjusted "
                    "DIR; window-attributed DIR remains diagnostic only."
                ),
                "paired": paired_stats(
                    [float(row.get("discard_hu_rate", 0.0)) for row in per_variant["L2_full"]],
                    [float(row.get("discard_hu_rate", 0.0)) for row in per_variant["L2_no_deceive"]],
                ),
            }
    if "L2_candidate" in per_variant:
        result["paired_candidate_net"] = {
            f"L2_candidate_vs_{variant}": paired_stats(
                [float(row["avg_net"]) for row in per_variant["L2_candidate"]],
                [float(row["avg_net"]) for row in per_variant[variant]],
            )
            for variant in args.variants
            if variant != "L2_candidate"
        }
    no_deceive_discard_rate = (
        float(result["aggregate"]["L2_no_deceive"].get("discard_hu_rate", 0.0))
        if "L2_no_deceive" in result["aggregate"] else None
    )
    result["paper_table"] = []
    for variant in args.variants:
        values = result["aggregate"][variant]
        calibration = values.get("B_phi_calibration") or {}
        result["paper_table"].append({
            "variant": variant,
            "S_bar": values.get("avg_net"),
            "DR": values.get("dealin_rate"),
            "ron_game_rate": values.get("ron_game_rate", values.get("discard_hu_rate")),
            "excess_DIR_per_game": (
                float(values.get("discard_hu_rate", 0.0)) - no_deceive_discard_rate
                if no_deceive_discard_rate is not None else None
            ),
            "oracle_FFR": values.get("FFR"),
            "conditional_defensive_switch_rate": values.get(
                "conditional_defensive_switch_rate"
            ),
            "BSE_cal": None if variant == "L2_no_belief" else calibration.get("bse_abs"),
            "Brier": None if variant == "L2_no_belief" else calibration.get("brier"),
            "ECE": None if variant == "L2_no_belief" else calibration.get("ece"),
            "CVaR_10pct": values.get("cvar_10pct"),
            "DeltaBelief_deceive": (values.get("belief_shift_by_mode") or {}).get("deceive", {}).get("mean_delta"),
        })
    by_variant = {row["variant"]: row for row in result["paper_table"]}
    result["e2_ladder_table"] = [
        by_variant[variant]
        for variant in ("L0", "L1", "L1_z_bphi", "L2_no_deceive", "L2_full")
        if variant in by_variant
    ]
    result["e3_ablation_table"] = [
        by_variant[variant]
        for variant in (
            "L2_full", "L2_no_z", "L2_no_belief", "L2_no_deceive",
            "L2_no_risk_gate", "L2_no_course",
        )
        if variant in by_variant
    ]
    write_json(args.output_dir / "e3_ablation_summary.json", result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
