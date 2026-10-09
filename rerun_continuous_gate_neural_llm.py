"""L0 / L1 / L2_continuous comparison against the distilled neural opponent,
using the real local_qwen LLM backend (not heuristic_fallback).

Mirrors h2_full_learned_route/evaluate_h2_neural_fast.py's subprocess +
ThreadPoolExecutor pattern (one run_gate1_experiments.py subprocess per
(variant, seed), safe to run concurrently since each is a separate process
with its own model load). L2_continuous uses --mask-gate-policy continuous
(the Chapter-5/6 alpha=f(u,rho) gate) instead of the discrete rule/learned
gate chain used by L2_retrained_gate/L2_value_gate in that script.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from experiment_trace import write_json
from rerun_v2_e2_ladder_3seeds import paired_stats


METHOD = {
    "L0": "llm_base", "L1": "llm_reactive_z",
    "L2_continuous": "llm_mask", "L2_continuous_v2": "llm_mask", "L2_continuous_v3": "llm_mask",
    "L2_continuous_v4": "llm_mask", "L2_continuous_v5": "llm_mask", "L2_continuous_v6": "llm_mask",
    "L2_continuous_v7": "llm_mask", "L2_neural_gate": "llm_mask",
    "L2_action_value": "llm_mask",
    "L2_value_residual": "llm_mask",
    "L2_learned_value_gate": "llm_mask",
}
DEFAULT_VARIANTS = ["L0", "L1", "L2_continuous"]


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.mean(values) if values else None


def optional_extend(command: list[str], option: str, value: Optional[str]) -> None:
    if value:
        command.extend([option, value])


def aggregate(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    mode_counts: Dict[str, int] = {"exploit": 0, "safe": 0, "deceive": 0}
    false_folds = 0
    false_fold_opportunities = 0
    defensive_switches = 0
    defensive_switch_opportunities = 0
    strict_event_false_folds = 0
    strict_event_opportunities = 0
    induced_dealin = 0
    deceive_windows = 0
    funnel_gate_rows = 0
    funnel_can_deceive = 0
    funnel_selected_deceive = 0
    funnel_margin_blocked = 0
    funnel_safe_margin_blocked = 0
    funnel_clip_count = 0
    funnel_max_abs_z = 0.0
    hu_available = 0
    hu_selected = 0
    missed_hu = 0
    missed_hu_value_floor = 0.0
    funnel_block_reasons: Dict[str, int] = {}
    funnel_available_sets: Dict[str, int] = {}
    for row in rows:
        for mode in mode_counts:
            mode_counts[mode] += int((row.get("mode_counts") or {}).get(mode, 0))
        counts = row.get("FFR_counts") or {}
        false_folds += int(counts.get("false_folds", 0))
        false_fold_opportunities += int(counts.get("false_fold_opportunities", 0))
        switch_counts = row.get("conditional_defensive_switch_counts") or {}
        defensive_switches += int(switch_counts.get("switches", 0))
        defensive_switch_opportunities += int(switch_counts.get("opportunities", 0))
        strict_event_counts = row.get("strict_FFR_event_counts") or {}
        strict_event_false_folds += int(strict_event_counts.get("false_folds", 0))
        strict_event_opportunities += int(strict_event_counts.get("opportunities", 0))
        dir_counts = row.get("DIR_counts") or {}
        induced_dealin += int(dir_counts.get("induced_dealin", 0))
        deceive_windows += int(dir_counts.get("deceive_windows", 0))
        hu_decision = row.get("hu_decision") or {}
        hu_available += int(hu_decision.get("hu_available", 0))
        hu_selected += int(hu_decision.get("hu_selected", 0))
        missed_hu += int(hu_decision.get("missed_hu", 0))
        missed_hu_value_floor += float(hu_decision.get("missed_hu_value_floor", 0.0))
        funnel = row.get("_funnel") or {}
        funnel_gate_rows += int(funnel.get("gate_rows", 0))
        funnel_can_deceive += int(funnel.get("can_deceive", 0))
        funnel_selected_deceive += int(funnel.get("selected_deceive", 0))
        funnel_margin_blocked += int(funnel.get("deceive_margin_blocked", 0))
        funnel_safe_margin_blocked += int(funnel.get("safe_margin_blocked", 0))
        funnel_clip_count += int(funnel.get("clip_count", 0))
        funnel_max_abs_z = max(funnel_max_abs_z, float(funnel.get("max_abs_normalized_feature", 0.0) or 0.0))
        for reason, count in (funnel.get("deceive_block_reasons") or {}).items():
            funnel_block_reasons[reason] = funnel_block_reasons.get(reason, 0) + int(count)
        for key, count in (funnel.get("available_mode_sets") or {}).items():
            funnel_available_sets[key] = funnel_available_sets.get(key, 0) + int(count)
    return {
        "seeds": len(rows),
        "games": sum(int(row["games"]) for row in rows),
        "avg_net": mean(float(row["avg_net"]) for row in rows),
        "hu_rate": mean(float(row["hu_rate"]) for row in rows),
        "discard_hu_rate": mean(float(row.get("discard_hu_rate", 0.0)) for row in rows),
        "ron_game_rate": mean(float(row.get("ron_game_rate", row.get("discard_hu_rate", 0.0))) for row in rows),
        "dealin_rate": mean(float(row["dealin_rate"]) for row in rows),
        "cvar_10pct": mean(float(row.get("cvar_10pct", 0.0)) for row in rows),
        "FFR": false_folds / false_fold_opportunities if false_fold_opportunities else 0.0,
        "FFR_counts": {
            "false_folds": false_folds,
            "false_fold_opportunities": false_fold_opportunities,
            "definition": (
                "first strict eligible decision per opponent-game; neural policies "
                "use the same progress-loss rule as a behavioral proxy"
            ),
        },
        "conditional_defensive_switch_rate": (
            defensive_switches / defensive_switch_opportunities
            if defensive_switch_opportunities else 0.0
        ),
        "conditional_defensive_switch_counts": {
            "switches": defensive_switches,
            "opportunities": defensive_switch_opportunities,
        },
        "strict_FFR_event_diagnostic": (
            strict_event_false_folds / strict_event_opportunities
            if strict_event_opportunities else 0.0
        ),
        "strict_FFR_event_counts": {
            "false_folds": strict_event_false_folds,
            "opportunities": strict_event_opportunities,
        },
        "DIR_totals": {"induced_dealin": induced_dealin, "deceive_windows": deceive_windows},
        "mode_counts": mode_counts,
        "latency_ms": {
            key: mean(float(row["decision_latency_ms"][key]) for row in rows)
            for key in ("p50", "p95", "p99")
        },
        "hu_decision": {
            "hu_available": hu_available,
            "hu_selected": hu_selected,
            "missed_hu": missed_hu,
            "missed_hu_rate": missed_hu / hu_available if hu_available else 0.0,
            "missed_hu_value_floor": missed_hu_value_floor,
            "missed_hu_value_floor_per_game": missed_hu_value_floor / sum(int(row["games"]) for row in rows),
        },
        "funnel": {
            "gate_rows": funnel_gate_rows,
            "can_deceive": funnel_can_deceive,
            "selected_deceive": funnel_selected_deceive,
            "deceive_margin_blocked": funnel_margin_blocked,
            "safe_margin_blocked": funnel_safe_margin_blocked,
            "can_deceive_rate": funnel_can_deceive / funnel_gate_rows if funnel_gate_rows else 0.0,
            "selected_deceive_rate": funnel_selected_deceive / funnel_gate_rows if funnel_gate_rows else 0.0,
            "deceive_margin_block_rate": funnel_margin_blocked / funnel_gate_rows if funnel_gate_rows else 0.0,
            "clip_count": funnel_clip_count,
            "clip_rate": funnel_clip_count / funnel_gate_rows if funnel_gate_rows else 0.0,
            "max_abs_normalized_feature": funnel_max_abs_z,
            "deceive_block_reasons": funnel_block_reasons,
            "available_mode_sets": funnel_available_sets,
        },
    }


def step_funnel(output: Path) -> Dict[str, Any]:
    path = output / "gate1_steps.jsonl"
    if not path.is_file():
        return {}
    total = gate_rows = can_deceive = selected_deceive = margin_blocked = safe_margin_blocked = clipped = 0
    max_abs_z = 0.0
    block_reasons: Dict[str, int] = {}
    available_counts: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            gate = row.get("learned_gate") or {}
            if gate.get("used"):
                gate_rows += 1
                modes_key = ",".join(gate.get("available_modes") or [])
                available_counts[modes_key] = available_counts.get(modes_key, 0) + 1
                if row.get("can_deceive"):
                    can_deceive += 1
                if row.get("mode") == "deceive":
                    selected_deceive += 1
                if gate.get("deceive_margin_blocked"):
                    margin_blocked += 1
                if gate.get("safe_margin_blocked"):
                    safe_margin_blocked += 1
                if gate.get("normalized_feature_clipped"):
                    clipped += 1
                max_abs_z = max(max_abs_z, float(gate.get("max_abs_normalized_feature", 0.0) or 0.0))
                reason = row.get("deceive_block_reason")
                if reason:
                    block_reasons[str(reason)] = block_reasons.get(str(reason), 0) + 1
    return {
        "step_rows": total,
        "gate_rows": gate_rows,
        "can_deceive": can_deceive,
        "selected_deceive": selected_deceive,
        "deceive_margin_blocked": margin_blocked,
        "safe_margin_blocked": safe_margin_blocked,
        "can_deceive_rate": can_deceive / gate_rows if gate_rows else 0.0,
        "selected_deceive_rate": selected_deceive / gate_rows if gate_rows else 0.0,
        "deceive_margin_block_rate": margin_blocked / gate_rows if gate_rows else 0.0,
        "clip_count": clipped,
        "clip_rate": clipped / gate_rows if gate_rows else 0.0,
        "max_abs_normalized_feature": max_abs_z,
        "deceive_block_reasons": block_reasons,
        "available_mode_sets": available_counts,
    }


def compact_row(seed: int, row: Dict[str, Any]) -> Dict[str, Any]:
    counts = row.get("FFR_counts") or {}
    modes = row.get("mode_counts") or {}
    dir_counts = row.get("DIR_counts") or {}
    hu_decision = row.get("hu_decision") or {}
    false_folds = int(counts.get("false_folds", 0))
    false_fold_opportunities = int(counts.get("false_fold_opportunities", 0))
    return {
        "seed": seed,
        "games": int(row["games"]),
        "avg_net": float(row["avg_net"]),
        "hu_rate": float(row["hu_rate"]),
        "discard_hu_rate": float(row.get("discard_hu_rate", 0.0)),
        "ron_game_rate": float(row.get("ron_game_rate", row.get("discard_hu_rate", 0.0))),
        "dealin_rate": float(row["dealin_rate"]),
        "cvar_10pct": float(row.get("cvar_10pct", 0.0)),
        "FFR": false_folds / false_fold_opportunities if false_fold_opportunities else 0.0,
        "false_folds": false_folds,
        "false_fold_opportunities": false_fold_opportunities,
        "conditional_defensive_switch_rate": float(
            row.get("conditional_defensive_switch_rate", 0.0)
        ),
        "exploit": int(modes.get("exploit", 0)),
        "safe": int(modes.get("safe", 0)),
        "deceive": int(modes.get("deceive", 0)),
        "induced_dealin": int(dir_counts.get("induced_dealin", 0)),
        "deceive_windows": int(dir_counts.get("deceive_windows", 0)),
        "hu_available": int(hu_decision.get("hu_available", 0)),
        "hu_selected": int(hu_decision.get("hu_selected", 0)),
        "missed_hu": int(hu_decision.get("missed_hu", 0)),
        "missed_hu_rate": float(hu_decision.get("missed_hu_rate", 0.0)),
        "missed_hu_value_floor": float(hu_decision.get("missed_hu_value_floor", 0.0)),
        "funnel": row.get("_funnel", {}),
    }


def common_command(args: argparse.Namespace, variant: str, seed: int, output: Path) -> list[str]:
    command = [
        str(args.python),
        "run_gate1_experiments.py",
        "--methods", METHOD[variant],
        "--games", str(args.games),
        "--seed", str(seed),
        "--opponent-style", "neural",
        "--neural-opponent-model-path", args.neural_opponent_model_path,
        "--neural-opponent-device", args.neural_opponent_device,
        "--danger-threshold", "1",
        "--ffr-hand-shanten", "1",
        "--sample-every", "0",
        "--auto-hu", args.auto_hu,
        "--snapshot-oracle-samples", str(args.snapshot_oracle_samples),
        "--snapshot-crn-seeds", "1",
        "--backend", "local_qwen",
        "--model-path", args.model_path,
        "--max-new-tokens", str(args.max_new_tokens),
        "--temperature", "0.0",
        "--output-dir", str(output),
    ]
    if args.mask_log_counterfactual:
        command.append("--mask-log-counterfactual")
    optional_extend(command, "--adapter-path", args.adapter_path)
    if variant == "L2_continuous":
        command.extend(["--mask-gate-policy", "continuous"])
    elif variant == "L2_continuous_v2":
        command.extend(["--mask-gate-policy", "continuous_v2"])
    elif variant == "L2_continuous_v3":
        command.extend(["--mask-gate-policy", "continuous_v3"])
    elif variant == "L2_continuous_v4":
        command.extend(["--mask-gate-policy", "continuous_v4"])
    elif variant == "L2_continuous_v5":
        command.extend(["--mask-gate-policy", "continuous_v5"])
    elif variant == "L2_continuous_v6":
        command.extend(["--mask-gate-policy", "continuous_v6"])
    elif variant == "L2_continuous_v7":
        command.extend(["--mask-gate-policy", "continuous_v7"])
    elif variant == "L2_neural_gate":
        command.extend([
            "--mask-gate-policy", "neural",
            "--neural-gate-model-path", args.neural_gate_model_path,
            "--neural-gate-device", args.neural_gate_device,
            "--neural-gate-deceive-margin", str(args.neural_gate_deceive_margin),
            "--neural-gate-safe-margin", str(args.neural_gate_safe_margin),
            "--neural-gate-safe-risk-threshold", str(args.neural_gate_safe_risk_threshold),
            "--mask-deceive-style", "threat",
            "--mask-threat-max-result-shanten", "0",
            "--mask-threat-max-shanten-regret", "0",
            "--mask-threat-min-ukeire-ratio", "1.0",
        ])
    elif variant == "L2_action_value":
        command.extend([
            "--mask-gate-policy", "neural_action_value",
            "--action-value-model-path", args.action_value_model_path,
            "--action-value-device", args.action_value_device,
            "--mask-threat-max-shanten-regret", "0",
            "--mask-threat-min-ukeire-ratio", "1.0",
        ])
    elif variant == "L2_value_residual":
        command.extend([
            "--mask-gate-policy", "neural_value_residual",
            "--action-value-model-path", args.action_value_model_path,
            "--value-residual-model-path", args.value_residual_model_path,
            "--action-value-device", args.action_value_device,
            "--mask-threat-max-shanten-regret", "0",
            "--mask-threat-min-ukeire-ratio", "1.0",
        ])
    elif variant == "L2_learned_value_gate":
        command.extend([
            "--mask-gate-policy", "neural_value_gate",
            "--gate-value-model-path", args.gate_value_model_path,
            "--neural-gate-device", args.neural_gate_device,
            "--mask-deceive-style", "threat",
            "--mask-threat-max-result-shanten", "0",
            "--mask-threat-max-shanten-regret", "0",
            "--mask-threat-min-ukeire-ratio", "1.0",
        ])
        if args.gate_value_safe_margin is not None:
            command.extend(["--gate-value-safe-margin", str(args.gate_value_safe_margin)])
        if args.gate_value_deceive_margin is not None:
            command.extend(["--gate-value-deceive-margin", str(args.gate_value_deceive_margin)])
    return command


def load_variant_row(output: Path, method: str) -> Dict[str, Any]:
    summary = json.loads((output / "gate1_summary.json").read_text(encoding="utf-8"))
    return summary["E2_ladder"][method]


def run_one(args: argparse.Namespace, output_dir: Path, variant: str, seed: int) -> tuple[str, int, Dict[str, Any]]:
    output = output_dir / variant / f"seed_{seed}"
    summary_path = output / "gate1_summary.json"
    if args.force or not summary_path.is_file():
        output.mkdir(parents=True, exist_ok=True)
        command = common_command(args, variant, seed, output)
        print(f"[Neural LLM eval] start variant={variant} seed={seed}", flush=True)
        env = dict(os.environ)
        env.setdefault("PYTHONHASHSEED", "0")
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        with (output / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=REPO_DIR, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    row = load_variant_row(output, METHOD[variant])
    row["_funnel"] = step_funnel(output)
    print(f"[Neural LLM eval] done variant={variant} seed={seed} net={row['avg_net']:.3f}", flush=True)
    return variant, seed, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("py10/bin/python3"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--neural-opponent-model-path", default="Neural_opponent_model/neural_opponent_policy.pth")
    parser.add_argument("--neural-opponent-device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260627, 20261627, 20262627])
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--snapshot-oracle-samples", type=int, default=1)
    parser.add_argument("--mask-log-counterfactual", action="store_true")
    parser.add_argument("--auto-hu", choices=["common", "off"], default="common")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--neural-gate-model-path", default="Neural_Gate_20260723_v2/model/neural_gate_policy.pth")
    parser.add_argument("--gate-value-model-path", default="")
    parser.add_argument("--gate-value-safe-margin", type=float, default=None)
    parser.add_argument("--gate-value-deceive-margin", type=float, default=None)
    parser.add_argument("--neural-gate-device", default="cpu")
    parser.add_argument("--neural-gate-deceive-margin", type=float, default=0.0)
    parser.add_argument("--neural-gate-safe-margin", type=float, default=0.0)
    parser.add_argument("--neural-gate-safe-risk-threshold", type=float, default=0.0)
    parser.add_argument("--action-value-model-path", default="")
    parser.add_argument("--value-residual-model-path", default="")
    parser.add_argument("--action-value-device", default="cpu")
    parser.add_argument("--variants", nargs="+", choices=list(METHOD), default=DEFAULT_VARIANTS)
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_variant: Dict[str, list[Dict[str, Any]]] = {variant: [] for variant in args.variants}
    jobs = [(variant, seed) for variant in args.variants for seed in args.seeds]
    if args.parallel_workers <= 1:
        completed = [run_one(args, output_dir, variant, seed) for variant, seed in jobs]
    else:
        completed = []
        with ThreadPoolExecutor(max_workers=args.parallel_workers) as pool:
            futures = [pool.submit(run_one, args, output_dir, variant, seed) for variant, seed in jobs]
            for future in as_completed(futures):
                completed.append(future.result())

    rows_by_key = {(variant, seed): row for variant, seed, row in completed}
    for variant in args.variants:
        for seed in args.seeds:
            per_variant[variant].append(rows_by_key[(variant, seed)])

    result = {
        "config": {
            "seeds": args.seeds,
            "variants": args.variants,
            "games_per_seed_per_variant": args.games,
            "opponent_style": "neural",
            "neural_opponent_model_path": args.neural_opponent_model_path,
            "neural_ffr": (
                "behavioral strict proxy: first opponent-game decision with "
                "own_shanten<=1 and P0_shanten>0; non-push action must increase "
                "shanten or reduce effective copies"
            ),
            "backend": "local_qwen",
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "gpu": args.gpu,
            "parallel_workers": args.parallel_workers,
            "auto_hu": args.auto_hu,
            "neural_gate_deceive_margin": args.neural_gate_deceive_margin,
            "neural_gate_safe_margin": args.neural_gate_safe_margin,
            "neural_gate_safe_risk_threshold": args.neural_gate_safe_risk_threshold,
            "gate_value_model_path": args.gate_value_model_path,
            "gate_value_safe_margin": args.gate_value_safe_margin,
            "gate_value_deceive_margin": args.gate_value_deceive_margin,
            "action_value_model_path": args.action_value_model_path,
            "value_residual_model_path": args.value_residual_model_path,
            "mask_log_counterfactual": args.mask_log_counterfactual,
        },
        "aggregate": {variant: aggregate(rows) for variant, rows in per_variant.items()},
        "per_seed": {
            variant: [compact_row(seed, row) for seed, row in zip(args.seeds, rows)]
            for variant, rows in per_variant.items()
        },
    }
    pairs = [
        ("L1", "L0"),
        ("L2_continuous", "L1"),
        ("L2_continuous", "L0"),
        ("L2_continuous_v2", "L1"),
        ("L2_continuous_v2", "L0"),
        ("L2_value_residual", "L2_continuous_v2"),
        ("L2_value_residual", "L1"),
        ("L2_value_residual", "L0"),
        ("L2_learned_value_gate", "L2_continuous_v2"),
        ("L2_learned_value_gate", "L1"),
        ("L2_learned_value_gate", "L0"),
    ]
    result["paired_net"] = {
        f"{left}_vs_{right}": paired_stats(
            [float(row["avg_net"]) for row in per_variant[left]],
            [float(row["avg_net"]) for row in per_variant[right]],
        )
        for left, right in pairs
        if left in per_variant and right in per_variant
    }
    summary_path = output_dir / "neural_llm_summary.json"
    write_json(summary_path, result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
