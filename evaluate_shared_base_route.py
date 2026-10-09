"""Paired evaluation of current SFT and one or more continued backbones."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

from experiment_trace import write_json
from rerun_continuous_gate_neural_llm import aggregate, compact_row
from rerun_v2_e2_ladder_3seeds import paired_stats


REPO_DIR = Path(__file__).resolve().parent


def aggregate_shared(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    result = aggregate(rows)
    distributions = [row.get("net_distribution") or {} for row in rows]
    result["robust_net"] = {
        "median_mean_across_seeds": sum(float(item.get("median", 0.0)) for item in distributions) / len(rows),
        "trimmed_mean_10pct_across_seeds": sum(
            float(item.get("trimmed_mean_10pct", 0.0)) for item in distributions
        ) / len(rows),
        "central_abs_500_mean_across_seeds": sum(
            float((item.get("central_abs_500") or {}).get("mean", 0.0) or 0.0)
            for item in distributions
        ) / len(rows),
        "positive_rate_across_seeds": sum(float(item.get("positive_rate", 0.0)) for item in distributions) / len(rows),
    }
    kinds = ("hu_self_draw", "hu_discard", "hua_zhu", "da_jiao")
    result["settlement_net_total"] = {
        kind: sum(
            float((row.get("settlement") or {}).get("net_total", {}).get(kind, 0.0))
            for row in rows
        )
        for kind in kinds
    }
    return result


def command_for(
    args: argparse.Namespace,
    model_path: str,
    adapter_path: Optional[str],
    seed: int,
    output: Path,
) -> list[str]:
    command = [
        str(args.python),
        "run_gate1_experiments.py",
        "--methods", "llm_base",
        "--games", str(args.games),
        "--seed", str(seed),
        "--opponent-style", "neural",
        "--neural-opponent-model-path", args.neural_opponent_model_path,
        "--neural-opponent-device", "cpu",
        "--sample-every", "0",
        "--auto-hu", "off",
        "--snapshot-oracle-samples", "1",
        "--snapshot-crn-seeds", "1",
        "--backend", "local_qwen",
        "--model-path", model_path,
        "--max-new-tokens", "64",
        "--temperature", "0.0",
        "--output-dir", str(output),
    ]
    if adapter_path:
        command.extend(["--adapter-path", adapter_path])
    return command


def run_one(
    args: argparse.Namespace,
    variant: str,
    model_path: str,
    adapter_path: Optional[str],
    seed: int,
) -> tuple[str, int, Dict[str, Any]]:
    output = args.output_dir / variant / f"seed_{seed}"
    summary_path = output / "gate1_summary.json"
    if args.force or not summary_path.is_file():
        output.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = "0"
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"[shared-base eval] start variant={variant} seed={seed}", flush=True)
        with (output / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command_for(args, model_path, adapter_path, seed, output),
                cwd=REPO_DIR,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = summary["E2_ladder"]["llm_base"]
    print(f"[shared-base eval] done variant={variant} seed={seed} net={row['avg_net']:.3f}", flush=True)
    return variant, seed, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("py10/bin/python3"))
    parser.add_argument("--current-model-path", default="models/Qwen-Mahjong-V1-Mixed-SFT-Merged")
    parser.add_argument("--sft-model-path", required=True)
    parser.add_argument("--grpo-adapter-path")
    parser.add_argument("--neural-opponent-model-path", default="Neural_opponent_model/neural_opponent_policy.pth")
    parser.add_argument("--seeds", nargs="+", type=int, default=[70260701, 70261701])
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = REPO_DIR / args.output_dir
    variants = {
        "current_sft": (args.current_model_path, None),
        "shared_sft": (args.sft_model_path, None),
    }
    if args.grpo_adapter_path:
        variants["shared_sft_grpo"] = (args.sft_model_path, args.grpo_adapter_path)
    jobs = [
        (variant, model_path, adapter_path, seed)
        for variant, (model_path, adapter_path) in variants.items()
        for seed in args.seeds
    ]
    completed = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel_workers)) as pool:
        futures = [pool.submit(run_one, args, *job) for job in jobs]
        for future in as_completed(futures):
            completed.append(future.result())

    by_key = {(variant, seed): row for variant, seed, row in completed}
    per_variant = {
        variant: [by_key[(variant, seed)] for seed in args.seeds]
        for variant in variants
    }
    pairs = [("shared_sft", "current_sft")]
    if "shared_sft_grpo" in variants:
        pairs.extend(
            [
                ("shared_sft_grpo", "shared_sft"),
                ("shared_sft_grpo", "current_sft"),
            ]
        )
    result = {
        "config": {
            "seeds": args.seeds,
            "games_per_seed_per_variant": args.games,
            "auto_hu": "off",
            "opponent_style": "neural",
            "parallel_workers": args.parallel_workers,
            "gpu": args.gpu,
            "variants": {
                variant: {"model_path": model, "adapter_path": adapter}
                for variant, (model, adapter) in variants.items()
            },
        },
        "aggregate": {variant: aggregate_shared(rows) for variant, rows in per_variant.items()},
        "per_seed": {
            variant: [compact_row(seed, row) for seed, row in zip(args.seeds, rows)]
            for variant, rows in per_variant.items()
        },
        "paired_net": {
            f"{left}_vs_{right}": paired_stats(
                [float(row["avg_net"]) for row in per_variant[left]],
                [float(row["avg_net"]) for row in per_variant[right]],
            )
            for left, right in pairs
        },
    }
    write_json(args.output_dir / "shared_base_eval_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
