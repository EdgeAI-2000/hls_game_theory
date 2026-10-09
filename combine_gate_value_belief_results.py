"""Combine disjoint GateValue seed families into one paired belief ablation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from experiment_trace import write_json
from rerun_v2_e2_ladder_3seeds import paired_stats


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    false_folds = sum(int(row.get("false_folds", 0)) for row in rows)
    opportunities = sum(int(row.get("false_fold_opportunities", 0)) for row in rows)
    induced = sum(int(row.get("induced_dealin", 0)) for row in rows)
    windows = sum(int(row.get("deceive_windows", 0)) for row in rows)
    mode_counts = {
        mode: sum(int(row.get(mode, 0)) for row in rows)
        for mode in ("exploit", "safe", "deceive")
    }
    return {
        "seeds": len(rows),
        "games": sum(int(row["games"]) for row in rows),
        "avg_net": statistics.mean(float(row["avg_net"]) for row in rows),
        "net_seed_std": statistics.stdev(float(row["avg_net"]) for row in rows),
        "hu_rate": statistics.mean(float(row["hu_rate"]) for row in rows),
        "dealin_rate": statistics.mean(float(row["dealin_rate"]) for row in rows),
        "FFR": false_folds / opportunities if opportunities else 0.0,
        "FFR_counts": {
            "false_folds": false_folds,
            "false_fold_opportunities": opportunities,
        },
        "mode_counts": mode_counts,
        "DIR": induced / windows if windows else 0.0,
        "DIR_counts": {"induced_dealin": induced, "deceive_windows": windows},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    combined: Dict[str, List[Dict[str, Any]]] = {variant: [] for variant in args.variants}
    seen_seeds = set()
    source_metadata = []
    for path in args.source:
        summary = json.loads(path.read_text(encoding="utf-8"))
        seeds = [int(seed) for seed in summary["config"]["seeds"]]
        overlap = seen_seeds.intersection(seeds)
        if overlap:
            raise ValueError(f"source seed families overlap: {sorted(overlap)}")
        seen_seeds.update(seeds)
        for variant in args.variants:
            rows = summary["per_seed"].get(variant)
            if rows is None:
                raise KeyError(f"{path} has no variant {variant}")
            if [int(row["seed"]) for row in rows] != seeds:
                raise ValueError(f"{path}: per-seed ordering mismatch for {variant}")
            combined[variant].extend(rows)
        source_metadata.append({"path": str(path.resolve()), "seeds": seeds})

    if args.anchor not in combined:
        raise ValueError(f"anchor is not in variants: {args.anchor}")
    anchor_values = [float(row["avg_net"]) for row in combined[args.anchor]]
    result = {
        "config": {
            "sources": source_metadata,
            "seeds": sorted(seen_seeds),
            "variants": args.variants,
            "anchor": args.anchor,
        },
        "aggregate": {
            variant: aggregate(rows) for variant, rows in combined.items()
        },
        "paired_net": {
            f"{args.anchor}_vs_{variant}": {
                **paired_stats(
                    anchor_values,
                    [float(row["avg_net"]) for row in combined[variant]],
                ),
                "anchor_wins": sum(
                    left > float(right["avg_net"])
                    for left, right in zip(anchor_values, combined[variant])
                ),
            }
            for variant in args.variants
            if variant != args.anchor
        },
        "per_seed": combined,
    }
    write_json(args.output, result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(json.dumps(result["paired_net"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
