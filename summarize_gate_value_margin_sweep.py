"""Summarize the frozen GateValue v1 safe/deceive margin sweep."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from experiment_trace import write_json


GATE_VARIANT = "L2_learned_value_gate"
CONFIGS = (
    ("current", 0.05, 0.05),
    ("medium", 0.15, 0.10),
    ("strong", 0.25, 0.15),
)


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.mean(values) if values else None


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def robust_metrics(stage_dir: Path, variant: str) -> Dict[str, Optional[float]]:
    rows = []
    methods = {
        "L0": "llm_base",
        "L1": "llm_reactive_z",
        GATE_VARIANT: "llm_mask",
    }
    for path in sorted((stage_dir / variant).glob("seed_*/gate1_summary.json")):
        summary = read_json(path)
        row = summary["E2_ladder"][methods[variant]]
        distribution = row.get("net_distribution") or {}
        central = distribution.get("central_abs_500") or {}
        rows.append(
            {
                "trimmed_mean_10pct": distribution.get("trimmed_mean_10pct"),
                "median": distribution.get("median"),
                "net_std": distribution.get("std"),
                "central_abs_500_mean": central.get("mean"),
                "avg_hu_fan": row.get("avg_hu_fan"),
                "fan_per_game": row.get("fan_per_game"),
            }
        )
    return {
        key: mean(float(row[key]) for row in rows if row.get(key) is not None)
        for key in (
            "trimmed_mean_10pct",
            "median",
            "net_std",
            "central_abs_500_mean",
            "avg_hu_fan",
            "fan_per_game",
        )
    }


def compact(
    name: str,
    stage_dir: Path,
    variant: str,
    aggregate: Dict[str, Any],
    safe_margin: Optional[float],
    deceive_margin: Optional[float],
) -> Dict[str, Any]:
    modes = aggregate.get("mode_counts") or {}
    funnel = aggregate.get("funnel") or {}
    dir_totals = aggregate.get("DIR_totals") or {}
    total_modes = sum(int(modes.get(mode, 0)) for mode in ("exploit", "safe", "deceive"))
    return {
        "name": name,
        "variant": variant,
        "safe_margin": safe_margin,
        "deceive_margin": deceive_margin,
        "seeds": aggregate.get("seeds"),
        "games": aggregate.get("games"),
        "avg_net": aggregate.get("avg_net"),
        "hu_rate": aggregate.get("hu_rate"),
        "dealin_rate": aggregate.get("dealin_rate"),
        "FFR_proxy": aggregate.get("FFR"),
        "mode_counts": modes,
        "mode_rates": {
            mode: int(modes.get(mode, 0)) / total_modes if total_modes else 0.0
            for mode in ("exploit", "safe", "deceive")
        },
        "induced_dealin": int(dir_totals.get("induced_dealin", 0)),
        "safe_margin_blocked": int(funnel.get("safe_margin_blocked", 0)),
        "deceive_margin_blocked": int(funnel.get("deceive_margin_blocked", 0)),
        "clip_rate": funnel.get("clip_rate"),
        "latency_ms": aggregate.get("latency_ms"),
        "robust": robust_metrics(stage_dir, variant),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    rows = []
    references = []
    current_summary = read_json(root / "current" / "neural_llm_summary.json")
    for variant in ("L0", "L1"):
        references.append(
            compact(
                variant,
                root / "current",
                variant,
                current_summary["aggregate"][variant],
                None,
                None,
            )
        )

    for name, safe_margin, deceive_margin in CONFIGS:
        stage_dir = root / name
        summary = read_json(stage_dir / "neural_llm_summary.json")
        rows.append(
            compact(
                name,
                stage_dir,
                GATE_VARIANT,
                summary["aggregate"][GATE_VARIANT],
                safe_margin,
                deceive_margin,
            )
        )

    eligible = [
        row for row in rows
        if float(row["hu_rate"]) >= 0.50 and float(row["dealin_rate"]) <= 0.14
    ]
    pool = eligible or rows
    selected = max(pool, key=lambda row: float(row["avg_net"]))
    result = {
        "protocol": {
            "paired": True,
            "only_gate_margins_change": True,
            "selection_constraints": {"hu_rate_min": 0.50, "dealin_rate_max": 0.14},
        },
        "references": references,
        "gate_margin_configs": rows,
        "selected": selected,
        "selection_used_constraints": bool(eligible),
    }
    output = args.output.resolve() if args.output else root / "margin_sweep_summary.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
