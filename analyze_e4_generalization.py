"""Combine formal train-split and frozen holdout evidence for E4."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

from analyze_e6_formal import iter_jsonl_paths, trace_paths
from rerun_v2_e2_ladder_3seeds import paired_stats


def lineup_summary(root: Path) -> Dict[str, Any]:
    grouped: Dict[str, list[float]] = defaultdict(list)
    spec_games: Counter[str] = Counter()
    for row in iter_jsonl_paths(trace_paths(root, "gate1_games.jsonl")):
        specs = sorted(
            str(spec.get("id"))
            for spec in (row.get("opponent_curriculum") or {}).values()
            if isinstance(spec, dict) and spec.get("id")
        )
        lineup = "+".join(specs) if specs else "unknown"
        grouped[lineup].append(float(row.get("agent_net", 0.0)))
        spec_games.update(set(specs))
    return {
        "lineups": {
            lineup: {
                "games": len(values),
                "avg_net": sum(values) / len(values),
                "cvar_10pct": sum(sorted(values)[:max(1, math.ceil(0.1 * len(values)))]) / max(1, math.ceil(0.1 * len(values))),
            }
            for lineup, values in sorted(grouped.items())
        },
        "spec_game_counts": dict(spec_games),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-summary", type=Path, required=True)
    parser.add_argument("--holdout-full-root", type=Path, required=True)
    parser.add_argument("--train-summary", type=Path, required=True)
    parser.add_argument("--train-full-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    holdout = json.loads(args.holdout_summary.read_text(encoding="utf-8"))
    train = json.loads(args.train_summary.read_text(encoding="utf-8"))
    holdout_full = holdout["aggregate"]["L2_full"]
    train_full = train["aggregate"]["L2_full"]
    holdout_seed = holdout["per_seed"]["L2_full"]
    train_seed = train["per_seed"]["L2_full"]
    if [row["seed"] for row in holdout_seed] != [row["seed"] for row in train_seed]:
        raise ValueError("train and holdout outer seeds are not aligned")

    train_vs_holdout = {
        "avg_net_train_minus_holdout": paired_stats(
            [float(row["avg_net"]) for row in train_seed],
            [float(row["avg_net"]) for row in holdout_seed],
        ),
        "cvar_train_minus_holdout": paired_stats(
            [float(row["cvar_10pct"]) for row in train_seed],
            [float(row["cvar_10pct"]) for row in holdout_seed],
        ),
    }
    comparisons = {
        name: {
            "score": holdout["paired_net"][f"L2_full_vs_{name}"],
            "dealin_rate": holdout["paired_behavior"][f"L2_full_vs_{name}"]["dealin_rate"],
            "ron_game_rate": holdout["paired_behavior"][f"L2_full_vs_{name}"].get(
                "ron_game_rate",
                holdout["paired_behavior"][f"L2_full_vs_{name}"]["discard_hu_rate"],
            ),
            "cvar_10pct": holdout["paired_behavior"][f"L2_full_vs_{name}"]["cvar_10pct"],
        }
        for name in ("L1", "L2_no_course")
    }
    retention = (
        float(holdout_full["avg_net"]) / float(train_full["avg_net"])
        if float(train_full["avg_net"]) > 0 else None
    )
    result = {
        "protocol": {
            "train_summary": str(args.train_summary),
            "holdout_summary": str(args.holdout_summary),
            "same_outer_seeds": True,
            "holdout_evidence_reused_from_frozen_e2_e3": True,
        },
        "train_full": train_full,
        "holdout_full": holdout_full,
        "holdout_to_train_score_retention": retention,
        "paired_train_vs_holdout": train_vs_holdout,
        "holdout_comparisons": comparisons,
        "train_lineups": lineup_summary(args.train_full_root),
        "holdout_lineups": lineup_summary(args.holdout_full_root),
        "H3_85pct_retention_pass": retention is not None and retention >= 0.85,
        "limitations": [
            "The nine-opponent course is small (6 train, 3 holdout).",
            "Each game contains three curriculum opponents, so lineup results are reported instead of attributing one payoff to a single opponent.",
            "Holdout results are reused from the frozen E2/E3 matrix; they were not selected after observing E4 train performance.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "e4_generalization.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# E4 Unseen-Style Generalization",
        "",
        f"- Train-split full score: {train_full['avg_net']:.4f}",
        f"- Holdout full score: {holdout_full['avg_net']:.4f}",
        f"- Holdout/train retention: {retention:.2%}" if retention is not None else "- Holdout/train retention: n/a",
        f"- 85% retention criterion: {'PASS' if result['H3_85pct_retention_pass'] else 'FAIL'}",
        "",
        "## Frozen Holdout Comparisons",
        "",
        "| Baseline | Score difference | p-value | DR difference | CVaR difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in comparisons.items():
        report.append(
            f"| {name} | {values['score']['mean_diff']:.4f} | {values['score']['t_p_value']:.3g} | "
            f"{values['dealin_rate']['mean_diff']:.5f} | {values['cvar_10pct']['mean_diff']:.4f} |"
        )
    report.extend(["", "See `e4_generalization.json` for paired statistics and lineup-level diagnostics."])
    (args.output_dir / "E4_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "train_score": train_full["avg_net"],
        "holdout_score": holdout_full["avg_net"],
        "retention": retention,
        "retention_pass": result["H3_85pct_retention_pass"],
        "holdout_comparisons": comparisons,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
