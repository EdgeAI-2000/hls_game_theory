"""Aggregate formal E5 runs with reused P3R holdout reference endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment_trace import write_json
from rerun_v2_e2_ladder_3seeds import paired_stats


ORDER = ["no_deceive", "tight", "default", "loose", "constant_deceive"]
P3R_VARIANTS = {
    "no_deceive": "L2_no_deceive",
    "default": "L2_full",
    "constant_deceive": "L2_no_risk_gate",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(row: dict[str, Any], name: str) -> float:
    if name == "avg_net":
        return float(row["avg_net"])
    if name == "dealin_rate":
        return float(row["dealin_rate"])
    if name == "cvar_10pct":
        return float(row["cvar_10pct"])
    if name in {"discard_hu_rate", "ron_game_rate"}:
        return float(row.get("ron_game_rate", row.get("discard_hu_rate", 0.0)))
    if name == "oracle_ffr":
        if row.get("FFR") is not None:
            return float(row["FFR"])
        counts = row.get("FFR_counts") or {}
        opportunities = int(counts.get("false_fold_opportunities", 0))
        return int(counts.get("false_folds", 0)) / opportunities if opportunities else 0.0
    if name == "deceive_rate":
        modes = row.get("mode_counts") or {}
        if not modes:
            modes = {mode: row.get(mode, 0) for mode in ("exploit", "safe", "deceive")}
        total = sum(int(modes.get(mode, 0)) for mode in ("exploit", "safe", "deceive"))
        return int(modes.get("deceive", 0)) / total if total else 0.0
    raise KeyError(name)


def aggregate_row(name: str, aggregate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    modes = aggregate.get("mode_counts") or {}
    total_modes = sum(int(modes.get(mode, 0)) for mode in ("exploit", "safe", "deceive"))
    induced = int((aggregate.get("DIR_totals") or {}).get("induced_dealin", 0))
    games = int(aggregate["games"])
    baseline_induced = int((baseline.get("DIR_totals") or {}).get("induced_dealin", 0))
    baseline_games = int(baseline["games"])
    return {
        "setting": name,
        "mean_score": aggregate["avg_net"],
        "score_gain_vs_no_deceive": float(aggregate["avg_net"]) - float(baseline["avg_net"]),
        "dealin_rate": aggregate["dealin_rate"],
        "ron_game_rate": aggregate.get(
            "ron_game_rate", aggregate.get("discard_hu_rate", 0.0)
        ),
        "cvar_10pct": aggregate["cvar_10pct"],
        "cvar_change_vs_no_deceive": float(aggregate["cvar_10pct"]) - float(baseline["cvar_10pct"]),
        "excess_dir_per_game": (
            float(aggregate.get("ron_game_rate", aggregate.get("discard_hu_rate", 0.0)))
            - float(baseline.get("ron_game_rate", baseline.get("discard_hu_rate", 0.0)))
        ),
        "window_induced_dealin_per_game_diagnostic": (
            induced / games - baseline_induced / baseline_games
        ),
        "oracle_ffr": aggregate.get("FFR"),
        "deceive_rate": int(modes.get("deceive", 0)) / total_modes if total_modes else 0.0,
        "delta_belief_deceive": (aggregate.get("belief_shift_by_mode") or {}).get("deceive", {}).get("mean_delta"),
    }


def pairwise(rows: dict[str, list[dict[str, Any]]], left: str, right: str) -> dict[str, Any]:
    metrics = (
        "avg_net",
        "dealin_rate",
        "cvar_10pct",
        "ron_game_rate",
        "oracle_ffr",
        "deceive_rate",
    )
    result = {}
    for name in metrics:
        xs = [metric(row, name) for row in rows[left]]
        ys = [metric(row, name) for row in rows[right]]
        stats = paired_stats(xs, ys)
        try:
            from scipy.stats import wilcoxon

            stats["wilcoxon_p_value"] = float(
                wilcoxon(xs, ys, alternative="two-sided").pvalue
            )
        except (ImportError, ValueError):
            stats["wilcoxon_p_value"] = None
        result[name] = stats
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e5-root", type=Path)
    parser.add_argument("--holdout-root", type=Path)
    parser.add_argument("--anti-deception-root", type=Path)
    parser.add_argument("--p3r-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.e5_root:
        loaded = read_json(args.e5_root / "e5_run_summary.json")
        e5_runs = {suite: loaded for suite in loaded["aggregate"]}
    else:
        if not args.holdout_root or not args.anti_deception_root:
            parser.error("provide --e5-root or both split formal roots")
        e5_runs = {
            "holdout": read_json(args.holdout_root / "e5_run_summary.json"),
            "anti_deception": read_json(args.anti_deception_root / "e5_run_summary.json"),
        }
    p3r = read_json(args.p3r_summary)
    p3r_seeds = list(p3r["config"]["seeds"])
    for suite, run in e5_runs.items():
        seeds = list(run["protocol"]["seeds"])
        if seeds != p3r_seeds:
            raise ValueError(f"{suite} and P3R outer seeds differ; paired reuse is invalid")

    suites: dict[str, Any] = {}
    for suite, run in e5_runs.items():
        if suite not in run["aggregate"]:
            raise ValueError(f"run does not contain requested suite: {suite}")
        aggregate = dict(run["aggregate"][suite])
        per_seed = dict(run["per_seed"][suite])
        if suite == "holdout":
            for setting, variant in P3R_VARIANTS.items():
                aggregate.setdefault(setting, p3r["aggregate"][variant])
                per_seed.setdefault(setting, p3r["per_seed"][variant])
        missing = [setting for setting in ORDER if setting not in aggregate or setting not in per_seed]
        if missing:
            raise ValueError(f"suite {suite} missing settings: {missing}")
        baseline = aggregate["no_deceive"]
        suites[suite] = {
            "frontier": [aggregate_row(setting, aggregate[setting], baseline) for setting in ORDER],
            "paired_vs_no_deceive": {
                setting: pairwise(per_seed, setting, "no_deceive")
                for setting in ORDER
                if setting != "no_deceive"
            },
            "paired_default_vs_no_deceive": pairwise(per_seed, "default", "no_deceive"),
            "paired_default_vs_tight": pairwise(per_seed, "default", "tight"),
            "paired_default_vs_loose": pairwise(per_seed, "default", "loose"),
            "paired_default_vs_constant_deceive": pairwise(per_seed, "default", "constant_deceive"),
        }

    result = {
        "protocol": {
            suite: run["protocol"]
            for suite, run in e5_runs.items()
        },
        "p3r_reuse": {
            "summary": str(args.p3r_summary.resolve()),
            "settings": P3R_VARIANTS,
            "paired_seed_match": True,
        },
        "suites": suites,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "e5_analysis.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
