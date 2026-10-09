"""Analyze the frozen V4 20-seed paired neural-opponent evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from experiment_trace import write_json


VARIANTS = ("L2_full", "L2_no_deceive")
METRICS = {
    "average_score": ("avg_net", 1),
    "excess_DIR": ("ron_game_rate", 1),
    "DR_difference": ("dealin_rate", -1),
    "CVaR_10pct": ("cvar_10pct", 1),
    "strict_behavior_FFR": ("FFR", 1),
}


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_summary(
    full_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    field: str,
    favorable_direction: int,
) -> dict[str, Any]:
    full = {int(row["seed"]): float(row[field]) for row in full_rows}
    baseline = {int(row["seed"]): float(row[field]) for row in baseline_rows}
    if full.keys() != baseline.keys():
        raise ValueError(f"Seed mismatch for {field}")
    seeds = sorted(full)
    differences = np.asarray([full[seed] - baseline[seed] for seed in seeds])
    sample_count = len(differences)
    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if sample_count > 1 else 0.0
    if sample_count > 1:
        radius = float(
            stats.t.ppf(0.975, sample_count - 1) * sd / math.sqrt(sample_count)
        )
        t_result = stats.ttest_1samp(differences, popmean=0.0)
        t_value = float(t_result.statistic)
        t_p_value = float(t_result.pvalue)
    else:
        radius = 0.0
        t_value = math.nan
        t_p_value = math.nan
    if np.allclose(differences, 0.0):
        wilcoxon_statistic = 0.0
        wilcoxon_p_value = 1.0
    else:
        wilcoxon = stats.wilcoxon(differences, alternative="two-sided")
        wilcoxon_statistic = float(wilcoxon.statistic)
        wilcoxon_p_value = float(wilcoxon.pvalue)
    favorable = differences * favorable_direction
    return {
        "n": sample_count,
        "seeds": seeds,
        "mean_difference": mean,
        "sd_difference": sd,
        "ci_95": [mean - radius, mean + radius],
        "t": t_value,
        "t_p_value": t_p_value,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_value": wilcoxon_p_value,
        "favorable_seed_count": int(np.sum(favorable > 0)),
        "tie_seed_count": int(np.sum(favorable == 0)),
        "direction_favorable": bool(mean * favorable_direction > 0),
    }


def audit_units(output_dir: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    evaluation = freeze["evaluation"]
    expected_units = (
        len(evaluation["profiles"])
        * len(evaluation["variants"])
        * len(evaluation["outer_seeds"])
    )
    summaries = sorted(output_dir.glob("predicted/*/L2_*/seed_*/gate1_summary.json"))
    failures: list[str] = []
    for path in summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        ladder = summary["E2_ladder"]["llm_mask"]
        if int(ladder["games"]) != int(evaluation["games_per_seed"]):
            failures.append(f"wrong game count: {path}")
        if float(ladder["end_state"]["max_steps_or_unsettled_rate"]) != 0.0:
            failures.append(f"unsettled game: {path}")
    return {
        "expected_units": expected_units,
        "completed_units": len(summaries),
        "expected_games": expected_units * int(evaluation["games_per_seed"]),
        "completed_games": sum(
            int(
                json.loads(path.read_text(encoding="utf-8"))["E2_ladder"]
                ["llm_mask"]["games"]
            )
            for path in summaries
        ),
        "failures": failures,
        "passed": len(summaries) == expected_units and not failures,
    }


def format_effect(value: float, metric: str) -> str:
    if metric in {"excess_DIR", "DR_difference", "strict_behavior_FFR"}:
        return f"{value * 100:+.3f}pp"
    return f"{value:+.3f}"


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# V4 风险感知神经对手 20-seed 扩展评测结果",
        "",
        "> 本报告由冻结的 fresh 20 outer-seed 评测生成；5-seed 功效预检未并入统计。",
        "",
        "## 1. 完整性",
        "",
        "| 单元 | 对局 | max-steps/未结算 | 判定 |",
        "|---:|---:|---:|---|",
        (
            f"| {result['audit']['completed_units']}/{result['audit']['expected_units']} "
            f"| {result['audit']['completed_games']}/{result['audit']['expected_games']} "
            f"| {len(result['audit']['failures'])} | "
            f"{'**通过**' if result['audit']['passed'] else '**不通过**'} |"
        ),
        "",
        "## 2. Full minus no-deceive",
        "",
        "| Profile | 平均分差 | excess-DIR | DR 差 | CVaR 差 | 严格行为 FFR 差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for profile, metrics in result["profiles"].items():
        cells = []
        for metric in (
            "average_score",
            "excess_DIR",
            "DR_difference",
            "CVaR_10pct",
            "strict_behavior_FFR",
        ):
            item = metrics[metric]
            cells.append(
                f"{format_effect(item['mean_difference'], metric)} "
                f"(`p={item['t_p_value']:.3g}`, Holm `{item['holm_t_p_value']:.3g}`)"
            )
        lines.append(f"| {profile} | " + " | ".join(cells) + " |")
    decisions = result["decisions"]
    lines.extend(
        [
            "",
            "## 3. 冻结判定",
            "",
            "| 主张 | 判定 |",
            "|---|---|",
            f"| DIR 方向稳定 | **{decisions['DIR_stability']}** |",
            f"| DR 方向稳定 | **{decisions['DR_stability']}** |",
            f"| 平均分稳定显著提高 | **{decisions['average_score_stability']}** |",
            "| 严格行为 FFR | 仅作诊断，不作为通过条件 |",
            "",
            "每个 profile 的完整效应、95% CI、配对 t-test、Wilcoxon 和 Holm 校正见同目录 JSON。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-eval-dir", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    profiles: dict[str, dict[str, Any]] = {}
    for profile in freeze["evaluation"]["profiles"]:
        summary_path = (
            args.paired_eval_dir / "predicted" / profile / "e3_ablation_summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        profile_result: dict[str, Any] = {}
        for metric, (field, favorable_direction) in METRICS.items():
            profile_result[metric] = paired_summary(
                summary["per_seed"][VARIANTS[0]],
                summary["per_seed"][VARIANTS[1]],
                field,
                favorable_direction,
            )
        profiles[profile] = profile_result

    for metric in METRICS:
        profile_names = list(profiles)
        adjusted = holm_adjust(
            [profiles[name][metric]["t_p_value"] for name in profile_names]
        )
        for name, p_value in zip(profile_names, adjusted):
            profiles[name][metric]["holm_t_p_value"] = p_value

    def stable(metric: str) -> bool:
        items = [profile[metric] for profile in profiles.values()]
        return all(item["direction_favorable"] for item in items) and sum(
            item["holm_t_p_value"] < 0.05 for item in items
        ) >= 2

    result = {
        "status": "complete",
        "audit": audit_units(args.paired_eval_dir, freeze),
        "profiles": profiles,
        "decisions": {
            "DIR_stability": "pass" if stable("excess_DIR") else "not_pass",
            "DR_stability": "pass" if stable("DR_difference") else "not_pass",
            "average_score_stability": (
                "pass" if stable("average_score") else "not_pass"
            ),
        },
    }
    if not result["audit"]["passed"]:
        raise RuntimeError(f"Integrity audit failed: {result['audit']}")
    write_json(args.output_json, result)
    write_report(args.output_report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
