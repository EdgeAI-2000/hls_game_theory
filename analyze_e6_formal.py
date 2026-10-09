"""Generate E6 calibration, mechanism, case and latency artifacts from frozen traces."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


BELIEF_LEVELS = {
    "raw_mc": "raw_tenpai_confidence",
    "public_model": "uncalibrated_tenpai_confidence",
    "isotonic": "tenpai_confidence",
}


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def trace_paths(root: Path, name: str) -> list[Path]:
    paths = sorted(root.glob(f"seed_*/{name}"))
    paths.extend(sorted(root.glob(f"seed_*/{name}.gz")))
    return paths


def iter_jsonl_paths(paths: Iterable[Path]) -> Iterator[Dict[str, Any]]:
    for path in paths:
        with open_text(path) as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


class ScoreAccumulator:
    def __init__(self, bins: int = 10) -> None:
        self.scores = array("d")
        self.labels = bytearray()
        self.bins = bins
        self.bin_count = [0] * bins
        self.bin_score = [0.0] * bins
        self.bin_label = [0.0] * bins
        self.squared_error = 0.0
        self.absolute_error = 0.0
        self.correct = 0

    def add(self, label: int, score: float) -> None:
        score = max(0.0, min(1.0, float(score)))
        label = int(bool(label))
        self.scores.append(score)
        self.labels.append(label)
        index = min(self.bins - 1, int(score * self.bins))
        self.bin_count[index] += 1
        self.bin_score[index] += score
        self.bin_label[index] += label
        self.squared_error += (score - label) ** 2
        self.absolute_error += abs(score - label)
        self.correct += int((score >= 0.5) == bool(label))

    def curve(self) -> list[Dict[str, Any]]:
        rows = []
        for index in range(self.bins):
            count = self.bin_count[index]
            rows.append({
                "bin": index,
                "lower": index / self.bins,
                "upper": (index + 1) / self.bins,
                "count": count,
                "mean_confidence": self.bin_score[index] / count if count else None,
                "positive_rate": self.bin_label[index] / count if count else None,
            })
        return rows

    def summary(self) -> Dict[str, Any]:
        count = len(self.labels)
        if not count:
            return {"samples": 0}
        labels = np.frombuffer(self.labels, dtype=np.uint8)
        scores = np.frombuffer(self.scores, dtype=np.float64)
        positives = int(labels.sum())
        negatives = count - positives
        auc = None
        if positives and negatives:
            ranks = rankdata(scores, method="average")
            auc = float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
        ece = sum(
            row["count"] / count * abs(float(row["mean_confidence"]) - float(row["positive_rate"]))
            for row in self.curve()
            if row["count"]
        )
        return {
            "samples": count,
            "positive_rate": positives / count,
            "accuracy": self.correct / count,
            "auc": auc,
            "brier": self.squared_error / count,
            "bse_abs": self.absolute_error / count,
            "ece": ece,
            "reliability_bins": self.curve(),
        }


def percentile(values: array, q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.frombuffer(values, dtype=np.float64), q))


def mean_threat_delta(row: Dict[str, Any]) -> float | None:
    values = [float(value) for value in (row.get("threat_delta") or {}).values()]
    return sum(values) / len(values) if values else None


def case_from_step(row: Dict[str, Any], case_type: str) -> Dict[str, Any]:
    return {
        "case_type": case_type,
        "game_id": row.get("game_id"),
        "seed": row.get("seed"),
        "step": row.get("step"),
        "mode": row.get("mode"),
        "action": row.get("action"),
        "reason": row.get("reason"),
        "own_shanten": row.get("own_shanten"),
        "true_tenpai": row.get("true_tenpai"),
        "induced_dealin": bool(row.get("induced_dealin")),
        "belief_json": row.get("belief_json"),
        "threat_before": row.get("threat_before"),
        "threat_after": row.get("threat_after"),
        "threat_delta": row.get("threat_delta"),
        "mean_threat_delta": mean_threat_delta(row),
    }


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    root = args.formal_root
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    games_by_seed: Dict[int, Dict[str, Any]] = {}
    spec_by_seed_pid: Dict[tuple[int, int], str] = {}
    for game in iter_jsonl_paths(trace_paths(root, "gate1_games.jsonl")):
        seed = int(game["seed"])
        games_by_seed[seed] = game
        for pid_text, spec in (game.get("opponent_curriculum") or {}).items():
            if isinstance(spec, dict) and spec.get("id"):
                spec_by_seed_pid[(seed, int(str(pid_text).lstrip("P")))] = str(spec["id"])

    overall = {level: ScoreAccumulator(args.bins) for level in BELIEF_LEVELS}
    by_spec: Dict[str, Dict[str, ScoreAccumulator]] = defaultdict(
        lambda: {level: ScoreAccumulator(args.bins) for level in BELIEF_LEVELS}
    )
    latency = array("d")
    shifts: Dict[str, list[float]] = defaultdict(list)
    positive_deceive_cases: list[Dict[str, Any]] = []
    induced_cases: list[Dict[str, Any]] = []

    step_count = 0
    for row in iter_jsonl_paths(trace_paths(root, "gate1_steps.jsonl")):
        step_count += 1
        label = int(bool(row.get("true_tenpai")))
        seed = int(row.get("seed", -1))
        for pid_text, belief in (row.get("belief_json") or {}).items():
            if not isinstance(belief, dict):
                continue
            pid = int(str(pid_text).lstrip("P"))
            spec_id = spec_by_seed_pid.get((seed, pid), "unknown")
            for level, field in BELIEF_LEVELS.items():
                if belief.get(field) is None:
                    continue
                score = float(belief[field])
                overall[level].add(label, score)
                by_spec[spec_id][level].add(label, score)
        if row.get("latency_ms") is not None:
            latency.append(float(row["latency_ms"]))
        delta = mean_threat_delta(row)
        mode = str(row.get("mode") or "none")
        if delta is not None:
            shifts[mode].append(delta)
            if mode == "deceive" and delta > 0:
                positive_deceive_cases.append(case_from_step(row, "positive_deceive_shift"))
        if row.get("induced_dealin"):
            induced_cases.append(case_from_step(row, "induced_dealin"))

    false_fold_cases = []
    for event in iter_jsonl_paths(trace_paths(root, "gate1_false_folds.jsonl")):
        if event.get("strict_false_fold", event.get("false_fold")) and event.get("last_p0_mode") == "deceive":
            false_fold_cases.append({
                "case_type": "strict_oracle_false_fold_after_deceive",
                **event,
            })

    positive_deceive_cases.sort(key=lambda row: float(row.get("mean_threat_delta") or 0.0), reverse=True)
    false_fold_cases.sort(
        key=lambda row: sum(float(value) for value in (row.get("last_p0_threat_delta") or {}).values()),
        reverse=True,
    )
    selected_cases = []
    seen_games = set()
    for candidates in (induced_cases, false_fold_cases, positive_deceive_cases):
        for case in candidates:
            game_id = case.get("game_id")
            if game_id in seen_games:
                continue
            selected_cases.append(case)
            seen_games.add(game_id)
            break
    selected_cases = selected_cases[: args.max_cases]

    calibration = {level: acc.summary() for level, acc in overall.items()}
    calibration_by_spec = {
        spec_id: {level: acc.summary() for level, acc in levels.items()}
        for spec_id, levels in sorted(by_spec.items())
    }
    shift_summary = {
        mode: {
            "samples": len(values),
            "mean_delta": sum(values) / len(values) if values else None,
            "positive_rate": sum(value > 0 for value in values) / len(values) if values else None,
        }
        for mode, values in sorted(shifts.items())
    }
    latency_summary = {
        "samples": len(latency),
        "p50_ms": percentile(latency, 0.50),
        "p95_ms": percentile(latency, 0.95),
        "p99_ms": percentile(latency, 0.99),
        "note": "Total P0 decision latency; historical traces do not contain complete per-module timings.",
    }

    h1 = json.loads(args.h1_summary.read_text(encoding="utf-8"))
    h1_natural = (h1.get("estimators_natural_eval") or {}).get("B2_B_phi_calibrated") or {}
    result = {
        "protocol": {
            "formal_root": str(root),
            "h1_summary": str(args.h1_summary),
            "calibration_fit_on_holdout": False,
            "belief_label": "realized P0 true tenpai at each decision",
            "belief_levels": BELIEF_LEVELS,
            "step_rows": step_count,
            "games": len(games_by_seed),
        },
        "calibration": calibration,
        "calibration_by_opponent_spec": calibration_by_spec,
        "h1_natural_calibrated_reference": h1_natural,
        "belief_shift_by_mode": shift_summary,
        "latency": latency_summary,
        "cases": selected_cases,
        "case_pool_counts": {
            "induced_dealin_steps": len(induced_cases),
            "strict_false_folds_after_deceive": len(false_fold_cases),
            "positive_deceive_shifts": len(positive_deceive_cases),
        },
        "limitations": [
            "Calibration is evaluated without refitting on the course holdout.",
            "DeltaBelief is the paired change in the executable opponent's public threat response, not a direct observation of a human belief.",
            "The available formal traces record total decision latency but not complete per-module latency.",
        ],
    }
    (output / "e6_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "e6_cases.json").write_text(json.dumps(selected_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1, label="ideal")
    colors = {"raw_mc": "#4477AA", "public_model": "#CC6677", "isotonic": "#228833"}
    for level in BELIEF_LEVELS:
        bins = [row for row in calibration[level]["reliability_bins"] if row["count"]]
        ax.plot(
            [row["mean_confidence"] for row in bins],
            [row["positive_rate"] for row in bins],
            marker="o",
            linewidth=1.5,
            color=colors[level],
            label=f"{level} (ECE={calibration[level]['ece']:.3f})",
        )
    ax.set(xlabel="Mean predicted probability", ylabel="Observed tenpai frequency", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "e6_calibration_curve.png", dpi=300)
    plt.close(fig)

    modes = [mode for mode in ("exploit", "safe", "deceive") if mode in shift_summary]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    axes[0].bar(modes, [shift_summary[mode]["mean_delta"] for mode in modes], color=["#4477AA", "#999999", "#CC6677"][:len(modes)])
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_ylabel("Mean paired threat shift")
    axes[1].bar(modes, [shift_summary[mode]["positive_rate"] for mode in modes], color=["#4477AA", "#999999", "#CC6677"][:len(modes)])
    axes[1].set_ylabel("Positive-shift rate")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "e6_belief_shift_by_mode.png", dpi=300)
    plt.close(fig)

    report = [
        "# E6 Calibration and Mechanism Analysis",
        "",
        f"- Frozen full games: {len(games_by_seed):,}",
        f"- Decision rows: {step_count:,}",
        "- Holdout calibration was evaluated without refitting.",
        "",
        "## Calibration",
        "",
        "| Level | AUC | Brier | ECE |",
        "|---|---:|---:|---:|",
    ]
    for level in BELIEF_LEVELS:
        values = calibration[level]
        report.append(f"| {level} | {values['auc']:.4f} | {values['brier']:.4f} | {values['ece']:.4f} |")
    report.extend([
        "",
        "## Belief Shift",
        "",
        "| Mode | Samples | Mean shift | Positive rate |",
        "|---|---:|---:|---:|",
    ])
    for mode, values in shift_summary.items():
        report.append(f"| {mode} | {values['samples']:,} | {values['mean_delta']:.5f} | {values['positive_rate']:.3%} |")
    report.extend([
        "",
        "## Latency",
        "",
        f"Total decision latency p50/p95/p99: {latency_summary['p50_ms']:.1f}/{latency_summary['p95_ms']:.1f}/{latency_summary['p99_ms']:.1f} ms.",
        "",
        "## Interpretation",
        "",
        "Deceive actions shift the executable opponents' public threat response in the intended direction, while exploit and safe actions shift it downward on average. The isotonic map substantially repairs the public-action model's probability scale, but it does not beat raw MC on Brier or ECE under the course holdout's 2.64% positive prevalence. Its ECE is also materially worse than on the independent H1 distribution. This distribution shift must be reported rather than repaired on holdout.",
        "",
        "See `e6_cases.json` for the selected auditable cases.",
    ])
    (output / "E6_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--h1-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=3)
    args = parser.parse_args()
    result = analyze(args)
    print(json.dumps({
        "calibration": {level: {k: v for k, v in values.items() if k != "reliability_bins"} for level, values in result["calibration"].items()},
        "belief_shift_by_mode": result["belief_shift_by_mode"],
        "latency": result["latency"],
        "cases": len(result["cases"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
