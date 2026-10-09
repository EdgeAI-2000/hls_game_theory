"""Analyze B_phi alignment with V3 neural risk heads from paired evaluation traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from experiment_trace import write_json
from train_neural_opponent import calibration_error


def safe_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if not labels or positives == 0 or positives == len(labels):
        return None
    return float(roc_auc_score(labels, scores))


def safe_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def summarize_trace_files(paths: list[Path]) -> dict[str, Any]:
    labels: list[int] = []
    neural_scores: list[float] = []
    bphi_raw_scores: list[float] = []
    bphi_calibrated_scores: list[float] = []
    deltas: dict[str, list[float]] = defaultdict(list)
    decisions = 0

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                decisions += 1
                beliefs = row.get("belief_json") or {}
                threat_before = row.get("threat_before") or {}
                threat_delta = row.get("threat_delta") or {}
                label = int(bool(row.get("true_tenpai")))
                for pid_text, neural_value in threat_before.items():
                    belief = beliefs.get(pid_text) or {}
                    raw_value = belief.get("uncalibrated_tenpai_confidence")
                    calibrated_value = belief.get("tenpai_confidence")
                    if raw_value is None or calibrated_value is None:
                        continue
                    labels.append(label)
                    neural_scores.append(float(neural_value))
                    bphi_raw_scores.append(float(raw_value))
                    bphi_calibrated_scores.append(float(calibrated_value))
                mode = str(row.get("mode") or "unknown")
                for value in threat_delta.values():
                    deltas[mode].append(float(value))

    result: dict[str, Any] = {
        "trace_files": len(paths),
        "decisions": decisions,
        "opponent_decision_samples": len(labels),
        "positive_rate": sum(labels) / len(labels) if labels else None,
        "alignment": {
            "bphi_raw_vs_neural_pearson": safe_correlation(
                bphi_raw_scores, neural_scores
            ),
            "bphi_calibrated_vs_neural_pearson": safe_correlation(
                bphi_calibrated_scores, neural_scores
            ),
            "bphi_raw_vs_neural_mae": (
                float(np.mean(np.abs(np.array(bphi_raw_scores) - np.array(neural_scores))))
                if labels
                else None
            ),
        },
        "true_tenpai_prediction": {},
        "neural_risk_delta_by_p0_mode": {
            mode: {
                "samples": len(values),
                "mean": float(np.mean(values)) if values else None,
                "positive_rate": (
                    sum(value > 0 for value in values) / len(values)
                    if values
                    else None
                ),
            }
            for mode, values in sorted(deltas.items())
        },
    }
    for name, scores in (
        ("neural_risk_head", neural_scores),
        ("bphi_raw", bphi_raw_scores),
        ("bphi_frozen_calibrated", bphi_calibrated_scores),
    ):
        result["true_tenpai_prediction"][name] = {
            "auc": safe_auc(labels, scores),
            "brier": float(brier_score_loss(labels, scores)) if labels else None,
            "ece": calibration_error(labels, scores) if labels else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-eval-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    output: dict[str, Any] = {"profiles": {}}
    for risk_mode_dir in sorted(path for path in args.paired_eval_dir.iterdir() if path.is_dir()):
        risk_mode = risk_mode_dir.name
        for profile_dir in sorted(path for path in risk_mode_dir.iterdir() if path.is_dir()):
            profile = profile_dir.name
            profile_result = output["profiles"].setdefault(profile, {})
            mode_result = profile_result.setdefault(risk_mode, {})
            for variant in ("L2_full", "L2_no_deceive"):
                paths = sorted((profile_dir / variant).glob("seed_*/gate1_steps.jsonl"))
                if paths:
                    mode_result[variant] = summarize_trace_files(paths)

    write_json(args.output_file, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
