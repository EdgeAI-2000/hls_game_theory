"""Audit V3 risk prediction and counterfactual risk-conditioned action response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score

from experiment_trace import write_json
from neural_opponent_policy import (
    ACTION_SPACE,
    ACTION_TO_INDEX,
    RiskAwareNeuralOpponentNet,
)
from train_neural_opponent import calibration_error


def load_eval_rows(data_path: Path, eval_seeds: set[int]) -> list[dict[str, Any]]:
    rows = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["seed"]) in eval_seeds:
                rows.append(row)
    return rows


def masked_probabilities(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.masked_fill(~legal, -1e9), dim=-1)


def safe_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def audit_profile(
    profile: dict[str, Any],
    low_risk: float,
    high_risk: float,
    batch_size: int,
) -> dict[str, Any]:
    checkpoint_path = Path(profile["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "risk_aware_multitask":
        raise ValueError(f"Not a risk-aware checkpoint: {checkpoint_path}")
    model = RiskAwareNeuralOpponentNet(
        int(checkpoint["input_dim"]),
        len(ACTION_SPACE),
        hidden_dims=tuple(checkpoint.get("hidden_dims", (256, 128))),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    training = profile["training_summary"]
    eval_seeds = {int(value) for value in training["eval_game_seed_values"]}
    data_path = Path(profile["data_summary"]["output_file"])
    rows = load_eval_rows(data_path, eval_seeds)
    if not rows:
        raise ValueError(f"No evaluation rows for profile {profile['profile']}")

    labels: list[float] = []
    scores: list[float] = []
    teacher_scores: list[float] = []
    model_scores_for_teacher: list[float] = []
    predicted_blind_changes = 0
    low_high_changes = 0
    response_opportunities = 0
    switches_toward_defense = 0
    defense_probability_deltas: list[float] = []
    push_probability_deltas: list[float] = []
    defense_vs_push_logit_deltas: list[float] = []
    total_variation_deltas: list[float] = []
    prevalence = float(checkpoint.get("risk_prevalence", 0.5))

    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            x = torch.tensor(
                [row["features"] for row in batch], dtype=torch.float32
            )
            legal = torch.zeros((len(batch), len(ACTION_SPACE)), dtype=torch.bool)
            for row_index, row in enumerate(batch):
                for action in row["legal_actions"]:
                    if action in ACTION_TO_INDEX:
                        legal[row_index, ACTION_TO_INDEX[action]] = True

            predicted_logits, risk_logits = model(x)
            blind_logits, _ = model(x, risk_override=prevalence)
            low_logits, _ = model(x, risk_override=low_risk)
            high_logits, _ = model(x, risk_override=high_risk)
            predicted_probs = masked_probabilities(predicted_logits, legal)
            blind_probs = masked_probabilities(blind_logits, legal)
            low_probs = masked_probabilities(low_logits, legal)
            high_probs = masked_probabilities(high_logits, legal)
            predicted_actions = torch.argmax(predicted_probs, dim=-1)
            blind_actions = torch.argmax(blind_probs, dim=-1)
            low_actions = torch.argmax(low_probs, dim=-1)
            high_actions = torch.argmax(high_probs, dim=-1)
            predicted_blind_changes += int(
                (predicted_actions != blind_actions).sum().item()
            )
            low_high_changes += int((low_actions != high_actions).sum().item())
            total_variation_deltas.extend(
                float(value)
                for value in (0.5 * torch.abs(high_probs - low_probs).sum(dim=-1)).tolist()
            )

            batch_scores = torch.sigmoid(risk_logits).tolist()
            for row_index, (row, score) in enumerate(zip(batch, batch_scores)):
                labels.append(float(row["risk_label"]))
                scores.append(float(score))
                teacher_score = row.get("teacher_risk_probability")
                if teacher_score is not None:
                    teacher_scores.append(float(teacher_score))
                    model_scores_for_teacher.append(float(score))

                push = row.get("reference_push_action")
                defense = row.get("reference_defensive_action")
                if (
                    row.get("phase") != "turn"
                    or push not in ACTION_TO_INDEX
                    or defense not in ACTION_TO_INDEX
                    or push == defense
                    or push not in row["legal_actions"]
                    or defense not in row["legal_actions"]
                ):
                    continue
                response_opportunities += 1
                push_index = ACTION_TO_INDEX[push]
                defense_index = ACTION_TO_INDEX[defense]
                defense_probability_deltas.append(
                    float(
                        high_probs[row_index, defense_index]
                        - low_probs[row_index, defense_index]
                    )
                )
                push_probability_deltas.append(
                    float(
                        high_probs[row_index, push_index]
                        - low_probs[row_index, push_index]
                    )
                )
                defense_vs_push_logit_deltas.append(
                    float(
                        (
                            high_logits[row_index, defense_index]
                            - high_logits[row_index, push_index]
                        )
                        - (
                            low_logits[row_index, defense_index]
                            - low_logits[row_index, push_index]
                        )
                    )
                )
                if (
                    int(high_actions[row_index]) == defense_index
                    and int(low_actions[row_index]) != defense_index
                ):
                    switches_toward_defense += 1

    positives = sum(labels)
    return {
        "profile": profile["profile"],
        "checkpoint": str(checkpoint_path),
        "eval_rows": len(rows),
        "eval_game_seeds": len(eval_seeds),
        "risk_prediction": {
            "positive_rate": positives / len(labels),
            "auc": (
                float(roc_auc_score(labels, scores))
                if 0 < positives < len(labels)
                else None
            ),
            "brier": float(brier_score_loss(labels, scores)),
            "ece": calibration_error(labels, scores),
            "teacher_risk_pearson": safe_correlation(
                model_scores_for_teacher, teacher_scores
            ),
        },
        "risk_condition_response": {
            "low_risk": low_risk,
            "high_risk": high_risk,
            "predicted_vs_blind_argmax_change_rate": (
                predicted_blind_changes / len(rows)
            ),
            "low_vs_high_argmax_change_rate": low_high_changes / len(rows),
            "mean_action_total_variation": float(np.mean(total_variation_deltas)),
            "reference_response_opportunities": response_opportunities,
            "mean_defense_probability_delta": (
                float(np.mean(defense_probability_deltas))
                if defense_probability_deltas
                else None
            ),
            "mean_push_probability_delta": (
                float(np.mean(push_probability_deltas))
                if push_probability_deltas
                else None
            ),
            "mean_defense_vs_push_logit_delta": (
                float(np.mean(defense_vs_push_logit_deltas))
                if defense_vs_push_logit_deltas
                else None
            ),
            "switch_toward_reference_defense_rate": (
                switches_toward_defense / response_opportunities
                if response_opportunities
                else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--low-risk", type=float, default=0.1)
    parser.add_argument("--high-risk", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--auc-threshold", type=float, default=0.70)
    parser.add_argument("--min-action-tv", type=float, default=0.005)
    args = parser.parse_args()

    manifest = json.loads(
        (args.training_dir / "training_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("failures"):
        raise RuntimeError("Cannot probe a failed training manifest")
    profiles = [
        audit_profile(profile, args.low_risk, args.high_risk, args.batch_size)
        for profile in manifest["profiles"]
    ]
    for result in profiles:
        prediction = result["risk_prediction"]
        response = result["risk_condition_response"]
        result["acceptance"] = {
            "risk_auc": bool(
                prediction["auc"] is not None
                and prediction["auc"] >= args.auc_threshold
            ),
            "action_condition_used": bool(
                response["mean_action_total_variation"] >= args.min_action_tv
            ),
            "defense_direction": bool(
                response["mean_defense_vs_push_logit_delta"] is not None
                and response["mean_defense_vs_push_logit_delta"] > 0
            ),
        }
        result["acceptance"]["passed"] = all(result["acceptance"].values())

    overall_passed = all(result["acceptance"]["passed"] for result in profiles)
    output = {
        "status": "passed" if overall_passed else "failed",
        "thresholds": {
            "risk_auc": args.auc_threshold,
            "minimum_mean_action_total_variation": args.min_action_tv,
            "defense_direction": "high-risk increases defense-vs-push relative logit",
        },
        "profiles": profiles,
    }
    write_json(args.output_file, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
