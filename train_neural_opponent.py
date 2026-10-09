"""Train a neural opponent policy from responsive+learned defender imitation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import brier_score_loss, roc_auc_score

from experiment_trace import write_json
from neural_opponent_policy import (
    ACTION_SPACE,
    ACTION_TO_INDEX,
    FEATURE_VERSION_V1,
    FEATURE_VERSION_V3,
    SUPPORTED_FEATURE_VERSIONS,
    NeuralOpponentNet,
    RiskAwareNeuralOpponentNet,
)


class OpponentDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        x = torch.tensor(row["features"], dtype=torch.float32)
        y = torch.tensor(ACTION_TO_INDEX[row["action"]], dtype=torch.long)
        legal = torch.zeros(len(ACTION_SPACE), dtype=torch.bool)
        for action in row["legal_actions"]:
            if action in ACTION_TO_INDEX:
                legal[ACTION_TO_INDEX[action]] = True
        risk = torch.tensor(float(row.get("risk_label", 0.0)), dtype=torch.float32)
        push = row.get("reference_push_action")
        defense = row.get("reference_defensive_action")
        response_eligible = bool(
            row.get("phase") == "turn"
            and push in ACTION_TO_INDEX
            and defense in ACTION_TO_INDEX
            and push != defense
            and push in row.get("legal_actions", [])
            and defense in row.get("legal_actions", [])
        )
        push_index = ACTION_TO_INDEX[push] if response_eligible else 0
        defense_index = ACTION_TO_INDEX[defense] if response_eligible else 0
        return (
            x,
            y,
            legal,
            risk,
            torch.tensor(push_index, dtype=torch.long),
            torch.tensor(defense_index, dtype=torch.long),
            torch.tensor(response_eligible, dtype=torch.bool),
        )


def load_rows(path: Path) -> tuple[List[Dict[str, Any]], str]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            feature_version = str(row.get("feature_version", FEATURE_VERSION_V1))
            if feature_version not in SUPPORTED_FEATURE_VERSIONS:
                raise ValueError(
                    f"Unsupported feature version in {path}: {feature_version}"
                )
            if row.get("action") not in ACTION_TO_INDEX:
                continue
            if row.get("action") not in row.get("legal_actions", []):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No valid rows loaded from {path}")
    feature_versions = {
        str(row.get("feature_version", FEATURE_VERSION_V1)) for row in rows
    }
    if len(feature_versions) != 1:
        raise ValueError(
            f"Mixed feature versions in {path}: {sorted(feature_versions)}"
        )
    dim = len(rows[0]["features"])
    bad = [idx for idx, row in enumerate(rows) if len(row["features"]) != dim]
    if bad:
        raise ValueError(f"Inconsistent feature dimensions; first bad row index={bad[0]}")
    feature_version = next(iter(feature_versions))
    if feature_version == FEATURE_VERSION_V3:
        missing_risk = [idx for idx, row in enumerate(rows) if "risk_label" not in row]
        if missing_risk:
            raise ValueError(
                f"V3 data requires risk_label; first missing row index={missing_risk[0]}"
            )
    return rows, feature_version


def calibration_error(labels: List[float], scores: List[float], bins: int = 10) -> float:
    if not labels:
        return 0.0
    total = len(labels)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            position
            for position, score in enumerate(scores)
            if lower <= score < upper or (index == bins - 1 and score == 1.0)
        ]
        if not selected:
            continue
        mean_score = sum(scores[position] for position in selected) / len(selected)
        mean_label = sum(labels[position] for position in selected) / len(selected)
        error += len(selected) / total * abs(mean_score - mean_label)
    return error


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    risk_aware: bool = False,
    blind_risk_value: float = 0.5,
    response_low_risk: float = 0.1,
    response_high_risk: float = 0.9,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    legal_correct = 0
    loss_sum = 0.0
    risk_labels: List[float] = []
    risk_scores: List[float] = []
    conditioned_action_changes = 0
    low_high_total_variation = 0.0
    response_relative_logit_delta = 0.0
    response_opportunities = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for x, y, legal, risk, push, defense, response_eligible in loader:
            x = x.to(device)
            y = y.to(device)
            legal = legal.to(device)
            if risk_aware:
                logits, risk_logits, counterfactual_logits = (
                    model.forward_with_risk_counterfactuals(
                        x,
                        [blind_risk_value, response_low_risk, response_high_risk],
                    )
                )
                blind_logits, low_logits, high_logits = counterfactual_logits
                risk_labels.extend(float(value) for value in risk.tolist())
                risk_scores.extend(
                    float(value) for value in torch.sigmoid(risk_logits).cpu().tolist()
                )
                conditioned_pred = torch.argmax(logits.masked_fill(~legal, -1e9), dim=-1)
                blind_pred = torch.argmax(
                    blind_logits.masked_fill(~legal, -1e9), dim=-1
                )
                conditioned_action_changes += int(
                    (conditioned_pred != blind_pred).sum().item()
                )
                low_probs = torch.softmax(low_logits.masked_fill(~legal, -1e9), dim=-1)
                high_probs = torch.softmax(
                    high_logits.masked_fill(~legal, -1e9), dim=-1
                )
                low_high_total_variation += float(
                    (0.5 * torch.abs(high_probs - low_probs).sum(dim=-1)).sum().item()
                )
                response_mask = response_eligible.to(device)
                if bool(response_mask.any()):
                    selected_low = low_logits[response_mask]
                    selected_high = high_logits[response_mask]
                    selected_push = push.to(device)[response_mask].unsqueeze(-1)
                    selected_defense = defense.to(device)[response_mask].unsqueeze(-1)
                    relative_delta = (
                        selected_high.gather(1, selected_defense)
                        - selected_high.gather(1, selected_push)
                        - selected_low.gather(1, selected_defense)
                        + selected_low.gather(1, selected_push)
                    )
                    response_relative_logit_delta += float(relative_delta.sum().item())
                    response_opportunities += int(response_mask.sum().item())
            else:
                logits = model(x)
            loss_sum += float(criterion(logits, y).item())
            pred = torch.argmax(logits, dim=-1)
            correct += int((pred == y).sum().item())
            masked_logits = logits.masked_fill(~legal, -1e9)
            legal_pred = torch.argmax(masked_logits, dim=-1)
            legal_correct += int((legal_pred == y).sum().item())
            total += int(y.numel())
    result = {
        "loss": loss_sum / max(1, total),
        "raw_acc": correct / max(1, total),
        "legal_masked_acc": legal_correct / max(1, total),
    }
    if risk_aware:
        positives = sum(risk_labels)
        result.update(
            {
                "risk_positive_rate": positives / max(1, len(risk_labels)),
                "risk_auc": (
                    float(roc_auc_score(risk_labels, risk_scores))
                    if 0 < positives < len(risk_labels)
                    else float("nan")
                ),
                "risk_brier": float(brier_score_loss(risk_labels, risk_scores)),
                "risk_ece": calibration_error(risk_labels, risk_scores),
                "risk_condition_action_change_rate": (
                    conditioned_action_changes / max(1, total)
                ),
                "risk_condition_total_variation": (
                    low_high_total_variation / max(1, total)
                ),
                "risk_defense_vs_push_logit_delta": (
                    response_relative_logit_delta / max(1, response_opportunities)
                ),
                "risk_response_opportunities": float(response_opportunities),
            }
        )
    return result


def train(args: argparse.Namespace) -> Dict[str, Any]:
    rows, feature_version = load_rows(args.data_file)
    rng = random.Random(args.seed)
    game_seeds = sorted({int(row["seed"]) for row in rows})
    if len(game_seeds) < 2:
        raise ValueError("Grouped train/eval split requires at least two game seeds")
    rng.shuffle(game_seeds)
    eval_game_count = max(1, min(len(game_seeds) - 1, round(len(game_seeds) * args.eval_ratio)))
    eval_game_seeds = set(game_seeds[:eval_game_count])
    train_game_seeds = set(game_seeds[eval_game_count:])
    train_rows = [row for row in rows if int(row["seed"]) in train_game_seeds]
    eval_rows = [row for row in rows if int(row["seed"]) in eval_game_seeds]
    input_dim = len(rows[0]["features"])
    architecture = args.architecture
    if architecture == "auto":
        architecture = (
            "risk_aware_multitask"
            if feature_version == FEATURE_VERSION_V3
            else "action_only"
        )
    risk_aware = architecture == "risk_aware_multitask"
    if risk_aware and feature_version != FEATURE_VERSION_V3:
        raise ValueError("risk-aware architecture requires V3 labeled data")
    risk_target_thresholds = {
        int(row["risk_target_max_shanten"])
        for row in rows
        if row.get("risk_target_max_shanten") is not None
    }
    if risk_aware and len(risk_target_thresholds) != 1:
        raise ValueError(
            f"Risk-aware data requires one risk target threshold, got {risk_target_thresholds}"
        )
    risk_target_max_shanten = (
        next(iter(risk_target_thresholds)) if risk_target_thresholds else None
    )
    train_risk_positive = sum(int(row.get("risk_label", 0)) for row in train_rows)
    train_risk_prevalence = train_risk_positive / max(1, len(train_rows))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        OpponentDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    eval_loader = DataLoader(OpponentDataset(eval_rows), batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    hidden_dims = tuple(int(value) for value in args.hidden_dims)
    if risk_aware:
        model = RiskAwareNeuralOpponentNet(
            input_dim, len(ACTION_SPACE), hidden_dims=hidden_dims
        ).to(device)
    else:
        model = NeuralOpponentNet(
            input_dim, len(ACTION_SPACE), hidden_dims=hidden_dims
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    negative_count = len(train_rows) - train_risk_positive
    positive_weight = negative_count / max(1, train_risk_positive)
    risk_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )

    best_metric = -1.0
    best_state = None
    best_legal_masked_acc = -1.0
    best_epoch = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y, _legal, risk, push, defense, response_eligible in train_loader:
            x = x.to(device)
            y = y.to(device)
            risk = risk.to(device)
            push = push.to(device)
            defense = defense.to(device)
            response_eligible = response_eligible.to(device)
            optimizer.zero_grad(set_to_none=True)
            if risk_aware:
                logits, risk_logits, counterfactual_logits = (
                    model.forward_with_risk_counterfactuals(
                        x,
                        [args.response_low_risk, args.response_high_risk],
                    )
                )
                action_loss = criterion(logits, y)
                auxiliary_risk_loss = risk_criterion(risk_logits, risk)
                loss = action_loss + args.risk_loss_weight * auxiliary_risk_loss
                if args.response_loss_weight > 0 and bool(response_eligible.any()):
                    low_logits, high_logits = counterfactual_logits
                    selected_low = low_logits[response_eligible]
                    selected_high = high_logits[response_eligible]
                    selected_push = push[response_eligible].unsqueeze(-1)
                    selected_defense = defense[response_eligible].unsqueeze(-1)
                    low_margin = selected_low.gather(1, selected_push) - selected_low.gather(
                        1, selected_defense
                    )
                    high_margin = selected_high.gather(
                        1, selected_defense
                    ) - selected_high.gather(1, selected_push)
                    response_loss = 0.5 * (
                        F.softplus(args.response_logit_margin - low_margin).mean()
                        + F.softplus(args.response_logit_margin - high_margin).mean()
                    )
                    loss = loss + args.response_loss_weight * response_loss
            else:
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())
        metrics = evaluate(
            model,
            eval_loader,
            device,
            risk_aware=risk_aware,
            blind_risk_value=train_risk_prevalence,
            response_low_risk=args.response_low_risk,
            response_high_risk=args.response_high_risk,
        )
        metrics["epoch"] = float(epoch)
        metrics["train_loss"] = total_loss / max(1, total)
        history.append(metrics)
        print(
            f"[train] epoch={epoch}/{args.epochs} train_loss={metrics['train_loss']:.4f} "
            f"eval_loss={metrics['loss']:.4f} legal_acc={metrics['legal_masked_acc']:.4f}",
            flush=True,
        )
        mechanism_eligible = bool(
            not risk_aware
            or (
                metrics["risk_auc"] >= args.selection_min_risk_auc
                and metrics["risk_condition_total_variation"]
                >= args.selection_min_action_tv
                and metrics["risk_defense_vs_push_logit_delta"] > 0
            )
        )
        metrics["mechanism_eligible"] = float(mechanism_eligible)
        selection_metric = metrics["legal_masked_acc"] + (
            0.0 if mechanism_eligible else -1.0
        )
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_legal_masked_acc = metrics["legal_masked_acc"]
            best_epoch = epoch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "neural_opponent_policy.pth"
    torch.save(
        {
            "model_state": best_state or model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "actions": ACTION_SPACE,
            "feature_version": feature_version,
            "architecture": architecture,
            "risk_target": (
                f"p0_shanten<={risk_target_max_shanten}" if risk_aware else None
            ),
            "risk_target_max_shanten": risk_target_max_shanten,
            "risk_prevalence": train_risk_prevalence,
            "response_low_risk": args.response_low_risk,
            "response_high_risk": args.response_high_risk,
            "response_loss_weight": args.response_loss_weight,
            "response_logit_margin": args.response_logit_margin,
            "profile": args.profile,
            "source_data": str(args.data_file),
        },
        checkpoint_path,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "data_file": str(args.data_file),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "split_unit": "game_seed",
        "train_game_seeds": len(train_game_seeds),
        "eval_game_seeds": len(eval_game_seeds),
        "eval_game_seed_values": sorted(eval_game_seeds),
        "eval_game_seed_sha256": hashlib.sha256(
            "\n".join(str(value) for value in sorted(eval_game_seeds)).encode("ascii")
        ).hexdigest(),
        "input_dim": input_dim,
        "feature_version": feature_version,
        "architecture": architecture,
        "profile": args.profile,
        "risk_loss_weight": args.risk_loss_weight if risk_aware else None,
        "response_loss_weight": args.response_loss_weight if risk_aware else None,
        "response_low_risk": args.response_low_risk if risk_aware else None,
        "response_high_risk": args.response_high_risk if risk_aware else None,
        "response_logit_margin": args.response_logit_margin if risk_aware else None,
        "risk_positive_weight": positive_weight if risk_aware else None,
        "train_risk_prevalence": train_risk_prevalence if risk_aware else None,
        "risk_target_max_shanten": risk_target_max_shanten,
        "hidden_dims": list(hidden_dims),
        "best_legal_masked_acc": best_legal_masked_acc,
        "best_epoch": best_epoch,
        "selection_min_risk_auc": args.selection_min_risk_auc if risk_aware else None,
        "selection_min_action_tv": args.selection_min_action_tv if risk_aware else None,
        "history": history,
    }
    write_json(args.output_dir / "training_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=Path("Neural_opponent_data/responsive_learned_teacher.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("Neural_opponent_model"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--architecture",
        choices=["auto", "action_only", "risk_aware_multitask"],
        default="auto",
    )
    parser.add_argument("--risk-loss-weight", type=float, default=1.0)
    parser.add_argument("--response-loss-weight", type=float, default=0.0)
    parser.add_argument("--response-low-risk", type=float, default=0.1)
    parser.add_argument("--response-high-risk", type=float, default=0.9)
    parser.add_argument("--response-logit-margin", type=float, default=0.5)
    parser.add_argument("--selection-min-risk-auc", type=float, default=0.70)
    parser.add_argument("--selection-min-action-tv", type=float, default=0.005)
    parser.add_argument("--profile", default="unspecified")
    args = parser.parse_args()
    summary = train(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
