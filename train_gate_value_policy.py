"""Train a three-output MASK mode-value network from paired CRN returns."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from experiment_trace import write_json
from gate_value_policy import (
    GATE_VALUE_FEATURE_ABLATIONS,
    GATE_VALUE_FEATURE_DIM,
    GATE_VALUE_FEATURE_NAMES,
    GateValueNet,
    apply_gate_value_feature_ablation,
    apply_gate_value_mode_mask,
)
from neural_gate_policy import (
    MODES,
    MODE_TO_INDEX,
    NORMALIZATION_STD_FLOOR,
    NORMALIZED_FEATURE_CLIP,
    prepare_normalization_tensors,
)


class ModeValueDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], mean: torch.Tensor, std: torch.Tensor):
        self.rows = rows
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        features = torch.tensor(row["features"], dtype=torch.float32)
        features = ((features - self.mean) / self.std).clamp(
            -NORMALIZED_FEATURE_CLIP, NORMALIZED_FEATURE_CLIP
        )
        return {
            "features": features,
            "values": torch.tensor(row["values"], dtype=torch.float32),
            "targets": torch.tensor(row["targets"], dtype=torch.float32),
            "mask": torch.tensor(row["mask"], dtype=torch.bool),
            "best": torch.tensor(row["best"], dtype=torch.long),
            "raw_best": torch.tensor(row["raw_best"], dtype=torch.long),
            "weight": torch.tensor(row["weight"], dtype=torch.float32),
        }


def paired_standard_error(source: Dict[str, Any], mode: str) -> float:
    stored = source.get("pairwise_standard_errors") or {}
    key = f"{mode}_vs_exploit"
    if key in stored:
        return max(0.0, float(stored[key]))
    rollouts = source.get("mode_rollouts") or {}
    left = (rollouts.get(mode) or {}).get("returns") or []
    right = (rollouts.get("exploit") or {}).get("returns") or []
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    if len(differences) <= 1:
        return 0.0
    return statistics.stdev(differences) / math.sqrt(len(differences))


def load_rows(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    skipped: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    raw_labels: Counter[str] = Counter()
    availability: Counter[str] = Counter()
    confidence_standard_errors: List[float] = []
    confidence_label_changes = 0
    uncertain_rows = 0
    opponent_distributions: Counter[str] = Counter()
    curriculum_ids: Counter[str] = Counter()
    allowed_modes = tuple(mode for mode in MODES if mode in set(args.allowed_modes))
    for input_path in args.input:
        for line in input_path.open("r", encoding="utf-8"):
            if not line.strip():
                continue
            source = json.loads(line)
            distribution = source.get("opponent_distribution") or {}
            opponent_distributions[str(distribution.get("style") or "unknown")] += 1
            curriculum_ids.update(
                str(value)
                for value in distribution.get("curriculum_ids", [])
                if value
            )
            features = source.get("features") or []
            available = [
                mode
                for mode in source.get("available_modes", [])
                if mode in MODE_TO_INDEX and mode in allowed_modes
            ]
            mode_values = source.get("mode_values") or {}
            if len(features) != GATE_VALUE_FEATURE_DIM:
                skipped["feature_dim"] += 1
                continue
            source_belief_max = float(
                features[GATE_VALUE_FEATURE_NAMES.index("belief_max")]
            )
            features = apply_gate_value_feature_ablation(features, args.feature_ablation)
            features = apply_gate_value_mode_mask(features, allowed_modes)
            if len(available) < 2 or any(mode not in mode_values for mode in available):
                skipped["mode_values"] += 1
                continue
            values = [0.0] * len(MODES)
            targets = [0.0] * len(MODES)
            mask = [False] * len(MODES)
            reference = float(
                mode_values.get(
                    "exploit",
                    sum(float(mode_values[mode]) for mode in available) / len(available),
                )
            )
            adjusted_values = {
                mode: float(mode_values[mode]) for mode in available
            }
            if "exploit" in available and args.confidence_z > 0.0:
                for mode in available:
                    if mode == "exploit":
                        continue
                    standard_error = paired_standard_error(source, mode)
                    confidence_standard_errors.append(standard_error)
                    adjusted_values[mode] -= args.confidence_z * standard_error
            raw_best_mode = max(available, key=lambda mode: float(mode_values[mode]))
            best_mode = max(available, key=lambda mode: adjusted_values[mode])
            if (
                best_mode != "exploit"
                and "exploit" in available
                and adjusted_values[best_mode]
                < adjusted_values["exploit"] + args.min_confidence_advantage
            ):
                best_mode = "exploit"
            label_changed = best_mode != raw_best_mode
            confidence_label_changes += int(label_changed)
            uncertain_rows += int(label_changed)
            available_values = []
            for mode in available:
                index = MODE_TO_INDEX[mode]
                value = float(mode_values[mode])
                values[index] = value
                targets[index] = max(
                    -args.target_clip,
                    min(
                        args.target_clip,
                        (adjusted_values[mode] - reference) / args.value_scale,
                    ),
                )
                mask[index] = True
                available_values.append(value)
                availability[mode] += 1
            value_range = max(available_values) - min(available_values)
            if value_range < args.min_value_range:
                skipped["low_value_range"] += 1
                continue
            labels[best_mode] += 1
            raw_labels[raw_best_mode] += 1
            confidence_weight = args.uncertain_weight if label_changed else 1.0
            rows.append(
                {
                    "features": [float(value) for value in features],
                    "values": values,
                    "targets": targets,
                    "mask": mask,
                    "best": MODE_TO_INDEX[best_mode],
                    "raw_best": MODE_TO_INDEX[raw_best_mode],
                    "weight": confidence_weight * (
                        1.0 + min(3.0, value_range / args.reward_weight_scale)
                    ),
                    "group": int(source.get("game_seed", len(rows))),
                    # Keep the pre-ablation value so full/no-belief variants can
                    # use exactly the same coverage weights.
                    "source_belief_max": source_belief_max,
                }
            )
    belief_balance = {"enabled": False}
    if rows and args.belief_balance_strength > 0.0:
        belief_index = GATE_VALUE_FEATURE_NAMES.index("belief_max")
        edges = sorted(float(edge) for edge in args.belief_bin_edges)

        def bucket(value: float) -> int:
            return sum(value >= edge for edge in edges)

        bucket_counts = Counter(
            bucket(float(row["source_belief_max"])) for row in rows
        )
        largest = max(bucket_counts.values())
        raw_multipliers = {
            bin_index: min(
                args.max_belief_balance_weight,
                (largest / count) ** args.belief_balance_strength,
            )
            for bin_index, count in bucket_counts.items()
        }
        mean_multiplier = sum(
            bucket_counts[bin_index] * multiplier
            for bin_index, multiplier in raw_multipliers.items()
        ) / len(rows)
        normalized_multipliers = {
            bin_index: multiplier / max(1e-9, mean_multiplier)
            for bin_index, multiplier in raw_multipliers.items()
        }
        for row in rows:
            bin_index = bucket(float(row["source_belief_max"]))
            row["weight"] *= normalized_multipliers[bin_index]
            row["belief_bin"] = bin_index
        belief_balance = {
            "enabled": True,
            "feature": "belief_max",
            "edges": edges,
            "counts": {str(key): value for key, value in sorted(bucket_counts.items())},
            "weight_multipliers": {
                str(key): value for key, value in sorted(normalized_multipliers.items())
            },
            "strength": args.belief_balance_strength,
            "max_weight": args.max_belief_balance_weight,
        }
    return rows, {
        "rows": len(rows),
        "skipped": dict(skipped),
        "best_mode_labels": dict(labels),
        "raw_best_mode_labels": dict(raw_labels),
        "mode_availability": dict(availability),
        "confidence_label_changes": confidence_label_changes,
        "uncertain_rows": uncertain_rows,
        "belief_balance": belief_balance,
        "paired_standard_error_mean": (
            statistics.mean(confidence_standard_errors)
            if confidence_standard_errors else 0.0
        ),
        "paired_standard_error_median": (
            statistics.median(confidence_standard_errors)
            if confidence_standard_errors else 0.0
        ),
        "feature_ablation": args.feature_ablation,
        "allowed_modes": list(allowed_modes),
        "opponent_distributions": dict(opponent_distributions),
        "curriculum_ids": dict(curriculum_ids),
        "input_files": [str(path.resolve()) for path in args.input],
    }


def group_split_indices(
    rows: List[Dict[str, Any]], eval_ratio: float, seed: int
) -> Tuple[List[int], List[int]]:
    groups: Dict[int, List[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(int(row["group"]), []).append(index)
    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    target_eval = max(1, int(len(rows) * eval_ratio))
    eval_indices: List[int] = []
    train_indices: List[int] = []
    for group_id in group_ids:
        destination = eval_indices if len(eval_indices) < target_eval else train_indices
        destination.extend(groups[group_id])
    if not train_indices:
        moved_group = group_ids[-1]
        moved = set(groups[moved_group])
        train_indices = list(moved)
        eval_indices = [index for index in eval_indices if index not in moved]
    return train_indices, eval_indices


def compute_loss(
    model: GateValueNet,
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(batch["features"])
    masked_outputs = outputs.masked_fill(~batch["mask"], -1e9)
    regression = F.smooth_l1_loss(outputs, batch["targets"], reduction="none")
    regression = (regression * batch["mask"] * batch["weight"].unsqueeze(1)).sum()
    regression = regression / batch["mask"].sum().clamp_min(1)
    ranking = F.cross_entropy(masked_outputs, batch["best"], reduction="none")
    ranking = (ranking * batch["weight"]).mean()
    probs = torch.softmax(masked_outputs / args.value_temperature, dim=-1)
    centered_values = batch["targets"] - (
        (batch["targets"] * batch["mask"]).sum(dim=1, keepdim=True)
        / batch["mask"].sum(dim=1, keepdim=True).clamp_min(1)
    )
    expected_value = (probs * centered_values * batch["mask"]).sum(dim=1).mean()
    entropy = -(probs * probs.clamp_min(1e-8).log() * batch["mask"]).sum(dim=1).mean()
    loss = (
        args.regression_weight * regression
        + args.ranking_weight * ranking
        - args.expected_value_weight * expected_value
        - args.entropy_weight * entropy
    )
    return loss, outputs


def evaluate(
    model: GateValueNet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    total = correct = 0
    loss_sum = selected_value = oracle_value = raw_oracle_value = exploit_value = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, outputs = compute_loss(model, batch, args)
            masked = outputs.masked_fill(~batch["mask"], -1e9)
            selected = torch.argmax(masked, dim=-1)
            selected_value += float(
                batch["values"].gather(1, selected.unsqueeze(1)).sum().item()
            )
            oracle_value += float(
                batch["values"].gather(1, batch["best"].unsqueeze(1)).sum().item()
            )
            raw_oracle_value += float(
                batch["values"].gather(1, batch["raw_best"].unsqueeze(1)).sum().item()
            )
            exploit_index = MODE_TO_INDEX["exploit"]
            exploit_value += float(batch["values"][:, exploit_index].sum().item())
            correct += int((selected == batch["best"]).sum().item())
            total += int(selected.numel())
            loss_sum += float(loss.item()) * int(selected.numel())
    return {
        "loss": loss_sum / max(1, total),
        "mode_accuracy": correct / max(1, total),
        "selected_utility": selected_value / max(1, total),
        "exploit_utility": exploit_value / max(1, total),
        "oracle_utility": oracle_value / max(1, total),
        "raw_oracle_utility": raw_oracle_value / max(1, total),
        "utility_gain_over_exploit": (selected_value - exploit_value) / max(1, total),
        "oracle_gap": (oracle_value - selected_value) / max(1, total),
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows, data_summary = load_rows(args)
    if len(rows) < args.min_examples:
        raise RuntimeError(f"not enough gate-value rows: {len(rows)} < {args.min_examples}")
    feature_matrix = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    raw_mean = feature_matrix.mean(dim=0)
    raw_std = feature_matrix.std(dim=0, unbiased=False)
    mean, std, near_constant = prepare_normalization_tensors(
        raw_mean, raw_std, NORMALIZATION_STD_FLOOR
    )
    dataset = ModeValueDataset(rows, mean, std)
    if args.group_split:
        train_indices, eval_indices = group_split_indices(rows, args.eval_ratio, args.seed)
        train_set = Subset(dataset, train_indices)
        eval_set = Subset(dataset, eval_indices)
    else:
        eval_size = max(1, int(len(dataset) * args.eval_ratio))
        train_set, eval_set = random_split(
            dataset,
            [len(dataset) - eval_size, eval_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    data_summary["train_rows"] = len(train_set)
    data_summary["eval_rows"] = len(eval_set)
    data_summary["group_split"] = bool(args.group_split)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = GateValueNet(hidden_dims=tuple(args.hidden_dims)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_metric = -float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = items = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, _ = compute_loss(model, batch, args)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(batch["best"])
            items += len(batch["best"])
        metrics = evaluate(model, eval_loader, device, args)
        metrics.update({"epoch": epoch, "train_loss": train_loss / max(1, items)})
        history.append(metrics)
        score = metrics["selected_utility"] - 0.1 * metrics["oracle_gap"]
        if score > best_metric:
            best_metric = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
    if best_state is None:
        raise RuntimeError("gate-value training produced no checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "gate_value_policy.pth"
    torch.save(
        {
            "model_state": best_state,
            "input_dim": GATE_VALUE_FEATURE_DIM,
            "hidden_dims": list(args.hidden_dims),
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "near_constant_features": near_constant.tolist(),
            "value_temperature": args.value_temperature,
            "min_mode_margin": args.min_mode_margin,
            "confidence_z": args.confidence_z,
            "min_confidence_advantage": args.min_confidence_advantage,
            "modes": list(MODES),
            "feature_ablation": args.feature_ablation,
            "allowed_modes": list(args.allowed_modes),
            "disable_belief_runtime": args.feature_ablation == "no_belief",
            "disable_z_runtime": args.feature_ablation == "no_z",
            "training_data": data_summary,
        },
        checkpoint_path,
    )
    best_eval = max(history, key=lambda item: item["selected_utility"] - 0.1 * item["oracle_gap"])
    result = {
        "checkpoint": str(checkpoint_path),
        "data": data_summary,
        "best_metric": best_metric,
        "best_eval": best_eval,
        "history": history,
        "config": vars(args),
    }
    write_json(args.output_dir / "gate_value_training_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-examples", type=int, default=500)
    parser.add_argument("--min-value-range", type=float, default=1.0)
    parser.add_argument("--reward-weight-scale", type=float, default=64.0)
    parser.add_argument("--value-scale", type=float, default=64.0)
    parser.add_argument("--target-clip", type=float, default=16.0)
    parser.add_argument("--eval-ratio", type=float, default=0.20)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--expected-value-weight", type=float, default=0.10)
    parser.add_argument("--entropy-weight", type=float, default=0.005)
    parser.add_argument("--value-temperature", type=float, default=1.0)
    parser.add_argument("--min-mode-margin", type=float, default=0.05)
    parser.add_argument(
        "--feature-ablation",
        choices=GATE_VALUE_FEATURE_ABLATIONS,
        default="none",
    )
    parser.add_argument("--allowed-modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--confidence-z", type=float, default=0.0)
    parser.add_argument("--min-confidence-advantage", type=float, default=0.0)
    parser.add_argument("--uncertain-weight", type=float, default=1.0)
    parser.add_argument(
        "--belief-bin-edges",
        nargs="+",
        type=float,
        default=[0.80, 0.90, 0.95],
        help="Raw belief_max boundaries used for coverage-aware sample weighting.",
    )
    parser.add_argument(
        "--belief-balance-strength",
        type=float,
        default=0.0,
        help="Inverse-frequency exponent for belief bins; zero disables weighting.",
    )
    parser.add_argument("--max-belief-balance-weight", type=float, default=4.0)
    parser.add_argument("--group-split", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=3431)
    args = parser.parse_args()
    args.allowed_modes = [mode for mode in MODES if mode in set(args.allowed_modes)]
    if "exploit" not in args.allowed_modes:
        raise ValueError("--allowed-modes must include exploit")
    if len(args.allowed_modes) < 2:
        raise ValueError("--allowed-modes must contain at least two modes")
    if args.confidence_z < 0.0:
        raise ValueError("--confidence-z must be non-negative")
    if not 0.0 < args.uncertain_weight <= 1.0:
        raise ValueError("--uncertain-weight must be in (0, 1]")
    if args.belief_balance_strength < 0.0:
        raise ValueError("--belief-balance-strength must be non-negative")
    if args.max_belief_balance_weight < 1.0:
        raise ValueError("--max-belief-balance-weight must be at least 1")
    print(json.dumps(train(args), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
