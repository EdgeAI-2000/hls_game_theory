"""Train a residual over the frozen action-value v1 scorer using CRN returns."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from action_value_policy import ACTION_DIM, STATE_DIM, ActionValueNet, ResidualScoreNet
from experiment_trace import write_json


class ResidualDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


def load_rows(path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    skipped: Dict[str, int] = {}
    candidate_counts = []
    target_change = 0
    for line in path.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        source = json.loads(line)
        candidates = source.get("candidates") or []
        actions = [str(item.get("action")) for item in candidates]
        target = str(source.get("target_action", ""))
        if len(source.get("state_features") or []) != STATE_DIM:
            skipped["state_dim"] = skipped.get("state_dim", 0) + 1
            continue
        if len(candidates) < 2 or target not in actions:
            skipped["candidate_target"] = skipped.get("candidate_target", 0) + 1
            continue
        if any(len(item.get("features") or []) != ACTION_DIM for item in candidates):
            skipped["action_dim"] = skipped.get("action_dim", 0) + 1
            continue
        utilities = [float(item["utility"]) for item in candidates]
        if max(utilities) - min(utilities) < 1e-9:
            skipped["utility_tie"] = skipped.get("utility_tie", 0) + 1
            continue
        student_action = str(source.get("student_action", ""))
        target_change += int(student_action != target)
        rows.append(
            {
                "state_features": [float(value) for value in source["state_features"]],
                "action_features": [
                    [float(value) for value in item["features"]] for item in candidates
                ],
                "utilities": utilities,
                "target_index": actions.index(target),
                "student_index": actions.index(student_action) if student_action in actions else -1,
            }
        )
        candidate_counts.append(len(candidates))
    return rows, {
        "rows": len(rows),
        "skipped": skipped,
        "target_changes_student_action": target_change,
        "target_change_rate": target_change / len(rows) if rows else 0.0,
        "avg_candidates": sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
        "max_candidates": max(candidate_counts, default=0),
    }


def collate_rows(rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_candidates = max(len(row["action_features"]) for row in rows)
    batch = len(rows)
    states = torch.zeros(batch, STATE_DIM)
    actions = torch.zeros(batch, max_candidates, ACTION_DIM)
    utilities = torch.full((batch, max_candidates), -1e9)
    mask = torch.zeros(batch, max_candidates, dtype=torch.bool)
    targets = torch.zeros(batch, dtype=torch.long)
    for index, row in enumerate(rows):
        count = len(row["action_features"])
        states[index] = torch.tensor(row["state_features"], dtype=torch.float32)
        actions[index, :count] = torch.tensor(row["action_features"], dtype=torch.float32)
        utilities[index, :count] = torch.tensor(row["utilities"], dtype=torch.float32)
        mask[index, :count] = True
        targets[index] = int(row["target_index"])
    return {
        "states": states,
        "actions": actions,
        "utilities": utilities,
        "mask": mask,
        "targets": targets,
    }


def load_base(path: Path, device: torch.device) -> ActionValueNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ActionValueNet(
        state_dim=int(checkpoint.get("state_dim", STATE_DIM)),
        action_dim=int(checkpoint.get("action_dim", ACTION_DIM)),
        hidden=int(checkpoint.get("hidden", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def batch_loss(
    residual_model: ResidualScoreNet,
    base_model: ActionValueNet,
    batch: Dict[str, torch.Tensor],
    temperature: float,
    soft_weight: float,
    residual_weight: float,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    with torch.no_grad():
        base_scores, _ = base_model(batch["states"], batch["actions"])
    residuals = residual_model(batch["states"], batch["actions"])
    final_scores = (base_scores + residuals).masked_fill(~batch["mask"], -1e9)
    hard_loss = F.cross_entropy(final_scores, batch["targets"])
    centered_utilities = batch["utilities"] - batch["utilities"].masked_fill(
        ~batch["mask"], -1e9
    ).max(dim=1, keepdim=True).values
    utility_probs = torch.softmax(centered_utilities / temperature, dim=-1)
    soft_loss = F.kl_div(
        torch.log_softmax(final_scores, dim=-1), utility_probs, reduction="batchmean"
    )
    residual_penalty = residuals.masked_select(batch["mask"]).square().mean()
    loss = hard_loss + soft_weight * soft_loss + residual_weight * residual_penalty
    return loss, {
        "base_scores": base_scores.masked_fill(~batch["mask"], -1e9),
        "final_scores": final_scores,
        "residuals": residuals,
    }


def evaluate(
    residual_model: ResidualScoreNet,
    base_model: ActionValueNet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    residual_model.eval()
    total = final_correct = base_correct = 0
    loss_sum = final_utility = base_utility = oracle_utility = residual_abs = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, outputs = batch_loss(
                residual_model,
                base_model,
                batch,
                args.utility_temperature,
                args.soft_weight,
                args.residual_weight,
            )
            final_pick = torch.argmax(outputs["final_scores"], dim=-1)
            base_pick = torch.argmax(outputs["base_scores"], dim=-1)
            final_correct += int((final_pick == batch["targets"]).sum().item())
            base_correct += int((base_pick == batch["targets"]).sum().item())
            final_utility += float(
                batch["utilities"].gather(1, final_pick.unsqueeze(1)).sum().item()
            )
            base_utility += float(
                batch["utilities"].gather(1, base_pick.unsqueeze(1)).sum().item()
            )
            oracle_utility += float(
                batch["utilities"].gather(1, batch["targets"].unsqueeze(1)).sum().item()
            )
            residual_abs += float(
                outputs["residuals"].masked_select(batch["mask"]).abs().sum().item()
            )
            loss_sum += float(loss.item()) * len(batch["targets"])
            total += len(batch["targets"])
    return {
        "loss": loss_sum / max(1, total),
        "action_accuracy": final_correct / max(1, total),
        "base_action_accuracy": base_correct / max(1, total),
        "selected_utility": final_utility / max(1, total),
        "base_selected_utility": base_utility / max(1, total),
        "oracle_utility": oracle_utility / max(1, total),
        "utility_gain_over_base": (final_utility - base_utility) / max(1, total),
        "avg_abs_residual": residual_abs / max(1, total),
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows, data_summary = load_rows(args.input)
    if len(rows) < args.min_examples:
        raise RuntimeError(f"not enough value-residual rows: {len(rows)} < {args.min_examples}")
    dataset = ResidualDataset(rows)
    eval_size = max(1, int(len(dataset) * args.eval_ratio))
    train_set, eval_set = random_split(
        dataset,
        [len(dataset) - eval_size, eval_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_rows
    )
    eval_loader = DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_rows
    )
    device = torch.device(args.device)
    base_model = load_base(args.base_action_value_model_path, device)
    residual_model = ResidualScoreNet(hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(
        residual_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_metric = -float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        residual_model.train()
        train_loss = items = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, _ = batch_loss(
                residual_model,
                base_model,
                batch,
                args.utility_temperature,
                args.soft_weight,
                args.residual_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(batch["targets"])
            items += len(batch["targets"])
        metrics = evaluate(residual_model, base_model, eval_loader, device, args)
        metrics.update({"epoch": epoch, "train_loss": train_loss / max(1, items)})
        history.append(metrics)
        score = metrics["selected_utility"] + 10.0 * metrics["action_accuracy"]
        if score > best_metric:
            best_metric = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in residual_model.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
    if best_state is None:
        raise RuntimeError("value-residual training produced no checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "value_residual_policy.pth"
    torch.save(
        {
            "model_state": best_state,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "hidden": args.hidden,
            "residual_scale": 1.0,
            "base_action_value_model_path": str(args.base_action_value_model_path),
        },
        checkpoint_path,
    )
    best_eval = max(
        history, key=lambda item: item["selected_utility"] + 10.0 * item["action_accuracy"]
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "data": data_summary,
        "best_metric": best_metric,
        "best_eval": best_eval,
        "history": history,
        "config": vars(args),
    }
    write_json(args.output_dir / "value_residual_training_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-action-value-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-examples", type=int, default=600)
    parser.add_argument("--eval-ratio", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--utility-temperature", type=float, default=64.0)
    parser.add_argument("--soft-weight", type=float, default=0.50)
    parser.add_argument("--residual-weight", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=3429)
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
