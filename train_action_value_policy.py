"""Train a variable-candidate action scorer from continuous-v2 score lists."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from action_value_policy import ACTION_DIM, MODES, STATE_DIM, ActionValueNet
from experiment_trace import write_json


MODE_TO_INDEX = {mode: index for index, mode in enumerate(MODES)}


class ActionValueDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


def collate_rows(rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_candidates = max(len(row["action_features"]) for row in rows)
    batch = len(rows)
    states = torch.zeros(batch, STATE_DIM)
    actions = torch.zeros(batch, max_candidates, ACTION_DIM)
    teacher_scores = torch.full((batch, max_candidates), -1e9)
    mask = torch.zeros(batch, max_candidates, dtype=torch.bool)
    targets = torch.zeros(batch, dtype=torch.long)
    modes = torch.zeros(batch, dtype=torch.long)
    for index, row in enumerate(rows):
        count = len(row["action_features"])
        states[index] = torch.tensor(row["state_features"], dtype=torch.float32)
        actions[index, :count] = torch.tensor(row["action_features"], dtype=torch.float32)
        teacher_scores[index, :count] = torch.tensor(row["teacher_scores"], dtype=torch.float32)
        mask[index, :count] = True
        targets[index] = int(row["target_index"])
        modes[index] = int(row["mode_index"])
    return {
        "states": states,
        "actions": actions,
        "teacher_scores": teacher_scores,
        "mask": mask,
        "targets": targets,
        "modes": modes,
    }


def load_rows(path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    skipped: Dict[str, int] = {}
    mode_counts: Dict[str, int] = {mode: 0 for mode in MODES}
    candidate_counts: List[int] = []
    for line in path.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        source = json.loads(line)
        candidates = source.get("candidates") or []
        actions = [str(item.get("action")) for item in candidates]
        target = str(source.get("teacher_action", ""))
        mode = str(source.get("teacher_mode", "exploit"))
        if len(source.get("state_features") or []) != STATE_DIM:
            skipped["state_dim"] = skipped.get("state_dim", 0) + 1
            continue
        if len(candidates) < 2 or target not in actions:
            skipped["candidate_target"] = skipped.get("candidate_target", 0) + 1
            continue
        if any(len(item.get("features") or []) != ACTION_DIM for item in candidates):
            skipped["action_dim"] = skipped.get("action_dim", 0) + 1
            continue
        if mode not in MODE_TO_INDEX:
            skipped["mode"] = skipped.get("mode", 0) + 1
            continue
        rows.append(
            {
                "state_features": [float(value) for value in source["state_features"]],
                "action_features": [[float(value) for value in item["features"]] for item in candidates],
                "teacher_scores": [float(item["teacher_score"]) for item in candidates],
                "target_index": actions.index(target),
                "mode_index": MODE_TO_INDEX[mode],
            }
        )
        mode_counts[mode] += 1
        candidate_counts.append(len(candidates))
    return rows, {
        "rows": len(rows),
        "skipped": skipped,
        "mode_counts": mode_counts,
        "avg_candidates": sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
        "max_candidates": max(candidate_counts, default=0),
    }


def evaluate(model: ActionValueNet, loader: DataLoader, device: torch.device, temperature: float) -> Dict[str, float]:
    model.eval()
    total = action_correct = mode_correct = 0
    regret_sum = loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            values, mode_logits = model(batch["states"], batch["actions"])
            values = values.masked_fill(~batch["mask"], -1e9)
            hard_loss = F.cross_entropy(values, batch["targets"])
            teacher_probs = torch.softmax(batch["teacher_scores"] / temperature, dim=-1)
            soft_loss = F.kl_div(torch.log_softmax(values, dim=-1), teacher_probs, reduction="batchmean")
            mode_loss = F.cross_entropy(mode_logits, batch["modes"])
            loss_sum += float((hard_loss + 0.25 * soft_loss + 0.20 * mode_loss).item()) * len(batch["targets"])
            selected = torch.argmax(values, dim=-1)
            action_correct += int((selected == batch["targets"]).sum().item())
            mode_correct += int((torch.argmax(mode_logits, dim=-1) == batch["modes"]).sum().item())
            best_teacher = batch["teacher_scores"].gather(1, batch["targets"].unsqueeze(1)).squeeze(1)
            selected_teacher = batch["teacher_scores"].gather(1, selected.unsqueeze(1)).squeeze(1)
            regret_sum += float((best_teacher - selected_teacher).clamp_min(0).sum().item())
            total += len(batch["targets"])
    return {
        "loss": loss_sum / max(1, total),
        "action_accuracy": action_correct / max(1, total),
        "mode_accuracy": mode_correct / max(1, total),
        "teacher_score_regret": regret_sum / max(1, total),
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows, data_summary = load_rows(args.input)
    if len(rows) < args.min_examples:
        raise RuntimeError(f"not enough action-value rows: {len(rows)} < {args.min_examples}")
    dataset = ActionValueDataset(rows)
    eval_size = max(1, int(len(dataset) * args.eval_ratio))
    train_size = len(dataset) - eval_size
    train_set, eval_set = random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_rows)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_rows)
    device = torch.device(args.device)
    model = ActionValueNet(hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_metric = -float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = items = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            values, mode_logits = model(batch["states"], batch["actions"])
            values = values.masked_fill(~batch["mask"], -1e9)
            hard_loss = F.cross_entropy(values, batch["targets"])
            teacher_probs = torch.softmax(batch["teacher_scores"] / args.teacher_temperature, dim=-1)
            soft_loss = F.kl_div(torch.log_softmax(values, dim=-1), teacher_probs, reduction="batchmean")
            mode_loss = F.cross_entropy(mode_logits, batch["modes"])
            loss = hard_loss + args.soft_weight * soft_loss + args.mode_weight * mode_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(batch["targets"])
            items += len(batch["targets"])
        metrics = evaluate(model, eval_loader, device, args.teacher_temperature)
        metrics.update({"epoch": epoch, "train_loss": train_loss / max(1, items)})
        history.append(metrics)
        score = metrics["action_accuracy"] - 0.001 * metrics["teacher_score_regret"]
        if score > best_metric:
            best_metric = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
    if best_state is None:
        raise RuntimeError("action-value training produced no checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "action_value_policy.pth"
    torch.save(
        {
            "model_state": best_state,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "hidden": args.hidden,
        },
        checkpoint_path,
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "data": data_summary,
        "best_metric": best_metric,
        "best_eval": max(history, key=lambda item: item["action_accuracy"] - 0.001 * item["teacher_score_regret"]),
        "history": history,
    }
    write_json(args.output_dir / "action_value_training_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-examples", type=int, default=2000)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-temperature", type=float, default=20.0)
    parser.add_argument("--soft-weight", type=float, default=0.25)
    parser.add_argument("--mode-weight", type=float, default=0.20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=3427)
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
