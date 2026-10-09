"""GRPO preference tuning for the shared Mahjong policy backbone."""

from __future__ import annotations

import os

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

from unsloth import FastLanguageModel, PatchFastRL

import argparse
import json
import multiprocessing
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset

PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOConfig, GRPOTrainer

from shared_base_reward import (
    extract_action,
    extract_text,
    format_reward_func,
    hu_decision_reward_func,
    legal_action_reward_func,
    legal_actions_from_prompt,
    parse_action_values,
    progress_reward_func,
    teacher_anchor_reward_func,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1800)
    parser.add_argument("--max-completion-length", type=int, default=32)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dataset-limit", type=int, default=6000)
    parser.add_argument(
        "--hu-prompt-fraction",
        type=float,
        default=0.2,
        help="Reserve this fraction of GRPO prompts for states where h is legal.",
    )
    parser.add_argument("--min-legal-actions", type=int, default=2)
    parser.add_argument(
        "--require-progress-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only curriculum states whose discard candidates have different shanten/ukeire values.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=3427)
    parser.add_argument("--num-workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_generations < 2:
        raise ValueError("--num-generations must be at least 2")
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch = args.batch_size * args.gradient_accumulation * world_size
    if effective_batch % args.num_generations:
        raise ValueError(
            f"effective batch {effective_batch} must be divisible by num_generations {args.num_generations}"
        )
    if not 0.0 <= args.hu_prompt_fraction < 1.0:
        raise ValueError("--hu-prompt-fraction must be in [0, 1)")


def load_dataset_rows(
    path: Path,
    limit: int,
    min_legal_actions: int,
    seed: int,
    require_progress_signal: bool,
    hu_prompt_fraction: float,
) -> Dataset:
    hu_rows: List[Dict[str, Any]] = []
    progress_rows: List[Dict[str, Any]] = []
    skipped = {"forced": 0, "duplicate": 0, "no_progress_values": 0, "flat_progress_values": 0}
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            messages = source["messages"]
            prompt = messages[:2]
            prompt_text = extract_text(prompt)
            legal = legal_actions_from_prompt(prompt_text)
            if len(set(legal)) < min_legal_actions:
                skipped["forced"] += 1
                continue
            reference, _ = extract_action(messages[-1].get("content", ""))
            dedupe_key = (prompt_text, reference)
            if dedupe_key in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(dedupe_key)
            meta = source.get("meta") or {}
            progress_values = parse_action_values(meta.get("action_values") or {})
            is_hu_decision = "h" in legal
            if require_progress_signal and not is_hu_decision:
                if len(progress_values) < 2:
                    skipped["no_progress_values"] += 1
                    continue
                signatures = {
                    (
                        value["shanten"],
                        value["effective_copies"],
                        value["effective_types"],
                    )
                    for value in progress_values.values()
                }
                if len(signatures) < 2:
                    skipped["flat_progress_values"] += 1
                    continue
            item = {
                "prompt": prompt,
                "reference_action": reference,
                "shared_base_source": str(
                    meta.get("shared_base_dataset_source")
                    or meta.get("shared_base_source")
                    or meta.get("v1_clean_source")
                    or "unknown"
                ),
                "action_values_json": json.dumps(progress_values, ensure_ascii=False),
                "decision_type": "hu" if is_hu_decision else "progress",
            }
            (hu_rows if is_hu_decision else progress_rows).append(item)

    rng = random.Random(seed)
    rng.shuffle(hu_rows)
    rng.shuffle(progress_rows)
    if limit > 0:
        hu_limit = min(len(hu_rows), round(limit * hu_prompt_fraction))
        progress_limit = min(len(progress_rows), limit - hu_limit)
        remaining = limit - hu_limit - progress_limit
        if remaining > 0:
            extra_hu = min(len(hu_rows) - hu_limit, remaining)
            hu_limit += extra_hu
            remaining -= extra_hu
            progress_limit += min(len(progress_rows) - progress_limit, remaining)
        rows = hu_rows[:hu_limit] + progress_rows[:progress_limit]
    else:
        rows = hu_rows + progress_rows
    rng.shuffle(rows)
    if not rows:
        raise ValueError("No GRPO prompts survived filtering")
    print(
        "[shared-base GRPO] "
        f"dataset_skipped={skipped} available_hu={len(hu_rows)} "
        f"available_progress={len(progress_rows)} selected_hu={sum(row['decision_type'] == 'hu' for row in rows)} "
        f"selected_progress={sum(row['decision_type'] == 'progress' for row in rows)}"
    )
    return Dataset.from_list(rows)


def reward_diversity(log_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    zero_std = [
        float(row["frac_reward_zero_std"])
        for row in log_history
        if isinstance(row.get("frac_reward_zero_std"), (int, float))
    ]
    reward_std = [
        float(row["reward_std"])
        for row in log_history
        if isinstance(row.get("reward_std"), (int, float))
    ]
    return {
        "logged_steps": len(zero_std),
        "mean_frac_reward_zero_std": sum(zero_std) / len(zero_std) if zero_std else None,
        "mean_reward_std": sum(reward_std) / len(reward_std) if reward_std else None,
        "max_reward_std": max(reward_std) if reward_std else None,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    repo = Path(__file__).resolve().parent
    model_path = Path(args.model_path)
    data_path = Path(args.data_file)
    output_dir = Path(args.output_dir)
    if not model_path.is_absolute():
        model_path = repo / model_path
    if not data_path.is_absolute():
        data_path = repo / data_path
    if not output_dir.is_absolute():
        output_dir = repo / output_dir

    dataset = load_dataset_rows(
        data_path,
        args.dataset_limit,
        args.min_legal_actions,
        args.seed,
        args.require_progress_signal,
        args.hu_prompt_fraction,
    )
    print(f"[shared-base GRPO] model={model_path}")
    print(f"[shared-base GRPO] data={data_path} prompts={len(dataset)}")
    print(f"[shared-base GRPO] output={output_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=False,
        max_lora_rank=args.lora_r,
        local_files_only=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    training_args = GRPOConfig(
        use_vllm=False,
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else None,
        beta=args.beta,
        max_steps=args.max_steps,
        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=2 if args.num_workers > 0 else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=8,
        logging_steps=args.logging_steps,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward_func,
            legal_action_reward_func,
            hu_decision_reward_func,
            progress_reward_func,
            teacher_anchor_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    diversity = reward_diversity(trainer.state.log_history)
    (output_dir / "shared_base_grpo_summary.json").write_text(
        json.dumps(
            {
                "created_for": "shared backbone preference tuning",
                "excludes": ["B_phi", "z_j(t)", "safe/deceive modes", "gate rewards"],
                "model_path": str(model_path),
                "data_file": str(data_path),
                "prompts": len(dataset),
                "decision_type_counts": dict(Counter(str(value) for value in dataset["decision_type"])),
                "reward_components": ["format", "legal", "hu", "shanten_ukeire", "teacher_anchor"],
                "config": vars(args),
                "reward_diversity": diversity,
                "signal_valid": (
                    diversity["mean_frac_reward_zero_std"] is not None
                    and diversity["mean_frac_reward_zero_std"] < 0.6
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    final_dir = output_dir / "best_grpo_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[shared-base GRPO] saved={final_dir}")


if __name__ == "__main__":
    main()
