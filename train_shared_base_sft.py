"""Continued SFT for the shared L0/L1/L2 policy backbone.

Unlike the legacy V1 trainer, this route trains only on assistant completion
tokens and selects the final adapter using held-out action quality.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from shared_base_reward import extract_action, legal_actions_from_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=0.75)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=3427)
    return parser.parse_args()


def resolve(repo: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo / path


def dataset_stats(path: Path) -> Dict[str, Any]:
    rows = 0
    hu_rows = 0
    sources: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            messages = row.get("messages") or []
            action = str(messages[-1].get("content", "")).strip() if messages else ""
            hu_rows += int(action == "h")
            meta = row.get("meta") or {}
            source = str(
                meta.get("shared_base_dataset_source")
                or meta.get("shared_base_source")
                or meta.get("v1_clean_source")
                or "unknown"
            )
            sources[source] = sources.get(source, 0) + 1
    return {
        "rows": rows,
        "hu_rows": hu_rows,
        "hu_rate": hu_rows / rows if rows else 0.0,
        "sources": sources,
    }


def evaluation_targets(rows) -> List[Dict[str, Any]]:
    targets = []
    for row in rows:
        messages = row.get("messages") or []
        prompt = str(messages[-2].get("content", "")) if len(messages) >= 2 else ""
        reference, _ = extract_action(messages[-1].get("content", "") if messages else "")
        meta = row.get("meta") or {}
        legal = list(meta.get("legal_actions") or legal_actions_from_prompt(prompt))
        phase = str(meta.get("phase", ""))
        if reference == "h":
            category = "hu"
        elif phase == "response" or reference in {"n", "p", "g"}:
            category = "response"
        else:
            category = "discard"
        targets.append({"reference": reference, "legal": legal, "category": category})
    return targets


def main() -> None:
    args = parse_args()
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    repo = Path(__file__).resolve().parent
    model_path = resolve(repo, args.model_path)
    train_file = resolve(repo, args.train_file)
    eval_file = resolve(repo, args.eval_file)
    output_dir = resolve(repo, args.output_dir)
    if args.save_steps != args.eval_steps:
        raise ValueError("--save-steps must equal --eval-steps when loading the best checkpoint")

    train_stats = dataset_stats(train_file)
    eval_stats = dataset_stats(eval_file)
    print(f"[shared-base SFT] model={model_path}")
    print(f"[shared-base SFT] train={json.dumps(train_stats, ensure_ascii=False)}")
    print(f"[shared-base SFT] eval={json.dumps(eval_stats, ensure_ascii=False)}")

    # Unsloth must be imported before transformers/trl.
    from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
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

    files = {"train": str(train_file), "eval": str(eval_file)}
    datasets = load_dataset("json", data_files=files)
    eval_targets = evaluation_targets(datasets["eval"])

    def to_prompt_completion(examples):
        prompts = []
        completions = []
        for messages in examples["messages"]:
            prompts.append(
                tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            completions.append(str(messages[-1]["content"]).strip() + tokenizer.eos_token)
        return {"prompt": prompts, "completion": completions}

    processed = {}
    for split in ("train", "eval"):
        processed[split] = datasets[split].map(
            to_prompt_completion,
            batched=True,
            num_proc=args.dataset_num_proc,
            remove_columns=datasets[split].column_names,
            desc=f"Building completion-only {split} rows",
        )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_action_metrics(eval_prediction):
        predictions, labels = eval_prediction
        records = []
        # Causal-LM logits at position t predict the label at position t+1.
        for index, (prediction_ids, label_ids) in enumerate(zip(predictions, labels)):
            shifted_predictions = prediction_ids[:-1]
            shifted_labels = label_ids[1:]
            mask = shifted_labels != -100
            predicted_text = tokenizer.decode(shifted_predictions[mask], skip_special_tokens=True).strip()
            predicted, _ = extract_action(predicted_text)
            target = eval_targets[index]
            records.append({
                "predicted": predicted,
                "reference": target["reference"],
                "legal": predicted in target["legal"],
                "category": target["category"],
            })

        def mean(values):
            values = list(values)
            return sum(values) / len(values) if values else 0.0

        exact = mean(row["predicted"] == row["reference"] for row in records)
        legal = mean(row["legal"] for row in records)
        hu_recall = mean(
            row["predicted"] == "h" for row in records if row["reference"] == "h"
        )
        discard_exact = mean(
            row["predicted"] == row["reference"]
            for row in records
            if row["category"] == "discard"
        )
        response_exact = mean(
            row["predicted"] == row["reference"]
            for row in records
            if row["category"] == "response"
        )
        return {
            "action_exact_match": exact,
            "legal_rate": legal,
            "hu_recall": hu_recall,
            "discard_exact_match": discard_exact,
            "response_exact_match": response_exact,
            "action_score": 0.5 * exact + 0.3 * hu_recall + 0.2 * legal,
        }

    config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_action_score",
        greater_is_better=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=args.seed,
        report_to="none",
        max_length=args.max_seq_length,
        dataset_num_proc=args.dataset_num_proc,
        packing=args.packing,
        eval_packing=False,
        completion_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=processed["train"],
        eval_dataset=processed["eval"],
        args=config,
        compute_metrics=compute_action_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    final_eval = trainer.evaluate()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    summary = {
        "created_for": "completion-only continued SFT of the shared policy backbone",
        "model_path": str(model_path),
        "train_file": str(train_file),
        "eval_file": str(eval_file),
        "train_stats": train_stats,
        "eval_stats": eval_stats,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "final_eval": final_eval,
        "config": vars(args),
    }
    (output_dir / "shared_base_sft_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[shared-base SFT] saved={output_dir} best={trainer.state.best_model_checkpoint}")


if __name__ == "__main__":
    main()
