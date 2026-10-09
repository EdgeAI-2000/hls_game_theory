"""Build an audited continued-SFT dataset for the shared L0/L1/L2 backbone."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from experiment_trace import write_json


BLOCKED_TEXT = (
    "B_phi",
    "belief_state",
    "mode_hint",
    "suspected_waits",
    "deceive",
    "欺骗模式",
)
BLOCKED_SOURCES = {"mask_l2_teacher", "gate_teacher", "reranker_teacher"}


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def assistant_action(row: Dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if not messages:
        return ""
    content = str(messages[-1].get("content", "")).strip()
    if content.startswith("{"):
        try:
            return str(json.loads(content).get("action", "")).strip()
        except json.JSONDecodeError:
            return content
    return content


def audit_row(row: Dict[str, Any]) -> Optional[str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return "invalid_messages"
    if [message.get("role") for message in messages[:3]] != ["system", "user", "assistant"]:
        return "invalid_roles"
    action = assistant_action(row)
    if not action:
        return "empty_action"
    meta = row.get("meta") or {}
    source = str(
        meta.get("shared_base_source")
        or meta.get("v1_clean_source")
        or meta.get("v1_source")
        or "unknown"
    )
    if source in BLOCKED_SOURCES:
        return f"blocked_source:{source}"
    mode = str(meta.get("mode", "")).lower()
    if mode in {"safe", "deceive"}:
        return f"blocked_mode:{mode}"
    raw = json.dumps(row, ensure_ascii=False)
    for token in BLOCKED_TEXT:
        if token in raw:
            return f"blocked_text:{token}"
    return None


def load_rows(
    path: Path,
    source_name: str,
    cap: Optional[int],
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    rejected = Counter()
    for _, row in iter_jsonl(path):
        reason = audit_row(row)
        if reason:
            rejected[reason] += 1
            continue
        valid.append(row)
    available = len(valid)
    if cap is not None and available > cap:
        valid = rng.sample(valid, cap)

    output = []
    for row in valid:
        item = copy.deepcopy(row)
        meta = item.setdefault("meta", {})
        meta["shared_base_dataset_source"] = source_name
        meta["uses_opponent_belief"] = False
        meta["uses_gate_mode"] = False
        output.append(item)
    return output, {
        "source": source_name,
        "path": str(path),
        "available_valid": available,
        "emitted": len(output),
        "cap": cap,
        "rejected": dict(rejected),
        "actions": dict(Counter(assistant_action(row) for row in output).most_common(30)),
    }


def dedupe(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen = set()
    output = []
    dropped = 0
    for row in rows:
        messages = row["messages"]
        key = (str(messages[1].get("content", "")), assistant_action(row))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        output.append(row)
    return output, dropped


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--clean-train-file",
        type=Path,
        default=Path("V1_Clean_20260715_v1_clean/v1_clean/v1_clean_sft.jsonl"),
    )
    parser.add_argument(
        "--clean-eval-file",
        type=Path,
        default=Path("V1_Clean_20260715_v1_clean/v1_clean/v1_clean_sft_eval.jsonl"),
    )
    parser.add_argument("--curriculum-train-file", type=Path, required=True)
    parser.add_argument("--curriculum-eval-file", type=Path, required=True)
    parser.add_argument("--clean-cap", type=int, default=30000)
    parser.add_argument("--curriculum-repeat", type=int, default=2)
    parser.add_argument("--hu-repeat", type=int, default=2)
    parser.add_argument("--dedupe-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=3427)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_cap = None if args.clean_cap < 0 else args.clean_cap
    clean_rows, clean_stats = load_rows(args.clean_train_file, "v1_clean_replay", clean_cap, rng)
    curriculum_rows, curriculum_stats = load_rows(
        args.curriculum_train_file,
        "exact_progress_curriculum",
        None,
        rng,
    )
    clean_eval_rows, clean_eval_stats = load_rows(args.clean_eval_file, "v1_clean_heldout", None, rng)
    curriculum_eval_rows, curriculum_eval_stats = load_rows(
        args.curriculum_eval_file,
        "exact_progress_curriculum_heldout",
        None,
        rng,
    )

    dropped_clean_duplicates = 0
    if args.dedupe_clean:
        clean_rows, dropped_clean_duplicates = dedupe(clean_rows)

    train_rows = list(clean_rows)
    for repeat_index in range(max(1, args.curriculum_repeat)):
        for row in curriculum_rows:
            repeats = args.hu_repeat if assistant_action(row) == "h" else 1
            for hu_repeat_index in range(max(1, repeats)):
                item = copy.deepcopy(row)
                item["meta"]["dataset_repeat"] = repeat_index
                item["meta"]["dataset_hu_repeat"] = hu_repeat_index
                train_rows.append(item)
    rng.shuffle(train_rows)

    eval_rows, dropped_eval_duplicates = dedupe(clean_eval_rows + curriculum_eval_rows)
    rng.shuffle(eval_rows)
    train_path = args.output_dir / "shared_base_sft_train.jsonl"
    eval_path = args.output_dir / "shared_base_sft_eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    final_rejections = Counter()
    for row in train_rows + eval_rows:
        reason = audit_row(row)
        if reason:
            final_rejections[reason] += 1
    if final_rejections:
        raise RuntimeError(f"Final shared-base dataset failed audit: {dict(final_rejections)}")

    summary = {
        "created_for": "continued SFT of the shared L0/L1/L2 backbone",
        "excludes": ["MASK teacher", "B_phi", "z_j(t)", "gate modes", "safe/deceive labels"],
        "seed": args.seed,
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "clean_duplicates_dropped": dropped_clean_duplicates,
        "eval_duplicates_dropped": dropped_eval_duplicates,
        "source_stats": [clean_stats, curriculum_stats, clean_eval_stats, curriculum_eval_stats],
        "final_source_counts": dict(
            Counter(str((row.get("meta") or {}).get("shared_base_dataset_source", "unknown")) for row in train_rows)
        ),
        "final_action_counts": dict(Counter(assistant_action(row) for row in train_rows)),
        "final_hu_rows": sum(assistant_action(row) == "h" for row in train_rows),
        "final_rows_with_action_values": sum(bool((row.get("meta") or {}).get("action_values")) for row in train_rows),
        "audit_rejections": dict(final_rejections),
    }
    write_json(args.output_dir / "shared_base_sft_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
