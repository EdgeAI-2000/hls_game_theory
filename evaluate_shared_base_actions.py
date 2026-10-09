"""Stratified held-out action audit for shared-backbone checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from experiment_trace import write_json
from shared_base_reward import extract_action


REPO = Path(__file__).resolve().parent


def parse_variant(value: str) -> Tuple[str, str, Optional[str]]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("variant must be NAME=MODEL_PATH or NAME=MODEL_PATH::ADAPTER_PATH")
    name, paths = parts
    model, separator, adapter = paths.partition("::")
    return name, model, adapter if separator else None


def action_of(row: Dict[str, Any]) -> str:
    return str((row.get("messages") or [{}])[-1].get("content", "")).strip()


def category_of(row: Dict[str, Any]) -> str:
    action = action_of(row)
    phase = str((row.get("meta") or {}).get("phase", ""))
    if action == "h":
        return "hu"
    if phase == "response" or action in {"n", "p", "g"}:
        return "response"
    return "discard"


def load_rows(path: Path, limit_per_category: int, seed: int) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            groups[category_of(row)].append(row)
    rng = random.Random(seed)
    output = []
    for category in ("hu", "response", "discard"):
        rows = groups.get(category, [])
        rng.shuffle(rows)
        output.extend(rows[:limit_per_category] if limit_per_category > 0 else rows)
    rng.shuffle(output)
    if not output:
        raise ValueError(f"No held-out rows found in {path}")
    return output


def resolved(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else REPO / path)


def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    categories = sorted({str(row["category"]) for row in records})

    def metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(rows)
        exact = sum(bool(row["exact"]) for row in rows)
        legal = sum(bool(row["legal"]) for row in rows)
        return {
            "rows": total,
            "exact_match": exact / total if total else None,
            "legal_rate": legal / total if total else None,
            "invalid": total - legal,
        }

    result = metrics(records)
    result["by_category"] = {
        category: metrics([row for row in records if row["category"] == category])
        for category in categories
    }
    hu = [row for row in records if row["reference"] == "h"]
    result["hu_recall"] = sum(row["predicted"] == "h" for row in hu) / len(hu) if hu else None
    result["prediction_counts"] = dict(Counter(str(row["predicted"]) for row in records))
    return result


def evaluate_variant(
    name: str,
    model_path: str,
    adapter_path: Optional[str],
    rows: List[Dict[str, Any]],
    max_new_tokens: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import torch
    from llm_backends import LocalQwenCallable

    backend = LocalQwenCallable(
        model_path=resolved(model_path),
        adapter_path=resolved(adapter_path) if adapter_path else None,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    records = []
    for index, row in enumerate(rows, start=1):
        messages = row["messages"]
        prompt = str(messages[-2]["content"])
        raw = backend(prompt)
        predicted, _ = extract_action(raw)
        reference = action_of(row)
        legal_actions = list((row.get("meta") or {}).get("legal_actions") or [])
        if not legal_actions:
            from shared_base_reward import legal_actions_from_prompt

            legal_actions = legal_actions_from_prompt(prompt)
        records.append({
            "variant": name,
            "index": index,
            "category": category_of(row),
            "reference": reference,
            "predicted": predicted,
            "raw": raw,
            "legal_actions": legal_actions,
            "exact": predicted == reference,
            "legal": predicted in legal_actions,
        })
        if index == 1 or index % 100 == 0:
            print(f"[action audit] variant={name} row={index}/{len(rows)}", flush=True)
    summary = summarize(records)
    del backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--variant", action="append", type=parse_variant, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-category", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3427)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--min-candidate-hu-recall", type=float, default=None)
    parser.add_argument("--max-hu-recall-drop", type=float, default=None)
    parser.add_argument("--min-candidate-legal-rate", type=float, default=None)
    args = parser.parse_args()

    eval_file = args.eval_file if args.eval_file.is_absolute() else REPO / args.eval_file
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(eval_file, args.limit_per_category, args.seed)
    summaries = {}
    all_records = []
    for name, model, adapter in args.variant:
        if name in summaries:
            raise ValueError(f"Duplicate variant name: {name}")
        summary, records = evaluate_variant(name, model, adapter, rows, args.max_new_tokens)
        summaries[name] = summary
        all_records.extend(records)

    result = {
        "config": {
            "eval_file": str(eval_file),
            "limit_per_category": args.limit_per_category,
            "seed": args.seed,
            "variants": [name for name, _, _ in args.variant],
        },
        "rows": len(rows),
        "summary": summaries,
    }
    write_json(output_dir / "shared_base_action_audit.json", result)
    with (output_dir / "shared_base_action_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.candidate:
        if args.candidate not in summaries:
            raise ValueError(f"Unknown candidate variant: {args.candidate}")
        candidate = summaries[args.candidate]
        failures = []
        if args.min_candidate_hu_recall is not None and candidate["hu_recall"] < args.min_candidate_hu_recall:
            failures.append(
                f"hu_recall={candidate['hu_recall']:.4f} < {args.min_candidate_hu_recall:.4f}"
            )
        if args.min_candidate_legal_rate is not None and candidate["legal_rate"] < args.min_candidate_legal_rate:
            failures.append(
                f"legal_rate={candidate['legal_rate']:.4f} < {args.min_candidate_legal_rate:.4f}"
            )
        if args.max_hu_recall_drop is not None:
            if not args.baseline or args.baseline not in summaries:
                raise ValueError("--max-hu-recall-drop requires a valid --baseline")
            drop = summaries[args.baseline]["hu_recall"] - candidate["hu_recall"]
            if drop > args.max_hu_recall_drop:
                failures.append(f"hu_recall_drop={drop:.4f} > {args.max_hu_recall_drop:.4f}")
        if failures:
            raise RuntimeError("Shared-base action audit failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
