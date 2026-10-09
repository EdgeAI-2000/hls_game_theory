"""Generate a bounded, executable MASK opponent curriculum with a local LLM."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from experiment_trace import write_json
from llm_backends import build_llm_callable
from opponent_curriculum import (
    assign_train_holdout,
    curriculum_fingerprint,
    greedy_diversity_filter,
)


SYSTEM_PROMPT = """你是四川麻将对手课程设计器。你只生成参数 JSON，不生成代码。
每个对手必须仅依据自己的手牌和公开牌桌推断 P0，不能读取 P0 手牌或牌墙。
输出必须是 JSON 数组，不要 markdown。"""


def prompt_for_batch(batch: int, count: int) -> str:
    return f"""生成 {count} 个差异明显、可被信念塑形也可能反制塑形的对手参数。批次={batch}。
每项字段严格为：
{{"id":"beta_b{batch}_NN","name":"中文短名","archetype":"push_reader|cautious_reader|late_switcher|skeptical_reader|anti_deception","rationale":"一句话",
"threat_model":"discard_tell|blend|mc","threat_threshold":0.25到0.70,
"oracle_samples":8到80,"beta":0.5到5.0,"danger_threshold":0到2,
"ffr_hand_shanten":0到2,"tell_weight":0到1,"tell_window":3到12}}
要求覆盖不同阈值、窗口、MC/弃牌信号混合比例；不要重复。"""


def parse_json_array(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("["), text.rfind("]")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError("LLM output is not a JSON array")
    return [item for item in value if isinstance(item, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--raw-input", type=Path,
                        help="Re-run validation/splitting from an audited *.raw.json without loading the LLM.")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--per-batch", type=int, default=8)
    parser.add_argument("--target-size", type=int, default=24)
    parser.add_argument("--min-distance", type=float, default=0.22)
    parser.add_argument("--holdout-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    args = parser.parse_args()

    candidates: List[Dict[str, Any]] = []
    raw_outputs = []
    failures = []
    if args.raw_input:
        raw_data = json.loads(args.raw_input.read_text(encoding="utf-8"))
        candidates = list(raw_data.get("parsed_candidates", []))
        raw_outputs = list(raw_data.get("outputs", []))
        failures = list(raw_data.get("parse_failures", []))
        raw_generation = raw_data.get("generation") or {}
        if not args.model_path:
            args.model_path = raw_generation.get("model_path")
        if not args.adapter_path:
            args.adapter_path = raw_generation.get("adapter_path") or ""
    else:
        if not args.model_path:
            parser.error("--model-path is required unless --raw-input is used")
        llm = build_llm_callable(
            "local_qwen",
            Path(__file__).resolve().parent,
            model_path=args.model_path,
            adapter_path=args.adapter_path or None,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            system_prompt=SYSTEM_PROMPT,
        )
        for batch in range(args.batches):
            raw = llm(prompt_for_batch(batch, args.per_batch))
            raw_outputs.append({"batch": batch, "text": raw})
            try:
                candidates.extend(parse_json_array(raw))
            except Exception as exc:
                failures.append({"batch": batch, "error": str(exc), "text": raw})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output.with_suffix(".raw.json"),
        {
            "generation": {
                "backend": "local_qwen",
                "model_path": str(Path(args.model_path).resolve()) if args.model_path else None,
                "adapter_path": str(Path(args.adapter_path).resolve()) if args.adapter_path else None,
            },
            "outputs": raw_outputs,
            "parsed_candidates": candidates,
            "parse_failures": failures,
        },
    )

    accepted = greedy_diversity_filter(
        candidates,
        min_distance=args.min_distance,
        limit=args.target_size,
    )
    if len(accepted) < max(8, args.target_size // 2):
        raise RuntimeError(
            f"only {len(accepted)} diverse valid opponents from {len(candidates)} candidates; "
            "increase --batches or lower --min-distance"
        )
    opponents = assign_train_holdout(accepted, holdout_ratio=args.holdout_ratio)
    manifest = {
        "schema_version": "mask_opponent_curriculum_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation": {
            "backend": "local_qwen",
            "model_path": str(Path(args.model_path).resolve()) if args.model_path else None,
            "raw_input": str(args.raw_input.resolve()) if args.raw_input else None,
            "adapter_path": str(Path(args.adapter_path).resolve()) if args.adapter_path else None,
            "candidate_count": len(candidates),
            "accepted_count": len(opponents),
            "parse_failures": failures,
            "diversity_min_distance": args.min_distance,
            "split_strategy": "behavior_parameter_farthest_point_holdout",
        },
        "opponents": opponents,
    }
    manifest["fingerprint"] = curriculum_fingerprint(opponents)
    write_json(args.output, manifest)
    print(json.dumps({"output": str(args.output), "fingerprint": manifest["fingerprint"], "count": len(opponents)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
