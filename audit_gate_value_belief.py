"""Audit whether a GateValue checkpoint uses B_phi on paired CRN rows."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch

from experiment_trace import write_json
from gate_value_policy import (
    GATE_VALUE_FEATURE_NAMES,
    GateValuePolicy,
    apply_gate_value_feature_ablation,
)
from neural_gate_policy import MODES, MODE_TO_INDEX, NORMALIZED_FEATURE_CLIP


def load_rows(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def select(policy: GateValuePolicy, row: Dict[str, Any], features: List[float]) -> tuple[str, torch.Tensor]:
    features = apply_gate_value_feature_ablation(features, policy.feature_ablation)
    x = torch.tensor(features, dtype=torch.float32)
    normalized = ((x - policy.mean.cpu()) / policy.std.cpu()).clamp(
        -NORMALIZED_FEATURE_CLIP, NORMALIZED_FEATURE_CLIP
    )
    with torch.no_grad():
        values = policy.model(normalized.unsqueeze(0)).squeeze(0).cpu()
    available = [mode for mode in row.get("available_modes", []) if mode in MODE_TO_INDEX]
    selected = max(available, key=lambda mode: float(values[MODE_TO_INDEX[mode]]))
    return selected, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = GateValuePolicy.load(args.model, device="cpu")
    rows = load_rows(args.input)
    belief_index = GATE_VALUE_FEATURE_NAMES.index("belief_max")
    bins: Counter[str] = Counter()
    changed = 0
    output_deltas: List[float] = []
    full_utilities: List[float] = []
    ablated_utilities: List[float] = []
    full_modes: Counter[str] = Counter()
    ablated_modes: Counter[str] = Counter()
    usable = 0
    for row in rows:
        features = row.get("features") or []
        available = row.get("available_modes") or []
        mode_values = row.get("mode_values") or {}
        if len(features) != len(GATE_VALUE_FEATURE_NAMES) or len(available) < 2:
            continue
        belief = float(features[belief_index])
        label = "lt_0.80" if belief < 0.80 else "0.80_0.90" if belief < 0.90 else "0.90_0.95" if belief < 0.95 else "ge_0.95"
        bins[label] += 1
        full_mode, full_values = select(policy, row, [float(value) for value in features])
        ablated = apply_gate_value_feature_ablation(features, "no_belief")
        ablated_mode, ablated_values = select(policy, row, ablated)
        changed += int(full_mode != ablated_mode)
        output_deltas.append(float((full_values - ablated_values).abs().max().item()))
        full_utilities.append(float(mode_values[full_mode]))
        ablated_utilities.append(float(mode_values[ablated_mode]))
        full_modes[full_mode] += 1
        ablated_modes[ablated_mode] += 1
        usable += 1

    result = {
        "model": str(args.model.resolve()),
        "inputs": [str(path.resolve()) for path in args.input],
        "rows": len(rows),
        "usable_rows": usable,
        "belief_bins": dict(bins),
        "selection_change_count": changed,
        "selection_change_rate": changed / usable if usable else 0.0,
        "full_mode_counts": dict(full_modes),
        "no_belief_mode_counts": dict(ablated_modes),
        "full_selected_utility": statistics.mean(full_utilities) if full_utilities else 0.0,
        "no_belief_selected_utility": statistics.mean(ablated_utilities) if ablated_utilities else 0.0,
        "belief_utility_gain": (
            statistics.mean(full_utilities) - statistics.mean(ablated_utilities)
            if full_utilities else 0.0
        ),
        "max_output_delta_mean": statistics.mean(output_deltas) if output_deltas else 0.0,
        "max_output_delta_max": max(output_deltas, default=0.0),
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
