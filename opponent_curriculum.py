"""Executable opponent-curriculum schema for formal MASK experiments.

The LLM proposes bounded parameter specifications rather than arbitrary Python
code.  Each accepted specification is interpreted by ``ResponsiveDefender`` so
legality and the public-information boundary remain enforced by the engine.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


THREAT_MODELS = ("discard_tell", "blend", "mc")
SPLITS = ("train", "holdout")
NUMERIC_BOUNDS = {
    "threat_threshold": (0.25, 0.70),
    "oracle_samples": (8, 80),
    "beta": (0.5, 5.0),
    "danger_threshold": (0, 2),
    "ffr_hand_shanten": (0, 2),
    "tell_weight": (0.0, 1.0),
    "tell_window": (3, 12),
}


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def normalize_spec(raw: Mapping[str, Any], index: int = 0) -> Dict[str, Any]:
    """Validate and normalize one generated curriculum specification."""
    model = str(raw.get("threat_model", "blend"))
    if model not in THREAT_MODELS:
        mentioned = [candidate for candidate in THREAT_MODELS if candidate in model]
        if not mentioned:
            raise ValueError(f"unsupported threat_model: {model}")
        # LLMs occasionally repeat the enum example. Resolve it deterministically
        # and keep the raw generation separately for audit.
        model = mentioned[index % len(mentioned)]

    spec_id = str(raw.get("id") or f"beta_{index:03d}")
    if not spec_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"invalid curriculum id: {spec_id}")

    archetypes = str(raw.get("archetype") or "belief_responsive").split("|")
    archetype = archetypes[index % len(archetypes)]
    spec: Dict[str, Any] = {
        "id": spec_id,
        "name": str(raw.get("name") or spec_id)[:80],
        "archetype": archetype[:40],
        "rationale": str(raw.get("rationale") or "")[:240],
        "threat_model": model,
    }
    integer_fields = {"oracle_samples", "danger_threshold", "ffr_hand_shanten", "tell_window"}
    defaults = {
        "threat_threshold": 0.4,
        "oracle_samples": 30,
        "beta": 2.0,
        "danger_threshold": 1,
        "ffr_hand_shanten": 1,
        "tell_weight": 1.0 if model == "discard_tell" else 0.6,
        "tell_window": 6,
    }
    for field, bounds in NUMERIC_BOUNDS.items():
        value = _clamp(float(raw.get(field, defaults[field])), bounds)
        spec[field] = int(round(value)) if field in integer_fields else round(value, 4)

    if model == "discard_tell":
        spec["tell_weight"] = 1.0
    elif model == "mc":
        spec["tell_weight"] = 0.0

    split = str(raw.get("split", ""))
    if split:
        if split not in SPLITS:
            raise ValueError(f"invalid curriculum split: {split}")
        spec["split"] = split
    return spec


def spec_vector(spec: Mapping[str, Any]) -> List[float]:
    model = str(spec["threat_model"])
    model_one_hot = [1.0 if model == candidate else 0.0 for candidate in THREAT_MODELS]
    values = []
    for field, (low, high) in NUMERIC_BOUNDS.items():
        values.append((float(spec[field]) - low) / max(1e-9, high - low))
    return model_one_hot + values


def behavior_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Parameter-space distance used only during proposal generation."""
    a, b = spec_vector(left), spec_vector(right)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def response_vector_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """RMS distance between responses to the same behavioral probe suite."""
    if len(left) != len(right) or not left:
        raise ValueError("response vectors must be non-empty and have equal length")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)) / len(left))


def greedy_diversity_filter(
    specs: Iterable[Mapping[str, Any]],
    min_distance: float = 0.28,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    accepted: List[Dict[str, Any]] = []
    seen_ids = set()
    for index, raw in enumerate(specs):
        try:
            spec = normalize_spec(raw, index=index)
        except (TypeError, ValueError):
            continue
        if spec["id"] in seen_ids:
            continue
        if accepted and min(behavior_distance(spec, item) for item in accepted) < min_distance:
            continue
        accepted.append(spec)
        seen_ids.add(spec["id"])
        if limit is not None and len(accepted) >= limit:
            break
    return accepted


def assign_train_holdout(
    specs: Sequence[Mapping[str, Any]], holdout_ratio: float = 1.0 / 3.0
) -> List[Dict[str, Any]]:
    """Create a fixed parameter-space holdout using farthest-point selection."""
    normalized = [normalize_spec(spec, index=i) for i, spec in enumerate(specs)]
    if len(normalized) < 4:
        raise ValueError("curriculum needs at least four accepted opponents")
    holdout_count = max(1, min(len(normalized) - 2, round(len(normalized) * holdout_ratio)))
    centroid = [sum(v[d] for v in map(spec_vector, normalized)) / len(normalized) for d in range(len(spec_vector(normalized[0])))]

    def centroid_distance(spec: Mapping[str, Any]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(spec_vector(spec), centroid)))

    first = max(range(len(normalized)), key=lambda i: centroid_distance(normalized[i]))
    chosen = [first]
    while len(chosen) < holdout_count:
        candidates = [i for i in range(len(normalized)) if i not in chosen]
        nxt = max(
            candidates,
            key=lambda i: min(behavior_distance(normalized[i], normalized[j]) for j in chosen),
        )
        chosen.append(nxt)
    holdout = set(chosen)
    for index, spec in enumerate(normalized):
        spec["split"] = "holdout" if index in holdout else "train"
    return normalized


def curriculum_fingerprint(specs: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(specs), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_curriculum(path: str | Path, split: str | None = None) -> Dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    specs = [normalize_spec(item, index=i) for i, item in enumerate(manifest.get("opponents", []))]
    if split and split != "all":
        if split not in SPLITS:
            raise ValueError(f"unknown curriculum split: {split}")
        specs = [item for item in specs if item.get("split") == split]
    if not specs:
        raise ValueError(f"curriculum has no opponents for split={split!r}: {path}")
    return {**manifest, "path": str(path.resolve()), "opponents": specs}


def select_specs_for_game(
    manifest: Mapping[str, Any], game_seed: int, player_ids: Sequence[int] = (1, 2, 3)
) -> Dict[int, Dict[str, Any]]:
    specs = list(manifest["opponents"])
    return {
        pid: dict(specs[(game_seed * 17 + pid * 31) % len(specs)])
        for pid in player_ids
    }


def defender_config(spec: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "threat_threshold": float(spec["threat_threshold"]),
        "oracle_samples": int(spec["oracle_samples"]),
        "beta": float(spec["beta"]),
        "danger_threshold": int(spec["danger_threshold"]),
        "ffr_hand_shanten": int(spec["ffr_hand_shanten"]),
        "threat_model": str(spec["threat_model"]),
        "tell_weight": float(spec["tell_weight"]),
        "tell_window": int(spec["tell_window"]),
    }
