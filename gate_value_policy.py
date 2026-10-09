"""Three-mode value gate over public state and concrete mode actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from action_value_policy import ACTION_DIM, action_features
from game import MahjongGame
from neural_gate_policy import (
    MODES,
    MODE_TO_INDEX,
    NORMALIZATION_STD_FLOOR,
    NORMALIZED_FEATURE_CLIP,
    prepare_normalization_tensors,
)
from ppo_features import STATE_DIM, extract_state_features


MODE_FEATURE_DIM = 1 + ACTION_DIM
GATE_VALUE_FEATURE_DIM = STATE_DIM + len(MODES) * MODE_FEATURE_DIM
STATE_FEATURE_NAMES = (
    "own_shanten", "own_effective_copies", "own_potential_fan", "own_tell",
    "missing_suit_count", "open_meld_count", "belief_avg", "belief_max",
    "belief_min", "belief_std", "z_drift_max", "z_cusum_max",
    "z_entropy_max", "z_opponent_count", "risk_budget", "uncertainty",
    "score_behind", "gate_tiles_left", "wall_remaining", "total_discards",
    "own_balance", "opponent_1_balance", "opponent_2_balance",
    "opponent_3_balance", "tell_before", "tell_after_proxy",
    "belief_tell_headroom", "near_tenpai_belief_headroom",
)
ACTION_FEATURE_NAMES = (
    "is_discard", "is_hu", "tile_number", "suit_0", "suit_1", "suit_2",
    "tile_count", "is_terminal", "result_shanten", "effective_types",
    "effective_copies", "potential_fan", "shape_direction", "tell_delta",
    "public_safe", "is_last_drawn",
)
GATE_VALUE_FEATURE_NAMES = tuple(STATE_FEATURE_NAMES) + tuple(
    name
    for mode in MODES
    for name in (f"{mode}.available",) + tuple(
        f"{mode}.{action_name}" for action_name in ACTION_FEATURE_NAMES
    )
)
GATE_VALUE_FEATURE_ABLATIONS = ("none", "no_belief", "no_z")


def apply_gate_value_feature_ablation(
    features: Sequence[float], ablation: str = "none"
) -> List[float]:
    """Apply the same raw-feature ablation during training and deployment."""
    if ablation not in GATE_VALUE_FEATURE_ABLATIONS:
        raise ValueError(f"unknown gate-value feature ablation: {ablation}")
    values = [float(value) for value in features]
    if len(values) != GATE_VALUE_FEATURE_DIM:
        raise ValueError(
            f"expected {GATE_VALUE_FEATURE_DIM} gate-value features, got {len(values)}"
        )
    if ablation == "none":
        return values

    index = {name: position for position, name in enumerate(GATE_VALUE_FEATURE_NAMES)}
    belief_max = values[index["belief_max"]]

    if ablation == "no_z":
        # RiskGate uncertainty is z_drift + 0.4 * (1-max(B_phi)).  Keep the
        # belief contribution while removing every z_j-derived quantity.
        values[index["uncertainty"]] = max(
            0.0, min(1.0, 0.4 * (1.0 - belief_max))
        )
        for name in (
            "z_drift_max", "z_cusum_max", "z_entropy_max", "z_opponent_count"
        ):
            values[index[name]] = 0.0
        return values

    # RiskGate uses 0.35 * max(B_phi) in rho and (1-max(B_phi))*0.4 in
    # uncertainty. Reconstruct the public/z-only quantities instead of merely
    # zeroing them, so the ablation removes B_phi without removing public risk.
    values[index["risk_budget"]] = max(
        0.0, min(1.0, values[index["risk_budget"]] - 0.35 * belief_max)
    )
    values[index["uncertainty"]] = max(
        0.0, min(1.0, values[index["z_drift_max"]] + 0.4)
    )
    for name in ("belief_avg", "belief_max", "belief_min", "belief_std"):
        values[index[name]] = 0.0

    # These two features are headroom terms. With an empty belief estimate,
    # extract_state_features defines them from public hand/tell state alone.
    values[index["belief_tell_headroom"]] = values[index["own_tell"]]
    values[index["near_tenpai_belief_headroom"]] = float(
        values[index["own_shanten"]] <= (1.0 / 6.0 + 1e-9)
    )
    return values


def apply_gate_value_mode_mask(
    features: Sequence[float], allowed_modes: Sequence[str]
) -> List[float]:
    """Remove excluded mode-action blocks from a stored feature vector."""
    allowed = set(allowed_modes)
    unknown = allowed.difference(MODES)
    if unknown:
        raise ValueError(f"unknown gate-value modes: {sorted(unknown)}")
    values = [float(value) for value in features]
    if len(values) != GATE_VALUE_FEATURE_DIM:
        raise ValueError(
            f"expected {GATE_VALUE_FEATURE_DIM} gate-value features, got {len(values)}"
        )
    for mode_index, mode in enumerate(MODES):
        if mode in allowed:
            continue
        start = STATE_DIM + mode_index * MODE_FEATURE_DIM
        values[start : start + MODE_FEATURE_DIM] = [0.0] * MODE_FEATURE_DIM
    return values


def build_gate_value_features(
    game: MahjongGame,
    player_id: int,
    z_state: Dict[int, Dict[str, Any]],
    beliefs: Dict[str, Any],
    gate: Dict[str, Any],
    available_modes: Sequence[str],
    mode_actions: Dict[str, str],
) -> List[float]:
    features = list(extract_state_features(game, player_id, z_state, beliefs, gate))
    available = set(available_modes)
    for mode in MODES:
        action = mode_actions.get(mode)
        is_available = mode in available and isinstance(action, str)
        features.append(float(is_available))
        features.extend(
            action_features(game, player_id, action) if is_available else [0.0] * ACTION_DIM
        )
    if len(features) != GATE_VALUE_FEATURE_DIM:
        raise ValueError(
            f"expected {GATE_VALUE_FEATURE_DIM} gate-value features, got {len(features)}"
        )
    return features


class GateValueNet(nn.Module):
    def __init__(
        self,
        input_dim: int = GATE_VALUE_FEATURE_DIM,
        hidden_dims: Sequence[int] = (128, 64),
    ):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.05)])
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(last_dim, len(MODES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(features))


class GateValuePolicy:
    def __init__(
        self,
        model: GateValueNet,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        value_temperature: float = 1.0,
        min_mode_margin: float = 0.0,
        safe_margin: Optional[float] = None,
        deceive_margin: Optional[float] = None,
        feature_ablation: str = "none",
        allowed_modes: Sequence[str] = MODES,
        disable_belief_runtime: bool = False,
        disable_z_runtime: bool = False,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        raw_mean = (
            mean if mean is not None else torch.zeros(GATE_VALUE_FEATURE_DIM)
        ).to(self.device)
        raw_std = (
            std if std is not None else torch.ones(GATE_VALUE_FEATURE_DIM)
        ).to(self.device)
        self.mean, self.std, self.near_constant_features = prepare_normalization_tensors(
            raw_mean, raw_std, NORMALIZATION_STD_FLOOR
        )
        self.feature_clip = float(NORMALIZED_FEATURE_CLIP)
        self.value_temperature = max(1e-6, float(value_temperature))
        self.min_mode_margin = max(0.0, float(min_mode_margin))
        self.safe_margin = max(
            0.0, float(min_mode_margin if safe_margin is None else safe_margin)
        )
        self.deceive_margin = max(
            0.0, float(min_mode_margin if deceive_margin is None else deceive_margin)
        )
        if feature_ablation not in GATE_VALUE_FEATURE_ABLATIONS:
            raise ValueError(f"unknown gate-value feature ablation: {feature_ablation}")
        self.feature_ablation = feature_ablation
        self.allowed_modes = tuple(mode for mode in MODES if mode in set(allowed_modes))
        if "exploit" not in self.allowed_modes:
            raise ValueError("gate-value policy must allow exploit")
        self.ablate_belief = feature_ablation == "no_belief"
        self.disable_belief_runtime = bool(disable_belief_runtime)
        self.disable_z_runtime = bool(disable_z_runtime)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "GateValuePolicy":
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        model = GateValueNet(
            input_dim=int(checkpoint.get("input_dim", GATE_VALUE_FEATURE_DIM)),
            hidden_dims=tuple(checkpoint.get("hidden_dims", (128, 64))),
        )
        model.load_state_dict(checkpoint["model_state"])
        return cls(
            model=model,
            mean=torch.tensor(checkpoint["feature_mean"], dtype=torch.float32),
            std=torch.tensor(checkpoint["feature_std"], dtype=torch.float32),
            value_temperature=float(checkpoint.get("value_temperature", 1.0)),
            min_mode_margin=float(checkpoint.get("min_mode_margin", 0.0)),
            safe_margin=checkpoint.get("safe_margin"),
            deceive_margin=checkpoint.get("deceive_margin"),
            feature_ablation=str(checkpoint.get("feature_ablation", "none")),
            allowed_modes=tuple(checkpoint.get("allowed_modes", MODES)),
            disable_belief_runtime=bool(checkpoint.get("disable_belief_runtime", False)),
            disable_z_runtime=bool(checkpoint.get("disable_z_runtime", False)),
            device=device,
        )

    def predict_with_actions(
        self,
        game: MahjongGame,
        player_id: int,
        z_state: Dict[int, Dict[str, Any]],
        beliefs: Dict[str, Any],
        gate: Dict[str, Any],
        available_modes: Sequence[str],
        mode_actions: Dict[str, str],
    ) -> Tuple[str, Dict[str, Any]]:
        available = [
            mode
            for mode in available_modes
            if mode in MODE_TO_INDEX and mode in self.allowed_modes
        ]
        if not available:
            available = ["exploit"]
        if len(available) == 1:
            selected = available[0]
            return selected, {
                "scores": {
                    mode: float(mode == selected) for mode in MODES
                },
                "mode_values": {mode: 0.0 for mode in MODES},
                "mode_actions": dict(mode_actions),
                "value_margin": float("inf"),
                "min_mode_margin": self.min_mode_margin,
                "safe_margin": self.safe_margin,
                "deceive_margin": self.deceive_margin,
                "safe_margin_blocked": False,
                "deceive_margin_blocked": False,
                "margin_fallback_to_exploit": False,
                "max_abs_normalized_feature": 0.0,
                "max_abs_feature_index": -1,
                "max_abs_feature_name": None,
                "clipped_feature_count": 0,
                "clipped_features": [],
                "normalized_feature_clipped": False,
                "feature_source": "single_available_mode_bypass",
                "feature_ablation": self.feature_ablation,
                "allowed_modes": list(self.allowed_modes),
                "disable_belief_runtime": self.disable_belief_runtime,
                "disable_z_runtime": self.disable_z_runtime,
            }
        features = build_gate_value_features(
            game,
            player_id,
            z_state,
            beliefs,
            gate,
            available,
            mode_actions,
        )
        features = apply_gate_value_feature_ablation(features, self.feature_ablation)
        features = apply_gate_value_mode_mask(features, self.allowed_modes)
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        normalized_raw = (x - self.mean) / self.std
        max_abs = float(normalized_raw.abs().max().item()) if normalized_raw.numel() else 0.0
        max_index = int(normalized_raw.abs().argmax().item()) if normalized_raw.numel() else -1
        clipped_indices = [
            int(index)
            for index in torch.nonzero(
                normalized_raw.abs() > self.feature_clip, as_tuple=False
            ).flatten().tolist()
        ]
        normalized = normalized_raw.clamp(-self.feature_clip, self.feature_clip)
        with torch.no_grad():
            values = self.model(normalized.unsqueeze(0)).squeeze(0)
        masked = torch.full_like(values, -1e9)
        for mode in available:
            if mode in MODE_TO_INDEX:
                masked[MODE_TO_INDEX[mode]] = values[MODE_TO_INDEX[mode]]
        selected_index = int(torch.argmax(masked).item())
        selected = MODES[selected_index]
        available_values = sorted(
            (
                float(values[MODE_TO_INDEX[mode]].item()),
                mode,
            )
            for mode in available
            if mode in MODE_TO_INDEX
        )
        margin = (
            available_values[-1][0] - available_values[-2][0]
            if len(available_values) >= 2
            else float("inf")
        )
        margin_fallback = False
        pre_margin_selected = selected
        selected_advantage = float("inf")
        applied_margin = 0.0
        if selected != "exploit" and "exploit" in available:
            selected_advantage = float(
                values[MODE_TO_INDEX[selected]].item()
                - values[MODE_TO_INDEX["exploit"]].item()
            )
            applied_margin = (
                self.safe_margin if selected == "safe" else self.deceive_margin
            )
            # Keep the frozen v1 checkpoint's original top-1/top-2 margin
            # semantics. Splitting the threshold by mode must not change the
            # 0.05/0.05 reference behavior used by prior evaluations.
            if margin < applied_margin:
                selected = "exploit"
                margin_fallback = True
        probs = torch.softmax(masked / self.value_temperature, dim=-1)
        return selected, {
            "scores": {
                mode: float(probs[MODE_TO_INDEX[mode]].item()) for mode in MODES
            },
            "mode_values": {
                mode: float(values[MODE_TO_INDEX[mode]].item()) for mode in MODES
            },
            "mode_actions": dict(mode_actions),
            "value_margin": margin,
            "min_mode_margin": self.min_mode_margin,
            "safe_margin": self.safe_margin,
            "deceive_margin": self.deceive_margin,
            "pre_margin_selected_mode": pre_margin_selected,
            "selected_advantage_vs_exploit": selected_advantage,
            "applied_mode_margin": applied_margin,
            "safe_margin_blocked": (
                margin_fallback and pre_margin_selected == "safe"
            ),
            "deceive_margin_blocked": (
                margin_fallback and pre_margin_selected == "deceive"
            ),
            "margin_fallback_to_exploit": margin_fallback,
            "max_abs_normalized_feature": max_abs,
            "max_abs_feature_index": max_index,
            "max_abs_feature_name": (
                GATE_VALUE_FEATURE_NAMES[max_index] if max_index >= 0 else None
            ),
            "clipped_feature_count": len(clipped_indices),
            "clipped_features": [
                {
                    "index": index,
                    "name": GATE_VALUE_FEATURE_NAMES[index],
                    "normalized_value": float(normalized_raw[index].item()),
                    "raw_value": float(x[index].item()),
                }
                for index in clipped_indices
            ],
            "normalized_feature_clipped": max_abs > self.feature_clip,
            "feature_source": "state_plus_mode_actions_v1",
            "feature_ablation": self.feature_ablation,
            "allowed_modes": list(self.allowed_modes),
            "disable_belief_runtime": self.disable_belief_runtime,
            "disable_z_runtime": self.disable_z_runtime,
        }
