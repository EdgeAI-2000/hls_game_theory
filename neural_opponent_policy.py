"""Neural opponent policy distilled from the responsive learned defender.

The model is intentionally small and action-masked.  It replaces the online
responsive rule + learned danger pipeline with a parameterized policy that maps
public/current-player features to one legal Mahjong action.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn

from game import MahjongGame
from prompt_builder import get_legal_actions
from rule_engine import ShantenCalculator
from tiles import Suit


SUITS: Sequence[Suit] = (Suit.WAN, Suit.TIAO, Suit.TONG)
SUIT_TO_OFFSET = {suit: idx * 9 for idx, suit in enumerate(SUITS)}
ACTION_SPACE: List[str] = ["h", "g", "p", "n"] + [
    f"d {number}{suit.value}"
    for suit in SUITS
    for number in range(1, 10)
]
ACTION_TO_INDEX: Dict[str, int] = {action: idx for idx, action in enumerate(ACTION_SPACE)}
FEATURE_VERSION_V1 = "neural_opponent_v1_public_current_player"
FEATURE_VERSION_V2 = "neural_opponent_v2_public_no_engineered_tell"
FEATURE_VERSION_V3 = "neural_opponent_v3_public_risk_aware"
SUPPORTED_FEATURE_VERSIONS = {
    FEATURE_VERSION_V1,
    FEATURE_VERSION_V2,
    FEATURE_VERSION_V3,
}
# Keep the legacy default so existing data and checkpoints remain loadable.
FEATURE_VERSION = FEATURE_VERSION_V1


def tile_index(tile) -> int:
    return SUIT_TO_OFFSET[tile.suit] + tile.number - 1


def one_hot(index: Optional[int], size: int) -> List[float]:
    values = [0.0] * size
    if index is not None and 0 <= index < size:
        values[index] = 1.0
    return values


def tile_counts(tiles: Iterable, scale: float = 4.0) -> List[float]:
    counts = [0.0] * 27
    for tile in tiles:
        counts[tile_index(tile)] += 1.0 / scale
    return counts


def open_meld_tile_counts(player) -> List[float]:
    counts = [0.0] * 27
    for meld in player.open_melds:
        for tile in meld:
            counts[tile_index(tile)] += 1.0 / 4.0
    return counts


def legal_action_mask(legal_actions: Sequence[str]) -> List[float]:
    legal = set(legal_actions)
    return [1.0 if action in legal else 0.0 for action in ACTION_SPACE]


def legal_action_indices(legal_actions: Sequence[str]) -> List[int]:
    return [ACTION_TO_INDEX[action] for action in legal_actions if action in ACTION_TO_INDEX]


def extract_policy_features(
    game: MahjongGame,
    player_id: int,
    legal_actions: Optional[Sequence[str]] = None,
    response_actions: Optional[Sequence[str]] = None,
    feature_version: str = FEATURE_VERSION,
) -> List[float]:
    """Build fixed-size features for a current-player neural opponent.

    The feature vector avoids hidden opponent hands.  It includes the acting
    player's own hand plus public table state, so it is valid for P1/P2/P3
    opponent decisions.
    """

    if feature_version not in SUPPORTED_FEATURE_VERSIONS:
        raise ValueError(f"Unsupported neural opponent feature version: {feature_version}")

    player = game.players[player_id]
    legal_actions = list(legal_actions or get_legal_actions(game, player_id, list(response_actions) if response_actions is not None else None))
    response_set = set(response_actions or [])

    features: List[float] = []

    features.extend(one_hot(player_id, 4))
    features.append(1.0 if response_actions is not None else 0.0)
    features.extend([
        1.0 if "hu" in response_set else 0.0,
        1.0 if "gang" in response_set else 0.0,
        1.0 if "peng" in response_set else 0.0,
    ])
    features.extend(legal_action_mask(legal_actions))

    features.extend(tile_counts(player.hand_tiles))
    missing_idx = SUITS.index(player.missing_suit) if player.missing_suit in SUITS else None
    features.extend(one_hot(missing_idx, 3))
    own_shanten = ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit)
    features.append(max(-1.0, min(8.0, float(own_shanten))) / 8.0)
    features.append(len(player.open_melds) / 4.0)
    features.append(len(player.discarded_tiles) / 30.0)
    features.append((player.balance - 10000.0) / 10000.0)
    features.append(1.0 if player.is_hu else 0.0)

    features.append(game.deck.remaining_count() / 108.0)

    last_discard = None
    last_actor: Optional[int] = None
    for table_player in game.players:
        if table_player.discarded_tiles:
            candidate = table_player.discarded_tiles[-1]
            # History ordering is not required for this coarse signal; using the
            # latest visible discard per player keeps the feature deterministic.
            last_discard = candidate
            last_actor = table_player.player_id
    features.extend(one_hot(tile_index(last_discard) if last_discard is not None else None, 27))
    features.extend(one_hot(last_actor, 4))

    for table_player in game.players:
        features.extend(tile_counts(table_player.discarded_tiles))
    total_public_discards = []
    for table_player in game.players:
        total_public_discards.extend(table_player.discarded_tiles)
    features.extend(tile_counts(total_public_discards, scale=16.0))

    for table_player in game.players:
        features.extend(open_meld_tile_counts(table_player))
    for table_player in game.players:
        features.append(len(table_player.open_melds) / 4.0)
    for table_player in game.players:
        features.append(1.0 if table_player.is_hu else 0.0)
    for table_player in game.players:
        idx = SUITS.index(table_player.missing_suit) if table_player.missing_suit in SUITS else None
        features.extend(one_hot(idx, 3))
    for table_player in game.players:
        features.append((table_player.balance - 10000.0) / 10000.0)

    if feature_version == FEATURE_VERSION_V1:
        p0 = game.players[0]
        recent = p0.discarded_tiles[-6:]
        middle = sum(1 for tile in recent if tile.number in (4, 5, 6))
        terminal = sum(1 for tile in recent if tile.number in (1, 9))
        public_safe = {
            (tile.suit, tile.number)
            for table_player in game.players
            if table_player.player_id != 0
            for tile in table_player.discarded_tiles
        }
        safe_discards = sum(1 for tile in recent if (tile.suit, tile.number) in public_safe)
        denom = max(1, len(recent))
        features.extend([
            middle / denom,
            terminal / denom,
            safe_discards / denom,
            len(p0.open_melds) / 4.0,
        ])

    return features


class NeuralOpponentNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int] = (256, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(0.05)])
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RiskAwareNeuralOpponentNet(nn.Module):
    """Joint public-risk estimator and risk-conditioned action policy."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.ReLU(),
                    nn.LayerNorm(hidden),
                    nn.Dropout(0.05),
                ]
            )
            prev = hidden
        self.encoder = nn.Sequential(*layers)
        self.risk_head = nn.Linear(prev, 1)
        self.action_head = nn.Sequential(
            nn.Linear(prev + 2, prev),
            nn.ReLU(),
            nn.Linear(prev, output_dim),
        )

    def action_logits(
        self,
        hidden: torch.Tensor,
        predicted_risk: torch.Tensor,
        risk_override: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        if risk_override is None:
            conditioned_risk = predicted_risk
        elif isinstance(risk_override, torch.Tensor):
            conditioned_risk = risk_override.to(
                device=hidden.device, dtype=hidden.dtype
            ).reshape(predicted_risk.shape)
        else:
            conditioned_risk = torch.full_like(predicted_risk, float(risk_override))
        conditioned_risk = conditioned_risk.clamp(0.0, 1.0)
        risk_features = torch.stack(
            [conditioned_risk, conditioned_risk.square()], dim=-1
        )
        return self.action_head(torch.cat([hidden, risk_features], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        risk_override: Optional[torch.Tensor | float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        risk_logits = self.risk_head(hidden).squeeze(-1)
        predicted_risk = torch.sigmoid(risk_logits)
        return (
            self.action_logits(hidden, predicted_risk, risk_override=risk_override),
            risk_logits,
        )

    def forward_with_risk_counterfactuals(
        self,
        x: torch.Tensor,
        risk_overrides: Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Reuse one encoded state so dropout cannot mimic a risk intervention."""
        hidden = self.encoder(x)
        risk_logits = self.risk_head(hidden).squeeze(-1)
        predicted_risk = torch.sigmoid(risk_logits)
        predicted_logits = self.action_logits(hidden, predicted_risk)
        counterfactuals = [
            self.action_logits(hidden, predicted_risk, risk_override=value)
            for value in risk_overrides
        ]
        return predicted_logits, risk_logits, counterfactuals


def min_shanten_action(game: MahjongGame, player_id: int, legal_actions: Sequence[str]) -> str:
    if "h" in legal_actions:
        return "h"
    if "g" in legal_actions:
        return "g"
    player = game.players[player_id]
    best_action = legal_actions[0] if legal_actions else "n"
    best_shanten = 99
    for action in legal_actions:
        if not action.startswith("d "):
            continue
        tile_text = action[2:]
        for tile in player.hand_tiles:
            if str(tile) != tile_text:
                continue
            temp = player.hand_tiles.copy()
            temp.remove(tile)
            shanten = ShantenCalculator.calculate_shanten(temp, player.missing_suit)
            if shanten < best_shanten:
                best_shanten = shanten
                best_action = action
            break
    return best_action


def discard_safety_score(game: MahjongGame, player_id: int, action: str) -> float:
    if not action.startswith("d "):
        return 0.0
    tile_text = action[2:]
    player = game.players[player_id]
    tile = next((candidate for candidate in player.hand_tiles if str(candidate) == tile_text), None)
    if tile is None:
        return 0.0
    public_discards = {
        (discard.suit, discard.number)
        for table_player in game.players
        for discard in table_player.discarded_tiles
    }
    score = 0.0
    if (tile.suit, tile.number) in public_discards:
        score += 2.0
    if tile.number in (1, 9):
        score += 1.0
    if tile.number in (2, 8):
        score += 0.5
    return score


def action_result_shanten(game: MahjongGame, player_id: int, action: str) -> Optional[int]:
    if not action.startswith("d "):
        return None
    tile_text = action[2:]
    player = game.players[player_id]
    tile = next((candidate for candidate in player.hand_tiles if str(candidate) == tile_text), None)
    if tile is None:
        return None
    temp = player.hand_tiles.copy()
    temp.remove(tile)
    return ShantenCalculator.calculate_shanten(temp, player.missing_suit)


class NeuralOpponentPolicy:
    """Runtime wrapper compatible with ResponsiveDefender's turn/response API."""

    def __init__(
        self,
        observer_pid: int,
        model_path: str,
        device: str = "cpu",
        force_hu: bool = True,
        danger_threshold: int = 1,
        ffr_hand_shanten: int = 1,
        risk_mode: str = "predicted",
    ):
        self.pid = observer_pid
        self.model_path = str(model_path)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.feature_version = str(checkpoint.get("feature_version", FEATURE_VERSION_V1))
        if self.feature_version not in SUPPORTED_FEATURE_VERSIONS:
            raise ValueError(
                f"Unsupported checkpoint feature version: {self.feature_version}"
            )
        self.actions = list(checkpoint.get("actions", ACTION_SPACE))
        if self.actions != ACTION_SPACE:
            raise ValueError("Neural opponent checkpoint action space does not match runtime ACTION_SPACE.")
        self.input_dim = int(checkpoint["input_dim"])
        hidden_dims = tuple(checkpoint.get("hidden_dims", (256, 128)))
        self.architecture = str(checkpoint.get("architecture", "action_only"))
        if self.architecture == "risk_aware_multitask":
            self.model = RiskAwareNeuralOpponentNet(
                self.input_dim,
                len(ACTION_SPACE),
                hidden_dims=hidden_dims,
            ).to(self.device)
        elif self.architecture == "action_only":
            self.model = NeuralOpponentNet(
                self.input_dim,
                len(ACTION_SPACE),
                hidden_dims=hidden_dims,
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported neural opponent architecture: {self.architecture}")
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.force_hu = force_hu
        self.force_gang = bool(checkpoint.get("runtime_force_gang", False))
        self.max_shanten_regret = checkpoint.get("runtime_max_shanten_regret")
        self.max_effective_copy_loss = checkpoint.get(
            "runtime_max_effective_copy_loss"
        )
        self.danger_threshold = danger_threshold
        self.ffr_hand_shanten = ffr_hand_shanten
        if risk_mode not in {"predicted", "blind"}:
            raise ValueError(f"Unsupported neural opponent risk mode: {risk_mode}")
        if risk_mode == "blind" and self.architecture != "risk_aware_multitask":
            raise ValueError("risk_mode=blind requires a risk-aware checkpoint")
        self.risk_mode = risk_mode
        self.risk_prevalence = float(checkpoint.get("risk_prevalence", 0.5))
        self.profile = str(checkpoint.get("profile", "unspecified"))
        self.last_risk_probability: Optional[float] = None

        self.ff_opportunities = 0
        self.ff_false = 0
        self.strict_ffr_event_opportunities = 0
        self.strict_ffr_event_false = 0
        self.strict_ffr_event_events: List[Dict] = []
        self.defensive_switch_opportunities = 0
        self.defensive_switches = 0
        self.defensive_switch_events: List[Dict] = []
        self.declined_self_gang_opportunities = 0
        self.declined_self_gangs = 0
        self.turn_folds = 0
        self.response_declines = 0
        self.ffr_events: List[Dict] = []
        self.false_fold_events: List[Dict] = []

    def threat(self, game: MahjongGame) -> float:
        if self.architecture != "risk_aware_multitask":
            return 0.0
        legal = get_legal_actions(game, self.pid)
        _, risk = self._predict(game, legal)
        return risk

    def threat_crn(self, game: MahjongGame, seeds: List[int], num_samples: int) -> float:
        return self.threat(game)

    @torch.inference_mode()
    def _predict(
        self,
        game: MahjongGame,
        legal_actions: Sequence[str],
        response_actions: Optional[Sequence[str]] = None,
        risk_mode: Optional[str] = None,
    ) -> tuple[torch.Tensor, float]:
        features = extract_policy_features(
            game,
            self.pid,
            legal_actions=legal_actions,
            response_actions=response_actions,
            feature_version=self.feature_version,
        )
        if len(features) != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: got {len(features)}, expected {self.input_dim}."
            )
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.architecture == "risk_aware_multitask":
            effective_mode = risk_mode or self.risk_mode
            override = self.risk_prevalence if effective_mode == "blind" else None
            logits, risk_logits = self.model(x, risk_override=override)
            risk_probability = float(torch.sigmoid(risk_logits).item())
        else:
            logits = self.model(x)
            risk_probability = 0.0
        return logits.squeeze(0), risk_probability

    @torch.inference_mode()
    def select_action(
        self,
        game: MahjongGame,
        legal_actions: Sequence[str],
        response_actions: Optional[Sequence[str]] = None,
    ) -> str:
        if self.force_hu and "h" in legal_actions:
            return "h"
        if self.force_gang and "g" in legal_actions:
            return "g"
        logits, risk_probability = self._predict(
            game,
            legal_actions,
            response_actions=response_actions,
        )
        self.last_risk_probability = risk_probability
        mask = torch.full_like(logits, -1e9)
        idxs = legal_action_indices(legal_actions)
        if not idxs:
            return "n"
        mask[idxs] = 0.0
        action_idx = int(torch.argmax(logits + mask).item())
        action = ACTION_SPACE[action_idx]
        return self._apply_progress_guard(game, legal_actions, action)

    def _apply_progress_guard(
        self,
        game: MahjongGame,
        legal_actions: Sequence[str],
        selected_action: str,
    ) -> str:
        """Keep the neural style choice inside the checkpoint's progress budget."""
        if self.max_shanten_regret is None or not selected_action.startswith("d "):
            return selected_action
        from run_gate1_experiments import discard_progress_metrics

        metrics = {
            action: value
            for action in legal_actions
            if action.startswith("d ")
            and (value := discard_progress_metrics(game, self.pid, action)) is not None
        }
        selected = metrics.get(selected_action)
        if selected is None or not metrics:
            return selected_action
        best_shanten = min(int(value["shanten"]) for value in metrics.values())
        best_effective = max(
            int(value["effective_copies"])
            for value in metrics.values()
            if int(value["shanten"]) == best_shanten
        )
        max_shanten = best_shanten + int(self.max_shanten_regret)
        max_copy_loss = int(self.max_effective_copy_loss or 0)
        selected_allowed = (
            int(selected["shanten"]) <= max_shanten
            and (
                int(selected["shanten"]) > best_shanten
                or int(selected["effective_copies"])
                >= best_effective - max_copy_loss
            )
        )
        if selected_allowed:
            return selected_action
        eligible = [
            action
            for action, value in metrics.items()
            if int(value["shanten"]) == best_shanten
            and int(value["effective_copies"]) == best_effective
        ]
        return min(eligible) if eligible else selected_action

    def turn(self, player, game: MahjongGame, step: int = -1, last_p0_state=None, in_deceive_window: bool = False) -> str:
        legal = get_legal_actions(game, self.pid)
        action = self.select_action(game, legal)
        self._record_ffr_proxy(game, player, legal, action, step, last_p0_state, in_deceive_window)
        return action

    def _record_ffr_proxy(
        self,
        game: MahjongGame,
        player,
        legal: Sequence[str],
        action: str,
        step: int,
        last_p0_state,
        in_deceive_window: bool,
    ) -> None:
        """Record the P4 strict behavioral FFR proxy for a neural opponent.

        The policy has no explicit threat threshold. A non-push choice therefore
        supplies the behavioral switch signal; oracle hand information is used
        only after action selection to classify progress loss.
        """

        if "h" in legal:
            return
        # Imported lazily because run_gate1_experiments loads this module only
        # when neural opponents are requested.
        from run_gate1_experiments import (
            choose_min_shanten_action,
            classify_strict_false_fold,
            discard_progress_metrics,
        )

        own_shanten = ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit)
        p0 = game.players[0]
        p0_shanten = ShantenCalculator.calculate_shanten(p0.hand_tiles, p0.missing_suit)
        push = choose_min_shanten_action(game, self.pid, list(legal))
        switched = action != push
        push_metrics = discard_progress_metrics(game, self.pid, push)
        chosen_metrics = discard_progress_metrics(game, self.pid, action)
        strict = classify_strict_false_fold(
            own_shanten=int(own_shanten),
            p0_shanten=int(p0_shanten),
            high_threat=switched,
            push_action=push,
            chosen_action=action,
            push_metrics=push_metrics,
            chosen_metrics=chosen_metrics,
        )
        event = {
            "step": step,
            "defender_pid": self.pid,
            "ffr_kind": "neural_behavioral_strict_oracle",
            "own_shanten": int(own_shanten),
            "ffr_hand_shanten": int(self.ffr_hand_shanten),
            "p0_shanten": int(p0_shanten),
            "danger_threshold": int(self.danger_threshold),
            "predicted_p0_risk": getattr(self, "last_risk_probability", None),
            "risk_mode": getattr(self, "risk_mode", "unavailable"),
            "risk_prevalence": getattr(self, "risk_prevalence", None),
            "push_action": push,
            "chosen_action": action,
            "defensive_switch": bool(switched),
            "false_fold": bool(strict["false_fold"]),
            "strict_false_fold": bool(strict["false_fold"]),
            "progress_loss": bool(strict["progress_loss"]),
            "shanten_loss": bool(strict["shanten_loss"]),
            "ukeire_loss": bool(strict["ukeire_loss"]),
            "chosen_progress": chosen_metrics,
            "push_progress": push_metrics,
            "chosen_safety_score": discard_safety_score(game, self.pid, action),
            "push_safety_score": discard_safety_score(game, self.pid, push),
            "last_p0_step": (last_p0_state or {}).get("step"),
            "last_p0_mode": (last_p0_state or {}).get("mode"),
            "last_p0_action": (last_p0_state or {}).get("action"),
            "in_deceive_window": bool(in_deceive_window),
        }

        switch_opportunity = (
            own_shanten <= self.ffr_hand_shanten
            and p0_shanten > self.danger_threshold
        )
        if switch_opportunity:
            self.defensive_switch_opportunities += 1
            self.defensive_switch_events.append(dict(event))
            if switched:
                self.defensive_switches += 1
                self.turn_folds += 1

        if strict["eligible"]:
            self.strict_ffr_event_opportunities += 1
            strict_event = dict(event)
            strict_event["ffr_counted_primary"] = self.ff_opportunities == 0
            self.strict_ffr_event_events.append(strict_event)
            if strict["false_fold"]:
                self.strict_ffr_event_false += 1
            if self.ff_opportunities == 0:
                self.ff_opportunities = 1
                self.ffr_events.append(strict_event)
                if strict["false_fold"]:
                    self.ff_false = 1
                    self.false_fold_events.append(strict_event)

    def response(self, player, acts, game: MahjongGame) -> str:
        legal = get_legal_actions(game, self.pid, response_actions=list(acts))
        action = self.select_action(game, legal, response_actions=list(acts))
        if action == "n":
            self.response_declines += 1
        return action


def load_policy(
    model_path: str,
    observer_pid: int,
    device: str = "cpu",
    danger_threshold: int = 1,
    ffr_hand_shanten: int = 1,
    risk_mode: str = "predicted",
) -> NeuralOpponentPolicy:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Neural opponent checkpoint not found: {path}")
    return NeuralOpponentPolicy(
        observer_pid=observer_pid,
        model_path=str(path),
        device=device,
        danger_threshold=danger_threshold,
        ffr_hand_shanten=ffr_hand_shanten,
        risk_mode=risk_mode,
    )
