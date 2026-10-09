"""Small public-information action scorer distilled from continuous-v2."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from game import MahjongGame
from policy_metrics import discard_progress_metrics
from rule_engine import FanCalculator


ACTION_DIM = 16
STATE_DIM = 28
MODES = ("exploit", "safe", "deceive")


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _candidate_potential_fan(player: Any, hand_after_discard: List[Any]) -> int:
    original_hand = player.hand_tiles
    player.hand_tiles = hand_after_discard
    try:
        fan, _ = player.calculate_potential_fan()
    finally:
        player.hand_tiles = original_hand
    return int(fan)


def _shape_direction(hand_tiles: List[Any], open_melds: List[List[Any]], missing_suit: Any) -> float:
    all_tiles = list(hand_tiles) + [tile for meld in open_melds for tile in meld]
    counted = [tile for tile in all_tiles if tile.suit != missing_suit]
    if not counted:
        return 0.0
    suit_counts = Counter(tile.suit for tile in counted)
    purity = max(suit_counts.values()) / len(counted)
    number_counts = Counter((tile.suit, tile.number) for tile in counted)
    proto_triplets = sum(count >= 2 for count in number_counts.values())
    pung_ratio = min(1.0, proto_triplets / max(1, len(counted) // 3))
    terminal_ratio = sum(tile.number in (1, 9) for tile in counted) / len(counted)
    return 2.0 * purity + pung_ratio + 2.0 * terminal_ratio


def _tell_threat(game: MahjongGame, player_id: int, extra_tile: Any = None, window: int = 6) -> float:
    player = game.players[player_id]
    recent = list(player.discarded_tiles)
    if extra_tile is not None:
        recent.append(extra_tile)
    recent = recent[-window:]
    other_discards = {
        (tile.suit, tile.number)
        for other in game.players
        if other.player_id != player_id
        for tile in other.discarded_tiles
    }
    push = fold = 0.0
    for tile in recent:
        if tile.number in (4, 5, 6):
            push += 1.0
        elif tile.number in (1, 9):
            fold += 1.0
        elif (tile.suit, tile.number) in other_discards:
            fold += 0.5
    tell = (push - fold) / max(1, len(recent))
    base = 0.30 + 0.06 * len(player.open_melds) + 0.10 * _clip((40 - game.deck.remaining_count()) / 40.0)
    return _clip(base + 0.35 * tell)


def _public_safe_score(game: MahjongGame, player_id: int, tile: Any) -> float:
    discards = {
        (discard.suit, discard.number)
        for player in game.players
        for discard in player.discarded_tiles
    }
    if (tile.suit, tile.number) in discards:
        return 1.0
    return 0.5 if tile.number in (1, 9) else 0.0


def _hu_fan(game: MahjongGame, player_id: int) -> int:
    player = game.players[player_id]
    try:
        if player.can_hu() and player.hand_tiles:
            win_tile = player.last_drawn_tile or player.hand_tiles[-1]
            fan, _ = FanCalculator.calculate_fan(
                player.hand_tiles.copy(),
                win_tile,
                open_melds=player.open_melds,
                is_self_drawn=True,
                gang_count=player.gang_count,
                concealed_kong_count=player.concealed_kong_count,
            )
            return int(fan)
    except Exception:
        pass
    return 0


def action_features(
    game: MahjongGame,
    player_id: int,
    action: str,
    metrics: Optional[Dict[str, int]] = None,
) -> List[float]:
    player = game.players[player_id]
    values = [0.0] * ACTION_DIM
    if action == "h":
        values[1] = 1.0
        values[11] = _clip(_hu_fan(game, player_id) / 8.0)
        return values
    if not action.startswith("d "):
        return values

    tile = next((item for item in player.hand_tiles if str(item) == action[2:]), None)
    if tile is None:
        return values
    metrics = metrics or discard_progress_metrics(game, player_id, action)
    if metrics is None:
        return values
    values[0] = 1.0
    values[2] = tile.number / 9.0
    suits = list(type(tile.suit))
    if tile.suit in suits:
        values[3 + suits.index(tile.suit)] = 1.0
    values[6] = sum(item == tile for item in player.hand_tiles) / 4.0
    values[7] = float(tile.number in (1, 9))
    values[8] = _clip((int(metrics["shanten"]) + 1) / 8.0)
    values[9] = _clip(int(metrics["effective_types"]) / 9.0)
    values[10] = _clip(int(metrics["effective_copies"]) / 36.0)
    hand_after = player.hand_tiles.copy()
    hand_after.remove(tile)
    if int(metrics["shanten"]) == 0:
        values[11] = _clip(_candidate_potential_fan(player, hand_after) / 8.0)
    values[12] = _clip(_shape_direction(hand_after, player.open_melds, player.missing_suit) / 5.0)
    values[13] = _tell_threat(game, player_id, tile) - _tell_threat(game, player_id)
    values[14] = _public_safe_score(game, player_id, tile)
    values[15] = float(player.last_drawn_tile == tile)
    return values


def build_action_candidates(
    game: MahjongGame,
    player_id: int,
    valid_actions: Sequence[str],
    max_shanten_regret: int = 0,
    min_ukeire_ratio: float = 1.0,
) -> List[Dict[str, Any]]:
    progress = {
        action: metrics
        for action in valid_actions
        if action.startswith("d ")
        for metrics in [discard_progress_metrics(game, player_id, action)]
        if metrics is not None
    }
    feasible: List[str] = []
    if progress:
        best_shanten = min(item["shanten"] for item in progress.values())
        best_ukeire = max(
            item["effective_copies"]
            for item in progress.values()
            if item["shanten"] == best_shanten
        )
        feasible = [
            action
            for action, item in progress.items()
            if item["shanten"] <= best_shanten + max(0, int(max_shanten_regret))
            and item["effective_copies"] >= best_ukeire * max(0.0, float(min_ukeire_ratio))
        ]
    if "h" in valid_actions:
        feasible.append("h")
    return [
        {
            "action": action,
            "features": action_features(game, player_id, action, progress.get(action)),
            "metrics": progress.get(action),
        }
        for action in feasible
    ]


class ActionValueNet(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden: int = 128):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 64), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.value_head = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))
        self.mode_head = nn.Linear(64, len(MODES))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_hidden = self.state_encoder(states)
        action_hidden = self.action_encoder(actions)
        expanded_state = state_hidden.unsqueeze(1).expand(-1, actions.shape[1], -1)
        values = self.value_head(torch.cat([expanded_state, action_hidden], dim=-1)).squeeze(-1)
        return values, self.mode_head(state_hidden)


class ResidualScoreNet(nn.Module):
    """Small correction head trained on paired settlement rollouts."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden: int = 96):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 48), nn.ReLU()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 48), nn.ReLU(), nn.Linear(48, 24), nn.ReLU()
        )
        self.residual_head = nn.Sequential(nn.Linear(72, 48), nn.ReLU(), nn.Linear(48, 1))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        state_hidden = self.state_encoder(states)
        action_hidden = self.action_encoder(actions)
        expanded_state = state_hidden.unsqueeze(1).expand(-1, actions.shape[1], -1)
        return self.residual_head(
            torch.cat([expanded_state, action_hidden], dim=-1)
        ).squeeze(-1)


class ActionValuePolicy:
    def __init__(self, model: ActionValueNet, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ActionValuePolicy":
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        model = ActionValueNet(
            state_dim=int(checkpoint.get("state_dim", STATE_DIM)),
            action_dim=int(checkpoint.get("action_dim", ACTION_DIM)),
            hidden=int(checkpoint.get("hidden", 128)),
        )
        model.load_state_dict(checkpoint["model_state"])
        return cls(model, device=device)

    def select(
        self,
        game: MahjongGame,
        player_id: int,
        valid_actions: Sequence[str],
        state_features: Sequence[float],
        max_shanten_regret: int = 0,
        min_ukeire_ratio: float = 1.0,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        candidates = build_action_candidates(
            game,
            player_id,
            valid_actions,
            max_shanten_regret=max_shanten_regret,
            min_ukeire_ratio=min_ukeire_ratio,
        )
        if not candidates:
            return None, {"used": False, "reason": "no_value_candidates"}
        state = torch.tensor([list(state_features)], dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            [[item["features"] for item in candidates]], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            values, mode_logits = self.model(state, actions)
        scores = values[0]
        selected_index = int(torch.argmax(scores).item())
        mode_index = int(torch.argmax(mode_logits[0]).item())
        selected = str(candidates[selected_index]["action"])
        return selected, {
            "used": True,
            "selected_action": selected,
            "selected_mode": MODES[mode_index],
            "candidate_scores": {
                str(item["action"]): float(scores[index].item())
                for index, item in enumerate(candidates)
            },
        }


class ValueResidualPolicy:
    """Frozen continuous-v2 distillation plus an on-policy return residual."""

    def __init__(
        self,
        base_model: ActionValueNet,
        residual_model: ResidualScoreNet,
        residual_scale: float = 1.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.base_model = base_model.to(self.device).eval()
        self.residual_model = residual_model.to(self.device).eval()
        self.residual_scale = float(residual_scale)

    @classmethod
    def load(
        cls,
        base_path: str | Path,
        residual_path: str | Path,
        device: str = "cpu",
    ) -> "ValueResidualPolicy":
        base_checkpoint = torch.load(Path(base_path), map_location=device, weights_only=False)
        base_model = ActionValueNet(
            state_dim=int(base_checkpoint.get("state_dim", STATE_DIM)),
            action_dim=int(base_checkpoint.get("action_dim", ACTION_DIM)),
            hidden=int(base_checkpoint.get("hidden", 128)),
        )
        base_model.load_state_dict(base_checkpoint["model_state"])
        residual_checkpoint = torch.load(
            Path(residual_path), map_location=device, weights_only=False
        )
        residual_model = ResidualScoreNet(
            state_dim=int(residual_checkpoint.get("state_dim", STATE_DIM)),
            action_dim=int(residual_checkpoint.get("action_dim", ACTION_DIM)),
            hidden=int(residual_checkpoint.get("hidden", 96)),
        )
        residual_model.load_state_dict(residual_checkpoint["model_state"])
        return cls(
            base_model,
            residual_model,
            residual_scale=float(residual_checkpoint.get("residual_scale", 1.0)),
            device=device,
        )

    def select(
        self,
        game: MahjongGame,
        player_id: int,
        valid_actions: Sequence[str],
        state_features: Sequence[float],
        max_shanten_regret: int = 0,
        min_ukeire_ratio: float = 1.0,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        candidates = build_action_candidates(
            game,
            player_id,
            valid_actions,
            max_shanten_regret=max_shanten_regret,
            min_ukeire_ratio=min_ukeire_ratio,
        )
        if not candidates:
            return None, {"used": False, "reason": "no_value_candidates"}
        state = torch.tensor([list(state_features)], dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            [[item["features"] for item in candidates]], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            base_values, mode_logits = self.base_model(state, actions)
            residuals = self.residual_model(state, actions)
            values = base_values + self.residual_scale * residuals
        scores = values[0]
        selected_index = int(torch.argmax(scores).item())
        mode_index = int(torch.argmax(mode_logits[0]).item())
        selected = str(candidates[selected_index]["action"])
        return selected, {
            "used": True,
            "selected_action": selected,
            "selected_mode": MODES[mode_index],
            "residual_scale": self.residual_scale,
            "candidate_scores": {
                str(item["action"]): float(scores[index].item())
                for index, item in enumerate(candidates)
            },
            "base_candidate_scores": {
                str(item["action"]): float(base_values[0, index].item())
                for index, item in enumerate(candidates)
            },
            "residual_scores": {
                str(item["action"]): float(residuals[0, index].item())
                for index, item in enumerate(candidates)
            },
        }
