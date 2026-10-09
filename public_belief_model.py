"""Deployable public-information belief model for MASK.

The feature vector contains only public actions, declared missing suit, open
meld count, wall phase and the public-information MC score. It never reads the
target player's concealed hand.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib


PUBLIC_BELIEF_FEATURE_VERSION = "public_belief_v1"
PUBLIC_BELIEF_FEATURE_NAMES = (
    "mc_raw",
    "discard_count",
    "open_meld_count",
    "missing_suit_discard_count",
    "tiles_left",
    "history_length",
    "peng_count",
    "gang_count",
    "distinct_discard_count",
    "max_discard_multiplicity",
    "repeated_discard_kinds",
    "terminal_discard_rate",
    "edge_discard_rate",
    "middle_discard_rate",
    "wan_discard_rate",
    "tiao_discard_rate",
    "tong_discard_rate",
    "discard_rank_std",
    "discard_rank_mean",
    "discard_history_rate",
    "peng_action_rate",
    "gang_action_rate",
    "last_rank_1",
    "last_rank_2",
    "last_rank_3",
    "last_rank_4",
    "last_rank_5",
    "last_rank_6",
)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _missing_suit_text(missing_suit: Any) -> str:
    if missing_suit is None:
        return ""
    return str(getattr(missing_suit, "value", missing_suit))


def build_public_belief_features(
    history: Iterable[Dict[str, Any]],
    *,
    target_pid: int,
    missing_suit: Any,
    open_meld_count: int,
    tiles_left: int,
    raw_mc: float,
) -> List[float]:
    records = list(history)
    target_actions = [row for row in records if int(row.get("pid", -1)) == target_pid]
    discards = [
        str(row.get("tile") or "")
        for row in target_actions
        if row.get("act") == "discard"
    ]
    action_counts = Counter(str(row.get("act") or "") for row in target_actions)
    tile_counts = Counter(discards)
    ranks = [int(tile[0]) for tile in discards if len(tile) >= 2 and tile[0].isdigit()]
    suits = Counter(tile[1] for tile in discards if len(tile) >= 2)
    discard_denominator = max(1, len(discards))
    action_denominator = max(1, len(target_actions))
    history_denominator = max(1, len(records))
    missing_text = _missing_suit_text(missing_suit)
    rank_mean = sum(ranks) / len(ranks) if ranks else 0.0
    rank_variance = (
        sum((rank - rank_mean) ** 2 for rank in ranks) / len(ranks)
        if ranks else 0.0
    )
    last_ranks = ([0] * 6 + ranks)[-6:]
    values = [
        float(raw_mc),
        float(len(discards)),
        float(open_meld_count),
        float(sum(1 for tile in discards if len(tile) >= 2 and tile[1] == missing_text)),
        float(tiles_left),
        float(len(records)),
        float(action_counts["peng"]),
        float(action_counts["gang"]),
        float(len(tile_counts)),
        float(max(tile_counts.values(), default=0)),
        float(sum(count > 1 for count in tile_counts.values())),
        sum(rank in (1, 9) for rank in ranks) / discard_denominator,
        sum(rank in (2, 8) for rank in ranks) / discard_denominator,
        sum(rank in (4, 5, 6) for rank in ranks) / discard_denominator,
        suits["万"] / discard_denominator,
        suits["条"] / discard_denominator,
        suits["筒"] / discard_denominator,
        math.sqrt(rank_variance),
        rank_mean,
        len(discards) / history_denominator,
        action_counts["peng"] / action_denominator,
        action_counts["gang"] / action_denominator,
        *[float(rank) for rank in last_ranks],
    ]
    if len(values) != len(PUBLIC_BELIEF_FEATURE_NAMES):
        raise RuntimeError("public belief feature dimension mismatch")
    return values


def features_from_h1_row(row: Dict[str, Any]) -> List[float]:
    public = row.get("public_features") or {}
    history = row.get("a_i_history") or []
    missing_suit = ""
    for action in history:
        tile = str(action.get("tile") or "")
        if action.get("act") == "discard" and len(tile) >= 2:
            # H1 stores the already-computed count, so preserve it below even
            # when the declared suit itself is unavailable in historical rows.
            missing_suit = tile[1]
            break
    values = build_public_belief_features(
        history,
        target_pid=0,
        missing_suit=missing_suit,
        open_meld_count=int(public.get("own_melds", 0)),
        tiles_left=int(public.get("tiles_left", 0)),
        raw_mc=float(
            (row.get("b_j_oracle") or {}).get("mc_opponent_posterior")
            if (row.get("b_j_oracle") or {}).get("mc_opponent_posterior") is not None
            else row.get("B2_B_phi", 0.0)
        ),
    )
    index = {name: position for position, name in enumerate(PUBLIC_BELIEF_FEATURE_NAMES)}
    history_length = max(1, int(row.get("t", len(history))))
    discard_count = sum(1 for action in history if action.get("act") == "discard")
    values[index["missing_suit_discard_count"]] = float(public.get("missing_discards", 0))
    values[index["history_length"]] = float(history_length)
    values[index["discard_history_rate"]] = discard_count / history_length
    return values


class PublicBeliefModel:
    def __init__(self, estimator: Any, metadata: Dict[str, Any], path: Optional[Path] = None):
        self.estimator = estimator
        self.metadata = metadata
        self.path = path
        names = tuple(metadata.get("feature_names") or ())
        if names != PUBLIC_BELIEF_FEATURE_NAMES:
            raise ValueError(
                f"public belief feature schema mismatch: artifact={names}, "
                f"runtime={PUBLIC_BELIEF_FEATURE_NAMES}"
            )

    @classmethod
    def load(cls, path: str | Path) -> "PublicBeliefModel":
        artifact_path = Path(path)
        artifact = joblib.load(artifact_path)
        if not isinstance(artifact, dict) or "estimator" not in artifact:
            raise ValueError(f"invalid public belief model artifact: {artifact_path}")
        return cls(artifact["estimator"], artifact.get("metadata") or {}, artifact_path)

    def predict_features(self, features: List[float]) -> float:
        probability = float(self.estimator.predict_proba([features])[0][1])
        return max(0.0, min(1.0, probability))

    def predict_game(self, game: Any, *, target_pid: int, raw_mc: float) -> float:
        target = game.players[target_pid]
        features = build_public_belief_features(
            game.history,
            target_pid=target_pid,
            missing_suit=target.missing_suit,
            open_meld_count=len(target.open_melds),
            tiles_left=game.deck.remaining_count(),
            raw_mc=raw_mc,
        )
        return self.predict_features(features)
