"""
Gate1 minimal experiments for LLM-native MASK.

This is the first executable bridge from the W0 skeleton to W1/W2 evidence:

E1: coarse B_phi validity
    Does the belief estimator's tenpai confidence track whether P0 is actually
    ready?  This is only a coarse oracle, not the final opponent-posterior
    oracle described in the paper plan.

E2: L0/L1/L2 ladder
    L0 = llm_base       : public table + legal action heuristic.
    L1 = llm_reactive_z : L0 plus public opponent drift z_j(t).
    L2 = llm_mask       : z_j(t) + B_phi + risk gate.

Current default backend is heuristic fallback so the pipeline can run without a
local LLM.  Reports carry that backend label explicitly.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from belief_oracle import opponent_view_posterior, within_shanten
from experiment_trace import (
    ensure_deterministic_hashing,
    paired_method_comparison,
    summarize_belief_samples,
    summarize_by_method,
    write_json,
    write_jsonl,
)
from game import (
    STYLE_BOTS,
    MahjongGame,
    bot_decide_exchange,
    bot_decide_missing_suit,
    bot_decide_response,
    bot_decide_turn_action,
    parse_console_tile,
    print_table_info,
)
from llm_backends import build_llm_callable
from mask_llm import LLMBeliefEstimator, MASKLLMAgent, PublicOpponentTracker, RiskGate, legalize_action
from opponent_curriculum import defender_config, load_curriculum, select_specs_for_game
from policy_metrics import discard_progress_metrics
from ppo_features import extract_state_features
from prompt_builder import (
    blocked_calls_while_missing,
    build_base_decision_prompt,
    build_state_prompt,
    get_legal_actions,
)
from public_belief_model import PublicBeliefModel, sha256_file
from rule_engine import ShantenCalculator


RESULTS_DIR = Path("Gate1_results")
METHODS = ("llm_base", "llm_reactive_z", "llm_reactive_z_bphi", "llm_mask")
OPPONENT_STYLES = tuple(sorted(STYLE_BOTS.keys())) + (
    "mixed", "responsive", "neural", "curriculum"
)
LLMCallable = Callable[[str], str]
GAME_SEED_STRIDE = 1_000_000


def derive_game_seed(outer_seed: int, game_index: int) -> int:
    """Encode an outer seed and game index without cross-run collisions."""
    if outer_seed < 0:
        raise ValueError("outer_seed must be non-negative")
    if not 0 <= game_index < GAME_SEED_STRIDE:
        raise ValueError(
            f"game_index must be in [0, {GAME_SEED_STRIDE}), got {game_index}"
        )
    return outer_seed * GAME_SEED_STRIDE + game_index


def serializable_mask_config(
    runtime_config: Dict[str, Any],
    *,
    belief_public_model_path: Optional[str],
    belief_public_model_sha256: str,
) -> Dict[str, Any]:
    """Replace the loaded public-belief model with its reproducible artifact identity."""
    result = dict(runtime_config)
    result["belief_public_model"] = (
        str(belief_public_model_path) if belief_public_model_path else None
    )
    result["belief_public_model_sha256"] = belief_public_model_sha256 or None
    json.dumps(result)
    return result


def load_belief_calibration_steps(
    path: Optional[str],
    *,
    oracle_samples: Optional[int] = None,
    oracle_beta: Optional[float] = None,
    danger_threshold: Optional[int] = None,
    b_phi_source: Optional[str] = None,
    belief_public_model_sha256: Optional[str] = None,
) -> List[Tuple[float, float]]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates = [
        (data.get("calibration") or {}).get("B2_B_phi_steps"),
        data.get("isotonic_steps"),
    ]
    steps = next((value for value in candidates if value), None)
    if not steps:
        raise ValueError(f"belief calibration file has no B2/isotonic steps: {path}")
    expected = {
        "oracle_samples": oracle_samples,
        "oracle_beta": oracle_beta,
        "danger_threshold": danger_threshold,
        "b_phi_source": b_phi_source,
        "belief_public_model_sha256": belief_public_model_sha256,
    }
    if any(value is not None for value in expected.values()):
        config = data.get("estimator_config")
        if not isinstance(config, dict):
            raise ValueError(
                f"belief calibration file has no estimator_config for deployment validation: {path}"
            )
        mismatches = []
        for key, value in expected.items():
            if value is None:
                continue
            actual = config.get(key)
            equal = (
                abs(float(actual) - float(value)) <= 1e-12
                if key == "oracle_beta" and actual is not None
                else actual == value
            )
            if not equal:
                mismatches.append(f"{key}: artifact={actual}, runtime={value}")
        if mismatches:
            raise ValueError(
                "belief calibration estimator mismatch: " + "; ".join(mismatches)
            )
    return [(float(threshold), float(value)) for threshold, value in steps]


def init_game(
    seed: int, opponent_style: str, game_id: str,
    defender_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[MahjongGame, Dict[int, Tuple], Dict[int, ResponsiveDefender]]:
    random.seed(seed)
    game = MahjongGame(game_id, ["Agent", "B1", "B2", "B3"], bots=[False, True, True, True])
    defenders: Dict[int, ResponsiveDefender] = {}
    if opponent_style == "curriculum":
        cfg = defender_cfg or {}
        manifest_path = cfg.get("curriculum_manifest")
        if not manifest_path:
            raise ValueError("--opponent-style curriculum requires --curriculum-manifest")
        manifest = load_curriculum(manifest_path, split=cfg.get("curriculum_split", "holdout"))
        explicit = cfg.get("curriculum_specs_by_pid") or {}
        selected = select_specs_for_game(manifest, seed)
        requested_id = cfg.get("curriculum_spec_id")
        if requested_id:
            requested = next(
                (item for item in manifest["opponents"] if item["id"] == requested_id),
                None,
            )
            if requested is None:
                raise ValueError(f"curriculum opponent id not found: {requested_id}")
            explicit = {pid: requested for pid in (1, 2, 3)}
        for pid in (1, 2, 3):
            spec = explicit.get(pid) or explicit.get(str(pid)) or selected[pid]
            spec_cfg = defender_config(spec)
            defenders[pid] = ResponsiveDefender(
                observer_pid=pid,
                threat_threshold=spec_cfg["threat_threshold"],
                oracle_samples=spec_cfg["oracle_samples"],
                beta=spec_cfg["beta"],
                danger_threshold=spec_cfg["danger_threshold"],
                ffr_hand_shanten=spec_cfg["ffr_hand_shanten"],
                rng=random.Random(seed * 100 + pid),
                threat_model=spec_cfg["threat_model"],
                tell_weight=spec_cfg["tell_weight"],
                tell_window=spec_cfg["tell_window"],
            )
            defenders[pid].curriculum_spec = dict(spec)
        opponent_funcs = {pid: (defenders[pid].turn, None) for pid in (1, 2, 3)}
    elif opponent_style == "responsive":
        cfg = defender_cfg or {}
        for pid in (1, 2, 3):
            defenders[pid] = ResponsiveDefender(
                observer_pid=pid,
                threat_threshold=cfg.get("threat_threshold", 0.4),
                oracle_samples=cfg.get("oracle_samples", 30),
                beta=cfg.get("beta", 2.0),
                danger_threshold=cfg.get("danger_threshold", 1),
                ffr_hand_shanten=cfg.get("ffr_hand_shanten", 1),
                rng=random.Random(seed * 100 + pid),
                threat_model=cfg.get("threat_model", "mc"),
                tell_weight=cfg.get("tell_weight", 1.0),
                tell_window=cfg.get("tell_window", 6),
                learned_model_path=cfg.get("learned_model_path"),
            )
        opponent_funcs = {pid: (defenders[pid].turn, None) for pid in (1, 2, 3)}
    elif opponent_style == "neural":
        from neural_opponent_policy import load_policy

        cfg = defender_cfg or {}
        model_path = cfg.get("neural_model_path")
        if not model_path:
            raise ValueError(
                "--opponent-style neural requires --neural-opponent-model-path "
                "pointing to a trained neural_opponent_policy.pth checkpoint."
            )
        for pid in (1, 2, 3):
            defenders[pid] = load_policy(
                model_path=model_path,
                observer_pid=pid,
                device=cfg.get("neural_device", "cpu"),
                danger_threshold=cfg.get("danger_threshold", 1),
                ffr_hand_shanten=cfg.get("ffr_hand_shanten", 1),
                risk_mode=cfg.get("neural_risk_mode", "predicted"),
            )
        opponent_funcs = {pid: (defenders[pid].turn, None) for pid in (1, 2, 3)}
    elif opponent_style == "mixed":
        styles = ["aggressive", "conservative", "random"]
        opponent_funcs = {pid: STYLE_BOTS[styles[pid - 1]] for pid in (1, 2, 3)}
    else:
        opponent_funcs = {pid: STYLE_BOTS.get(opponent_style, STYLE_BOTS["greedy"]) for pid in (1, 2, 3)}
    game.start_game()

    for player in game.players:
        game.select_exchange_tiles(player.player_id, bot_decide_exchange(player))

    for player in game.players:
        game.set_missing_suit(player.player_id, bot_decide_missing_suit(player))

    return game, opponent_funcs, defenders


def true_tenpai_label(game: MahjongGame, player_id: int = 0) -> Tuple[bool, List[str], int]:
    player = game.players[player_id]
    ready, waits = player.is_ready_with_missing_suit()
    shanten = ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit)
    return bool(ready), [str(t) for t in waits], int(shanten)


def action_result_shanten(game: MahjongGame, player_id: int, action: str) -> Optional[int]:
    metrics = discard_progress_metrics(game, player_id, action)
    return metrics["shanten"] if metrics else None


def classify_strict_false_fold(
    *,
    own_shanten: int,
    p0_shanten: int,
    high_threat: bool,
    push_action: str,
    chosen_action: str,
    push_metrics: Optional[Dict[str, int]],
    chosen_metrics: Optional[Dict[str, int]],
) -> Dict[str, Any]:
    """Classify a real false fold without treating every safe switch as regret."""
    eligible = bool(
        own_shanten <= 1
        and p0_shanten > 0
        and push_action.startswith("d ")
        and chosen_action.startswith("d ")
        and push_metrics is not None
        and chosen_metrics is not None
    )
    shanten_loss = bool(
        eligible and chosen_metrics["shanten"] > push_metrics["shanten"]
    )
    ukeire_loss = bool(
        eligible
        and chosen_metrics["shanten"] == push_metrics["shanten"]
        and chosen_metrics["effective_copies"] < push_metrics["effective_copies"]
    )
    progress_loss = shanten_loss or ukeire_loss
    return {
        "eligible": eligible,
        "false_fold": bool(
            eligible
            and high_threat
            and chosen_action != push_action
            and progress_loss
        ),
        "progress_loss": progress_loss,
        "shanten_loss": shanten_loss,
        "ukeire_loss": ukeire_loss,
    }


def best_discard_result_shanten(
    game: MahjongGame,
    player_id: int,
    valid_actions: List[str],
) -> Optional[int]:
    values = [
        value
        for action in valid_actions
        if (value := action_result_shanten(game, player_id, action)) is not None
    ]
    return min(values) if values else None


def estimate_hu_value_floor(game: MahjongGame, player_id: int = 0) -> float:
    player = game.players[player_id]
    fan = 0
    try:
        if player.can_hu() and player.hand_tiles:
            from rule_engine import FanCalculator

            win_tile = player.hand_tiles[-1]
            fan, _ = FanCalculator.calculate_fan(
                player.hand_tiles.copy(),
                win_tile,
                open_melds=player.open_melds,
                is_self_drawn=True,
                gang_count=player.gang_count,
                is_last_tile=(game.deck.remaining_count() == 0),
            )
        else:
            fan, _ = player.calculate_potential_fan()
    except Exception:
        fan = 0
    fan = max(1, int(fan))
    return float(getattr(game, "base_score", 10) * (2 ** min(fan, getattr(game, "max_fan", 8))))


def choose_min_shanten_action(
    game: MahjongGame,
    player_id: int,
    valid_actions: Optional[List[str]] = None,
    allow_hu: bool = True,
) -> str:
    valid_actions = valid_actions if valid_actions is not None else get_legal_actions(game, player_id)
    player = game.players[player_id]

    if allow_hu and "h" in valid_actions:
        return "h"
    if "g" in valid_actions:
        return "g"

    best_action = next(
        (action for action in valid_actions if allow_hu or action != "h"),
        (valid_actions[0] if valid_actions else "n"),
    )
    best_shanten = 99
    for action in valid_actions:
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


def choose_public_safe_no_regret(
    game: MahjongGame,
    player_id: int,
    valid_actions: List[str],
    fallback_action: str,
) -> str:
    """Apply a public-risk shield without sacrificing immediate shanten."""
    if fallback_action in {"h", "g"}:
        return fallback_action
    public_discards = {
        (tile.suit, tile.number)
        for table_player in game.players
        for tile in table_player.discarded_tiles
    }
    metrics = {
        action: value
        for action in valid_actions
        if (value := discard_progress_metrics(game, player_id, action)) is not None
    }
    if not metrics:
        return fallback_action
    best_shanten = min(value["shanten"] for value in metrics.values())
    best_effective_copies = max(
        value["effective_copies"]
        for value in metrics.values()
        if value["shanten"] == best_shanten
    )
    player = game.players[player_id]
    candidates = []
    for action in valid_actions:
        value = metrics.get(action)
        if (
            value is None
            or value["shanten"] != best_shanten
            or value["effective_copies"] < best_effective_copies
        ):
            continue
        tile = next((tile for tile in player.hand_tiles if str(tile) == action[2:]), None)
        if tile is not None and (tile.suit, tile.number) in public_discards:
            candidates.append(action)
    if fallback_action in candidates:
        return fallback_action
    return candidates[0] if candidates else fallback_action


def defensive_discard_action(game: MahjongGame, pid: int, valid_actions: List[str]) -> str:
    """Safest discard: a tile already publicly discarded (won't deal in), else a
    terminal (1/9), else the first legal discard.  Used when folding under threat."""
    player = game.players[pid]
    discards = [a for a in valid_actions if a.startswith("d ")]
    if not discards:
        return valid_actions[0] if valid_actions else "n"
    public = {(t.suit, t.number) for p in game.players for t in p.discarded_tiles}
    for action in discards:
        tile_text = action[2:]
        for tile in player.hand_tiles:
            if str(tile) == tile_text and (tile.suit, tile.number) in public:
                return action
    for action in discards:
        try:
            num = int(action[2])
        except (ValueError, IndexError):
            num = 5
        if num in (1, 9):
            return action
    return discards[0]


def discard_tell_threat(game: MahjongGame, target_pid: int = 0, window: int = 6) -> float:
    """Deceivable, deterministic threat from P0's RECENT discard pattern.

    Unlike the pool-based MC posterior (which P0 can barely move), this reads the
    classic tell and IS manipulable by P0's discard choice:
      * middle tiles (4-6) discarded  -> looks like pushing  -> threat UP;
      * terminals (1/9) or already-public-safe tiles -> looks like folding -> threat DOWN.
    A structural base (melds + tile depletion) anchors it. No sampling -> the
    before/after threat-delta from this model is noise-free.
    """
    p0 = game.players[target_pid]
    recent = list(p0.discarded_tiles)[-window:]
    others = {(t.suit, t.number) for pl in game.players if pl.player_id != target_pid for t in pl.discarded_tiles}
    push = fold = 0.0
    for t in recent:
        if t.number in (4, 5, 6):
            push += 1.0
        elif t.number in (1, 9):
            fold += 1.0
        elif (t.suit, t.number) in others:
            fold += 0.5
        # 2,3,7,8 are neutral
    tell = (push - fold) / max(1, len(recent))           # roughly [-1.5, 1]
    melds = len(p0.open_melds)
    tiles_left = game.deck.remaining_count()
    base = 0.30 + 0.06 * melds + 0.10 * max(0.0, min(1.0, (40 - tiles_left) / 40.0))
    return max(0.0, min(1.0, base + 0.35 * tell))


class ResponsiveDefender:
    """Experiment-only, rule-based, deceivable opponent.

    Reads only PUBLIC info to estimate how dangerous P0 looks (cached MC
    posterior), then:
      * turn policy: threat low  -> greedy push (may deal into P0 -> feeds DIR);
                     threat high -> defensive safe discard (folds a good hand
                     when it wrongly believes P0 is dangerous -> feeds FFR).
      * response policy: threat high -> decline peng/gang (commitment); always
                     take a hu (declining a win is not realistic).

    Strict FFR is measured on the first eligible discard decision per
    opponent-game. P0 must truly be non-tenpai, the defender must be at most one
    shanten away, and the defensive discard must lose shanten or effective
    copies relative to push. The broader defensive-switch rate remains a
    diagnostic. Oracle information is never used by the policy decision.
    """

    def __init__(self, observer_pid: int, threat_threshold: float, oracle_samples: int,
                 beta: float, danger_threshold: int, ffr_hand_shanten: int, rng: random.Random,
                 threat_model: str = "mc", tell_weight: float = 1.0, tell_window: int = 6,
                 learned_model_path: Optional[str] = None):
        self.pid = observer_pid
        self.threshold = threat_threshold
        self.oracle_samples = oracle_samples
        self.beta = beta
        self.danger_threshold = danger_threshold
        self.ffr_hand_shanten = ffr_hand_shanten
        self.rng = rng
        self.threat_model = threat_model      # "mc" | "discard_tell" | "blend" | "learned"
        self.tell_weight = tell_weight        # blend: tell_weight*tell + (1-tell_weight)*mc
        self.tell_window = tell_window
        self.learned_model_path = learned_model_path
        self._learned_model = None            # lazy-loaded on first "learned" inference
        self._cache: Dict[int, float] = {}
        self.ff_opportunities = 0   # first strict opportunity; max one per game
        self.ff_false = 0           # first strict opportunity was a real false fold
        self.strict_ffr_event_opportunities = 0
        self.strict_ffr_event_false = 0
        self.strict_ffr_event_events: List[Dict[str, Any]] = []
        self.defensive_switch_opportunities = 0
        self.defensive_switches = 0
        self.defensive_switch_events: List[Dict[str, Any]] = []
        self.declined_self_gang_opportunities = 0
        self.declined_self_gangs = 0
        self.turn_folds = 0
        self.response_declines = 0
        self.ffr_events: List[Dict[str, Any]] = []
        self.false_fold_events: List[Dict[str, Any]] = []

    def _mc_threat(self, game: MahjongGame, num_samples: int, rng: random.Random) -> float:
        post = opponent_view_posterior(
            game, target_pid=0, observer_pid=self.pid,
            num_samples=num_samples, rng=rng, beta=self.beta, max_shanten=self.danger_threshold,
        )
        return float(post["tenpai_prob"])

    def _learned_threat(self, game: MahjongGame) -> float:
        """Learned danger channel: same public inputs discard_tell_threat() reads
        (P0's own discard window, meld count, others' safe-discard set), scored by
        the trained defender_danger_model checkpoint instead of a fixed formula."""
        import numpy as np

        from defender_danger_model import WINDOW_SIZE, load_model, predict_proba
        from opponent_classifier import extract_action_features

        if self._learned_model is None:
            self._learned_model = load_model(self.learned_model_path, device="cpu")

        p0 = game.players[0]
        others = {(t.suit, t.number) for p in game.players if p.player_id != 0 for t in p.discarded_tiles}
        meld_count = len(p0.open_melds)
        total = len(p0.discarded_tiles)
        start_idx = max(0, total - WINDOW_SIZE)
        recent = p0.discarded_tiles[start_idx:]
        window = []
        for offset, tile in enumerate(recent):
            global_idx = start_idx + offset + 1  # true running discard count, matches training features
            is_safe = (tile.suit, tile.number) in others
            window.append(extract_action_features(tile, "discard", global_idx, meld_count, is_safe, meld_count))
        if not window:
            return 0.5
        arr = np.stack(window).astype(np.float32)
        if len(arr) < WINDOW_SIZE:
            pad = np.zeros((WINDOW_SIZE - len(arr), arr.shape[1]), dtype=np.float32)
            arr = np.vstack([pad, arr])
        return predict_proba(self._learned_model, arr, device="cpu")

    def _combine(self, game: MahjongGame, num_samples: int, rng: random.Random) -> float:
        if self.threat_model == "discard_tell":
            return discard_tell_threat(game, 0, self.tell_window)
        if self.threat_model == "blend":
            tell = discard_tell_threat(game, 0, self.tell_window)
            mc = self._mc_threat(game, num_samples, rng)
            return self.tell_weight * tell + (1.0 - self.tell_weight) * mc
        if self.threat_model == "learned":
            return self._learned_threat(game)
        return self._mc_threat(game, num_samples, rng)  # "mc"

    def threat(self, game: MahjongGame) -> float:
        key = len(game.history)  # public state advances with history; cache per state
        if key not in self._cache:
            self._cache[key] = self._combine(game, self.oracle_samples, self.rng)
        return self._cache[key]

    def threat_crn(self, game: MahjongGame, seeds: List[int], num_samples: int) -> float:
        """Measurement-only threat with common random numbers (paired sampling).

        Pass the SAME seeds for the before- and after-action snapshot so MC
        sampling noise largely cancels and the delta reflects P0's action, not
        sampling jitter.  The discard_tell model is deterministic, so its delta is
        exact.  Does not affect the bot's policy (which still uses threat())."""
        vals = [self._combine(game, num_samples, random.Random(s)) for s in seeds]
        return sum(vals) / len(vals) if vals else 0.0

    def turn(
        self,
        player,
        game: MahjongGame,
        step: int = -1,
        last_p0_state: Optional[Dict[str, Any]] = None,
        in_deceive_window: bool = False,
    ) -> str:
        valid = get_legal_actions(game, self.pid)
        if "h" in valid:
            return "h"  # take a win when offered
        threat = self.threat(game)
        push = choose_min_shanten_action(game, self.pid, valid)
        high = threat >= self.threshold

        own_shanten = ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit)
        p0 = game.players[0]
        p0_shanten = ShantenCalculator.calculate_shanten(p0.hand_tiles, p0.missing_suit)
        p0_dangerous = within_shanten(p0.hand_tiles, p0.missing_suit, self.danger_threshold)  # oracle (measurement only)
        switch_opportunity = (own_shanten <= self.ffr_hand_shanten) and (not p0_dangerous)

        if high:
            action = defensive_discard_action(game, self.pid, valid)
            switched = action != push
        else:
            action = push
            switched = False

        if switched:
            self.turn_folds += 1
        push_metrics = discard_progress_metrics(game, self.pid, push)
        chosen_metrics = discard_progress_metrics(game, self.pid, action)
        strict = classify_strict_false_fold(
            own_shanten=int(own_shanten),
            p0_shanten=int(p0_shanten),
            high_threat=high,
            push_action=push,
            chosen_action=action,
            push_metrics=push_metrics,
            chosen_metrics=chosen_metrics,
        )

        def event_payload() -> Dict[str, Any]:
            return {
                "step": step,
                "defender_pid": self.pid,
                "opponent_id": (getattr(self, "curriculum_spec", {}) or {}).get("id"),
                "threat_at_fold": round(threat, 4),
                "threat_threshold": self.threshold,
                "own_shanten": int(own_shanten),
                "ffr_hand_shanten": self.ffr_hand_shanten,
                "p0_shanten": int(p0_shanten),
                "p0_dangerous": bool(p0_dangerous),
                "danger_threshold": self.danger_threshold,
                "push_action": push,
                "folded_action": action if switched else None,
                "chosen_action": action,
                "defensive_switch": bool(switched),
                "push_progress": push_metrics,
                "chosen_progress": chosen_metrics,
                "progress_loss": bool(strict["progress_loss"]),
                "shanten_loss": bool(strict["shanten_loss"]),
                "ukeire_loss": bool(strict["ukeire_loss"]),
                "strict_false_fold": bool(strict["false_fold"]),
                "false_fold": bool(strict["false_fold"]),
                "last_p0_step": (last_p0_state or {}).get("step"),
                "last_p0_mode": (last_p0_state or {}).get("mode"),
                "last_p0_action": (last_p0_state or {}).get("action"),
                "last_p0_reason": (last_p0_state or {}).get("reason"),
                "last_p0_threat_before": (last_p0_state or {}).get("threat_before"),
                "last_p0_threat_after": (last_p0_state or {}).get("threat_after"),
                "last_p0_threat_delta": (last_p0_state or {}).get("threat_delta"),
                "in_deceive_window": bool(in_deceive_window),
            }

        if switch_opportunity:
            self.defensive_switch_opportunities += 1
            switch_event = event_payload()
            self.defensive_switch_events.append(switch_event)
            if switched:
                self.defensive_switches += 1

        if push == "g":
            self.declined_self_gang_opportunities += 1
            if high and action != push:
                self.declined_self_gangs += 1

        if strict["eligible"]:
            self.strict_ffr_event_opportunities += 1
            strict_event = event_payload()
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
        return action

    def response(self, player, acts, game: MahjongGame) -> str:
        if "hu" in acts:
            return "h"
        if self.threat(game) >= self.threshold:
            self.response_declines += 1
            return "n"  # decline peng/gang under perceived threat (commitment risk)
        if "gang" in acts:
            return "g"
        if "peng" in acts:
            return "p"
        return "n"


def choose_llm_base_action(
    game: MahjongGame,
    player_id: int,
    valid_actions: List[str],
    llm: Optional[LLMCallable],
    auto_hu: str = "common",
) -> Tuple[str, Dict[str, Any]]:
    if "h" in valid_actions and auto_hu == "common":
        return "h", {
            "mode": "exploit",
            "reason": "common take-win rule",
            "common_take_win": True,
            "auto_hu": auto_hu,
        }

    if llm is None:
        return choose_min_shanten_action(
            game,
            player_id,
            valid_actions,
            allow_hu=(auto_hu == "common"),
        ), {
            "mode": "exploit",
            "reason": "L0 base min shanten fallback",
            "auto_hu": auto_hu,
        }

    prompt = build_base_decision_prompt(game, player_id, valid_actions=valid_actions)
    raw = llm(prompt)
    return legalize_action(
        raw,
        valid_actions,
        prefer_h_fallback=(auto_hu == "common"),
    ), {
        "mode": "exploit",
        "reason": "L0 local LLM base decision",
        "llm_raw": raw,
        "prompt_schema": "shared_base_action_v1",
        "auto_hu": auto_hu,
    }


def choose_reactive_z_action(
    game: MahjongGame,
    player_id: int,
    tracker: PublicOpponentTracker,
    risk_gate: RiskGate,
    llm: Optional[LLMCallable],
    valid_actions: Optional[List[str]] = None,
    auto_hu: str = "common",
    beliefs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    valid_actions = valid_actions if valid_actions is not None else get_legal_actions(game, player_id)
    z_state = tracker.update_from_game(game)
    gate = risk_gate.compute(game, player_id, z_state, beliefs=beliefs or {})

    if "h" in valid_actions and auto_hu == "common":
        return "h", {
            "z_state": z_state,
            "beliefs": beliefs or {},
            "gate": gate,
            "mode": "exploit",
            "reason": "common take-win rule",
            "common_take_win": True,
            "auto_hu": auto_hu,
        }

    if llm is not None:
        prompt = build_base_decision_prompt(game, player_id, valid_actions=valid_actions)
        raw = llm(prompt)
        baseline_action = legalize_action(
            raw,
            valid_actions,
            prefer_h_fallback=(auto_hu == "common"),
        )
        action = baseline_action
        mode = "exploit"
        reason = "L1 local LLM shared base decision"
        if baseline_action == "h":
            reason = "L1 local LLM selected hu"
        elif gate["mode_hint"] == "safe":
            action = choose_public_safe_no_regret(game, player_id, valid_actions, baseline_action)
            if action != baseline_action:
                mode = "safe"
                reason = "L1 public z/belief no-regret safety shield" if beliefs is not None else "L1 public z/risk no-regret safety shield"
        return action, {
            "z_state": z_state,
            "beliefs": beliefs or {},
            "gate": gate,
            "public_gate": gate,
            "mode": mode,
            "reason": reason,
            "llm_raw": raw,
            "baseline_action": baseline_action,
            "safe_override_applied": action != baseline_action,
            "prompt_schema": "shared_base_action_v1",
            "auto_hu": auto_hu,
        }

    if gate["mode_hint"] == "safe":
        public_discards = {(t.suit, t.number) for p in game.players for t in p.discarded_tiles}
        player = game.players[player_id]
        for action in valid_actions:
            if not action.startswith("d "):
                continue
            tile_text = action[2:]
            for tile in player.hand_tiles:
                if str(tile) == tile_text and (tile.suit, tile.number) in public_discards:
                    return action, {
                        "z_state": z_state,
                        "beliefs": beliefs or {},
                        "gate": gate,
                        "mode": "safe",
                        "reason": "reactive-z safe public discard",
                    }

    action = choose_min_shanten_action(
        game,
        player_id,
        valid_actions,
        allow_hu=(auto_hu == "common"),
    )
    return action, {
        "z_state": z_state,
        "beliefs": beliefs or {},
        "gate": gate,
        "mode": "exploit",
        "reason": "reactive-z min shanten",
        "auto_hu": auto_hu,
    }


def execute_action(game: MahjongGame, pid: int, action: str, drawn_tile=None) -> Optional[object]:
    player = game.players[pid]
    action = (action or "").strip()
    discarded_tile = None

    if action == "h" and player.can_hu():
        win_tile = drawn_tile if drawn_tile else (player.hand_tiles[-1] if player.hand_tiles else None)
        if win_tile is not None:
            game.hu(pid, win_tile, True)
            game.check_game_over()
        return None

    if action == "g":
        gang_info = game.can_self_gang(pid)
        if gang_info.get("can_gang"):
            gang_succeeded, _ = game.gang(pid, gang_info["gang_tiles"][0])
            if gang_succeeded:
                return None

    if action.startswith("d "):
        tile = parse_console_tile(action[2:])
        if tile and game.discard_tile(pid, tile):
            discarded_tile = tile

    if discarded_tile is None:
        legal_discards = [a for a in get_legal_actions(game, pid) if a.startswith("d ")]
        if legal_discards:
            tile = parse_console_tile(legal_discards[0][2:])
            if tile and game.discard_tile(pid, tile):
                discarded_tile = tile

    return discarded_tile


def settlement_breakdown_for_player(game: MahjongGame, player_id: int) -> Dict[str, Any]:
    net = Counter()
    paid = Counter()
    received = Counter()
    event_counts = Counter()
    for entry in getattr(game, "settlement_ledger", []):
        kind = str(entry.get("kind", "unknown"))
        score = float(entry.get("score", 0.0))
        if int(entry.get("payer_id", -1)) == player_id:
            net[kind] -= score
            paid[kind] += score
            event_counts[f"paid_{kind}"] += 1
        if int(entry.get("receiver_id", -1)) == player_id:
            net[kind] += score
            received[kind] += score
            event_counts[f"received_{kind}"] += 1
    return {
        "net": dict(net),
        "paid": dict(paid),
        "received": dict(received),
        "event_counts": dict(event_counts),
        "total_net": sum(net.values()),
    }


def resolve_responses(
    game: MahjongGame, discarded_tile, acting_pid: int, agent: Optional[MASKLLMAgent],
    defenders: Optional[Dict[int, "ResponsiveDefender"]] = None,
    p0_response_decider: Optional[Callable[[List[str]], Tuple[str, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    defenders = defenders or {}
    responses = game.check_responses(discarded_tile, acting_pid)
    response_trace: List[Dict[str, Any]] = []
    responded = False
    agent_won_by_discard = False
    agent_dealt_in = False
    response_false_fold_opportunities = 0
    response_false_folds = 0
    last_p0_state: Dict[str, Any] = {}
    p0_hu_available = 0
    p0_hu_selected = 0
    p0_missed_hu = 0
    p0_missed_hu_value_floor = 0.0
    p0_blocked_call_states = 0
    p0_blocked_calls: Counter[str] = Counter()

    for rid, acts in responses.items():
        if responded:
            break

        player = game.players[rid]
        if rid == 0:
            blocked_calls = blocked_calls_while_missing(game, rid, response_actions=acts)
            if blocked_calls:
                p0_blocked_call_states += 1
                p0_blocked_calls.update(blocked_calls)
            valid = get_legal_actions(game, rid, response_actions=acts)
            if "h" in valid:
                p0_hu_available += 1
            if p0_response_decider is not None:
                response_action, response_state = p0_response_decider(valid)
            elif agent is not None:
                response_action = agent.decide(game, valid)
                response_state = dict(agent.last_decision)
            else:
                response_action = bot_decide_response(player, acts)
                response_state = {"reason": "bot response fallback"}
            if "h" in valid:
                if response_action == "h":
                    p0_hu_selected += 1
                else:
                    p0_missed_hu += 1
                    p0_missed_hu_value_floor += estimate_hu_value_floor(game, 0)
        elif rid in defenders:
            response_action = defenders[rid].response(player, acts, game)
            response_state = {}
        else:
            response_action = bot_decide_response(player, acts)
            response_state = {}

        # Legacy response-phase "decline to hu" counter (near-zero: declining a
        # win is not realistic). The meaningful FFR is the turn-phase one below.
        if "hu" in acts and rid != 0:
            response_false_fold_opportunities += 1
            if response_action != "h":
                response_false_folds += 1

        response_trace.append(
            {
                "responder": rid,
                "available": list(acts),
                "chosen": response_action,
                "decision_state": response_state,
            }
        )

        if response_action == "h" and "hu" in acts:
            game.hu(rid, discarded_tile, False, acting_pid)
            game.check_game_over()
            responded = True
            agent_won_by_discard = rid == 0
            agent_dealt_in = acting_pid == 0 and rid != 0
        elif response_action == "g" and "gang" in acts:
            game.gang(rid, discarded_tile, acting_pid)
            game.current_player_id = rid
            responded = True
        elif response_action == "p" and "peng" in acts:
            game.peng(rid, discarded_tile, acting_pid)
            game.current_player_id = rid
            responded = True

    return {
        "responded": responded,
        "agent_won_by_discard": agent_won_by_discard,
        "agent_dealt_in": agent_dealt_in,
        "response_false_fold_opportunities": response_false_fold_opportunities,
        "response_false_folds": response_false_folds,
        "responses": response_trace,
        "p0_hu_available": p0_hu_available,
        "p0_hu_selected": p0_hu_selected,
        "p0_missed_hu": p0_missed_hu,
        "p0_missed_hu_value_floor": p0_missed_hu_value_floor,
        "p0_blocked_call_states": p0_blocked_call_states,
        "p0_blocked_calls": dict(p0_blocked_calls),
    }


def make_step_trace(
    game: MahjongGame,
    method: str,
    seed: int,
    step: int,
    pid: int,
    action: str,
    legal_actions: List[str],
    decision_ms: float,
    decision_state: Dict[str, Any],
    backend: str,
) -> Dict[str, Any]:
    player = game.players[0]
    true_ready, true_waits, true_shanten = true_tenpai_label(game, 0)
    chosen_result_shanten = action_result_shanten(game, 0, action)
    best_result_shanten = best_discard_result_shanten(game, 0, legal_actions)
    shanten_regret = (
        chosen_result_shanten - best_result_shanten
        if chosen_result_shanten is not None and best_result_shanten is not None
        else None
    )
    gate_features = None
    if decision_state.get("gate") is not None:
        try:
            gate_features = extract_state_features(
                game,
                0,
                decision_state.get("z_state", {}),
                decision_state.get("beliefs", {}),
                decision_state.get("gate", {}),
            )
        except Exception:
            gate_features = None
    return {
        "game_id": game.game_id,
        "seed": seed,
        "method": method,
        "backend": backend,
        "step": step,
        "actor": pid,
        "action": action,
        "legal_actions": legal_actions,
        "auto_hu": decision_state.get("auto_hu"),
        "hu_available": "h" in legal_actions,
        "hu_selected": action == "h",
        "missed_hu": ("h" in legal_actions and action != "h"),
        "hu_selected_by": decision_state.get("hu_selected_by"),
        "mode": decision_state.get("mode", "none"),
        "z_state": decision_state.get("z_state", {}),
        "belief_json": decision_state.get("beliefs", {}),
        "risk_budget": decision_state.get("gate", {}).get("risk_budget"),
        "gate": decision_state.get("gate", {}),
        "gate_features": gate_features,
        "public_gate": decision_state.get("public_gate", {}),
        "score_delta": player.balance - 10000,
        "true_tenpai": true_ready,
        "true_waits": true_waits,
        "true_shanten": true_shanten,
        "dealin": False,
        "induced_dealin": False,
        "false_fold": False,
        "latency_ms": decision_ms,
        "reason": decision_state.get("reason", ""),
        "blocked_calls_while_missing": decision_state.get("blocked_calls_while_missing", []),
        "own_shanten": decision_state.get("own_shanten"),
        "own_danger_threshold": decision_state.get("own_danger_threshold"),
        "dir_ready_threshold": decision_state.get("dir_ready_threshold"),
        "deceive_ready": decision_state.get("deceive_ready"),
        "ffr_ready": decision_state.get("ffr_ready"),
        "deceive_attempted": decision_state.get("deceive_attempted"),
        "deceive_action": decision_state.get("deceive_action"),
        "can_deceive": decision_state.get("can_deceive"),
        "deceive_block_reason": decision_state.get("deceive_block_reason"),
        "deceive_signal": decision_state.get("deceive_signal", {}),
        "counterfactual_exploit_action": decision_state.get("counterfactual_exploit_action"),
        "disguise_equals_exploit": decision_state.get("disguise_equals_exploit"),
        "rule_action": decision_state.get("rule_action"),
        "baseline_action": decision_state.get("baseline_action"),
        "safe_override_applied": decision_state.get("safe_override_applied", False),
        "candidate_reranker": decision_state.get("candidate_reranker", {}),
        "gate_policy": decision_state.get("gate_policy", "rule"),
        "learned_gate": decision_state.get("learned_gate", {}),
        "neural_gate_deceive_margin": (
            decision_state.get("learned_gate", {}).get("deceive_margin")
            if isinstance(decision_state.get("learned_gate"), dict)
            else None
        ),
        "neural_gate_deceive_margin_blocked": (
            decision_state.get("learned_gate", {}).get("deceive_margin_blocked")
            if isinstance(decision_state.get("learned_gate"), dict)
            else None
        ),
        "neural_gate_safe_margin_blocked": (
            decision_state.get("learned_gate", {}).get("safe_margin_blocked")
            if isinstance(decision_state.get("learned_gate"), dict)
            else None
        ),
        "decision_llm_raw": decision_state.get("decision_llm_raw"),
        "decision_llm_parsed": decision_state.get("decision_llm_parsed"),
        "prompt_schema": decision_state.get("prompt_schema"),
        "chosen_result_shanten": chosen_result_shanten,
        "best_result_shanten": best_result_shanten,
        "shanten_regret": shanten_regret,
        "belief_shaping": decision_state.get("belief_shaping", False),
        "avg_belief_conf": decision_state.get("avg_belief_conf"),
        "b_term": decision_state.get("b_term"),
        "d_term": decision_state.get("d_term"),
        "f_term": decision_state.get("f_term"),
    }


def collect_belief_samples(game: MahjongGame, method: str, seed: int, step: int, estimator: LLMBeliefEstimator, backend: str) -> List[Dict[str, Any]]:
    true_ready, true_waits, true_shanten = true_tenpai_label(game, 0)
    rows = []
    for target_pid in (1, 2, 3):
        belief = estimator.infer(game, 0, target_pid)
        rows.append(
            {
                "game_id": game.game_id,
                "seed": seed,
                "method": method,
                "backend": backend,
                "step": step,
                "target_opponent": f"P{target_pid}",
                "think_i_am_tenpai": belief.get("think_i_am_tenpai"),
                "tenpai_confidence": float(belief.get("tenpai_confidence", 0.5)),
                "suspected_waits": belief.get("suspected_waits", []),
                "danger_tiles_for_me": belief.get("danger_tiles_for_me", []),
                "true_tenpai": true_ready,
                "true_waits": true_waits,
                "true_shanten": true_shanten,
                "label_scope": "coarse true P0 tenpai",
            }
        )
    return rows


def summarize_false_fold_attribution(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not events:
        return {
            "ffr_opportunity_events": 0,
            "note": "No first eligible opponent-game FFR events were recorded.",
        }

    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    by_opponent: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        by_mode.setdefault(str(event.get("last_p0_mode") or "none"), []).append(event)
        by_opponent.setdefault(str(event.get("opponent_id") or "unknown"), []).append(event)

    def avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary: Dict[str, Any] = {
        "ffr_opportunity_events": len(events),
        "false_fold_events": sum(1 for event in events if event.get("false_fold")),
        "ffr": sum(1 for event in events if event.get("false_fold")) / len(events),
        "unit": "first eligible decision per opponent-game",
        "opportunities_in_deceive_window": sum(1 for event in events if event.get("in_deceive_window")),
        "false_folds_in_deceive_window": sum(
            1 for event in events if event.get("in_deceive_window") and event.get("false_fold")
        ),
        "by_last_p0_mode": {},
        "by_opponent": {},
    }
    for mode, rows in sorted(by_mode.items()):
        deltas = []
        threat_at_fold = []
        for row in rows:
            threat_at_fold.append(float(row.get("threat_at_fold", 0.0)))
            delta_map = row.get("last_p0_threat_delta") or {}
            if isinstance(delta_map, dict):
                key = f"P{row.get('defender_pid')}"
                if key in delta_map:
                    try:
                        deltas.append(float(delta_map[key]))
                    except (TypeError, ValueError):
                        pass
        summary["by_last_p0_mode"][mode] = {
            "opportunities": len(rows),
            "false_folds": sum(1 for row in rows if row.get("false_fold")),
            "ffr": sum(1 for row in rows if row.get("false_fold")) / len(rows),
            "avg_threat_at_fold": round(avg(threat_at_fold), 4),
            "avg_last_p0_threat_delta_for_defender": round(avg(deltas), 4),
            "positive_delta_rate": (
                sum(1 for value in deltas if value > 0.0) / len(deltas)
                if deltas else None
            ),
        }
    for opponent_id, rows in sorted(by_opponent.items()):
        false_folds = sum(1 for row in rows if row.get("false_fold"))
        summary["by_opponent"][opponent_id] = {
            "opportunities": len(rows),
            "false_folds": false_folds,
            "ffr": false_folds / len(rows),
        }
    return summary


def play_one_game(
    method: str,
    seed: int,
    opponent_style: str,
    max_steps: int,
    sample_every: int,
    backend: str,
    llm: Optional[LLMCallable],
    belief_llm: Optional[LLMCallable],
    reranker_llm: Optional[LLMCallable] = None,
    gate_llm: Optional[LLMCallable] = None,
    neural_gate: Optional[Any] = None,
    action_value_policy: Optional[Any] = None,
    defender_cfg: Optional[Dict[str, Any]] = None,
    mask_cfg: Optional[Dict[str, Any]] = None,
    auto_hu: str = "common",
    debug_game: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    game, opponent_funcs, defenders = init_game(seed, opponent_style, f"Gate1_{method}_{seed}", defender_cfg)
    start_balances = [p.balance for p in game.players]
    mask_cfg = mask_cfg or {}
    snapshot_samples = int(mask_cfg.get("snapshot_samples", 120))
    snapshot_crn_seeds = max(1, int(mask_cfg.get("snapshot_crn_seeds", 1)))
    mask_agent = (
        MASKLLMAgent(
            player_id=0,
            decision_llm=llm,
            belief_llm=belief_llm,
            reranker_llm=reranker_llm,
            gate_llm=gate_llm,
            neural_gate=neural_gate,
            action_value_policy=action_value_policy,
            gate_policy=mask_cfg.get("gate_policy", "rule"),
            mc_seed=seed * 13 + 1,
            mc_oracle_samples=mask_cfg.get("oracle_samples", 30),
            mc_beta=mask_cfg.get("beta", 2.0),
            mc_danger_threshold=mask_cfg.get("danger_threshold", 1),
            belief_public_model=mask_cfg.get("belief_public_model"),
            belief_calibration_steps=mask_cfg.get("belief_calibration_steps"),
            dir_ready_threshold=mask_cfg.get("dir_ready_threshold", 0),
            deceive_threat_ceiling=mask_cfg.get("deceive_threat_ceiling", 0.5),
            forced_deceive=mask_cfg.get("forced_deceive", "off"),
            risk_gate_ablation=mask_cfg.get("risk_gate_ablation", "off"),
            deceive_style=mask_cfg.get("deceive_style", "safe"),
            threat_allow_break_ready=mask_cfg.get("threat_allow_break_ready", False),
            threat_max_result_shanten=mask_cfg.get("threat_max_result_shanten", 0),
            threat_max_shanten_regret=mask_cfg.get("threat_max_shanten_regret", 0),
            threat_min_ukeire_ratio=mask_cfg.get("threat_min_ukeire_ratio", 1.0),
            threat_gate_threshold=mask_cfg.get("threat_gate_threshold", 0.4),
            threat_gate_margin=mask_cfg.get("threat_gate_margin", 0.12),
            threat_min_delta=mask_cfg.get("threat_min_delta", 0.03),
            threat_gate_mode=mask_cfg.get("threat_gate_mode", "cross"),
            threat_response_model=mask_cfg.get("threat_response_model", "tell"),
            threat_response_tell_weight=mask_cfg.get("threat_response_tell_weight", 1.0),
            threat_tell_window=mask_cfg.get("threat_tell_window", 6),
            threat_max_start_shanten=mask_cfg.get("threat_max_start_shanten", 3),
            threat_require_non_exploit=mask_cfg.get("threat_require_non_exploit", True),
            threat_require_real_target=mask_cfg.get("threat_require_real_target", False),
            threat_target_max_shanten=mask_cfg.get("threat_target_max_shanten", 0),
            threat_target_signal=mask_cfg.get("threat_target_signal", "oracle"),
            threat_target_prob_threshold=mask_cfg.get("threat_target_prob_threshold", 0.5),
            log_counterfactual=mask_cfg.get("log_counterfactual", False),
            use_candidate_reranker=mask_cfg.get("candidate_reranker", False),
            use_candidate_scoring=mask_cfg.get("candidate_scoring", False),
            reranker_max_candidates=mask_cfg.get("reranker_max_candidates", 6),
            auto_hu=auto_hu,
            neural_gate_deceive_margin=mask_cfg.get("neural_gate_deceive_margin", 0.0),
            neural_gate_safe_margin=mask_cfg.get("neural_gate_safe_margin", 0.0),
            neural_gate_safe_risk_threshold=mask_cfg.get("neural_gate_safe_risk_threshold", 0.0),
        )
        if method in {"llm_mask", "llm_reactive_z_bphi"} else None
    )
    z_tracker = PublicOpponentTracker([1, 2, 3])
    risk_gate = RiskGate()
    belief_estimator = LLMBeliefEstimator(llm=belief_llm)

    skip_draw = True
    steps = 0
    decision_times: List[float] = []
    mode_counts: Counter[str] = Counter()
    step_rows: List[Dict[str, Any]] = []
    belief_rows: List[Dict[str, Any]] = []
    deceive_active_until = -1
    deceive_windows = 0
    induced_dealin = 0
    response_false_fold_opportunities = 0
    response_false_folds = 0
    last_p0_state: Dict[str, Any] = {}
    shanten_regrets: List[int] = []
    safe_overrides = 0
    reranker_enabled_states = 0
    reranker_used_states = 0
    reranker_changed_actions = 0
    reranker_parsed_outputs = 0
    learned_gate_used_states = 0
    learned_gate_parsed_outputs = 0
    learned_gate_mode_counts: Counter[str] = Counter()
    blocked_call_states = 0
    blocked_call_counts: Counter[str] = Counter()
    hu_available_turns = 0
    hu_selected_turns = 0
    missed_hu_turns = 0
    hu_available_responses = 0
    hu_selected_responses = 0
    missed_hu_responses = 0
    missed_hu_value_floor = 0.0

    def p0_response_decider(valid_actions: List[str]) -> Tuple[str, Dict[str, Any]]:
        if method == "llm_base":
            return choose_llm_base_action(game, 0, valid_actions, llm, auto_hu=auto_hu)
        if method in {"llm_reactive_z", "llm_reactive_z_bphi"}:
            return choose_reactive_z_action(
                game,
                0,
                z_tracker,
                risk_gate,
                llm,
                valid_actions,
                auto_hu=auto_hu,
                beliefs=(mask_agent._mc_beliefs(game) if method == "llm_reactive_z_bphi" else None),
            )
        if method == "llm_mask":
            assert mask_agent is not None
            action = mask_agent.decide(game, valid_actions)
            return action, dict(mask_agent.last_decision)
        action_map = {"h": "hu", "g": "gang", "p": "peng", "n": "pass"}
        response_acts = [action_map[action] for action in valid_actions if action in action_map]
        return bot_decide_response(game.players[0], response_acts), {"reason": "unknown method response fallback"}

    while not game.is_game_over and steps < max_steps:
        steps += 1
        if game.deck.remaining_count() == 0 or sum(1 for p in game.players if p.is_hu) >= 3:
            game.check_game_over()
            break

        pid = game.current_player_id
        player = game.players[pid]
        if player.is_hu:
            game.next_player()
            skip_draw = False
            continue

        drawn_tile = None
        if not skip_draw:
            drawn_tile = game.draw_tile(pid)
            if not drawn_tile:
                game.check_game_over()
                break
        else:
            skip_draw = False

        decision_state: Dict[str, Any] = {}
        decision_ms = 0.0
        legal_actions: List[str] = []

        if pid == 0:
            blocked_calls = blocked_calls_while_missing(game, 0)
            if blocked_calls:
                blocked_call_states += 1
                blocked_call_counts.update(blocked_calls)
            legal_actions = get_legal_actions(game, 0)
            hu_available_now = "h" in legal_actions
            if hu_available_now:
                hu_available_turns += 1
            if sample_every > 0 and (steps == 1 or steps % sample_every == 0):
                belief_rows.extend(collect_belief_samples(game, method, seed, steps, belief_estimator, backend))

            # Common-random-numbers snapshot for causal threat-delta (paired
            # before/after, fixed seeds, high samples) -- measurement only.
            snap_seeds = [seed * 100003 + steps * 31 + k for k in range(snapshot_crn_seeds)]
            threat_before = {
                f"P{rid}": defenders[rid].threat_crn(game, snap_seeds, snapshot_samples) for rid in defenders
            }
            t0 = time.perf_counter()
            if "h" in legal_actions and auto_hu == "common":
                action = "h"
                decision_state = {
                    "mode": "exploit",
                    "action": "h",
                    "reason": "common take-win rule",
                    "common_take_win": True,
                    "auto_hu": auto_hu,
                }
            elif method == "llm_base":
                action, decision_state = choose_llm_base_action(game, 0, legal_actions, llm, auto_hu=auto_hu)
            elif method in {"llm_reactive_z", "llm_reactive_z_bphi"}:
                action, decision_state = choose_reactive_z_action(
                    game,
                    0,
                    z_tracker,
                    risk_gate,
                    llm,
                    legal_actions,
                    auto_hu=auto_hu,
                    beliefs=(mask_agent._mc_beliefs(game) if method == "llm_reactive_z_bphi" else None),
                )
            elif method == "llm_mask":
                assert mask_agent is not None
                action = mask_agent.decide(game, legal_actions)
                decision_state = dict(mask_agent.last_decision)
            else:
                raise ValueError(f"Unknown method: {method}")
            if hu_available_now:
                if action == "h":
                    hu_selected_turns += 1
                else:
                    missed_hu_turns += 1
                    missed_hu_value_floor += estimate_hu_value_floor(game, 0)
                decision_state["hu_available"] = True
                decision_state["missed_hu"] = action != "h"
            decision_state["auto_hu"] = auto_hu
            decision_state["blocked_calls_while_missing"] = blocked_calls
            decision_ms = (time.perf_counter() - t0) * 1000.0
            decision_times.append(decision_ms)

            mode = decision_state.get("mode", "exploit")
            mode_counts[mode] += 1
            if decision_state.get("safe_override_applied"):
                safe_overrides += 1
            reranker_state = decision_state.get("candidate_reranker") or {}
            if reranker_state.get("enabled"):
                reranker_enabled_states += 1
            if reranker_state.get("used"):
                reranker_used_states += 1
                if reranker_state.get("changed_action"):
                    reranker_changed_actions += 1
                if reranker_state.get("parsed"):
                    reranker_parsed_outputs += 1
            learned_gate_state = decision_state.get("learned_gate") or {}
            if learned_gate_state.get("used"):
                learned_gate_used_states += 1
                learned_gate_mode_counts[str(learned_gate_state.get("selected_mode", "unknown"))] += 1
                if learned_gate_state.get("parsed"):
                    learned_gate_parsed_outputs += 1
            if mode == "deceive":
                deceive_active_until = steps + 4
                deceive_windows += 1

            step_trace = make_step_trace(
                game, method, seed, steps, pid, action, legal_actions, decision_ms, decision_state, backend
            )
            step_rows.append(step_trace)
            if step_trace["shanten_regret"] is not None:
                shanten_regrets.append(int(step_trace["shanten_regret"]))
        else:
            if pid in defenders:
                action = defenders[pid].turn(
                    player,
                    game,
                    step=steps,
                    last_p0_state=last_p0_state,
                    in_deceive_window=deceive_active_until >= steps,
                )
            else:
                action = opponent_funcs[pid][0](player, game)

        discarded_tile = execute_action(game, pid, action, drawn_tile)
        if debug_game:
            print_table_info(game, pid, drawn_tile)
            print(f"  P{pid} 动作: {action}")
            if pid == 0 and decision_state:
                print(f"  P0 模式: {decision_state.get('mode', '?')} | 原因: {decision_state.get('reason', '')}")
        if game.is_game_over:
            break

        if pid == 0:
            # SAME snap_seeds as the before-snapshot (paired CRN): only the
            # post-action pool change drives the delta, not sampling noise.
            threat_after = {
                f"P{rid}": defenders[rid].threat_crn(game, snap_seeds, snapshot_samples) for rid in defenders
            }
            threat_delta = {
                key: round(threat_after.get(key, 0.0) - threat_before.get(key, 0.0), 4)
                for key in set(threat_before) | set(threat_after)
            }
            last_p0_state = {
                "step": steps,
                "mode": decision_state.get("mode", "none"),
                "action": action,
                "reason": decision_state.get("reason", ""),
                "threat_before": {k: round(v, 4) for k, v in threat_before.items()},
                "threat_after": {k: round(v, 4) for k, v in threat_after.items()},
                "threat_delta": threat_delta,
            }
            if step_rows and step_rows[-1]["actor"] == 0:
                step_rows[-1]["threat_before"] = last_p0_state["threat_before"]
                step_rows[-1]["threat_after"] = last_p0_state["threat_after"]
                step_rows[-1]["threat_delta"] = last_p0_state["threat_delta"]

        if discarded_tile is None:
            if action == "g":
                skip_draw = True
                continue
            game.next_player()
            skip_draw = False
            continue

        response_info = resolve_responses(
            game,
            discarded_tile,
            pid,
            mask_agent,
            defenders,
            p0_response_decider=p0_response_decider,
        )
        response_false_fold_opportunities += response_info["response_false_fold_opportunities"]
        response_false_folds += response_info["response_false_folds"]
        blocked_call_states += int(response_info.get("p0_blocked_call_states", 0))
        blocked_call_counts.update(response_info.get("p0_blocked_calls", {}))
        hu_available_responses += int(response_info.get("p0_hu_available", 0))
        hu_selected_responses += int(response_info.get("p0_hu_selected", 0))
        missed_hu_responses += int(response_info.get("p0_missed_hu", 0))
        missed_hu_value_floor += float(response_info.get("p0_missed_hu_value_floor", 0.0))

        if response_info["agent_dealt_in"]:
            for row in reversed(step_rows):
                if row["actor"] == 0:
                    row["dealin"] = True
                    break

        if response_info["agent_won_by_discard"] and deceive_active_until >= steps:
            induced_dealin += 1
            for row in reversed(step_rows):
                if row.get("mode") == "deceive":
                    row["induced_dealin"] = True
                    break

        if game.is_game_over:
            break
        if response_info["responded"]:
            skip_draw = True
        else:
            game.next_player()
            skip_draw = False

    if not game.is_game_over:
        game.check_game_over()

    end_balances = [p.balance for p in game.players]
    net = [end_balances[i] - start_balances[i] for i in range(4)]
    settlement = settlement_breakdown_for_player(game, 0)
    agent_player = game.players[0]
    agent_dealin = any(p.is_hu and not p.hu_is_self_drawn and p.hu_discard_player_id == 0 for p in game.players)
    agent_final_ready, _ = agent_player.is_ready_with_missing_suit()
    agent_final_shanten = (
        None
        if agent_player.is_hu
        else int(ShantenCalculator.calculate_shanten(agent_player.hand_tiles, agent_player.missing_suit))
    )
    agent_hua_zhu = bool(
        not agent_player.is_hu
        and agent_player.missing_suit
        and any(tile.suit == agent_player.missing_suit for tile in agent_player.hand_tiles)
    )
    if game.deck.remaining_count() == 0:
        end_reason = "wall_exhausted"
    elif sum(1 for player in game.players if player.is_hu) >= 3:
        end_reason = "three_winners"
    else:
        end_reason = "max_steps_or_unsettled"

    # Primary FFR uses at most one strict opportunity per opponent-game. The
    # broader repeated turn-level switch rate is retained only as a diagnostic.
    turn_ff_opportunities = sum(d.ff_opportunities for d in defenders.values())
    turn_ff_false = sum(d.ff_false for d in defenders.values())
    strict_event_opportunities = sum(
        int(getattr(d, "strict_ffr_event_opportunities", d.ff_opportunities))
        for d in defenders.values()
    )
    strict_event_false = sum(
        int(getattr(d, "strict_ffr_event_false", d.ff_false))
        for d in defenders.values()
    )
    defensive_switch_opportunities = sum(
        int(getattr(d, "defensive_switch_opportunities", 0))
        for d in defenders.values()
    )
    defensive_switches = sum(
        int(getattr(d, "defensive_switches", 0))
        for d in defenders.values()
    )
    declined_self_gang_opportunities = sum(
        int(getattr(d, "declined_self_gang_opportunities", 0))
        for d in defenders.values()
    )
    declined_self_gangs = sum(
        int(getattr(d, "declined_self_gangs", 0))
        for d in defenders.values()
    )
    response_declines = sum(d.response_declines for d in defenders.values())
    ffr_events = [event for defender in defenders.values() for event in defender.ffr_events]
    false_fold_events = [event for defender in defenders.values() for event in defender.false_fold_events]
    strict_ffr_event_events = [
        event
        for defender in defenders.values()
        for event in getattr(defender, "strict_ffr_event_events", defender.ffr_events)
    ]
    defensive_switch_events = [
        event
        for defender in defenders.values()
        for event in getattr(defender, "defensive_switch_events", [])
    ]
    for event in ffr_events + strict_ffr_event_events + defensive_switch_events:
        event["game_id"] = game.game_id
        event["seed"] = seed
        event["method"] = method
        event["backend"] = backend
    false_folds_after_deceive = sum(1 for event in false_fold_events if event.get("last_p0_mode") == "deceive")
    false_folds_after_exploit = sum(1 for event in false_fold_events if event.get("last_p0_mode") == "exploit")
    false_folds_in_deceive_window = sum(1 for event in false_fold_events if event.get("in_deceive_window"))

    # Tag steps where an opponent false-folded its hand this game (for trace).
    if turn_ff_false > 0:
        for row in step_rows:
            if row["actor"] == 0:
                row["false_fold_round"] = True

    game_row = {
        "seed": seed,
        "method": method,
        "backend": backend,
        "opponent_style": opponent_style,
        "opponent_curriculum": {
            f"P{pid}": dict(getattr(defender, "curriculum_spec", {}))
            for pid, defender in defenders.items()
            if getattr(defender, "curriculum_spec", None)
        },
        "auto_hu": auto_hu,
        "steps": steps,
        "agent_hu": bool(agent_player.is_hu),
        "agent_hu_by_discard": bool(agent_player.is_hu and not agent_player.hu_is_self_drawn),
        "agent_hu_fan": int(agent_player.hu_fan if agent_player.is_hu else 0),
        "agent_net": net[0],
        "settlement_breakdown": settlement,
        "settlement_reconciled": abs(float(settlement["total_net"]) - float(net[0])) < 1e-9,
        "agent_dealin": bool(agent_dealin),
        "agent_final_ready": bool(agent_final_ready),
        "agent_final_shanten": agent_final_shanten,
        "agent_hua_zhu": agent_hua_zhu,
        "end_reason": end_reason,
        "winners": list(game.winners),
        "decision_ms": decision_times,
        "mode_counts": dict(mode_counts),
        "deceive_windows": deceive_windows,
        "induced_dealin": induced_dealin,
        # Primary strict FFR: first eligible discard decision per opponent-game.
        "false_fold_opportunities": turn_ff_opportunities,
        "false_folds": turn_ff_false,
        "strict_ffr_event_opportunities": strict_event_opportunities,
        "strict_false_fold_events": strict_event_false,
        "conditional_defensive_switch_opportunities": defensive_switch_opportunities,
        "conditional_defensive_switches": defensive_switches,
        "declined_self_gang_opportunities": declined_self_gang_opportunities,
        "declined_self_gangs": declined_self_gangs,
        "false_folds_after_deceive": false_folds_after_deceive,
        "false_folds_after_exploit": false_folds_after_exploit,
        "false_folds_in_deceive_window": false_folds_in_deceive_window,
        "ffr_events": ffr_events,
        "false_fold_events": false_fold_events,
        "strict_ffr_event_events": strict_ffr_event_events,
        "conditional_defensive_switch_events": defensive_switch_events,
        "opponent_response_declines": response_declines,
        # legacy response-phase "decline to hu" (near-zero), kept for reference
        "response_false_fold_opportunities": response_false_fold_opportunities,
        "response_false_folds": response_false_folds,
        "history_len": len(game.history),
        "reranker_enabled_states": reranker_enabled_states,
        "reranker_used_states": reranker_used_states,
        "reranker_changed_actions": reranker_changed_actions,
        "reranker_parsed_outputs": reranker_parsed_outputs,
        "learned_gate_used_states": learned_gate_used_states,
        "learned_gate_parsed_outputs": learned_gate_parsed_outputs,
        "learned_gate_mode_counts": dict(learned_gate_mode_counts),
        "avg_shanten_regret": (
            sum(shanten_regrets) / len(shanten_regrets) if shanten_regrets else 0.0
        ),
        "positive_shanten_regret_rate": (
            sum(1 for value in shanten_regrets if value > 0) / len(shanten_regrets)
            if shanten_regrets else 0.0
        ),
        "safe_overrides": safe_overrides,
        "blocked_call_while_missing_states": blocked_call_states,
        "blocked_call_while_missing_counts": dict(blocked_call_counts),
        "hu_available_turns": hu_available_turns,
        "hu_selected_turns": hu_selected_turns,
        "missed_hu_turns": missed_hu_turns,
        "hu_available_responses": hu_available_responses,
        "hu_selected_responses": hu_selected_responses,
        "missed_hu_responses": missed_hu_responses,
        "hu_available_count": hu_available_turns + hu_available_responses,
        "hu_selected_count": hu_selected_turns + hu_selected_responses,
        "missed_hu_count": missed_hu_turns + missed_hu_responses,
        "missed_hu_value_floor": missed_hu_value_floor,
    }
    return game_row, step_rows, belief_rows


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo_dir = Path(__file__).resolve().parent
    auto_hu = getattr(args, "auto_hu", "common")
    llm = build_llm_callable(
        backend=args.backend,
        repo_dir=repo_dir,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=getattr(args, "temperature", 0.0),
    )
    if not args.belief_adapter_path or args.belief_adapter_path == args.adapter_path:
        belief_llm = llm
    else:
        belief_llm = build_llm_callable(
            backend=args.backend,
            repo_dir=repo_dir,
            model_path=args.model_path,
            adapter_path=args.belief_adapter_path,
            max_new_tokens=args.max_new_tokens,
            temperature=getattr(args, "temperature", 0.0),
        )

    reranker_model_path = getattr(args, "reranker_model_path", None)
    reranker_adapter_path = getattr(args, "reranker_adapter_path", None)
    if getattr(args, "mask_candidate_reranker", False) and (
        reranker_model_path or reranker_adapter_path
    ):
        reranker_llm = build_llm_callable(
            backend=args.backend,
            repo_dir=repo_dir,
            model_path=reranker_model_path or args.model_path,
            adapter_path=reranker_adapter_path,
            max_new_tokens=getattr(args, "reranker_max_new_tokens", args.max_new_tokens),
            temperature=0.0,
        )
    else:
        reranker_llm = llm

    gate_policy = getattr(args, "mask_gate_policy", "rule")
    gate_model_path = getattr(args, "gate_model_path", None)
    gate_adapter_path = getattr(args, "gate_adapter_path", None)
    neural_gate_model_path = getattr(args, "neural_gate_model_path", None)
    gate_value_model_path = getattr(args, "gate_value_model_path", None)
    action_value_model_path = getattr(args, "action_value_model_path", None)
    value_residual_model_path = getattr(args, "value_residual_model_path", None)
    if gate_policy == "learned":
        if not (gate_model_path or gate_adapter_path):
            raise ValueError("learned gate requires --gate-model-path or --gate-adapter-path")
        gate_llm = build_llm_callable(
            backend=args.backend,
            repo_dir=repo_dir,
            model_path=gate_model_path or args.model_path,
            adapter_path=gate_adapter_path,
            max_new_tokens=getattr(args, "gate_max_new_tokens", 8),
            temperature=0.0,
            system_prompt=(
                "你是四川麻将 MASK 风险门控器。只能从给出的 "
                "exploit、safe、deceive 模式中选择一个。"
            ),
        )
    else:
        gate_llm = None
    if gate_policy == "neural":
        if not neural_gate_model_path:
            raise ValueError("neural gate requires --neural-gate-model-path")
        from neural_gate_policy import NeuralGatePolicy

        neural_gate = NeuralGatePolicy.load(
            neural_gate_model_path,
            device=getattr(args, "neural_gate_device", "cpu"),
        )
    elif gate_policy == "neural_value_gate":
        if not gate_value_model_path:
            raise ValueError("neural_value_gate requires --gate-value-model-path")
        from gate_value_policy import GateValuePolicy

        neural_gate = GateValuePolicy.load(
            gate_value_model_path,
            device=getattr(args, "neural_gate_device", "cpu"),
        )
        if getattr(args, "gate_value_safe_margin", None) is not None:
            neural_gate.safe_margin = max(0.0, float(args.gate_value_safe_margin))
        if getattr(args, "gate_value_deceive_margin", None) is not None:
            neural_gate.deceive_margin = max(0.0, float(args.gate_value_deceive_margin))
    else:
        neural_gate = None
    if gate_policy == "neural_action_value":
        if not action_value_model_path:
            raise ValueError("neural_action_value requires --action-value-model-path")
        from action_value_policy import ActionValuePolicy

        action_value_policy = ActionValuePolicy.load(
            action_value_model_path,
            device=getattr(args, "action_value_device", "cpu"),
        )
    elif gate_policy == "neural_value_residual":
        if not action_value_model_path or not value_residual_model_path:
            raise ValueError(
                "neural_value_residual requires --action-value-model-path and "
                "--value-residual-model-path"
            )
        from action_value_policy import ValueResidualPolicy

        action_value_policy = ValueResidualPolicy.load(
            action_value_model_path,
            value_residual_model_path,
            device=getattr(args, "action_value_device", "cpu"),
        )
    else:
        action_value_policy = None

    defender_cfg = {
        "threat_threshold": args.threat_fold_threshold,
        "oracle_samples": args.oracle_samples,
        "beta": args.oracle_beta,
        "danger_threshold": args.danger_threshold,
        "ffr_hand_shanten": args.ffr_hand_shanten,
        "threat_model": args.defender_threat_model,
        "tell_weight": args.defender_tell_weight,
        "tell_window": args.defender_tell_window,
        "learned_model_path": args.defender_learned_model_path,
        "neural_model_path": args.neural_opponent_model_path,
        "neural_device": args.neural_opponent_device,
        "neural_risk_mode": args.neural_opponent_risk_mode,
        "curriculum_manifest": getattr(args, "curriculum_manifest", None),
        "curriculum_split": getattr(args, "curriculum_split", "holdout"),
        "curriculum_spec_id": getattr(args, "curriculum_spec_id", None),
    }
    belief_public_model = (
        PublicBeliefModel.load(args.belief_public_model)
        if args.belief_public_model else None
    )
    belief_public_model_sha256 = (
        sha256_file(args.belief_public_model)
        if args.belief_public_model else ""
    )
    mask_cfg = {
        "belief_public_model": belief_public_model,
        "oracle_samples": args.mask_oracle_samples,
        "beta": args.mask_oracle_beta,
        "danger_threshold": args.mask_danger_threshold,
        "belief_calibration_steps": load_belief_calibration_steps(
            getattr(args, "belief_calibration_summary", None),
            oracle_samples=args.mask_oracle_samples,
            oracle_beta=args.mask_oracle_beta,
            danger_threshold=args.mask_danger_threshold,
            b_phi_source=("public_model" if belief_public_model else "mc"),
            belief_public_model_sha256=belief_public_model_sha256,
        ),
        "dir_ready_threshold": args.mask_dir_ready_threshold,
        "deceive_threat_ceiling": args.mask_deceive_threat_ceiling,
        "forced_deceive": args.mask_forced_deceive,
        "risk_gate_ablation": args.mask_risk_gate_ablation,
        "deceive_style": args.mask_deceive_style,
        "threat_allow_break_ready": args.mask_threat_allow_break_ready,
        "threat_max_result_shanten": args.mask_threat_max_result_shanten,
        "threat_max_shanten_regret": getattr(args, "mask_threat_max_shanten_regret", 0),
        "threat_min_ukeire_ratio": getattr(args, "mask_threat_min_ukeire_ratio", 1.0),
        "threat_gate_threshold": args.mask_threat_gate_threshold,
        "threat_gate_margin": args.mask_threat_gate_margin,
        "threat_min_delta": args.mask_threat_min_delta,
        "threat_gate_mode": args.mask_threat_gate_mode,
        "threat_response_model": getattr(args, "mask_threat_response_model", "tell"),
        "threat_response_tell_weight": getattr(args, "mask_threat_response_tell_weight", 1.0),
        "threat_tell_window": args.mask_threat_tell_window,
        "threat_max_start_shanten": args.mask_threat_max_start_shanten,
        "threat_require_non_exploit": not args.mask_threat_allow_exploit_overlap,
        "threat_require_real_target": args.mask_threat_require_real_target,
        "threat_target_max_shanten": args.mask_threat_target_max_shanten,
        "threat_target_signal": args.mask_threat_target_signal,
        "threat_target_prob_threshold": args.mask_threat_target_prob_threshold,
        "log_counterfactual": args.mask_log_counterfactual,
        "candidate_reranker": getattr(args, "mask_candidate_reranker", False),
        "candidate_scoring": getattr(args, "mask_candidate_scoring", False),
        "reranker_max_candidates": getattr(args, "mask_reranker_max_candidates", 6),
        "gate_policy": gate_policy,
        "snapshot_samples": args.snapshot_oracle_samples,
        "snapshot_crn_seeds": args.snapshot_crn_seeds,
        "neural_gate_deceive_margin": getattr(args, "neural_gate_deceive_margin", 0.0),
        "neural_gate_safe_margin": getattr(args, "neural_gate_safe_margin", 0.0),
        "neural_gate_safe_risk_threshold": getattr(args, "neural_gate_safe_risk_threshold", 0.0),
    }
    mask_cfg_summary = serializable_mask_config(
        mask_cfg,
        belief_public_model_path=args.belief_public_model,
        belief_public_model_sha256=belief_public_model_sha256,
    )
    runtime_artifact_sha256 = {
        name: sha256_file(path)
        for name, path in {
            "belief_public_model": args.belief_public_model,
            "gate_value_model": gate_value_model_path,
            "neural_opponent_model": args.neural_opponent_model_path,
        }.items()
        if path and Path(path).is_file()
    }

    all_games: List[Dict[str, Any]] = []
    all_steps: List[Dict[str, Any]] = []
    all_beliefs: List[Dict[str, Any]] = []
    all_ffr_events: List[Dict[str, Any]] = []
    all_false_folds: List[Dict[str, Any]] = []
    all_defensive_switch_events: List[Dict[str, Any]] = []

    for method in args.methods:
        for i in range(args.games):
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == args.games:
                print(
                    f"[Gate1] output={args.output_dir} method={method} "
                    f"game={i + 1}/{args.games}",
                    flush=True,
                )
            seed = derive_game_seed(args.seed, i)
            game_row, step_rows, belief_rows = play_one_game(
                method=method,
                seed=seed,
                opponent_style=args.opponent_style,
                max_steps=args.max_steps,
                sample_every=args.sample_every,
                backend=args.backend,
                llm=llm,
                belief_llm=belief_llm,
                reranker_llm=reranker_llm,
                gate_llm=gate_llm,
                neural_gate=neural_gate,
                action_value_policy=action_value_policy,
                defender_cfg=defender_cfg,
                mask_cfg=mask_cfg,
                auto_hu=auto_hu,
                debug_game=(i < args.debug_game),
            )
            all_games.append(game_row)
            all_steps.extend(step_rows)
            all_beliefs.extend(belief_rows)
            all_ffr_events.extend(game_row.get("ffr_events", []))
            all_false_folds.extend(game_row.get("false_fold_events", []))
            all_defensive_switch_events.extend(
                game_row.get("conditional_defensive_switch_events", [])
            )

    cf_rows = [r for r in all_steps if r.get("disguise_equals_exploit") is not None]
    accepted_cf_rows = [r for r in cf_rows if r.get("mode") == "deceive"]
    n_cf = len(cf_rows)
    n_eq = sum(1 for r in cf_rows if r.get("disguise_equals_exploit"))
    n_accepted_cf = len(accepted_cf_rows)
    n_accepted_eq = sum(1 for r in accepted_cf_rows if r.get("disguise_equals_exploit"))
    deceive_counterfactual = {
        "candidate_decisions_logged": n_cf,
        "candidate_disguise_equals_exploit": n_eq,
        "candidate_overlap_rate": (n_eq / n_cf) if n_cf else None,
        "accepted_deceive_logged": n_accepted_cf,
        "accepted_disguise_equals_exploit": n_accepted_eq,
        "accepted_overlap_rate": (n_accepted_eq / n_accepted_cf) if n_accepted_cf else None,
        "note": ("candidate_* includes rejected threat-gate candidates; accepted_* is the "
                 "actual executed deceive action overlap. High accepted overlap means the "
                 "disguise is a no-op at the action level. Requires --mask-log-counterfactual."),
    }

    if args.opponent_style == "neural":
        ffr_definition = (
            "Neural behavioral strict FFR = strict false folds at the first eligible "
            "decision per opponent-game / first eligible opponent-game opportunities. "
            "Eligibility requires opponent shanten<=1, oracle P0 shanten>0, and "
            "discard actions for both push and choice. A false fold requires a "
            "non-push choice with higher post-discard shanten or fewer effective "
            "copies than push. It remains a behavioral proxy because the neural "
            "policy has no explicit perceived-threat threshold."
        )
        ffr_limitation = (
            "Against --opponent-style neural, strict FFR is oracle-grounded but "
            "behavioral; without an explicit threat threshold it does not identify "
            "the neural policy's internal reason for choosing the non-push action."
        )
    else:
        ffr_definition = (
            "Strict FFR = first strict false fold / first eligible opportunity per opponent-game. "
            "Eligibility requires defender shanten<=1, oracle P0 shanten>0, and discard actions "
            "for both push and defense. A false fold additionally requires high perceived threat, "
            "a defensive action different from push, and either "
            "higher post-discard shanten or fewer effective copies than push. Declined self-gang "
            "and the repeated conditional defensive-switch rate are reported separately."
        )
        ffr_limitation = (
            "FFR/DIR are only meaningful against belief-responsive opponents "
            "(--opponent-style responsive or curriculum); "
            "the default rule bots (greedy/aggressive/...) do not fold on perceived threat."
        )

    summary = {
        "created_for": "Gate1 minimal H1/H2 pipeline",
        "backend": args.backend,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "temperature": getattr(args, "temperature", 0.0),
        "auto_hu": auto_hu,
        "belief_adapter_path": args.belief_adapter_path,
        "belief_calibration_summary": getattr(args, "belief_calibration_summary", None),
        "reranker_model_path": reranker_model_path,
        "reranker_adapter_path": reranker_adapter_path,
        "gate_policy": gate_policy,
        "gate_model_path": gate_model_path,
        "gate_adapter_path": gate_adapter_path,
        "neural_gate_model_path": neural_gate_model_path,
        "gate_value_model_path": gate_value_model_path,
        "action_value_model_path": action_value_model_path,
        "value_residual_model_path": value_residual_model_path,
        "opponent_style": args.opponent_style,
        "methods": list(args.methods),
        "games_per_method": args.games,
        "outer_seed": args.seed,
        "game_seed_scheme": "outer_seed * 1000000 + game_index",
        "game_seed_first": derive_game_seed(args.seed, 0),
        "game_seed_last": derive_game_seed(args.seed, args.games - 1),
        "opponent_style": args.opponent_style,
        "defender_cfg": defender_cfg if args.opponent_style in {"responsive", "curriculum"} else None,
        "mask_cfg": mask_cfg_summary,
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "E1_belief": summarize_belief_samples(all_beliefs),
        "E2_ladder": summarize_by_method(all_games),
        "FFR_attribution": summarize_false_fold_attribution(all_ffr_events),
        "conditional_defensive_switch_definition": (
            "Repeated turn-level diagnostic: among opponent-specific pushable/non-dangerous "
            "states, the defensive action differed from the policy's push action."
        ),
        "deceive_counterfactual": deceive_counterfactual,
        "Gate1_pairwise": paired_method_comparison(all_games, "llm_reactive_z", "llm_mask"),
        "ffr_definition": ffr_definition,
        "limitations": [
            "If backend=heuristic_fallback, numbers are pipeline checks rather than trained LLM conclusions.",
            "E1 label is coarse true P0 tenpai; final paper needs opponent-view posterior oracle.",
            ffr_limitation,
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "gate1_summary.json", summary)
    write_jsonl(args.output_dir / "gate1_games.jsonl", all_games)
    write_jsonl(args.output_dir / "gate1_steps.jsonl", all_steps)
    write_jsonl(args.output_dir / "gate1_beliefs.jsonl", all_beliefs)
    write_jsonl(args.output_dir / "gate1_ffr_events.jsonl", all_ffr_events)
    write_jsonl(args.output_dir / "gate1_false_folds.jsonl", all_false_folds)
    write_jsonl(
        args.output_dir / "gate1_defensive_switches.jsonl",
        all_defensive_switch_events,
    )
    return summary


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--opponent-style", default="greedy", choices=list(OPPONENT_STYLES))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument(
        "--auto-hu",
        choices=["common", "off"],
        default="common",
        help=(
            "common: L0/L1/L2 all take legal hu automatically (upper-bound/control). "
            "off: no automatic hu; P0 must explicitly select h through the model/policy."
        ),
    )
    # ResponsiveDefender knobs (only used when --opponent-style responsive).
    parser.add_argument("--threat-fold-threshold", type=float, default=0.4,
                        help="Opponent folds/declines when its public threat estimate of P0 >= this.")
    parser.add_argument("--oracle-samples", type=int, default=30,
                        help="MC samples for the opponent's public threat estimate (cached per state).")
    parser.add_argument("--oracle-beta", type=float, default=2.0)
    parser.add_argument("--danger-threshold", type=int, default=1,
                        help="P0 'dangerous' = shanten<=this (also the threat-posterior target).")
    parser.add_argument("--ffr-hand-shanten", type=int, default=1,
                        help="Opponent has a 'real hand' worth pushing when its shanten<=this.")
    parser.add_argument("--neural-opponent-model-path", default="Neural_opponent_model/neural_opponent_policy.pth",
                        help="Checkpoint used when --opponent-style neural.")
    parser.add_argument("--neural-opponent-device", default="cpu",
                        help="Inference device for --opponent-style neural.")
    parser.add_argument(
        "--neural-opponent-risk-mode",
        choices=["predicted", "blind"],
        default="predicted",
        help=(
            "For risk-aware neural checkpoints, predicted conditions the action head on the "
            "model's public P0-risk estimate; blind replaces only that condition with the "
            "frozen training prevalence."
        ),
    )
    parser.add_argument("--curriculum-manifest", default=None,
                        help="Executable opponent curriculum JSON used by --opponent-style curriculum.")
    parser.add_argument("--curriculum-split", choices=["train", "holdout", "all"], default="holdout")
    parser.add_argument("--curriculum-spec-id", default=None,
                        help="Use one curriculum opponent for P1-P3; intended for screening only.")
    # MASK-side MC-B_phi and deceive controls.
    parser.add_argument("--mask-oracle-samples", type=int, default=30,
                        help="MC samples for MASK's own B_phi estimate.")
    parser.add_argument("--mask-oracle-beta", type=float, default=2.0)
    parser.add_argument("--mask-danger-threshold", type=int, default=1,
                        help="MASK belief target: P0 dangerous when shanten<=this.")
    parser.add_argument("--mask-dir-ready-threshold", type=int, default=0,
                        help="Strict readiness required for disguise/DIR actions.")
    parser.add_argument("--mask-deceive-threat-ceiling", type=float, default=0.5,
                        help="Natural deceive triggers only when perceived threat <= this.")
    parser.add_argument("--mask-forced-deceive", choices=["off", "eligible", "always"], default="off",
                        help=("Ablation switch. eligible uses style-specific eligibility: "
                              "safe/DIR when ready; threat/FFR when P0 is not dangerous."))
    parser.add_argument(
        "--mask-risk-gate-ablation",
        choices=["off", "constant_deceive"],
        default="off",
        help=("E3 single-variable ablation. constant_deceive keeps the normal "
              "candidate generator/guards, always selects deceive when available, "
              "and otherwise falls back to exploit without the public safe gate."),
    )
    parser.add_argument("--mask-deceive-style", choices=["safe", "threat"], default="safe",
                        help="safe=old ready-preserving safe discard; threat=visible middle/non-safe discard for FFR ablation.")
    parser.add_argument("--mask-threat-allow-break-ready", action="store_true",
                        help="Allow deceive-threat to break ready. Default is safer: only ready-preserving threat signals.")
    parser.add_argument("--mask-threat-max-result-shanten", type=int, default=0,
                        help=("For deceive-threat from a dangerous hand, allow resulting hand up to this shanten. "
                              "When threat/FFR starts from a non-dangerous hand, the action instead preserves "
                              "non-dangerous status so FFR opportunities remain countable."))
    parser.add_argument("--mask-threat-max-shanten-regret", type=int, default=0,
                        help=("Maximum shanten loss of a threat/deceive discard relative to the "
                              "best legal discard. Zero permits only no-immediate-tile-efficiency-cost deception."))
    parser.add_argument("--mask-threat-min-ukeire-ratio", type=float, default=1.0,
                        help="Minimum effective-tile-copy ratio versus the best same-shanten discard.")
    parser.add_argument("--mask-threat-gate-threshold", type=float, default=0.4,
                        help="Deceive-threat only fires when its projected discard-tell threat crosses this fold threshold.")
    parser.add_argument("--mask-threat-gate-margin", type=float, default=0.12,
                        help="Deceive-threat only considers states with tell threat in [threshold-margin, threshold).")
    parser.add_argument("--mask-threat-min-delta", type=float, default=0.03,
                        help="Minimum projected discard-tell threat increase required for deceive-threat.")
    parser.add_argument("--mask-threat-gate-mode", choices=["cross", "delta_only"], default="cross",
                        help=("cross (default): only fire when tell_before is already near the fold "
                              "threshold AND this single discard crosses it (narrow, can collapse trigger "
                              "rate). delta_only: fire on any meaningfully-pushing discard regardless of "
                              "distance to the threshold, letting pressure build across turns."))
    parser.add_argument("--mask-threat-response-model", choices=["tell", "blend"], default="tell",
                        help="Public response model used to decide whether a threat discard changes opponent behavior.")
    parser.add_argument("--mask-threat-response-tell-weight", type=float, default=1.0,
                        help="Tell weight for --mask-threat-response-model blend; must match the evaluated defender.")
    parser.add_argument("--mask-threat-tell-window", type=int, default=6,
                        help="Recent P0 discard window used by MASK's public tell gate.")
    parser.add_argument("--mask-threat-require-real-target", action="store_true",
                        help=("ORACLE ABLATION (upper bound, not a deployable signal): only allow "
                              "deceive-threat to fire when at least one opponent's true hand "
                              "(ground-truth peek) is within --mask-threat-target-max-shanten. Tests "
                              "whether deceiving only genuinely-close opponents converts FFR into net "
                              "benefit, vs. the default which fires on MASK's own tell projection alone."))
    parser.add_argument("--mask-threat-target-max-shanten", type=int, default=0,
                        help="Shanten threshold for --mask-threat-require-real-target (0 = opponent must be tenpai).")
    parser.add_argument("--mask-threat-target-signal", choices=["oracle", "mc"], default="oracle",
                        help=("Which signal --mask-threat-require-real-target uses to judge a target: "
                              "oracle (default) peeks at the opponent's true hand (upper-bound ablation, "
                              "not deployable). mc estimates each opponent's tenpai-ish probability from "
                              "public info only (open melds + discards) via belief_oracle.opponent_view_posterior "
                              "-- the deployable substitute, thresholded by --mask-threat-target-prob-threshold."))
    parser.add_argument("--mask-threat-target-prob-threshold", type=float, default=0.5,
                        help="Probability threshold on the mc signal's tenpai_prob for counting an opponent as a real target.")
    parser.add_argument("--mask-threat-max-start-shanten", type=int, default=3,
                        help="Do not use FFR deceive-threat from hopeless hands above this starting shanten.")
    parser.add_argument("--mask-threat-allow-exploit-overlap", action="store_true",
                        help="Allow deceive-threat even if the chosen disguise equals the exploit action. Default requires a real action change.")
    parser.add_argument("--mask-log-counterfactual", action="store_true",
                        help="On deceive decisions, also compute the exploit action and log whether disguise==exploit (measures no-op).")
    parser.add_argument("--mask-candidate-reranker", action="store_true",
                        help="Let the decision LLM rank rule-constrained candidates inside the rule-selected MASK mode.")
    parser.add_argument("--mask-candidate-scoring", action="store_true",
                        help="Rank candidates by conditional log probability instead of free generation.")
    parser.add_argument("--mask-reranker-max-candidates", type=int, default=6)
    parser.add_argument("--mask-gate-policy", choices=["rule", "learned", "neural", "neural_value_gate", "neural_action_value", "neural_value_residual", "continuous", "continuous_v2", "continuous_v3", "continuous_v4", "continuous_v5", "continuous_v6", "continuous_v7"], default="rule")
    parser.add_argument("--gate-model-path", default=None)
    parser.add_argument("--gate-adapter-path", default=None)
    parser.add_argument("--gate-max-new-tokens", type=int, default=8)
    parser.add_argument("--neural-gate-model-path", default=None)
    parser.add_argument("--gate-value-model-path", default=None)
    parser.add_argument("--gate-value-safe-margin", type=float, default=None)
    parser.add_argument("--gate-value-deceive-margin", type=float, default=None)
    parser.add_argument("--neural-gate-device", default="cpu")
    parser.add_argument("--neural-gate-deceive-margin", type=float, default=0.0)
    parser.add_argument("--neural-gate-safe-margin", type=float, default=0.0)
    parser.add_argument("--neural-gate-safe-risk-threshold", type=float, default=0.0)
    parser.add_argument("--action-value-model-path", default=None)
    parser.add_argument("--value-residual-model-path", default=None)
    parser.add_argument("--action-value-device", default="cpu")
    parser.add_argument("--snapshot-oracle-samples", type=int, default=120,
                        help="MC samples for the CRN threat before/after snapshot (measurement only; higher = less noise).")
    parser.add_argument("--snapshot-crn-seeds", type=int, default=1,
                        help="Number of common-random-number seeds to average for the threat-delta snapshot.")
    parser.add_argument("--defender-threat-model", choices=["mc", "discard_tell", "blend", "learned"], default="mc",
                        help="Opponent threat model. discard_tell is deceivable by P0's discard choice (middle=push/up, terminal/safe=fold/down).")
    parser.add_argument("--defender-tell-weight", type=float, default=1.0,
                        help="blend model: tell_weight*discard_tell + (1-tell_weight)*mc.")
    parser.add_argument("--defender-tell-window", type=int, default=6,
                        help="How many of P0's most recent discards the tell reads.")
    parser.add_argument("--defender-learned-model-path", default="Defender_danger_model/danger_model.pth",
                        help="Checkpoint for --defender-threat-model learned (see train_defender_danger_model.py).")
    parser.add_argument("--backend", default="heuristic_fallback", choices=["heuristic_fallback", "local_qwen"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--belief-adapter-path", default=None)
    parser.add_argument("--belief-calibration-summary", default=None,
                        help="H1 summary containing calibration.B2_B_phi_steps used by the deployed MC B_phi.")
    parser.add_argument("--belief-public-model", default=None,
                        help="Frozen MC + public-action B_phi joblib artifact.")
    parser.add_argument("--reranker-model-path", default=None,
                        help="Optional dedicated reranker base/merged model; the policy model remains unchanged.")
    parser.add_argument("--reranker-adapter-path", default=None,
                        help="Optional dedicated reranker LoRA adapter loaded on --reranker-model-path.")
    parser.add_argument("--reranker-max-new-tokens", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--debug-game", type=int, default=0,
                        help="Print god's-eye view for the first N games (0=off).")
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved Gate1 outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
