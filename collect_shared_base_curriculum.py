"""Collect clean shared-backbone SFT states from the Sichuan Mahjong engine.

The teacher uses only legal actions, the acting player's hand, exact shanten,
and public effective-tile counts.  It deliberately excludes opponent beliefs,
MASK modes, safe/deceive labels, and hidden opponent hands.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from experiment_trace import ensure_deterministic_hashing, write_json
from game import (
    MahjongGame,
    bot_decide_exchange,
    bot_decide_missing_suit,
    parse_console_tile,
)
from policy_metrics import discard_progress_metrics
from prompt_builder import ACTION_SYSTEM_PROMPT, build_base_decision_prompt, get_legal_actions
from rule_engine import ShantenCalculator


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def action_values(game: MahjongGame, player_id: int, legal_actions: List[str]) -> Dict[str, Dict[str, int]]:
    values: Dict[str, Dict[str, int]] = {}
    for action in legal_actions:
        metrics = discard_progress_metrics(game, player_id, action)
        if metrics is not None:
            values[action] = {key: int(value) for key, value in metrics.items()}
    return values


def choose_turn_action(
    game: MahjongGame,
    player_id: int,
    legal_actions: List[str],
    min_wall_for_gang: int = 4,
) -> Tuple[str, str, Dict[str, Dict[str, int]]]:
    if "h" in legal_actions:
        return "h", "take_legal_hu", {}

    values = action_values(game, player_id, legal_actions)
    player = game.players[player_id]
    missing_discards = [
        action
        for action in values
        if action.startswith("d ")
        and any(
            str(tile) == action[2:] and tile.suit == player.missing_suit
            for tile in player.hand_tiles
        )
    ]
    if missing_discards:
        action = min(
            missing_discards,
            key=lambda item: (
                values[item]["shanten"],
                -values[item]["effective_copies"],
                -values[item]["effective_types"],
                item,
            ),
        )
        return action, "clear_missing_suit_first", values

    if "g" in legal_actions and game.deck.remaining_count() >= min_wall_for_gang:
        return "g", "take_legal_self_gang_with_wall_buffer", values

    if values:
        action = min(
            values,
            key=lambda item: (
                values[item]["shanten"],
                -values[item]["effective_copies"],
                -values[item]["effective_types"],
                item,
            ),
        )
        best_shanten = min(value["shanten"] for value in values.values())
        best_copies = max(
            value["effective_copies"]
            for value in values.values()
            if value["shanten"] == best_shanten
        )
        reason = "min_shanten_max_ukeire"
        if values[action]["shanten"] == best_shanten and values[action]["effective_copies"] == best_copies:
            reason = "exact_progress_teacher"
        return action, reason, values

    fallback = next((action for action in legal_actions if action != "h"), legal_actions[0])
    return fallback, "legal_fallback", values


def choose_response_action(
    game: MahjongGame,
    player_id: int,
    legal_actions: List[str],
    min_wall_for_call: int = 4,
) -> Tuple[str, str]:
    if "h" in legal_actions:
        return "h", "take_legal_hu_response"
    player = game.players[player_id]
    has_missing = bool(
        player.missing_suit
        and any(tile.suit == player.missing_suit for tile in player.hand_tiles)
    )
    if has_missing and "n" in legal_actions:
        return "n", "pass_to_clear_missing_suit"
    if game.deck.remaining_count() >= min_wall_for_call:
        if "g" in legal_actions:
            return "g", "take_legal_gang_response_with_wall_buffer"
        if "p" in legal_actions:
            return "p", "take_legal_peng_response_with_wall_buffer"
    if "n" in legal_actions:
        return "n", "pass_response"
    return legal_actions[0], "legal_response_fallback"


def make_row(
    game: MahjongGame,
    player_id: int,
    legal_actions: List[str],
    action: str,
    reason: str,
    phase: str,
    seed: int,
    step: int,
    values: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    prompt = build_base_decision_prompt(game, player_id, valid_actions=legal_actions)
    player = game.players[player_id]
    current_shanten = int(ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit))
    return {
        "messages": [
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": action},
        ],
        "meta": {
            "shared_base_source": "exact_progress_curriculum",
            "teacher": "exact_shanten_ukeire",
            "game_id": game.game_id,
            "seed": seed,
            "step": step,
            "player_id": player_id,
            "phase": phase,
            "target_reason": reason,
            "legal_actions": list(legal_actions),
            "current_shanten": current_shanten,
            "action_values": values or {},
            "prompt_schema": "shared_base_action_v1",
            "shanten_engine": "exact_dfs_v2",
            "uses_opponent_belief": False,
            "uses_gate_mode": False,
        },
    }


def should_keep(row: Dict[str, Any], sample_stride: int) -> bool:
    action = str(row["messages"][-1]["content"])
    if action == "h":
        return True
    if row["meta"]["phase"] == "response":
        return True
    return int(row["meta"]["step"]) % max(1, sample_stride) == 0


def play_game(game_index: int, seed: int, max_steps: int, sample_stride: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    random.seed(seed)
    game = MahjongGame(
        f"SHARED_BASE_{seed}_{game_index}",
        [f"C{i}" for i in range(4)],
        bots=[True] * 4,
    )
    start_balances = [player.balance for player in game.players]
    game.start_game()
    for player in game.players:
        if not game.select_exchange_tiles(player.player_id, bot_decide_exchange(player)):
            raise RuntimeError(f"exchange failed for P{player.player_id}")
    for player in game.players:
        if not game.set_missing_suit(player.player_id, bot_decide_missing_suit(player)):
            raise RuntimeError(f"missing-suit selection failed for P{player.player_id}")

    rows: List[Dict[str, Any]] = []
    skip_draw = True
    steps = 0
    while not game.is_game_over and steps < max_steps:
        steps += 1
        if game.deck.remaining_count() == 0 or sum(player.is_hu for player in game.players) >= 3:
            game.check_game_over()
            break

        player_id = game.current_player_id
        player = game.players[player_id]
        if player.is_hu:
            game.next_player()
            skip_draw = False
            continue

        drawn = None
        if not skip_draw:
            drawn = game.draw_tile(player_id)
            if drawn is None:
                game.check_game_over()
                break
        else:
            skip_draw = False

        legal = get_legal_actions(game, player_id)
        action, reason, values = choose_turn_action(game, player_id, legal)
        row = make_row(game, player_id, legal, action, reason, "turn", seed, steps, values)
        if should_keep(row, sample_stride):
            rows.append(row)

        if action == "h" and player.can_hu():
            win_tile = drawn if drawn is not None else player.hand_tiles[-1]
            game.hu(player_id, win_tile, True)
            game.check_game_over()
            if not game.is_game_over:
                game.next_player()
                skip_draw = False
            continue

        if action == "g":
            gang_info = game.can_self_gang(player_id)
            if gang_info.get("can_gang"):
                game.gang(player_id, gang_info["gang_tiles"][0])
                skip_draw = True
                continue

        if action == "n":
            game.next_player()
            skip_draw = False
            continue

        tile = parse_console_tile(action[2:]) if action.startswith("d ") else None
        if tile is None or not game.discard_tile(player_id, tile):
            raise RuntimeError(f"teacher produced illegal action {action!r} in {game.game_id}")

        responses = game.check_responses(tile, player_id)
        responded = False
        for responder_id, available in responses.items():
            if responded:
                break
            response_legal = get_legal_actions(game, responder_id, response_actions=available)
            response, response_reason = choose_response_action(game, responder_id, response_legal)
            response_row = make_row(
                game,
                responder_id,
                response_legal,
                response,
                response_reason,
                "response",
                seed,
                steps,
            )
            if should_keep(response_row, sample_stride):
                rows.append(response_row)

            if response == "h" and "hu" in available:
                game.hu(responder_id, tile, False, player_id)
                game.check_game_over()
                responded = True
            elif response == "g" and "gang" in available:
                game.gang(responder_id, tile, player_id)
                game.current_player_id = responder_id
                responded = True
            elif response == "p" and "peng" in available:
                game.peng(responder_id, tile, player_id)
                game.current_player_id = responder_id
                responded = True

        if game.is_game_over:
            break
        if responded:
            skip_draw = True
        else:
            game.next_player()
            skip_draw = False

    if not game.is_game_over:
        game.check_game_over()

    final = {
        "steps": steps,
        "rows": len(rows),
        "winners": [player.player_id for player in game.players if player.is_hu],
        "net": [player.balance - start_balances[player.player_id] for player in game.players],
    }
    for row in rows:
        pid = int(row["meta"]["player_id"])
        row["meta"]["game_net"] = final["net"][pid]
        row["meta"]["game_hu"] = pid in final["winners"]
    return rows, final


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2026072701)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--hu-repeat", type=int, default=4)
    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be positive")
    if not 0.0 <= args.eval_fraction < 0.5:
        raise ValueError("--eval-fraction must be in [0, 0.5)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    game_stats: List[Dict[str, Any]] = []
    for game_index in range(args.games):
        seed = args.seed + game_index
        rows, final = play_game(game_index, seed, args.max_steps, args.sample_stride)
        target = eval_rows if random.Random(seed * 1009).random() < args.eval_fraction else train_rows
        for row in rows:
            repeats = args.hu_repeat if row["messages"][-1]["content"] == "h" else 1
            for repeat_index in range(max(1, repeats)):
                item = json.loads(json.dumps(row, ensure_ascii=False))
                item["meta"]["curriculum_repeat"] = repeat_index
                target.append(item)
        game_stats.append(final)
        if (game_index + 1) % 50 == 0 or game_index + 1 == args.games:
            print(
                f"[shared-base curriculum] game={game_index + 1}/{args.games} "
                f"train_rows={len(train_rows)} eval_rows={len(eval_rows)}",
                flush=True,
            )

    train_path = args.output_dir / "shared_base_curriculum_train.jsonl"
    eval_path = args.output_dir / "shared_base_curriculum_eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    def row_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "rows": len(rows),
            "actions": dict(Counter(str(row["messages"][-1]["content"]) for row in rows)),
            "phases": dict(Counter(str(row["meta"]["phase"]) for row in rows)),
            "target_reasons": dict(Counter(str(row["meta"]["target_reason"]) for row in rows)),
            "hu_rows": sum(row["messages"][-1]["content"] == "h" for row in rows),
            "rows_with_action_values": sum(bool(row["meta"].get("action_values")) for row in rows),
        }

    summary = {
        "created_for": "shared base SFT curriculum",
        "excludes": ["B_phi", "z_j(t)", "safe/deceive mode", "hidden opponent hands"],
        "games": args.games,
        "seed": args.seed,
        "sample_stride": args.sample_stride,
        "eval_fraction": args.eval_fraction,
        "hu_repeat": args.hu_repeat,
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "train": row_stats(train_rows),
        "eval": row_stats(eval_rows),
        "avg_steps": sum(item["steps"] for item in game_stats) / len(game_stats),
        "avg_rows_before_repeat": sum(item["rows"] for item in game_stats) / len(game_stats),
    }
    write_json(args.output_dir / "shared_base_curriculum_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
