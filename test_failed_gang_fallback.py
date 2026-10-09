from __future__ import annotations

import unittest
import random
from types import SimpleNamespace
from unittest.mock import patch

from game import MahjongGame
from run_gate1_experiments import (
    ResponsiveDefender,
    classify_strict_false_fold,
    execute_action,
)
from tiles import Suit, Tile


class _FakeGame:
    def __init__(self) -> None:
        self.players = [object()]
        self.discarded = None

    def can_self_gang(self, player_id: int):
        return {"can_gang": True, "gang_tiles": [object()]}

    def gang(self, player_id: int, tile):
        return False, None

    def discard_tile(self, player_id: int, tile) -> bool:
        self.discarded = tile
        return True


class FailedGangFallbackTest(unittest.TestCase):
    def test_strict_ffr_counts_only_first_opportunity_per_opponent_game(self):
        defender = ResponsiveDefender(
            observer_pid=1,
            threat_threshold=0.5,
            oracle_samples=1,
            beta=1.0,
            danger_threshold=1,
            ffr_hand_shanten=2,
            rng=random.Random(1),
        )
        defender.threat = lambda game: 0.8
        game = SimpleNamespace(
            players=[SimpleNamespace(hand_tiles=[], missing_suit=None), SimpleNamespace(hand_tiles=[], missing_suit=None)],
        )
        progress = {
            "d push": {"shanten": 1, "effective_copies": 8},
            "d safe": {"shanten": 1, "effective_copies": 4},
        }

        with (
            patch("run_gate1_experiments.get_legal_actions", return_value=["d push", "d safe"]),
            patch("run_gate1_experiments.choose_min_shanten_action", return_value="d push"),
            patch("run_gate1_experiments.defensive_discard_action", return_value="d safe"),
            patch("run_gate1_experiments.within_shanten", return_value=False),
            patch("run_gate1_experiments.ShantenCalculator.calculate_shanten", side_effect=[1, 1, 1, 1]),
            patch("run_gate1_experiments.discard_progress_metrics", side_effect=lambda game, pid, action: progress.get(action)),
        ):
            defender.turn(game.players[1], game)
            defender.turn(game.players[1], game)

        self.assertEqual(defender.ff_opportunities, 1)
        self.assertEqual(defender.ff_false, 1)
        self.assertEqual(defender.strict_ffr_event_opportunities, 2)
        self.assertEqual(defender.strict_ffr_event_false, 2)
        self.assertEqual(defender.defensive_switch_opportunities, 2)
        self.assertEqual(defender.defensive_switches, 2)

    def test_later_false_fold_does_not_replace_first_primary_opportunity(self):
        defender = ResponsiveDefender(
            observer_pid=1,
            threat_threshold=0.5,
            oracle_samples=1,
            beta=1.0,
            danger_threshold=1,
            ffr_hand_shanten=2,
            rng=random.Random(1),
        )
        defender.threat = lambda game: next(threats)
        threats = iter([0.2, 0.8])
        game = SimpleNamespace(
            players=[SimpleNamespace(hand_tiles=[], missing_suit=None), SimpleNamespace(hand_tiles=[], missing_suit=None)],
        )
        progress = {
            "d push": {"shanten": 1, "effective_copies": 8},
            "d safe": {"shanten": 1, "effective_copies": 4},
        }

        with (
            patch("run_gate1_experiments.get_legal_actions", return_value=["d push", "d safe"]),
            patch("run_gate1_experiments.choose_min_shanten_action", return_value="d push"),
            patch("run_gate1_experiments.defensive_discard_action", return_value="d safe"),
            patch("run_gate1_experiments.within_shanten", return_value=False),
            patch("run_gate1_experiments.ShantenCalculator.calculate_shanten", side_effect=[1, 1, 1, 1]),
            patch("run_gate1_experiments.discard_progress_metrics", side_effect=lambda game, pid, action: progress.get(action)),
        ):
            defender.turn(game.players[1], game)
            defender.turn(game.players[1], game)

        self.assertEqual(defender.ff_opportunities, 1)
        self.assertEqual(defender.ff_false, 0)
        self.assertEqual(defender.strict_ffr_event_opportunities, 2)
        self.assertEqual(defender.strict_ffr_event_false, 1)

    def test_strict_ffr_requires_real_progress_loss(self):
        common = {
            "own_shanten": 1,
            "p0_shanten": 1,
            "high_threat": True,
            "push_action": "d push",
            "chosen_action": "d safe",
            "push_metrics": {"shanten": 1, "effective_copies": 8},
        }

        no_loss = classify_strict_false_fold(
            **common,
            chosen_metrics={"shanten": 1, "effective_copies": 8},
        )
        ukeire_loss = classify_strict_false_fold(
            **common,
            chosen_metrics={"shanten": 1, "effective_copies": 4},
        )
        shanten_loss = classify_strict_false_fold(
            **common,
            chosen_metrics={"shanten": 2, "effective_copies": 12},
        )

        self.assertFalse(no_loss["false_fold"])
        self.assertTrue(ukeire_loss["false_fold"])
        self.assertTrue(ukeire_loss["ukeire_loss"])
        self.assertTrue(shanten_loss["false_fold"])
        self.assertTrue(shanten_loss["shanten_loss"])

    def test_strict_ffr_excludes_tenpai_p0_and_non_discard_push(self):
        metrics = {"shanten": 1, "effective_copies": 4}
        p0_tenpai = classify_strict_false_fold(
            own_shanten=1,
            p0_shanten=0,
            high_threat=True,
            push_action="d push",
            chosen_action="d safe",
            push_metrics={"shanten": 1, "effective_copies": 8},
            chosen_metrics=metrics,
        )
        declined_gang = classify_strict_false_fold(
            own_shanten=1,
            p0_shanten=1,
            high_threat=True,
            push_action="g",
            chosen_action="d safe",
            push_metrics=None,
            chosen_metrics=metrics,
        )

        self.assertFalse(p0_tenpai["eligible"])
        self.assertFalse(declined_gang["eligible"])

    def test_strict_ffr_keeps_low_threat_state_only_in_the_denominator(self):
        result = classify_strict_false_fold(
            own_shanten=1,
            p0_shanten=1,
            high_threat=False,
            push_action="d push",
            chosen_action="d push",
            push_metrics={"shanten": 1, "effective_copies": 8},
            chosen_metrics={"shanten": 1, "effective_copies": 8},
        )

        self.assertTrue(result["eligible"])
        self.assertFalse(result["false_fold"])

    def test_self_gang_is_not_offered_without_replacement_and_discard_tiles(self):
        game = MahjongGame("wall_guard", ["P0", "P1", "P2", "P3"])
        player = game.players[0]
        player.missing_suit = Suit.TIAO
        player.hand_tiles = [Tile(Suit.WAN, 1) for _ in range(4)]
        game.deck.tiles = [Tile(Suit.TONG, 1)]

        self.assertFalse(game.can_self_gang(0)["can_gang"])

    def test_failed_gang_falls_back_to_legal_discard(self):
        game = _FakeGame()
        discard_tile = object()

        with (
            patch(
                "run_gate1_experiments.get_legal_actions",
                return_value=["g", "d tile"],
            ),
            patch(
                "run_gate1_experiments.parse_console_tile",
                return_value=discard_tile,
            ),
        ):
            result = execute_action(game, 0, "g")

        self.assertIs(result, discard_tile)
        self.assertIs(game.discarded, discard_tile)


if __name__ == "__main__":
    unittest.main()
