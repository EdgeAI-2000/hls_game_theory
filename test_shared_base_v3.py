import unittest

from collect_shared_base_curriculum import choose_response_action, choose_turn_action
from experiment_trace import summarize_games
from game import MahjongGame
from prompt_builder import blocked_calls_while_missing, get_legal_actions
from tiles import Suit, Tile


class SharedBaseV3Test(unittest.TestCase):
    def make_game(self) -> MahjongGame:
        return MahjongGame("shared_base_v3_test", ["P0", "P1", "P2", "P3"])

    def test_teacher_clears_missing_suit_before_self_gang(self):
        game = self.make_game()
        player = game.players[0]
        player.missing_suit = Suit.WAN
        player.hand_tiles = [
            Tile(Suit.WAN, 1),
            Tile(Suit.TONG, 9), Tile(Suit.TONG, 9),
            Tile(Suit.TONG, 9), Tile(Suit.TONG, 9),
            Tile(Suit.TIAO, 1), Tile(Suit.TIAO, 2), Tile(Suit.TIAO, 3),
            Tile(Suit.TIAO, 4), Tile(Suit.TIAO, 5), Tile(Suit.TIAO, 6),
            Tile(Suit.TONG, 1), Tile(Suit.TONG, 2), Tile(Suit.TONG, 3),
        ]
        action, reason, _ = choose_turn_action(game, 0, ["g", "d 1万"])
        self.assertEqual(action, "d 1万")
        self.assertEqual(reason, "clear_missing_suit_first")

    def test_teacher_passes_calls_while_missing_suit_remains(self):
        game = self.make_game()
        player = game.players[1]
        player.missing_suit = Suit.WAN
        player.hand_tiles = [Tile(Suit.WAN, 2), Tile(Suit.TONG, 3), Tile(Suit.TONG, 3)]
        action, reason = choose_response_action(game, 1, ["g", "p", "n"])
        self.assertEqual(action, "n")
        self.assertEqual(reason, "pass_to_clear_missing_suit")

    def test_common_candidates_block_calls_until_missing_suit_is_clear(self):
        game = self.make_game()
        player = game.players[0]
        player.missing_suit = Suit.WAN
        player.hand_tiles = [
            Tile(Suit.WAN, 1),
            Tile(Suit.TONG, 9), Tile(Suit.TONG, 9),
            Tile(Suit.TONG, 9), Tile(Suit.TONG, 9),
        ]
        self.assertEqual(blocked_calls_while_missing(game, 0), ["g"])
        self.assertNotIn("g", get_legal_actions(game, 0))
        self.assertEqual(
            blocked_calls_while_missing(game, 0, response_actions=["gang", "peng"]),
            ["g", "p"],
        )
        self.assertEqual(get_legal_actions(game, 0, response_actions=["gang", "peng"]), ["n"])

    def test_final_wall_kong_is_rejected_without_mutation(self):
        game = self.make_game()
        tile = Tile(Suit.TONG, 5)
        game.players[0].hand_tiles = [tile, tile, tile, tile]
        game.deck.tiles = [Tile(Suit.WAN, 1)]
        success, replacement = game.gang(0, tile)
        self.assertFalse(success)
        self.assertIsNone(replacement)
        self.assertEqual(len(game.players[0].hand_tiles), 4)

    def test_settlement_ledger_reconciles_and_is_summarized(self):
        game = self.make_game()
        game._transfer_score(1, 0, 20, "hu_discard", fan=1)
        game._transfer_score(0, 2, 2560, "hua_zhu")
        self.assertEqual(game.players[0].balance, 7460)
        row = {
            "agent_net": -2540,
            "steps": 1,
            "decision_ms": [],
            "settlement_breakdown": {
                "net": {"hu_discard": 20, "hua_zhu": -2560},
                "paid": {"hua_zhu": 2560},
                "received": {"hu_discard": 20},
            },
            "settlement_reconciled": True,
        }
        summary = summarize_games([row])
        self.assertEqual(summary["settlement"]["net_total"]["hu_discard"], 20)
        self.assertEqual(summary["settlement"]["net_total"]["hua_zhu"], -2560)
        self.assertEqual(summary["net_distribution"]["extreme_negative_2500"]["games"], 1)


if __name__ == "__main__":
    unittest.main()
