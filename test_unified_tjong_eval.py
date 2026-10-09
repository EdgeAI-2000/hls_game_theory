import unittest
from types import SimpleNamespace

from tiles import Tile, Suit
from unified_tjong_eval import macro_mask, common_runner, p4, tile_index


class MappingTests(unittest.TestCase):
    def test_last_wall_response_gang_is_preserved(self):
        tile = Tile(list(Suit)[0], 1)
        game = SimpleNamespace(last_discarded_tile=tile,
                               deck=SimpleNamespace(remaining_count=lambda:1))
        mask, decode = macro_mask(game, 0, ['g', 'p', 'n', 'h'], True)
        self.assertTrue(mask[3, tile_index(tile)])
        self.assertEqual(set(decode.values()), {'g','p','n','h'})

    def test_own_gang_uses_engine_first_tile_not_sorted_order(self):
        a, b = Tile(list(Suit)[0], 9), Tile(list(Suit)[0], 1)
        game = SimpleNamespace(can_self_gang=lambda pid:dict(
            gang_tiles=[a,b], gang_types={f'{a.number}{a.suit.value}':'bu',f'{b.number}{b.suit.value}':'an'}))
        mask, decode = macro_mask(game, 0, ['g'], False)
        self.assertEqual(mask.sum(), 1)
        self.assertEqual(decode[(5,tile_index(a))], 'g')

    def test_runner_does_not_patch_shared_module(self):
        original = p4.play_one_game
        decision = p4.choose_llm_base_action
        a, b = common_runner(), common_runner()
        self.assertIs(p4.play_one_game, original)
        self.assertIs(p4.choose_llm_base_action, decision)
        self.assertIsNot(a.__globals__, b.__globals__)
        self.assertIs(a.__globals__['execute_action'], p4.execute_action)
        self.assertIs(a.__globals__['resolve_responses'], p4.resolve_responses)

    def test_opponent_private_pass_is_not_observed(self):
        from unified_baseline_eval import Policy
        policy = Policy('douzero', None, 1)
        policy.record(None, 1, 'n', True)
        self.assertEqual(policy.events, [])
        policy.record(None, 0, 'n', True)
        self.assertEqual(len(policy.events), 1)


if __name__ == '__main__':
    unittest.main()
