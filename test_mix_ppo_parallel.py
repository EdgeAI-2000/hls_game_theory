import copy
import unittest
from types import SimpleNamespace
import torch
from train_mix_ppo_adapted import collect
from mix_ppo_parallel import EnvironmentPool, collect_parallel
from mix_ppo_baseline import ACTIONS, public_planes, privileged_planes
from mix_ppo_search import public_search, progress_reward, paper_terminal_reward, progress_value
from mix_ppo_selfplay import goal_threshold, promotion_decision
from run_gate1_experiments import init_game
from prompt_builder import get_legal_actions
from tiles import Tile, Suit


class UniformModel:
    def __call__(self, obs, masks):
        return torch.distributions.Categorical(logits=torch.zeros_like(masks, dtype=torch.float).masked_fill(~masks, -torch.inf)), torch.zeros(len(obs))


class ParallelTests(unittest.TestCase):
    def test_process_vs_serial_exact_trajectories(self):
        torch.set_num_threads(1)
        args = SimpleNamespace(rollout_steps=1, envs=2, seed=2026091810, opponents="random",
            manifest="Opponent_Curriculum_formal/screened.json", max_steps=300, reward_scale=100.,
            context_inputs=True, prospective_inputs=False, device="cpu", gamma=.925, gae_lambda=.95,
            reward_mode="net")
        torch.manual_seed(11)
        reference, games, index = collect(UniformModel(), args, 0)
        torch.manual_seed(11)
        with EnvironmentPool(2) as pool:
            actual, parallel_games, next_index = collect_parallel(UniformModel(), args, 0, pool)
            self.assertEqual(len({r["worker_pid"] for r in parallel_games}), 2)
        for row in parallel_games:
            row.pop("worker_pid")
        self.assertEqual(games, parallel_games)
        self.assertEqual(index, next_index)
        for game in games:
            self.assertAlmostEqual(game["normalized_net"], game["net"] / 100.)
        for expected, found in zip(reference, actual):
            for key in expected:
                if isinstance(expected[key], torch.Tensor):
                    self.assertTrue(torch.equal(expected[key], found[key]), key)
                else:
                    self.assertEqual(expected[key], found[key])

    def test_worker_failure_propagates(self):
        with EnvironmentPool(1) as pool:
            with self.assertRaisesRegex(RuntimeError, "reward_scale"):
                pool.exchange({0: ("reset", dict(seed=1, reward_scale=-1))})

    def test_reward_and_goal_direction(self):
        self.assertAlmostEqual(progress_reward(2, 1, 1, 2), .02)
        self.assertEqual(paper_terminal_reward(80, 10), 3)
        self.assertEqual(paper_terminal_reward(-80, 10), -3)
        self.assertEqual(paper_terminal_reward(0, 10, True), -1)
        self.assertEqual(goal_threshold(0), 1)
        self.assertAlmostEqual(goal_threshold(5), .5)
        self.assertEqual(goal_threshold(20), .1)

    def test_search_public_invariance_and_no_mutation(self):
        game, _, _ = init_game(2026091810000000, "random", "test")
        # Two exposed melds keep the exact three-ply test small.
        player = game.players[0]
        player.hand_tiles = [Tile(Suit.WAN, n) for n in [1, 1, 2, 3, 4, 5, 6, 7]]
        player.missing_suit = Suit.TONG
        player.open_melds = [[Tile(Suit.TIAO, n)] * 3 for n in [1, 9]]
        legal = ["d 7万"]
        before = copy.deepcopy(game)
        result = public_search(game, 0, legal)
        game.players[1].hand_tiles, game.players[2].hand_tiles = game.players[2].hand_tiles, game.players[1].hand_tiles
        game.deck.tiles.reverse()
        self.assertEqual(result, public_search(game, 0, legal))
        self.assertEqual(game.players[0].hand_tiles, before.players[0].hand_tiles)
        self.assertGreaterEqual(result[legal[0]]["ukeire"], 0)

    def test_zero_privileged_input_is_public(self):
        game, _, _ = init_game(2026091810000001, "random", "test")
        obs = public_planes(game, 0)
        before = privileged_planes(game, 0, obs, 0.)
        game.players[1].hand_tiles = []
        self.assertTrue(torch.equal(before, privileged_planes(game, 0, obs, 0.)))
        self.assertEqual(float(before[-12:].abs().sum()), 0.)

    def test_progress_and_promotion_semantics(self):
        self.assertGreater(progress_value(0, 4, 80, 1), 0)
        self.assertGreater(progress_value(1, 8, 80), progress_value(2, 8, 80))
        self.assertGreater(progress_value(2, 12, 80), progress_value(2, 8, 80))
        self.assertFalse(promotion_decision([0.] * 200)["promote"])
        self.assertFalse(promotion_decision([-10.] * 200)["promote"])
        self.assertTrue(promotion_decision([10.] * 200)["promote"])

    def test_zero_fan_win_has_positive_search_value(self):
        from unittest.mock import patch
        game, _, _ = init_game(2026091810000000, "random", "test")
        player = game.players[0]
        player.hand_tiles = [Tile(Suit.WAN, n) for n in [1, 1, 2, 3, 4, 5, 6, 7]]
        player.missing_suit = Suit.TONG
        player.open_melds = [[Tile(Suit.TIAO, n)] * 3 for n in [1, 9]]
        with patch("mix_ppo_search.FanCalculator.calculate_fan", return_value=(0, [])):
            result = public_search(game, 0, ["d 7万"])["d 7万"]
        self.assertGreater(result["expectimax"], 0.)

    def test_validation_uses_separate_public_greedy_games(self):
        from unittest.mock import patch
        from mix_ppo_selfplay import validate_promotion
        args = SimpleNamespace(seed=10, promotion_seed=20, privileged_scale=1.)
        indices = []
        def collect_probe(model, probe, index, pool):
            self.assertNotEqual(probe.seed, args.seed)
            self.assertEqual(probe.privileged_scale, 0.)
            dist, _ = model(torch.zeros(1, 2), torch.ones(1, 3, dtype=torch.bool))
            self.assertEqual(int(dist.sample()[0]), 0)
            indices.append(index)
            return [], [{"net":10.}] * 8, index + 8
        before = torch.get_rng_state().clone()
        with patch("mix_ppo_parallel.collect_parallel", side_effect=collect_probe):
            result = validate_promotion(UniformModel(), args, None, 50)
        self.assertTrue(result["promote"])
        self.assertEqual(result["games"], 200)
        self.assertEqual(len(set(indices)), 25)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(args.privileged_scale, 1.)


if __name__ == "__main__":
    unittest.main()
