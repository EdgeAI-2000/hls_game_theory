"""Tests of migration infrastructure and source-formula ambiguity, not playing strength."""

import copy
import unittest

import torch

from evaluate_neural_opponent_selfplay import init_selfplay_game
from mix_ppo_baseline import (
    CHANNELS, ACTIONS, SichuanResNet101, discard_mask, action_mask, public_planes,
    paper_equation3_diagnostic, paper_equation4_diagnostic,
    adapted_loss, update_kl_coefficient, episode_gae,
)
from prompt_builder import get_legal_actions


class MixPPOInfrastructureTests(unittest.TestCase):
    def test_collector_retains_zero_decision_game(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from train_mix_ppo_adapted import collect

        def fake_episode(seed, *args, **kwargs):
            decisions = seed % 1000000
            if decisions:
                yield torch.zeros(1), torch.ones(1, dtype=torch.bool), 10000
            return {"seed": seed, "decisions": decisions, "net": 0, "final_balance": 10000}

        def model(obs, masks):
            return torch.distributions.Categorical(logits=torch.zeros(len(obs), 1)), torch.zeros(len(obs))

        args = SimpleNamespace(rollout_steps=1, envs=2, seed=1, opponents="curriculum",
                               manifest="unused", max_steps=300, reward_scale=100,
                               device="cpu", gamma=.925, gae_lambda=.95)
        with patch("train_mix_ppo_adapted.episode", fake_episode):
            samples, games, index = collect(model, args, 0)
        self.assertEqual(index, 2)
        self.assertEqual(len(samples), 1)
        self.assertEqual([g["decisions"] for g in games], [0, 1])

    def test_actual_zero_decision_seed_settles(self):
        from mix_ppo_rollout import episode
        with self.assertRaises(StopIteration) as finished:
            next(episode(2026091601003491))
        self.assertEqual(finished.exception.value["decisions"], 0)
        self.assertEqual(finished.exception.value["max_steps_or_unsettled"], 0)

    def test_context_does_not_change_rules_or_rng(self):
        from mix_ppo_rollout import episode

        def play(context):
            stream = episode(2026091705000000, context_inputs=context)
            try:
                _, mask, _ = next(stream)
                while True:
                    chosen = ACTIONS.index("h") if mask[ACTIONS.index("h")] else int(mask.nonzero()[0])
                    _, mask, _ = stream.send(chosen)
            except StopIteration as end:
                return end.value

        self.assertEqual(play(False), play(True))

    def test_public_prospective_features_are_public_and_shape_checked(self):
        from mix_ppo_baseline import PROSPECTIVE_CHANNELS
        from prompt_builder import get_legal_actions
        legal = get_legal_actions(self.game, 0)
        before = public_planes(self.game, 0, legal, False, True)
        self.assertEqual(tuple(before.shape), (PROSPECTIVE_CHANNELS, 3, 9))
        altered = copy.deepcopy(self.game)
        altered.players[1].hand_tiles.reverse()
        altered.deck.tiles.reverse()
        self.assertTrue(torch.equal(before, public_planes(altered, 0, legal, False, True)))

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(2026091601)
        cls.game = init_selfplay_game(2026091601000000, "mix_ppo_infrastructure")

    def test_hidden_information_invariance(self):
        before = public_planes(self.game, 0)
        altered = copy.deepcopy(self.game)
        altered.players[1].hand_tiles, altered.players[2].hand_tiles = (
            altered.players[2].hand_tiles, altered.players[1].hand_tiles)
        altered.deck.tiles.reverse()
        self.assertTrue(torch.equal(before, public_planes(altered, 0)))
        self.assertEqual(tuple(before.shape), (CHANNELS, 3, 9))

    def test_context_legality_phase_and_response_tile(self):
        from mix_ppo_baseline import CONTEXT_CHANNELS
        from tiles import Suit, Tile
        game = copy.deepcopy(self.game)
        game.last_discarded_tile = Tile(Suit.WAN, 1)
        game.last_discard_player_id = 2
        a = public_planes(game, 0, ["n", "p"], True)
        self.assertEqual(tuple(a.shape), (CONTEXT_CHANNELS, 3, 9))
        self.assertFalse(torch.equal(a, public_planes(game, 0, ["n", "h"], True)))
        self.assertFalse(torch.equal(a, public_planes(game, 0, ["n", "p"], False)))
        game.last_discarded_tile = Tile(Suit.TONG, 8)
        self.assertFalse(torch.equal(a, public_planes(game, 0, ["n", "p"], True)))
        before = public_planes(game, 0, ["n", "p"], True)
        game.players[1].hand_tiles, game.players[2].hand_tiles = game.players[2].hand_tiles, game.players[1].hand_tiles
        game.deck.tiles.reverse()
        self.assertTrue(torch.equal(before, public_planes(game, 0, ["n", "p"], True)))
        model = SichuanResNet101(context_inputs=True).eval()
        with torch.no_grad():
            dist, value = model(before[None], action_mask(["n", "p"])[None])
        self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(float(dist.probs[0, ACTIONS.index("h")]), 0)

    def test_public_discard_order_is_preserved(self):
        altered = copy.deepcopy(self.game)
        from tiles import Suit, Tile
        altered.players[1].discarded_tiles = [Tile(Suit.WAN, 1), Tile(Suit.WAN, 2)]
        before = public_planes(altered, 0)
        altered.players[1].discarded_tiles.reverse()
        self.assertFalse(torch.equal(before, public_planes(altered, 0)))

    def test_response_phase_is_rejected(self):
        with self.assertRaises(ValueError):
            discard_mask(["h", "n"])

    def test_network_mask_gradients_and_reload(self):
        model = SichuanResNet101().eval()
        self.assertFalse(any(isinstance(m, (torch.nn.MaxPool2d, torch.nn.AvgPool2d,
                                            torch.nn.AdaptiveAvgPool2d)) for m in model.modules()))
        obs = public_planes(self.game, 0).unsqueeze(0)
        legal = get_legal_actions(self.game, 0)
        mask = action_mask(legal).unsqueeze(0)
        distribution, value = model(obs, mask)
        self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(distribution.probs[~mask].sum().item(), 0.0)
        action = distribution.sample()
        self.assertIn(ACTIONS[action.item()], legal)
        loss = -distribution.log_prob(action).mean() + value.square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            reloaded, reloaded_value = model(obs, mask)
        self.assertTrue(torch.equal(distribution.probs, reloaded.probs))
        self.assertTrue(torch.equal(value, reloaded_value))

    def test_printed_kl_term_can_reward_divergence(self):
        kl = torch.tensor([2.0], requires_grad=True)
        objective = paper_equation4_diagnostic(
            torch.ones(1), torch.zeros(1), torch.zeros(1), kl, kl_coef=1.0)
        objective.backward()
        self.assertGreater(kl.grad.item(), 0)
        self.assertLess(paper_equation3_diagnostic(0.005, 0.1, 0.01, 0.01), 0.005)

    def test_schulman_coefficient_update(self):
        self.assertEqual(update_kl_coefficient(0.01, 0.001, 0.01), 0.005)
        self.assertEqual(update_kl_coefficient(0.01, 0.02, 0.01), 0.02)
        self.assertEqual(update_kl_coefficient(0.01, 0.01, 0.01), 0.01)

    def test_clip_both_advantage_signs_and_kl_penalty(self):
        from torch.distributions import Categorical
        old = Categorical(probs=torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
        new = Categorical(probs=torch.tensor([[0.9, 0.1], [0.1, 0.9]]))
        actions = torch.tensor([0, 0])
        loss, stats = adapted_loss(new, torch.zeros(2), actions, old.logits,
                                   old.log_prob(actions), torch.tensor([1., -1.]),
                                   torch.zeros(2), 0., entropy_coef=0.)
        self.assertAlmostEqual(stats["policy_objective"].item(), (1.2 - 0.8) / 2, places=6)
        penalized, _ = adapted_loss(new, torch.zeros(2), actions, old.logits,
                                    old.log_prob(actions), torch.tensor([1., -1.]),
                                    torch.zeros(2), 1., entropy_coef=0.)
        self.assertGreater(penalized.item(), loss.item())

    def test_gae_settlement_is_included(self):
        advantage, returns = episode_gae([1., 2., 3.], [0., 0., 0.], gamma=1., lam=1.)
        self.assertTrue(torch.equal(returns, torch.tensor([6., 5., 3.])))
        self.assertTrue(torch.equal(advantage, returns))

    def test_masked_objective_gradients_are_finite(self):
        from torch.distributions import Categorical
        logits = torch.tensor([[1., 2., 0.]], requires_grad=True)
        masked = logits.masked_fill(torch.tensor([[False, False, True]]), -torch.inf)
        distribution = Categorical(logits=masked)
        action = torch.tensor([0])
        loss, _ = adapted_loss(distribution, torch.zeros(1), action,
                               distribution.logits.detach(), distribution.log_prob(action).detach(),
                               torch.ones(1), torch.zeros(1), .005)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_rollout_repeatability_under_interleaving(self):
        from mix_ppo_rollout import episode

        def finish(stream):
            try:
                request = next(stream)
                while True:
                    request = stream.send(int(request[1].nonzero()[0]))
            except StopIteration as result:
                return result.value

        expected = finish(episode(2026091601000001, opponent_style="random"))
        stream = episode(2026091601000001, opponent_style="random")
        request = next(stream)
        finish(episode(2026091601000002, opponent_style="random"))
        try:
            while True:
                request = stream.send(int(request[1].nonzero()[0]))
        except StopIteration as result:
            actual = result.value
        self.assertEqual(expected, actual)
        self.assertEqual(actual["max_steps_or_unsettled"], 0)
        self.assertGreater(actual["decisions"], 0)


if __name__ == "__main__":
    unittest.main()
