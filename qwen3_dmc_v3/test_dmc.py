import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3Model

from run_gate1_experiments import init_game
from prompt_builder import get_legal_actions
from douzero_sichuan.core import STATE_DIM, encode as encode_structured
from .model import QwenValue, observation, pad_sequences
from .rollout import Policy, game_episode, return_to_go
from .train import regress


class FakeTokenizer:
    eos_token_id = pad_token_id = 0

    def encode(self, text, **kwargs):
        return [1 + ord(c) % 100 for c in text]


class LegalPolicy:
    def score(self, ids, candidates, structured=None):
        # Favor wins in this execution test, without imposing this on training.
        return candidates[:, 1] * 100 + candidates[:, 2] * 10 + candidates[:, 3] * 5 + np.arange(len(candidates)) * .001


class Tests(unittest.TestCase):
    def test_backbone_gradient_padding_and_reload(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, head_dim=8,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
        model = QwenValue(Qwen3Model(cfg), 4, 8).eval()
        sequences = [[1, 2, 3], [4, 5]]
        ids, mask = pad_sequences(sequences, 0, 'cpu')
        actions = torch.zeros(2, 32)
        actions[:, 0] = 1
        with torch.no_grad():
            together = model(ids, mask, actions)
            alone = model(ids[1:2, :2], mask[1:2, :2], actions[1:2])
        torch.testing.assert_close(together[1:], alone, atol=1e-6, rtol=1e-5)
        before = copy.deepcopy(model.weights())
        optimizer = torch.optim.RMSprop([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        loss = (model(ids, mask, actions) - torch.tensor([1., -1.])).square().mean()
        loss.backward()
        self.assertGreater(sum(p.grad.square().sum().item() for p in model.backbone.parameters()
                               if p.grad is not None), 0)
        optimizer.step()
        after = model.weights()
        self.assertTrue(any(not torch.equal(v, before['adapter'][k]) for k, v in after['adapter'].items()))
        self.assertTrue(any(not torch.equal(v, before['head'][k]) for k, v in after['head'].items()))
        loaded = QwenValue(Qwen3Model(cfg), 4, 8).eval()
        # Independent initialization would have different frozen base weights.
        loaded.backbone.base_model.model.load_state_dict(model.backbone.base_model.model.state_dict())
        loaded.load_weights(after)
        torch.testing.assert_close(model(ids, mask, actions), loaded(ids, mask, actions))
        broken = copy.deepcopy(after)
        broken['adapter'].pop(next(iter(broken['adapter'])))
        with self.assertRaises(RuntimeError):
            loaded.load_weights(broken)

    def test_hidden_information_invariance(self):
        game = init_game(26100200001, 'random', 'test')[0]
        actions = get_legal_actions(game, 0)
        first = observation(game, 0, actions, False)
        numeric = encode_structured(game, 0, actions, [], False)[0]
        for p in game.players[1:]:
            p.hand_tiles = [game.players[0].hand_tiles[0]] * len(p.hand_tiles)
        game.deck.tiles.reverse()
        game.history.append(dict(pid=1, act='private_pass', tile='hidden'))
        self.assertEqual(first, observation(game, 0, actions, False))
        np.testing.assert_array_equal(numeric, encode_structured(game, 0, actions, [], False)[0])

    def test_text_checkpoint_expansion_preserves_scores_and_learns_numeric_input(self):
        from qwen3_dmc_v2.model import QwenValue as TextValue
        torch.set_num_threads(1)
        torch.manual_seed(9)
        cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, head_dim=8,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
        backbone = Qwen3Model(cfg)
        old = TextValue(copy.deepcopy(backbone), 4, 8).eval()
        old.update_return_stats([-12, 4, 28])
        model = QwenValue(backbone, 4, 8).eval()
        saved = old.weights()
        original_first = saved['head']['0.weight'].clone()
        model.load_text_weights(saved)
        torch.testing.assert_close(saved['head']['0.weight'], original_first)
        ids, mask = pad_sequences([[1, 2, 3], [4, 5]], 0, 'cpu')
        actions = torch.zeros(2, 32)
        actions[0, 0], actions[1, 1] = 1, 1
        numeric = torch.randn(2, STATE_DIM)
        torch.testing.assert_close(model(ids, mask, actions, numeric), old(ids, mask, actions),
                                   atol=1e-6, rtol=1e-5)
        scores = model.score([1, 2, 3], actions.numpy(), numeric[0].numpy())
        np.testing.assert_allclose(scores, old.score([1, 2, 3], actions.numpy()), atol=1e-5, rtol=1e-5)
        first = model.head[0].weight[:, 32:32 + STATE_DIM]
        self.assertEqual(first.count_nonzero().item(), 0)
        model(ids, mask, actions, numeric).sum().backward()
        self.assertGreater(model.head[0].weight.grad[:, 32:32 + STATE_DIM].abs().sum().item(), 0)

    def test_complete_future_return_is_per_seat(self):
        policies = []
        class TrackingPolicy(Policy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                policies.append(self)
        with patch('qwen3_dmc_v3.rollout.Policy', TrackingPolicy):
            samples, row = game_episode(LegalPolicy(), FakeTokenizer(), 26100200002,
                                       max_length=4096, epsilon=0)
        self.assertGreater(len(samples), 0)
        self.assertEqual(sum(row['four_seat_net']), 0)
        self.assertEqual(set(s[4] for s in samples), {0, 1, 2, 3})
        self.assertTrue(all(len(s[3]) == STATE_DIM for s in samples))
        balances = [(pid, before) for pid, records in enumerate(policies[0].records)
                    for _, _, before, _ in records]
        for (_, _, target, _, pid), (seat, before) in zip(samples, balances):
            self.assertEqual(pid, seat)
            self.assertEqual(target + (before - 10000)/10, row['four_seat_net'][pid]/10)
        self.assertTrue(any(before != 10000 for _, before in balances))
        self.assertNotEqual(row['end_reason'], 'max_steps_or_unsettled')

    def test_small_positive_and_negative_future_payoffs_survive(self):
        self.assertEqual(return_to_go(10010, 10000, 10), 1)
        self.assertEqual(return_to_go(9990, 10000, 10), -1)
        self.assertEqual(return_to_go(10050, 10020, 10), 3)
        with self.assertRaises(ValueError):
            return_to_go(10010, 10000, 0)

    def test_train_rule_samples_only_contain_p0(self):
        samples, row = game_episode(LegalPolicy(), FakeTokenizer(), 26100400003,
                                   mode='train_rules', max_length=4096, epsilon=0)
        self.assertGreater(len(samples), 0)
        self.assertEqual(set(s[4] for s in samples), {0})
        self.assertTrue(all(len(s[3]) == STATE_DIM for s in samples))
        self.assertEqual(row['sampling_mode'], 'train_rules')

    def test_popart_preserves_raw_values_and_checkpoint_scale(self):
        torch.set_num_threads(1)
        cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, head_dim=8,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
        model = QwenValue(Qwen3Model(cfg), 4, 8).eval()
        ids, mask = pad_sequences([[1, 2], [3, 4]], 0, 'cpu')
        actions = torch.zeros(2, 32)
        actions[0, 0], actions[1, 1] = 1, 1
        before = model.denormalize(model(ids, mask, actions)).detach()
        for targets in ([768, -256, 1, -1], [2, 4, -3, -10]):
            model.update_return_stats(targets)
            torch.testing.assert_close(model.denormalize(model(ids, mask, actions)), before,
                                       atol=3e-5, rtol=1e-5)
        saved = copy.deepcopy(model.weights())
        model.update_return_stats([4096, 0])
        model.load_weights(saved)
        torch.testing.assert_close(model.denormalize(model(ids, mask, actions)), before,
                                   atol=3e-5, rtol=1e-5)
        saved.pop('return_stats')
        model.load_weights(saved)
        self.assertEqual(model.return_std.item(), 1)
        self.assertEqual(model.return_mean.item(), 0)

    def test_normalized_mc_updates_backbone_without_clipping_labels(self):
        torch.set_num_threads(1)
        cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, head_dim=8,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
        model = QwenValue(Qwen3Model(cfg), 4, 8)
        adapter = [p for p in model.backbone.parameters() if p.requires_grad]
        optimizer = torch.optim.RMSprop([dict(params=adapter, lr=2e-6),
                                        dict(params=model.head.parameters(), lr=1e-4)])
        before = copy.deepcopy(model.weights())
        actions = np.zeros((4, 32), np.float32)
        actions[:, 0] = 1
        structured = np.zeros(STATE_DIM, np.float32)
        samples = [([1, 2, 3], a, target, structured, 0)
                   for a, target in zip(actions, [768, -256, 1, -1])]
        stats = regress(model, FakeTokenizer(), optimizer, samples, np.random.default_rng(1), 2, 4)
        self.assertTrue(np.isfinite(stats['mse']))
        self.assertGreater(stats['return_std'], 100)
        self.assertGreater(stats['adapter_grad_sq'], 0)
        self.assertTrue(any(not torch.equal(v, before['adapter'][k])
                            for k, v in model.weights()['adapter'].items()))
        label = torch.tensor([768., -256., 1., -1.])
        torch.testing.assert_close(model.denormalize(model.normalize(label)), label)
        square = optimizer.state[model.head[-1].weight]['square_avg'].clone()
        old_std = model.return_std.clone()
        model.update_return_stats([2048, -4096], optimizer, decay=0)
        torch.testing.assert_close(optimizer.state[model.head[-1].weight]['square_avg'],
                                   square * (old_std/model.return_std).square())

    def test_unmodified_runner_parity(self):
        from unified_tjong_eval import common_runner, kwargs, p4, stable
        seed = 26100200003
        self.assertEqual(stable(p4.play_one_game(**kwargs(seed))[0]),
                         stable(common_runner()(**kwargs(seed))[0]))


if __name__ == '__main__':
    unittest.main()
