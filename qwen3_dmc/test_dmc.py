import copy
import unittest

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3Model

from run_gate1_experiments import init_game
from prompt_builder import get_legal_actions
from .model import QwenValue, observation, pad_sequences
from .rollout import game_episode


class FakeTokenizer:
    eos_token_id = pad_token_id = 0

    def encode(self, text, **kwargs):
        return [1 + ord(c) % 100 for c in text]


class LegalPolicy:
    def score(self, ids, candidates):
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
        for p in game.players[1:]:
            p.hand_tiles = [game.players[0].hand_tiles[0]] * len(p.hand_tiles)
        game.deck.tiles.reverse()
        game.history.append(dict(pid=1, act='private_pass', tile='hidden'))
        self.assertEqual(first, observation(game, 0, actions, False))

    def test_complete_game_return_is_per_seat(self):
        samples, row = game_episode(LegalPolicy(), FakeTokenizer(), 26100200002, max_length=4096)
        self.assertGreater(len(samples), 0)
        self.assertEqual(sum(row['four_seat_net']), 0)
        self.assertEqual(set(s[3] for s in samples), {0, 1, 2, 3})
        for _, _, target, pid in samples:
            self.assertEqual(target, row['four_seat_net'][pid] / 10)
        self.assertNotEqual(row['end_reason'], 'max_steps_or_unsettled')

    def test_unmodified_runner_parity(self):
        from unified_tjong_eval import common_runner, kwargs, p4, stable
        seed = 26100200003
        self.assertEqual(stable(p4.play_one_game(**kwargs(seed))[0]),
                         stable(common_runner()(**kwargs(seed))[0]))


if __name__ == '__main__':
    unittest.main()
