import copy
import random
import unittest

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3Model

from douzero_sichuan import core
from prompt_builder import get_legal_actions
from qwen3_dmc_v3.model import QwenValue
from qwen3_dmc_v3.test_dmc import FakeTokenizer, LegalPolicy
from run_gate1_experiments import init_game
from unified_baseline_eval import Policy
from .collect import PairedPolicy, paired_game
from .common import holm, teacher_values
from .pilot import relative_distribution, train_batch, values_for_rows
from .report import reference_metrics


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_teacher_scores_match_frozen_greedy_path(self):
        game = init_game(2610090001, 'random', 'test')[0]
        actions = get_legal_actions(game, 0)
        encoded = core.encode(game, 0, actions, [], False)
        teacher = core.ValueNetwork().eval()
        self.assertEqual(int(teacher_values(teacher, *encoded).argmax()),
                         teacher.choose(*encoded, random.Random(0), 0))

    def test_private_opponent_pass_never_enters_history(self):
        game = init_game(2610090002, 'random', 'test')[0]
        policy = Policy('douzero', None, 1)
        policy.record(game, 1, 'n', True)
        self.assertEqual(policy.events, [])
        policy.record(game, 0, 'n', True)
        self.assertEqual(len(policy.events), 1)
        self.assertEqual(policy.events[0], (0, 'n', None, True))

    def test_paired_scores_are_hidden_information_invariant(self):
        game = init_game(2610090003, 'random', 'test')[0]
        actions = get_legal_actions(game, 0)
        changed = copy.deepcopy(game)
        for player in changed.players[1:]:
            player.hand_tiles = [changed.players[0].hand_tiles[0]] * len(player.hand_tiles)
        changed.deck.tiles.reverse()
        changed.history.append(dict(pid=1, act='private_pass', tile='hidden'))
        teacher = core.ValueNetwork().eval()
        policies = [PairedPolicy(teacher, LegalPolicy(), FakeTokenizer(), 1, 'student', 4096) for _ in range(2)]
        a = policies[0].decide(game, 0, actions, False)[0]
        b = policies[1].decide(changed, 0, actions, False)[0]
        self.assertEqual(a, b)
        self.assertEqual(policies[0].diagnostics, policies[1].diagnostics)
        self.assertEqual(policies[0].samples[0]['observation_sha256'], policies[1].samples[0]['observation_sha256'])

    def test_full_game_targets_and_ledger(self):
        samples, diagnostic, row = paired_game(core.ValueNetwork().eval(), LegalPolicy(),
            FakeTokenizer(), 2610090004, 'student', 4096)
        self.assertGreater(len(samples), 0)
        self.assertEqual(row['game_seed'], 2610090004)
        self.assertEqual(sum(row['four_seat_net']), 0)
        self.assertNotEqual(row['end_reason'], 'max_steps_or_unsettled')
        self.assertTrue(all(len(r['actions']) >= 2 and np.isfinite(r['target']) for r in samples))
        self.assertTrue(all(0 <= r['chosen_index'] < len(r['legal']) for r in diagnostic))

    def test_relative_targets_ignore_full_vs_future_state_offset(self):
        a = relative_distribution([10., 12., 9.], 1.)
        b = relative_distribution([1010., 1012., 1009.], 1.)
        torch.testing.assert_close(a, b)
        self.assertGreater(a[1], a[0])
        with self.assertRaises(ValueError):
            relative_distribution([1., 2.], 0)
        np.testing.assert_allclose(holm([.04, .01]), [.04, .02])

    def test_historical_outcome_fields_preserve_deal_in(self):
        old = [dict(net=-10, hu=False, ron=False, dealt_in=True)]
        current = [dict(agent_net=-10, agent_hu=False, agent_hu_by_discard=False, agent_dealin=True)]
        self.assertEqual(reference_metrics(old), reference_metrics(current))
        self.assertEqual(reference_metrics(old)['dr'], 1.)

    def test_ragged_candidates_and_both_objectives_update_parameters(self):
        cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, head_dim=8,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
        initial = QwenValue(Qwen3Model(cfg), 4, 8).eval()
        rows = []
        for i, n in enumerate((2, 5)):
            actions = np.zeros((n, 32), np.float32)
            actions[:, 0] = 1
            actions[np.arange(n), np.arange(n) + 5] = 1
            rows.append(dict(ids=[1, 2, 3] if i == 0 else [4, 5],
                state=np.random.default_rng(i).normal(size=1006).astype(np.float32),
                actions=actions, teacher_q=np.arange(n, dtype=np.float32), chosen=0, target=1. if i else -1.))
        with torch.no_grad():
            batched = values_for_rows(initial, FakeTokenizer(), rows)
            self.assertEqual([len(x) for x in batched], [2, 5])
            for row, scores in zip(rows, batched):
                np.testing.assert_allclose(initial.denormalize(scores).numpy(),
                    initial.score(row['ids'], row['actions'], row['state']), atol=1e-6, rtol=1e-5)
        before = initial.weights()
        for arm in ('mc', 'rank'):
            model = copy.deepcopy(initial)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.RMSprop(parameters, lr=1e-4)
            result = train_batch(model, FakeTokenizer(), optimizer, rows, arm, 1, 1.)
            self.assertTrue(np.isfinite(result['loss']))
            self.assertGreater(result['adapter_grad_norm'], 0)
            self.assertGreater(result['head_grad_norm'], 0)
            after = model.weights()
            self.assertTrue(any(not torch.equal(before['adapter'][k], v) for k, v in after['adapter'].items()))
            self.assertTrue(any(not torch.equal(before['head'][k], v) for k, v in after['head'].items()))


if __name__ == '__main__':
    unittest.main()
