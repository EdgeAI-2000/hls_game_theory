import copy
import unittest

import numpy as np
import torch

from .core import ValueNetwork, encode, episode, init_game, get_legal_actions


class MonteCarloTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.model = ValueNetwork().eval()

    def test_private_information_isolation(self):
        game, _, _ = init_game(92345, 'random', 'privacy')
        pid = game.current_player_id
        legal = get_legal_actions(game, pid)
        before = encode(game, pid, legal, [])
        other = copy.deepcopy(game)
        # Preserve observable counts; arbitrarily replace hidden tile identities.
        for i, p in enumerate(other.players):
            if i != pid:
                p.hand_tiles = [game.players[pid].hand_tiles[0]] * len(p.hand_tiles)
        other.deck.tiles.reverse()
        after = encode(other, pid, legal, [])
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_complete_returns_and_reproducibility(self):
        nonzero = False
        for seed in range(926020000000, 926020000020):
            x, z, y, metrics = episode(self.model, seed, epsilon=1)
            cursor = 0
            for pid, size in enumerate(metrics['decisions']):
                np.testing.assert_array_equal(y[cursor:cursor + size], np.full(size, metrics['net'][pid] / 10))
                cursor += size
            self.assertEqual(cursor, len(y))
            self.assertEqual(sum(metrics['net']), 0)
            nonzero |= any(metrics['net'])
        self.assertTrue(nonzero, 'No nonzero training signal in fixture games')
        a = episode(self.model, 926020000003, epsilon=1)
        b = episode(self.model, 926020000003, epsilon=1)
        self.assertEqual(a[3], b[3])
        np.testing.assert_array_equal(a[2], b[2])

    def test_regression_learns_signed_return(self):
        model = ValueNetwork()
        x, z, y, _ = episode(self.model, 926020000003, epsilon=1)
        x, z = torch.from_numpy(x[:8]), torch.from_numpy(z[:8])
        target = torch.ones(len(x)) * -1
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=.99, eps=1e-5)
        initial = float((model(x, z) - target).square().mean().detach())
        for _ in range(15):
            loss = (model(x, z) - target).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 40, error_if_nonfinite=True)
            optimizer.step()
        final = float((model(x, z) - target).square().mean().detach())
        self.assertLess(final, initial * .5)


if __name__ == '__main__':
    unittest.main()
