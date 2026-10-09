import numpy as np
import torch
from torch import nn

from douzero_sichuan.core import encode, tile_index, parse_console_tile, STATE_DIM

OPP_DIM = 3 * 8 * 31
N_ACTIONS = 31


def action_id(action):
    return tile_index(parse_console_tile(action[2:])) if action.startswith('d ') else 27 + ['h', 'g', 'p', 'n'].index(action)


def observe(game, pid, legal, events, response=False):
    # Filter before truncating: private pass opportunities must not change the window.
    public = [e for e in events if e[1] in ('d', 'h', 'g', 'p')]
    state = encode(game, pid, legal, public, response)[0]
    history = np.zeros((3, 8, 31), np.uint8)
    for relative in range(1, 4):
        rows = [e for e in public if e[0] == (pid + relative) % 4][-8:]
        for i, (_, kind, tile, _) in enumerate(rows, 8 - len(rows)):
            history[relative - 1, i, ['d', 'h', 'g', 'p'].index(kind)] = 1
            if tile is not None:
                history[relative - 1, i, 4 + tile_index(tile)] = 1
    mask = np.zeros(N_ACTIONS, bool)
    for action in legal:
        mask[action_id(action)] = True
    return state, history.ravel(), mask


class DRON(nn.Module):
    def __init__(self):
        super().__init__()
        # Author SoccerONeuralQLearner.createNetwork_moe: shared trunk, linear
        # experts, and state+opponent-conditioned softmax weights.
        self.trunk = nn.Sequential(nn.Linear(STATE_DIM, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU())
        self.experts = nn.Linear(50, 3 * N_ACTIONS)
        self.gater = nn.Sequential(nn.Linear(STATE_DIM + OPP_DIM, 50), nn.ReLU(), nn.Linear(50, 3))

    def forward(self, state, opponent):
        weights = self.gater(torch.cat((state, opponent), -1)).softmax(-1)
        experts = self.experts(self.trunk(state)).reshape(-1, 3, N_ACTIONS)
        return (weights.unsqueeze(-1) * experts).sum(1)

    @torch.no_grad()
    def choose(self, observation, rng, epsilon):
        state, opponent, mask = observation
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise RuntimeError('No legal decision')
        if rng.random() < epsilon:
            return int(rng.choice(legal))
        values = self(torch.from_numpy(state[None]), torch.from_numpy(opponent[None]).float())[0]
        if not torch.isfinite(values).all():
            raise RuntimeError('Nonfinite Q values')
        return int(values.masked_fill(~torch.from_numpy(mask), -torch.inf).argmax())


def td_target(reward, done, next_q, next_mask, gamma=1.0):
    if ((~done) & (~next_mask.any(-1))).any():
        raise RuntimeError('Nonterminal transition without legal actions')
    maximum = next_q.masked_fill(~next_mask, -torch.inf).max(-1).values
    # Avoid 0 * -inf for terminal states with an empty action mask.
    return reward + gamma * torch.where(done, torch.zeros_like(maximum), maximum)
