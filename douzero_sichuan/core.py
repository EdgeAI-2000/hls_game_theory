import random
from collections import Counter

import numpy as np
import torch
from torch import nn

from tiles import Suit
from prompt_builder import get_legal_actions
from run_gate1_experiments import init_game, execute_action, parse_console_tile

SUITS = list(Suit)
KINDS = ['d', 'h', 'g', 'p', 'n']
HISTORY = 32
EVENT_DIM = 37  # relative seat, action type, tile, response flag
STATE_DIM = 108 * 9 + 12 + 4 * 4 + 4 + 2
ACTION_DIM = 32


def tile_index(tile):
    return SUITS.index(tile.suit) * 9 + tile.number - 1


def tiles_vector(tiles):
    counts = Counter(tile_index(t) for t in tiles)
    out = np.zeros((4, 27), dtype=np.float32)
    for index, count in counts.items():
        out[:min(count, 4), index] = 1
    return out.ravel()


def action_tile(game, pid, action, response):
    if action.startswith('d '):
        return parse_console_tile(action[2:])
    if response and action in ('h', 'g', 'p'):
        return game.last_discarded_tile
    if action == 'g':
        info = game.can_self_gang(pid)
        return info['gang_tiles'][0] if info['can_gang'] else None
    return None


def encode(game, pid, actions, events, response=False):
    players = [game.players[(pid + i) % 4] for i in range(4)]
    parts = [tiles_vector(players[0].hand_tiles)]
    parts += [tiles_vector(p.discarded_tiles) for p in players]
    parts += [tiles_vector([t for meld in p.open_melds for t in meld]) for p in players]
    parts += [np.array([float(p.missing_suit == s) for p in players for s in SUITS], np.float32)]
    parts += [np.array([v for p in players for v in (
        len(p.hand_tiles) / 14, float(p.is_hu), (p.balance - 10000) / 1000,
        len(p.open_melds) / 4)], np.float32)]
    parts += [np.eye(4, dtype=np.float32)[(game.current_player_id - pid) % 4]]
    parts += [np.array([game.deck.remaining_count() / 108, float(response)], np.float32)]
    state = np.concatenate(parts)
    history = np.zeros((HISTORY, EVENT_DIM), np.float32)
    for row, (actor, kind, tile, is_response) in enumerate(events[-HISTORY:], HISTORY - min(len(events), HISTORY)):
        history[row, (actor - pid) % 4] = 1
        history[row, 4 + KINDS.index(kind)] = 1
        if tile is not None:
            history[row, 9 + tile_index(tile)] = 1
        history[row, -1] = float(is_response)
    candidates = np.zeros((len(actions), ACTION_DIM), np.float32)
    for i, action in enumerate(actions):
        candidates[i, KINDS.index(action[0])] = 1
        tile = action_tile(game, pid, action, response)
        if tile is not None:
            candidates[i, 5 + tile_index(tile)] = 1
    assert state.shape == (STATE_DIM,)
    return state, history, candidates


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(EVENT_DIM, 128, batch_first=True)
        layers = []
        width = STATE_DIM + ACTION_DIM + 128
        for _ in range(5):
            layers += [nn.Linear(width, 512), nn.ReLU()]
            width = 512
        self.mlp = nn.Sequential(*layers, nn.Linear(512, 1))

    def forward(self, state_action, history):
        _, (hidden, _) = self.lstm(history)
        return self.mlp(torch.cat((state_action, hidden[-1]), -1)).squeeze(-1)

    @torch.no_grad()
    def choose(self, state, history, candidates, rng, epsilon):
        if rng.random() < epsilon:
            return rng.randrange(len(candidates))
        # Encode the identical history only once for all legal actions.
        _, (hidden, _) = self.lstm(torch.from_numpy(history[None]))
        x = torch.from_numpy(np.concatenate((np.repeat(state[None], len(candidates), axis=0), candidates), 1))
        values = self.mlp(torch.cat((x, hidden[-1].expand(len(candidates), -1)), 1)).squeeze(-1)
        if not torch.isfinite(values).all():
            raise ValueError('Non-finite actor value')
        return int(values.argmax())


def episode(model, seed, epsilon=0.01, evaluation=False, split='train'):
    if split not in ('train', 'holdout'):
        raise ValueError(split)
    cfg = {'curriculum_manifest': 'Opponent_Curriculum_formal/screened.json', 'curriculum_split': split}
    game, opponents, defenders = init_game(seed, 'curriculum' if evaluation else 'random', f'dmc_{seed}', cfg)
    initial = [p.balance for p in game.players]
    rng = random.Random(seed + 17)
    records = [[] for _ in range(4)]
    events, ron, dealin = [], [False] * 4, [False] * 4
    misses = [0, 0]

    def decide(pid, response_actions=None):
        response = response_actions is not None
        legal = get_legal_actions(game, pid, response_actions)
        if evaluation and pid != 0:
            player = game.players[pid]
            action = defenders[pid].response(player, response_actions, game) if response else opponents[pid][0](player, game)
            # Existing rule opponents use engine responses, not P0's stricter
            # missing-suit candidate filter. Preserve that benchmark convention.
            if response:
                legal = ['n'] + [a for a, name in [('h', 'hu'), ('g', 'gang'), ('p', 'peng')] if name in response_actions]
        else:
            state, history, candidates = encode(game, pid, legal, events, response)
            choice = model.choose(state, history, candidates, rng, epsilon)
            action = legal[choice]
            records[pid].append((np.concatenate((state, candidates[choice])), history))
            if 'h' in legal:
                misses[1] += 1
                misses[0] += action != 'h'
        if action not in legal:
            raise ValueError(f'Illegal action {action!r}; allowed {legal}')
        tile = action_tile(game, pid, action, response)
        events.append((pid, action[0], tile, response))
        return action

    skip_draw = True
    for steps in range(1, 301):
        if game.is_game_over:
            break
        if game.deck.remaining_count() == 0 or sum(p.is_hu for p in game.players) >= 3:
            game.check_game_over()
            break
        pid = game.current_player_id
        if game.players[pid].is_hu:
            game.next_player()
            skip_draw = False
            continue
        drawn = None
        if not skip_draw:
            drawn = game.draw_tile(pid)
            if drawn is None:
                game.check_game_over()
                break
        skip_draw = False
        action = decide(pid)
        discarded = execute_action(game, pid, action, drawn)
        if game.is_game_over:
            break
        if discarded is None:
            if action == 'g':
                skip_draw = True
            else:
                game.next_player()
            continue
        responded = False
        for rid, acts in game.check_responses(discarded, pid).items():
            # The engine rejects kongs without a replacement draw. Its response
            # enumerator does not apply this constraint, unlike can_self_gang.
            if game.deck.remaining_count() <= 1:
                acts = [a for a in acts if a != 'gang']
            if not acts:
                continue
            response = decide(rid, acts)
            if response == 'h':
                game.hu(rid, discarded, False, pid)
                game.check_game_over()
                ron[rid], dealin[pid], responded = True, True, True
            elif response == 'g':
                success, _ = game.gang(rid, discarded, pid)
                if not success:
                    raise RuntimeError('Legal gang rejected')
                game.current_player_id, responded = rid, True
            elif response == 'p':
                if not game.peng(rid, discarded, pid):
                    raise RuntimeError('Legal peng rejected')
                game.current_player_id, responded = rid, True
            if responded:
                break
        if game.is_game_over:
            break
        if responded:
            skip_draw = True
        else:
            game.next_player()
    if not game.is_game_over:
        raise RuntimeError(f'Unsettled game {seed}')
    net = [p.balance - initial[i] for i, p in enumerate(game.players)]
    ledger_net = [0] * 4
    settlement = {}
    for entry in game.settlement_ledger:
        ledger_net[entry['payer_id']] -= entry['score']
        ledger_net[entry['receiver_id']] += entry['score']
        settlement[entry['kind']] = settlement.get(entry['kind'], 0) + entry['score']
    assert sum(net) == 0 and ledger_net == net, (net, ledger_net)
    xs, zs, ys = [], [], []
    for pid, trajectory in enumerate(records):
        for x, z in trajectory:
            xs.append(x)
            zs.append(z)
            ys.append(net[pid] / game.base_score)
    metrics = dict(seed=seed, net=net, ron=ron, dr=dealin, hu=[p.is_hu for p in game.players],
                   decisions=[len(r) for r in records], missed_hu=misses, steps=steps, settlement=settlement)
    return np.asarray(xs, np.float32), np.asarray(zs, np.float32), np.asarray(ys, np.float32), metrics
