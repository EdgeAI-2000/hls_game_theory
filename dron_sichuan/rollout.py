"""P0 decision-interval transitions, using the existing frozen runner semantics."""
import random
import numpy as np

from douzero_sichuan.core import init_game, get_legal_actions, action_tile, execute_action
from .model import observe, action_id


def collect(model, seed, epsilon=0, split='train', training=True):
    if split not in ('train', 'holdout'):
        raise ValueError(split)
    cfg = {'curriculum_manifest': 'Opponent_Curriculum_formal/screened.json', 'curriculum_split': split}
    game, opponents, defenders = init_game(seed, 'curriculum', f'dron_{seed}', cfg)
    initial = [p.balance for p in game.players]
    rng = random.Random(seed + 17)
    events, records = [], []
    ron, dr, decisions = [False] * 4, [False] * 4, [0] * 4
    missed = [0, 0]

    def decide(pid, acts=None):
        response = acts is not None
        legal = get_legal_actions(game, pid, acts)
        if pid == 0:
            obs = observe(game, pid, legal, events, response)
            choice = model.choose(obs, rng, epsilon)
            action = next(a for a in legal if action_id(a) == choice)
            records.append((obs, choice, game.players[0].balance))
            if 'h' in legal:
                missed[1] += 1
                missed[0] += action != 'h'
        else:
            player = game.players[pid]
            action = defenders[pid].response(player, acts, game) if response else opponents[pid][0](player, game)
            if response:
                legal = ['n'] + [a for a, name in [('h', 'hu'), ('g', 'gang'), ('p', 'peng')] if name in acts]
        if action not in legal:
            raise RuntimeError(f'Illegal action: {action}, {legal}')
        decisions[pid] += 1
        if action != 'n':
            events.append((pid, action[0], action_tile(game, pid, action, response), response))
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
            game.next_player(); skip_draw = False
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
            if game.deck.remaining_count() <= 1:
                acts = [a for a in acts if a != 'gang']
            if not acts:
                continue
            response = decide(rid, acts)
            if response == 'h':
                game.hu(rid, discarded, False, pid)
                game.check_game_over()
                ron[rid], dr[pid], responded = True, True, True
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
    ledger = [0] * 4
    for entry in game.settlement_ledger:
        ledger[entry['payer_id']] -= entry['score']
        ledger[entry['receiver_id']] += entry['score']
    if ledger != net or sum(net):
        raise RuntimeError('Settlement mismatch')
    metrics = dict(seed=seed, net=net, ron=ron, dr=dr, hu=[p.is_hu for p in game.players],
                   decisions=decisions, missed_hu=missed, steps=steps,
                   opponent_ids={str(pid): defender.curriculum_spec['id'] for pid, defender in defenders.items()},
                   settlement_ledger=game.settlement_ledger, max_steps_or_unsettled=0)
    if not training:
        return metrics
    rows = []
    for i, (obs, action, balance) in enumerate(records):
        terminal = i == len(records) - 1
        next_obs = tuple(np.zeros_like(v) for v in obs) if terminal else records[i + 1][0]
        next_balance = game.players[0].balance if terminal else records[i + 1][2]
        rows.append((obs, action, (next_balance - balance) / game.base_score, next_obs, terminal))
    if records and abs(sum(r[2] for r in rows) * game.base_score - (game.players[0].balance - records[0][2])) > 1e-6:
        raise RuntimeError('Transition rewards do not telescope')
    return rows, metrics
