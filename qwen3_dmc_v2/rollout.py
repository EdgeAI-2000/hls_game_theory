import random

import numpy as np

import run_gate1_experiments as p4
from prompt_builder import get_legal_actions
from unified_tjong_eval import common_runner, kwargs
from .model import tokens, action_vectors


def return_to_go(final_balance, decision_balance, base_score):
    if base_score <= 0:
        raise ValueError('base_score must be positive')
    return (final_balance - decision_balance) / base_score


class Policy:
    def __init__(self, model, tokenizer, seed, epsilon=0, max_length=1024):
        self.model, self.tokenizer = model, tokenizer
        self.rng, self.epsilon, self.max_length = random.Random(seed + 17), epsilon, max_length
        self.records = [[] for _ in range(4)]
        self.hu_available = self.missed_hu = 0

    def decide(self, game, pid, actions, response):
        ids = tokens(self.tokenizer, game, pid, actions, response, self.max_length)
        candidates = action_vectors(game, pid, actions, response)
        if len(actions) == 1:
            choice = 0
        elif self.rng.random() < self.epsilon:
            choice = self.rng.randrange(len(actions))
        else:
            choice = int(self.model.score(ids, candidates).argmax())
        action = actions[choice]
        assert action in actions
        self.records[pid].append((ids, candidates[choice].tolist(), game.players[pid].balance))
        self.hu_available += 'h' in actions
        self.missed_hu += 'h' in actions and action != 'h'
        return action, dict(mode='exploit', reason='Qwen complete-return MC action value')


def game_episode(model, tokenizer, seed, mode='selfplay', max_length=1024, epsilon=.05):
    if mode not in ('selfplay', 'train_rules', 'dev', 'holdout'):
        raise ValueError(mode)
    policy = Policy(model, tokenizer, seed, epsilon if mode in ('selfplay', 'train_rules') else 0, max_length)
    capture = {}
    runner = common_runner(policy, capture)
    options = kwargs(seed)
    if mode in ('dev', 'train_rules'):
        options['defender_cfg']['curriculum_split'] = 'train'
        options['defender_cfg'].pop('curriculum_specs_by_pid', None)
        options['defender_cfg'].pop('curriculum_spec_id', None)
    if mode == 'selfplay':
        initialize = runner.__globals__['init_game']

        def init(*args, **kw):
            game, _, _ = initialize(*args, **kw)
            opponents = {pid: (lambda player, g: policy.decide(
                g, player.player_id, get_legal_actions(g, player.player_id), False)[0], None)
                         for pid in (1, 2, 3)}
            return game, opponents, {}

        class Responder:
            def response(self, player, actions, game):
                legal = get_legal_actions(game, player.player_id, actions)
                return policy.decide(game, player.player_id, legal, True)[0]

        def resolve(game, tile, pid, agent, defenders=None, p0_response_decider=None):
            return p4.resolve_responses(game, tile, pid, agent,
                                        {i: Responder() for i in (1, 2, 3)}, p0_response_decider)

        runner.__globals__.update(init_game=init, resolve_responses=resolve)
        options['opponent_style'] = 'random'
    row, _, _ = runner(**options)
    game = capture['game']
    net = [p.balance - b for p, b in zip(game.players, capture['initial'])]
    ledger = [0] * 4
    for event in game.settlement_ledger:
        ledger[event['payer_id']] -= event['score']
        ledger[event['receiver_id']] += event['score']
    if ledger != net or sum(net) != 0 or row['agent_net'] != net[0]:
        raise RuntimeError(f'Ledger reconciliation failed: {seed}')
    if not game.is_game_over or row['end_reason'] == 'max_steps_or_unsettled':
        raise RuntimeError(f'Unsettled game: {seed}')
    samples = [(ids, a, return_to_go(game.players[pid].balance, balance_before, game.base_score), pid)
               for pid, records in enumerate(policy.records) for ids, a, balance_before in records]
    row.update(four_seat_net=net, mc_samples=len(samples),
               qwen_hu_available=policy.hu_available, qwen_missed_hu=policy.missed_hu,
               evaluator='p4_macro_v1', game_seed=seed, sampling_mode=mode)
    row['decisions_with_past_settlement'] = sum(balance_before != capture['initial'][pid]
        for pid, records in enumerate(policy.records) for _, _, balance_before in records)
    return samples, row
