"""DouZero, DouZero+ and DRON adapters for the unchanged P4 game executor."""
import argparse
import json
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

import unified_tjong_eval as shared
from douzero_sichuan import core

CHECKPOINTS = {
    'douzero': 'DouZero_Sichuan_Formal_20260926/selected.pt',
    'douzero_plus': 'DouZeroPlus_Sichuan_Formal_20260927/selected.pt',
    'dron': 'DRON_MoE_Sichuan_Extended_20260928/selected.pt',
}


def load_model(name, expected):
    torch.set_num_threads(1)
    path = shared.ROOT / CHECKPOINTS[name]
    assert shared.digest(path) == expected
    data = torch.load(path, map_location='cpu', weights_only=True)
    if name == 'douzero': model = core.ValueNetwork()
    elif name == 'douzero_plus':
        from douzero_plus_sichuan.models import PlusPolicy
        model = PlusPolicy()
        data = data['policy']
    else:
        from dron_sichuan.model import DRON
        model = DRON()
    model.load_state_dict(data)
    return model.eval()


class Policy:
    def __init__(self, name, model, seed):
        self.name, self.model = name, model
        self.events, self.decisions = [], []
        self.rng = random.Random(seed+17)

    def record(self, game, pid, action, response):
        # An opponent's declined private response opportunity is not public.
        if pid != 0 and action == 'n': return
        if self.name == 'dron' and action == 'n': return
        self.events.append((pid, action[0], core.action_tile(game, pid, action, response), response))

    def decide(self, game, pid, actions, response):
        if self.name == 'dron':
            from dron_sichuan.model import observe, action_id
            obs = observe(game, pid, actions, self.events, response)
            choice = self.model.choose(obs, self.rng, 0)
            action = next(a for a in actions if action_id(a) == choice)
        else:
            obs = core.encode(game, pid, actions, self.events, response)
            action = actions[self.model.choose(*obs, self.rng, 0)]
        assert action in actions
        self.decisions.append(dict(action=action, response=response, legal=list(actions)))
        if response: self.record(game, pid, action, True)
        return action, dict(mode='exploit', reason=f'frozen {self.name} P4 adapter')


def runner(policy, capture):
    fn = shared.common_runner(policy, capture)
    ns = fn.__globals__
    initialize, execute = ns['init_game'], ns['execute_action']

    def init(*args, **kwargs):
        game, opponents, defenders = initialize(*args, **kwargs)
        for pid, defender in defenders.items():
            response = defender.response
            def wrapped(player, acts, g, original=response, actor=pid):
                action = original(player, acts, g)
                policy.record(g, actor, action, True)
                return action
            defender.response = wrapped
        return game, opponents, defenders

    def act(game, pid, action, drawn_tile=None):
        policy.record(game, pid, action, False)
        return execute(game, pid, action, drawn_tile)

    ns.update(init_game=init, execute_action=act)
    return fn


def game_row(name, model, seed, sha):
    capture, policy = {}, Policy(name, model, seed)
    row = runner(policy, capture)(**shared.kwargs(seed))[0]
    game = capture['game']
    net = [p.balance-b for p,b in zip(game.players,capture['initial'])]
    ledger = [0]*4
    for e in game.settlement_ledger:
        ledger[e['payer_id']] -= e['score']; ledger[e['receiver_id']] += e['score']
    assert ledger == net and sum(net) == 0 and net[0] == row['agent_net']
    assert game.is_game_over and row['end_reason'] != 'max_steps_or_unsettled'
    assert row['settlement_reconciled']
    row.update(checkpoint_sha256=sha, evaluator='p4_macro_v1', policy_name=name,
               four_seat_net=net, settlement_ledger=game.settlement_ledger, policy_trace=policy.decisions)
    return row


def legacy(name, model, seed):
    if name == 'dron':
        from dron_sichuan.rollout import collect
        result = collect(model, seed, epsilon=0, split='holdout', training=False)
        return result
    return core.episode(model, seed, epsilon=0, evaluation=True, split='holdout')[3]


def unit(out, name, seed, games, pilot):
    shared.verify_sources(out)
    protocol = json.loads((Path(out)/'PROTOCOL.json').read_text())
    sha = protocol['checkpoints'][name]['sha256']
    model = load_model(name, sha)
    folder = Path(out)/name/('pilot' if pilot else 'holdout')/f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    rows, old_rows = [], []
    with (folder/'games.jsonl').open('x') as f:
        for i in range(games):
            gs = seed*1000000+i
            row = game_row(name, model, gs, sha)
            if pilot:
                old = legacy(name, model, gs)
                old_rows.append(dict(seed=gs, agent_net=old['net'][0], agent_dealin=old['dr'][0],
                                     agent_hu_by_discard=old['ron'][0], agent_hu=old['hu'][0]))
                if i == 0:
                    assert shared.stable(row) == shared.stable(game_row(name, model, gs, sha))
            rows.append(row)
            f.write(json.dumps(row)+'\n'); f.flush()
            if (i+1)%10 == 0: print(f'{name}/{seed}: {i+1}/{games}', flush=True)
    shared.verify_sources(out)
    result = dict(name=name, seed=seed, metrics=shared.metrics(rows), checkpoint_sha256=sha,
                  games_sha256=shared.digest(folder/'games.jsonl'))
    if pilot:
        result.update(legacy=shared.metrics(old_rows), paired_legacy=old_rows,
                      changed_net_games=sum(a['agent_net']!=b['agent_net'] for a,b in zip(rows,old_rows)))
    shared.save(folder/'summary.json', result)
    return result


def phase(out, pilot, workers):
    seeds = [2026082001,2026082002] if pilot else range(2026082001,2026082021)
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        jobs = [pool.submit(unit,str(out),name,seed,10 if pilot else 1000,pilot)
                for name in CHECKPOINTS for seed in seeds]
        for job in as_completed(jobs): results.append(job.result())
    shared.save(out/('PILOT_PASSED.json' if pilot else 'FORMAL_COMPLETE.json'),
                sorted(results,key=lambda r:(r['name'],r['seed'])))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--continue-formal',action='store_true')
    args = parser.parse_args()
    os.chdir(shared.ROOT)
    if args.continue_formal:
        assert (args.output/'PILOT_PASSED.json').exists()
        shared.verify_sources(args.output)
        phase(args.output,False,args.workers)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    protocol = json.loads((shared.ROOT/'Tjong_Unified_P4_20260929/PROTOCOL.json').read_text())
    files = set(protocol['sources']) | {'unified_baseline_eval.py'}
    for package in ('douzero_sichuan','douzero_plus_sichuan','dron_sichuan'):
        files.update(str(f.relative_to(shared.ROOT)) for f in (shared.ROOT/package).glob('*.py'))
    protocol['sources'] = {name:shared.digest(shared.ROOT/name) for name in sorted(files)}
    protocol.pop('checkpoint'); protocol.pop('checkpoint_sha256')
    protocol['checkpoints'] = {name:dict(path=path,sha256=shared.digest(shared.ROOT/path)) for name,path in CHECKPOINTS.items()}
    protocol['scope'] = '60,000 fresh games, three frozen baselines; DRON selected after extended training'
    protocol['history'] = 'No opponent private pass events; DouZero retains own pass; DRON public events only. Observation repair, no retraining.'
    shared.save(args.output/'PROTOCOL.json',protocol)
    phase(args.output,True,args.workers)


if __name__ == '__main__':
    main()
