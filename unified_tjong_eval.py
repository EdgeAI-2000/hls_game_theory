"""Frozen Tjong inference through the P4 execution loop; no training updates."""
import argparse
import ast
import hashlib
import inspect
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

import run_gate1_experiments as p4
from douzero_sichuan.core import encode, parse_console_tile, tile_index
from tjong_sichuan.model import MEMORY, STATE_DIM, Tjong, tensor_batch

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'Tjong_Sichuan_Formal_20260928_v2/selected.pt'
EXPECTED = 'f31afa7dfd3310c41d9010c76cd08c47a20939b33eaad3a431f0bf397b4e0412'
CONFIG = ROOT / 'Formal_E2_E3_P4_STRICT_FFR_20260823/formal_e3/L2_full/seed_2026082001/gate1_summary.json'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def macro_mask(game, pid, actions, response):
    """One network action per P4 macro; g selects P4's first legal kong tile."""
    mask = np.zeros((7, 28), bool)
    decode = {}
    for a in actions:
        if a.startswith('d '): key = (0, tile_index(parse_console_tile(a[2:])))
        elif a == 'h': key = (1, 27)
        elif a == 'n': key = (6, 27)
        elif a == 'p': key = (2, tile_index(game.last_discarded_tile))
        elif a == 'g':
            if response: key = (3, tile_index(game.last_discarded_tile))
            else:
                info = game.can_self_gang(pid)
                tile = info['gang_tiles'][0]
                kind = info['gang_types'][f'{tile.number}{tile.suit.value}']
                key = (4 if kind == 'an' else 5, tile_index(tile))
        else: raise ValueError(f'Unsupported P4 action: {a}')
        mask[key] = True
        decode[key] = a
    if not mask.any(): raise RuntimeError('Empty P4 action set')
    return mask, decode


class Policy:
    def __init__(self, model):
        self.model = model
        self.context = []
        self.previous_action, self.previous_tile = 7, 28
        self.previous_balance = None
        self.decisions = []

    def decide(self, game, pid, actions, response):
        legal, decode = macro_mask(game, pid, actions, response)
        state = encode(game, pid, actions, [], response)[0]
        balance = game.players[pid].balance
        if self.previous_balance is None: self.previous_balance = balance
        item = (state, self.previous_action, self.previous_tile,
                (balance - self.previous_balance) / game.base_score)
        seq = (self.context + [item])[-MEMORY:]
        states = np.zeros((MEMORY, STATE_DIM), np.float32)
        pa, pt = np.full(MEMORY, 7, np.int64), np.full(MEMORY, 28, np.int64)
        pr = np.zeros(MEMORY, np.float32)
        for j, (s, a, t, r) in enumerate(seq, MEMORY-len(seq)):
            states[j], pa[j], pt[j], pr[j] = s, a, t, r
        batch = tensor_batch({k: v[None] for k, v in dict(
            states=states, prev_actions=pa, prev_tiles=pt, prev_rewards=pr, legal=legal).items()})
        with torch.inference_mode():
            kind, tile, _ = self.model.choose(batch, np.random.default_rng(0), greedy=True)
        action = decode[(kind, tile)]
        assert action in actions
        self.context = (self.context + [item])[-(MEMORY-1):]
        self.previous_action, self.previous_tile, self.previous_balance = kind, tile, balance
        self.decisions.append(dict(response=response, action=action, legal=actions,
                                   observation_sha256=hashlib.sha256(state.tobytes()).hexdigest()))
        return action, dict(mode='exploit', reason='frozen Tjong P4 macro adapter')


def common_runner(policy=None, capture=None):
    # Compile only the existing function into a private namespace. Neither the
    # imported P4 module nor a concurrently running evaluator is monkey-patched.
    tree = ast.parse(inspect.getsource(p4.play_one_game))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'choose_llm_base_action':
            valid_name = node.args[2].id
            if valid_name not in ('valid_actions', 'legal_actions'): raise RuntimeError('P4 hook changed')
            node.func.id = '_policy_decision'
            node.keywords.append(ast.keyword(arg='response', value=ast.Constant(valid_name == 'valid_actions')))
            changed += 1
    if changed != 2: raise RuntimeError(f'Expected exactly two P4 hooks, found {changed}')
    namespace = dict(vars(p4))

    def decision(game, pid, actions, llm, auto_hu, response):
        if policy is None:
            return p4.choose_llm_base_action(game, pid, actions, llm, auto_hu=auto_hu)
        return policy.decide(game, pid, actions, response)

    def initialize(*args, **kwargs):
        result = p4.init_game(*args, **kwargs)
        if capture is not None:
            capture['game'] = result[0]
            capture['initial'] = [p.balance for p in result[0].players]
        return result

    namespace.update(_policy_decision=decision, init_game=initialize)
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / 'run_gate1_experiments.py'), 'exec'), namespace)
    return namespace['play_one_game']


def kwargs(seed):
    cfg = json.loads(CONFIG.read_text())
    return dict(method='llm_base', seed=seed, opponent_style='curriculum', max_steps=300,
                sample_every=0, backend='local_qwen', llm=None, belief_llm=None,
                defender_cfg=cfg['defender_cfg'], mask_cfg={}, auto_hu='off')


def stable(row):
    return {k: v for k, v in row.items() if k != 'decision_ms'}


def run_game(model, seed):
    policy, capture = Policy(model), {}
    row, _, _ = common_runner(policy, capture)(**kwargs(seed))
    game = capture['game']
    net = [p.balance - b for p, b in zip(game.players, capture['initial'])]
    ledger = [0] * 4
    for e in game.settlement_ledger:
        ledger[e['payer_id']] -= e['score']
        ledger[e['receiver_id']] += e['score']
    assert ledger == net and sum(net) == 0 and row['agent_net'] == net[0]
    assert game.is_game_over and row['end_reason'] != 'max_steps_or_unsettled'
    assert row['settlement_reconciled']
    row.update(checkpoint_sha256=EXPECTED, evaluator='p4_macro_v1',
               four_seat_net=net, settlement_ledger=game.settlement_ledger,
               policy_trace=policy.decisions)
    return row


def metrics(rows):
    scores = sorted(r['agent_net'] for r in rows)
    return dict(games=len(rows), avg_net=float(np.mean(scores)),
                ron=float(np.mean([r['agent_hu_by_discard'] for r in rows])),
                dr=float(np.mean([r['agent_dealin'] for r in rows])),
                hu=float(np.mean([r['agent_hu'] for r in rows])),
                cvar10=float(np.mean(scores[:max(1, len(scores)//10)])))


def load_model():
    torch.set_num_threads(1)
    assert digest(CHECKPOINT) == EXPECTED, 'Frozen checkpoint changed'
    model = Tjong().eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=True))
    return model


def verify_sources(out):
    protocol = json.loads((Path(out)/'PROTOCOL.json').read_text())
    for name, sha in protocol['sources'].items():
        if digest(ROOT/name) != sha: raise RuntimeError(f'Frozen evaluation source changed: {name}')


def unit(out, seed, games, pilot):
    verify_sources(out)
    model = load_model()
    folder = Path(out)/('pilot' if pilot else 'holdout')/f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/'games.jsonl'
    if target.exists(): raise RuntimeError(f'Refusing to overwrite {target}')
    rows, legacy, parity = [], [], []
    with target.open('x') as f:
        for i in range(games):
            gs = seed*1000000+i
            if pilot:
                native = p4.play_one_game(**kwargs(gs))[0]
                wrapped = common_runner()(**kwargs(gs))[0]
                assert stable(native) == stable(wrapped), f'P4 delegation parity failed: {gs}'
                parity.append(gs)
                from tjong_sichuan.rollout import collect
                old = collect(model, gs, mode='eval', split='holdout')
                legacy.append(dict(seed=gs, agent_net=old['net'][0], agent_hu_by_discard=old['ron'][0],
                                   agent_dealin=old['dr'][0], agent_hu=old['hu'][0]))
            row = run_game(model, gs)
            if pilot and i == 0:
                assert stable(row) == stable(run_game(model, gs)), 'Tjong deterministic replay failed'
            f.write(json.dumps(row)+'\n'); f.flush()
            rows.append(row)
            if (i+1)%10 == 0: print(f'{seed}: {i+1}/{games}', flush=True)
    verify_sources(out)
    summary = dict(seed=seed, metrics=metrics(rows), checkpoint_sha256=EXPECTED,
                   games_sha256=digest(target), parity_games=parity)
    if pilot:
        summary.update(legacy=metrics(legacy), paired_legacy=legacy,
                       changed_net_games=sum(a['agent_net'] != b['agent_net'] for a,b in zip(rows,legacy)))
    save(folder/'summary.json', summary)
    return summary


def phase(out, seeds, games, workers, pilot):
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        jobs = [pool.submit(unit, str(out), seed, games, pilot) for seed in seeds]
        for job in as_completed(jobs):
            results.append(job.result())
    results.sort(key=lambda r:r['seed'])
    save(out/('PILOT_PASSED.json' if pilot else 'FORMAL_COMPLETE.json'), results)
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--continue-formal', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.continue_formal:
        assert (args.output/'PILOT_PASSED.json').exists()
        verify_sources(args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        previous = json.loads((CHECKPOINT.parent/'PROTOCOL.json').read_text())
        names = set(previous['hashes']) | {str(CONFIG.relative_to(ROOT)), 'unified_tjong_eval.py'}
        sources = {name:digest(ROOT/name) for name in sorted(names)}
        save(args.output/'PROTOCOL.json', dict(checkpoint=str(CHECKPOINT), checkpoint_sha256=EXPECTED,
             sources=sources, workers=args.workers, auto_hu='off', engine='unchanged legacy P4',
             mapping='P4 macro g: first engine gang tile; no extra wall mask; native opponents',
             pilot_seeds=[2026082001,2026082002], pilot_games_per_seed=10,
             formal_seeds=list(range(2026082001,2026082021)), formal_games_per_seed=1000,
             gate='exact native/delegated P4 parity; deterministic Tjong; settled zero-sum reconciled ledger',
             selection='frozen development-selected checkpoint; no holdout tuning'))
        phase(args.output, [2026082001,2026082002], 10, args.workers, True)
    if args.formal or args.continue_formal:
        phase(args.output, list(range(2026082001,2026082021)), 1000, args.workers, False)


if __name__ == '__main__':
    main()
