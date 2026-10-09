"""Archive train-dev legacy parity, then re-evaluate frozen DouZero under P4."""
import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from douzero_sichuan import core
from unified_baseline_eval import Policy, runner
from unified_tjong_eval import kwargs, metrics, save, digest
from .common import (TEACHER_SHA, confidence, read_protocol, reconcile, seed_digest,
                     teacher_model, train_options, verify)

MODEL = None


def initialize():
    global MODEL
    MODEL = teacher_model()


def play(seed, split):
    capture = {}
    policy = Policy('douzero', MODEL, seed)
    row = runner(policy, capture)(**(train_options(seed) if split == 'train' else kwargs(seed)))[0]
    reconcile(capture['game'], capture, row)
    row.update(game_seed=seed, checkpoint_sha256=TEACHER_SHA, policy_trace=policy.decisions,
               settlement_ledger=capture['game'].settlement_ledger)
    return row


def parity_job(seed):
    current = play(seed, 'train')
    old = core.episode(MODEL, seed, epsilon=0, evaluation=True, split='train')[3]
    legacy = dict(game_seed=seed, agent_net=old['net'][0], agent_hu=old['hu'][0],
                  agent_dealin=old['dr'][0], agent_hu_by_discard=old['ron'][0],
                  four_seat_net=old['net'])
    changed = any(current[k] != legacy[k] for k in ('agent_net', 'agent_hu', 'agent_dealin', 'agent_hu_by_discard'))
    return dict(game_seed=seed, unified=current, legacy=legacy, changed_outcome=changed,
                changed_net=current['agent_net'] != legacy['agent_net'])


def unit_job(args):
    out, outer, count = args
    protocol = read_protocol(out)
    verify(protocol['hashes'])
    folder = Path(out) / 'teacher_holdout' / f'seed_{outer}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'games.jsonl'
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for i, row in enumerate(rows):
            if row['game_seed'] != outer * 1000000 + i or row['checkpoint_sha256'] != TEACHER_SHA:
                raise RuntimeError('Invalid resumable teacher unit')
        if len(rows) > count:
            raise RuntimeError('Oversized teacher unit')
    with path.open('a') as handle:
        for i in range(len(rows), count):
            row = play(outer * 1000000 + i, 'holdout')
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            rows.append(row)
            if (i + 1) % 50 == 0:
                print(f'TEACHER {outer}: {i + 1}/{count}', flush=True)
    verify(protocol['hashes'])
    assert len({r['game_seed'] for r in rows}) == count
    summary = dict(outer_seed=outer, metrics=metrics(rows), checkpoint_sha256=TEACHER_SHA,
                   seed_sha256=seed_digest([r['game_seed'] for r in rows]),
                   games_sha256=digest(path), failures=0)
    save(folder / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    protocol = read_protocol(args.out)
    verify(protocol['hashes'])
    with ProcessPoolExecutor(8, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        parity_path = args.out / 'LEGACY_PARITY_128.json'
        if not parity_path.exists():
            count = 8 if args.smoke else protocol['audit_games']
            rows = list(pool.map(parity_job, range(protocol['audit_start'], protocol['audit_start'] + count)))
            save(parity_path, dict(games=count, teacher_sha256=TEACHER_SHA,
                 changed_net_games=sum(r['changed_net'] for r in rows),
                 changed_outcome_games=sum(r['changed_outcome'] for r in rows),
                 unified=metrics([r['unified'] for r in rows]), legacy=metrics([r['legacy'] for r in rows]),
                 rows=rows, caveat='Train-dev diagnostic, not formal performance evidence; combined executor/history change.'))
            print('PARITY ' + json.dumps({k: v for k, v in json.loads(parity_path.read_text()).items() if k != 'rows'}), flush=True)
        seeds = protocol['formal_outer_seeds'][:2] if args.smoke else protocol['formal_outer_seeds']
        games = 4 if args.smoke else protocol['formal_games_per_seed']
        jobs = [pool.submit(unit_job, (str(args.out), outer, games)) for outer in seeds]
        units = sorted([j.result() for j in as_completed(jobs)], key=lambda r: r['outer_seed'])
    result = dict(status='SMOKE_COMPLETE' if args.smoke else 'COMPLETE',
                  games=len(units) * games, units=units, checkpoint_sha256=TEACHER_SHA,
                  metrics={k: float(np.mean([u['metrics'][k] for u in units])) for k in units[0]['metrics'] if k != 'games'},
                  ci95={k: confidence([u['metrics'][k] for u in units]) for k in units[0]['metrics'] if k != 'games'},
                  cvar_aggregation='mean per-outer-seed CVaR; outer seed is formal inference unit',
                  scoring='unchanged legacy P4, not a corrected real-rule benchmark')
    verify(protocol['hashes'])
    save(args.out / 'TEACHER_HOLDOUT_SUMMARY.json', result)
    print('TEACHER_COMPLETE ' + json.dumps(result['metrics']), flush=True)


if __name__ == '__main__':
    main()
