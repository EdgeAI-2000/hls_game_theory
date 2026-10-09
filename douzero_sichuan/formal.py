"""Fixed-budget training, development-only selection and paired P4 holdout."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy import stats
import torch

from .core import ValueNetwork, episode
from run_mix_ppo_formal_holdout import summarize, read_rows, METRICS

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(2026082001, 2026082021))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def holdout_worker(job):
    checkpoint, expected_hash, output, seed, count = job
    torch.set_num_threads(1)
    if sha(checkpoint) != expected_hash:
        raise RuntimeError('Checkpoint hash mismatch')
    model = ValueNetwork().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    folder = Path(output) / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'games.jsonl'
    # A unit is atomic: incomplete units can be rerun without mixing checkpoints.
    tmp = folder / 'games.jsonl.tmp'
    rows = []
    with tmp.open('w') as stream:
        for index in range(count):
            game_seed = seed * 1000000 + index
            _, _, _, m = episode(model, game_seed, 0, evaluation=True, split='holdout')
            row = dict(seed=game_seed, net=m['net'][0], ron=m['ron'][0], dealt_in=m['dr'][0],
                       hu=m['hu'][0], max_steps_or_unsettled=0, settlement_reconciled=True,
                       four_seat_metrics=m)
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
            if (index + 1) % 100 == 0:
                stream.flush()
                save(folder / 'progress.json', dict(completed=index + 1, target=count))
    os.replace(tmp, path)
    metrics = summarize(rows, seed) if count == 1000 else None
    save(folder / 'summary.json', dict(outer_seed=seed, games=count, split='holdout', metrics=metrics,
                                      checkpoint_sha256=expected_hash, games_sha256=sha(path)))
    return seed


def compare(output, checkpoint_hash):
    series = {'DouZero-adapted': []}
    sources = {}
    for seed in SEEDS:
        folder = output / f'seed_{seed}'
        metadata = json.loads((folder / 'summary.json').read_text())
        assert metadata['checkpoint_sha256'] == checkpoint_hash and metadata['split'] == 'holdout'
        path = folder / 'games.jsonl'
        assert sha(path) == metadata['games_sha256']
        series['DouZero-adapted'].append(summarize(read_rows(path), seed))
        sources[str(path)] = sha(path)
    base = ROOT / 'Formal_E2_E3_P4_STRICT_FFR_20260823/formal_e3'
    for name, variant in [('L0', 'L0'), ('L1-z', 'L1'), ('no-deceive', 'L2_no_deceive'),
                          ('no-course', 'L2_no_course'), ('MASK-full', 'L2_full')]:
        series[name] = []
        for seed in SEEDS:
            path = base / variant / f'seed_{seed}/gate1_games.jsonl'
            series[name].append(summarize(read_rows(path), seed, legacy=True))
            sources[str(path)] = sha(path)
    means = {name: {m: float(np.mean([r[m] for r in rows])) for m in METRICS} for name, rows in series.items()}
    paired = {}
    for name in series:
        if name == 'DouZero-adapted':
            continue
        paired[name] = {}
        for metric in METRICS:
            differences = np.array([a[metric] - b[metric] for a, b in zip(series['DouZero-adapted'], series[name])])
            mean, se = float(differences.mean()), float(stats.sem(differences))
            half = float(stats.t.ppf(.975, 19)) * se
            p = float(stats.ttest_1samp(differences, 0).pvalue) if se else float(mean == 0)
            paired[name][metric] = dict(difference=mean, ci95=[mean-half, mean+half], p_nominal=p)
        ordered = sorted(METRICS, key=lambda m: paired[name][m]['p_nominal'])
        previous = 0.
        for rank, metric in enumerate(ordered):
            previous = max(previous, min(1., (len(ordered)-rank) * paired[name][metric]['p_nominal']))
            paired[name][metric]['p_holm_within_comparison'] = previous
    for row in means.values():
        row['excess_DIR'] = row['ron_game_rate'] - means['no-deceive']['ron_game_rate']
    save(output / 'comparison.json', dict(means=means, by_seed=series, paired_douzero_minus_baseline=paired,
                                          sources=sources, checkpoint_sha256=checkpoint_hash, outer_seeds=SEEDS))
    lines = ['# DouZero-adapted: P4 holdout comparison', '',
             '20 paired outer seeds x 1000 games. Checkpoint selected on TRAIN development games only.', '',
             '| System | Mean | Ron rate | DR | CVaR-10% | Hu rate | Excess-DIR |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in means.items():
        lines.append(f"| {name} | {r['avg_net']:.3f} | {r['ron_game_rate']:.3%} | {r['dealin_game_rate']:.3%} | {r['cvar10']:.3f} | {r['hu_game_rate']:.3%} | {100*r['excess_DIR']:+.3f}pp |")
    lines += ['', '## DouZero minus MASK-full', '',
              '| Metric | Difference | 95% CI | Nominal p | Holm p (five metrics) |',
              '|---|---:|---|---:|---:|']
    for metric, r in paired['MASK-full'].items():
        factor = 100 if metric.endswith('rate') else 1
        lines.append(f"| {metric} | {r['difference']*factor:+.4f} | {[round(v*factor,4) for v in r['ci95']]} | {r['p_nominal']:.6g} | {r['p_holm_within_comparison']:.6g} |")
    lines += ['', 'Rates differences are percentage points. CVaR averages per-seed worst 100 games.',
              'One training seed and a fixed budget do not establish convergence or an RL performance ceiling.',
              'Excess-DIR is relative to frozen no-deceive; this alone does not causally identify deception.',
              'Strict FFR is not evaluated here. Different actions can produce different trajectories despite paired initial seeds.',
              'The local adapter masks impossible response kongs when wall<=1; this boundary differs from the legacy runner and is disclosed in the protocol.']
    (output / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='DouZero_Sichuan_Formal_20260926')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--device', default='cuda:1')
    args = p.parse_args()
    os.chdir(ROOT)
    out = Path(args.out).resolve()
    out.mkdir(exist_ok=False)
    status = out / 'status.json'
    files = [Path(__file__), ROOT/'douzero_sichuan/core.py', ROOT/'douzero_sichuan/train.py',
             ROOT/'game.py', ROOT/'run_gate1_experiments.py', ROOT/'prompt_builder.py',
             ROOT/'Opponent_Curriculum_formal/screened.json']
    hashes = {str(f): sha(f) for f in files}
    protocol = dict(training_frames=5000000, training_seed=92602, training_start='random initialization',
                    workers=args.workers, device=args.device, dev_games=1000, dev_interval=500000,
                    selection='highest mean development net; then earliest checkpoint; exclude initialization',
                    dev_split='train', dev_seed_start=926990000000,
                    outer_seeds=SEEDS, games_per_seed=1000, game_seed_formula='outer_seed*1000000+game_index',
                    holdout_selection=False, source_hashes=hashes, formal_training_seeds=1,
                    caveat='Fixed budget, not a convergence claim. Local impossible-kong mask is an explicit adapter difference.')
    save(out / 'PROTOCOL.json', protocol)
    def update(stage, **details):
        save(status, dict(stage=stage, timestamp=time.time(), **details))
    try:
        # Validate all frozen baseline units before spending the training budget.
        base = ROOT / 'Formal_E2_E3_P4_STRICT_FFR_20260823/formal_e3'
        for variant in ['L0', 'L1', 'L2_no_deceive', 'L2_no_course', 'L2_full']:
            for seed in SEEDS:
                summarize(read_rows(base / variant / f'seed_{seed}/gate1_games.jsonl'), seed, legacy=True)
        update('training', target_frames=5000000, workers=args.workers)
        train = out / 'train'
        with (out / 'training_console.log').open('w') as log:
            subprocess.run([sys.executable, '-u', '-m', 'douzero_sichuan.train', '--out', str(train),
                            '--device', args.device, '--workers', str(args.workers), '--frames', '5000000',
                            '--games-per-update', '32', '--batch-size', '3200', '--eval-every', '500000',
                            '--eval-games', '1000', '--seed', '92602'], check=True, stdout=log, stderr=subprocess.STDOUT)
        if not (train / 'PILOT_COMPLETE').exists():
            raise RuntimeError('Training did not finish')
        for path, expected in hashes.items():
            if sha(path) != expected:
                raise RuntimeError(f'Frozen source changed: {path}')
        candidates = [json.loads(f.read_text()) for f in train.glob('dev_*.json') if f.name != 'dev_0.json']
        selected = max(candidates, key=lambda r: (r['mean'], -r['frames']))
        checkpoint = out / 'selected.pt'
        shutil.copyfile(train / f"model_{selected['frames']}.pt", checkpoint)
        checkpoint_hash = sha(checkpoint)
        save(out / 'SELECTION.json', dict(frames=selected['frames'], dev_mean=selected['mean'],
                                         checkpoint_sha256=checkpoint_hash, holdout_seen=False))
        evaluation = out / 'holdout'
        evaluation.mkdir()
        update('holdout', checkpoint_sha256=checkpoint_hash, completed_seeds=0, target_games=20000)
        jobs = [(str(checkpoint), checkpoint_hash, str(evaluation), seed, 1000) for seed in SEEDS]
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
            for count, seed in enumerate(pool.map(holdout_worker, jobs), 1):
                update('holdout', completed_seeds=count, last_seed=seed, target_games=20000)
        for path, expected in hashes.items():
            if sha(path) != expected:
                raise RuntimeError(f'Frozen source changed: {path}')
        compare(evaluation, checkpoint_hash)
        update('complete', games=20000, report=str(evaluation / 'COMPARISON.md'))
    except Exception as exc:
        update('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
