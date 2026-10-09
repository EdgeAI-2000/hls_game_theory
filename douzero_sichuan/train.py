"""Parallel complete-episode Monte Carlo regression; no bootstrapped targets."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .core import ValueNetwork, episode

_actor = None
_version = None


def actor_job(job):
    global _actor, _version
    path, version, seed, evaluation = job
    if _actor is None:
        torch.set_num_threads(1)
        _actor = ValueNetwork().eval()
    if _version != (path, version):
        _actor.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        _version = (path, version)
    result = episode(_actor, seed, 0 if evaluation else 0.01, evaluation)
    return result[3] if evaluation else result


def atomic_save(data, path):
    tmp = str(path) + '.tmp'
    torch.save(data, tmp)
    os.replace(tmp, path)


def evaluate(pool, snapshot, version, frames, count, out):
    rows = list(pool.map(actor_job, [(str(snapshot), version, 926990000000 + i, True) for i in range(count)]))
    scores = np.array([r['net'][0] for r in rows])
    report = dict(frames=frames, games=len(rows), mean=float(scores.mean()),
                  cvar10=float(np.sort(scores)[:max(1, len(rows)//10)].mean()),
                  ron=float(np.mean([r['ron'][0] for r in rows])), dr=float(np.mean([r['dr'][0] for r in rows])),
                  hu=float(np.mean([r['hu'][0] for r in rows])), rows=rows)
    (out / f'dev_{frames}.json').write_text(json.dumps(report, indent=2))
    print('DEV ' + json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--frames', type=int, default=300000)
    parser.add_argument('--games-per-update', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=3200)
    parser.add_argument('--eval-every', type=int, default=50000)
    parser.add_argument('--eval-games', type=int, default=100)
    parser.add_argument('--seed', type=int, default=92601)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists() and not args.resume:
        raise RuntimeError('Existing experiment: use --resume or a fresh output directory')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = ValueNetwork().to(args.device)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=.99, eps=1e-5, momentum=0)
    frames = games = update = 0
    checkpoint = out / 'latest.pt'
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        Path(__file__), Path(__file__).with_name('core.py'), Path('game.py'), Path('run_gate1_experiments.py'), Path('prompt_builder.py')]}
    if args.resume:
        saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
        if saved['hashes'] != hashes:
            raise RuntimeError('Source changed since checkpoint')
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        frames, games, update = saved['frames'], saved['games'], saved['update']
        rng.bit_generator.state = saved['rng']
    else:
        protocol = dict(vars(args), hashes=hashes, algorithm='DouZero-adapted, full-return MC MSE',
                        architecture='shared four-seat LSTM128 + five FC512 layers',
                        reward='own final net / base_score; gamma=1; no shaping or clipping',
                        epsilon=.01, optimizer='RMSprop lr1e-4 alpha.99 eps1e-5 grad_clip40',
                        official_commit='718a5c920bf3361e34178a38f3b80458e176b351',
                        adaptation='synchronous parallel full-game actors; fresh data one regression pass; fixed heuristic exchange/dingque',
                        training_seed_start=args.seed * 10000000, dev_seed_start=926990000000,
                        evaluation='train curriculum only, fixed dev seeds; no formal holdout')
        (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    snapshot = out / 'actor.pt'
    start, initial_frames = time.monotonic(), frames
    next_eval = (frames // args.eval_every + 1) * args.eval_every
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        if frames == 0:
            atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()}, snapshot)
            atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / 'model_0.pt')
            evaluate(pool, snapshot, update, 0, args.eval_games, out)
        while frames < args.frames:
            atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()}, snapshot)
            jobs = [(str(snapshot), update, args.seed * 10000000 + games + i, False) for i in range(args.games_per_update)]
            batch = list(pool.map(actor_job, jobs))
            x, z, y = [np.concatenate([result[i] for result in batch]) for i in range(3)]
            losses = []
            model.train()
            order = rng.permutation(len(y))
            for offset in range(0, len(y), args.batch_size):
                idx = order[offset:offset + args.batch_size]
                prediction = model(torch.from_numpy(x[idx]).to(args.device), torch.from_numpy(z[idx]).to(args.device))
                target = torch.from_numpy(y[idx]).to(args.device)
                loss = (prediction - target).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss')
                opt.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 40, error_if_nonfinite=True)
                opt.step()
                losses.append(float(loss.detach()))
            frames += len(y)
            games += len(batch)
            update += 1
            summary = dict(update=update, frames=frames, games=games, loss=float(np.mean(losses)),
                           grad_norm=float(norm), target_mean=float(y.mean()), target_abs_max=float(abs(y).max()),
                           frames_per_second=(frames - initial_frames) / (time.monotonic() - start),
                           hu_rate=float(np.mean([r[3]['hu'] for r in batch])),
                           missed_hu=sum(r[3]['missed_hu'][0] for r in batch), hu_opportunities=sum(r[3]['missed_hu'][1] for r in batch))
            print(json.dumps(summary), flush=True)
            with (out / 'training.jsonl').open('a') as f:
                f.write(json.dumps(summary) + '\n')
            with (out / 'games.jsonl').open('a') as f:
                for result in batch:
                    f.write(json.dumps(dict(result[3], policy_version=update - 1)) + '\n')
            atomic_save(dict(model=model.state_dict(), optimizer=opt.state_dict(), frames=frames, games=games,
                             update=update, rng=rng.bit_generator.state, hashes=hashes), checkpoint)
            if frames >= next_eval or frames >= args.frames:
                atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()}, snapshot)
                evaluate(pool, snapshot, update, frames, args.eval_games, out)
                atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / f'model_{frames}.pt')
                next_eval += args.eval_every
    (out / 'PILOT_COMPLETE').write_text(f'{frames} decision transitions, {games} self-play games\n')


if __name__ == '__main__':
    main()
