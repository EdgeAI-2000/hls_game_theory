"""Parallel actor collection, replay TD learning and frozen holdout evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
from scipy import stats
import torch
from torch.nn import functional as F

from douzero_sichuan.train import atomic_save
from douzero_sichuan.formal import SEEDS, summarize, read_rows, METRICS
from .model import DRON, STATE_DIM, OPP_DIM, N_ACTIONS, td_target
from .rollout import collect

ROOT = Path(__file__).resolve().parents[1]
_actor = _version = None


def save(path, obj):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_frozen(source, destination):
    temporary = Path(str(destination) + '.tmp')
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def worker(job):
    global _actor, _version
    path, version, seed, epsilon, split, training = job
    if _actor is None:
        torch.set_num_threads(1)
        _actor = DRON().eval()
    if _version != (path, version):
        _actor.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        _version = (path, version)
    return collect(_actor, seed, epsilon, split, training)


class Replay:
    def __init__(self, capacity):
        self.capacity, self.n, self.position = capacity, 0, 0
        specs = {'s': (STATE_DIM, np.float32), 'h': (OPP_DIM, np.uint8),
                 'ns': (STATE_DIM, np.float32), 'nh': (OPP_DIM, np.uint8),
                 'mask': (N_ACTIONS, bool), 'a': (None, np.int64),
                 'r': (None, np.float32), 'done': (None, bool)}
        self.arrays = {k: np.zeros((capacity,) if dim is None else (capacity, dim), dtype=dtype)
                       for k, (dim, dtype) in specs.items()}

    def add(self, rows):
        for (s, h, _), a, r, (ns, nh, mask), done in rows:
            for key, value in dict(s=s, h=h, a=a, r=r, ns=ns, nh=nh, mask=mask, done=done).items():
                self.arrays[key][self.position] = value
            self.position = (self.position + 1) % self.capacity
            self.n = min(self.capacity, self.n + 1)

    def sample(self, rng, device):
        idx = rng.integers(self.n, size=64)
        data = {k: torch.from_numpy(v[idx]).to(device) for k, v in self.arrays.items()}
        data['h'], data['nh'] = data['h'].float(), data['nh'].float()
        return data

    def state(self):
        return dict(n=self.n, position=self.position, arrays={k: v[:self.n].copy() for k, v in self.arrays.items()})

    def restore(self, data):
        self.n, self.position = data['n'], data['position']
        for k, v in data['arrays'].items():
            self.arrays[k][:self.n] = v


def measures(rows):
    scores = np.array([r['net'][0] for r in rows])
    return dict(avg_net=float(scores.mean()), cvar10=float(np.sort(scores)[:max(1, len(scores)//10)].mean()),
                ron_game_rate=float(np.mean([r['ron'][0] for r in rows])),
                dealin_game_rate=float(np.mean([r['dr'][0] for r in rows])),
                hu_game_rate=float(np.mean([r['hu'][0] for r in rows])))


def compare(out, dron):
    series = {'DRON-MoE-adapted': dron}
    base = ROOT / 'Formal_E2_E3_P4_STRICT_FFR_20260823/formal_e3'
    for name, variant in [('MASK-full', 'L2_full'), ('no-deceive', 'L2_no_deceive'), ('L0', 'L0')]:
        series[name] = [summarize(read_rows(base / variant / f'seed_{s}/gate1_games.jsonl'), s, True) for s in SEEDS]
    series['DouZero-adapted'] = [summarize(read_rows(ROOT / 'DouZero_Sichuan_Formal_20260926/holdout' / f'seed_{s}/games.jsonl'), s) for s in SEEDS]
    means = {name: {m: float(np.mean([r[m] for r in rows])) for m in METRICS} for name, rows in series.items()}
    paired = {}
    for name, rows in list(series.items())[1:]:
        paired[name] = {}
        for m in METRICS:
            d = np.array([a[m] - b[m] for a, b in zip(dron, rows)])
            se = float(stats.sem(d)); mean = float(d.mean())
            half = float(stats.t.ppf(.975, 19)) * se
            p = float(stats.ttest_1samp(d, 0).pvalue) if se else float(mean == 0)
            paired[name][m] = dict(delta=mean, ci95=[mean-half, mean+half], p_nominal=p)
        running = 0.
        for rank, m in enumerate(sorted(METRICS, key=lambda x: paired[name][x]['p_nominal'])):
            running = max(running, min(1., (len(METRICS)-rank)*paired[name][m]['p_nominal']))
            paired[name][m]['p_holm'] = running
    save(out/'comparison.json', dict(by_seed=series, means=means, paired_dron_minus_baseline=paired))
    lines = ['# DRON-MoE-adapted: legacy-environment comparison', '',
             'One training seed and fixed budget; no convergence claim. Known engine/runner issues remain unresolved.',
             'This is not a rules-validated ranking. Training resources differ across method families.', '',
             '| System | Mean | Ron | DR | CVaR-10% | Hu | Excess-DIR vs no-deceive |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in means.items():
        excess = 100 * (r['ron_game_rate'] - means['no-deceive']['ron_game_rate'])
        lines.append(f"| {name} | {r['avg_net']:.4f} | {r['ron_game_rate']:.3%} | {r['dealin_game_rate']:.3%} | {r['cvar10']:.4f} | {r['hu_game_rate']:.3%} | {excess:+.3f}pp |")
    lines += ['', 'Paired seed-level confidence intervals and nominal/Holm p-values: comparison.json.',
              'FFR is not evaluated; excess-DIR alone is not evidence of deception.']
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='DRON_MoE_Sichuan_Formal_20260927')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--frames', type=int, default=1000000)
    p.add_argument('--dev-every', type=int, default=100000)
    p.add_argument('--dev-games', type=int, default=1000)
    p.add_argument('--holdout-games', type=int, default=1000)
    p.add_argument('--units', type=int, default=20)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not args.smoke and not args.resume and (args.frames, args.dev_every, args.dev_games, args.holdout_games, args.units) != (1000000,100000,1000,1000,20):
        raise ValueError('Budget changes need a new protocol; integration tests use --smoke')
    os.chdir(ROOT)
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(2); torch.manual_seed(92731)
    rng = np.random.default_rng(92731)
    files = list((ROOT/'dron_sichuan').glob('*.*')) + [ROOT / s for s in (
        'douzero_sichuan/core.py', 'douzero_sichuan/train.py', 'douzero_sichuan/formal.py',
        'game.py', 'rule_engine.py', 'tiles.py', 'prompt_builder.py', 'run_gate1_experiments.py',
        'belief_oracle.py', 'public_belief_model.py', 'policy_metrics.py', 'mask_llm.py',
        'ppo_features.py', 'experiment_trace.py', 'llm_backends.py',
        'opponent_curriculum.py', 'run_mix_ppo_formal_holdout.py', 'Opponent_Curriculum_formal/screened.json')]
    hashes = {str(f.relative_to(ROOT)): digest(f) for f in files}
    config = {k: v for k,v in vars(args).items() if k != 'resume'}
    protocol = dict(config=config, hashes=hashes, author_commit='5ab544ca150013cbe5549227107eed64b9928da8',
        software=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda),
        architecture='Author soccer MoE: hidden50, shared two-layer trunk, 3 linear experts, state+opponent gater',
        train_seed=92731, train_seed_start=927310000000, dev_seed_start=927390000000,
        outer_seeds=SEEDS if not args.smoke else [9273800001+i for i in range(args.units)],
        training='P0 versus six frozen train opponents; no self-play and no Qwen initialization',
        selection='maximum raw mean on fixed TRAIN development seeds; earliest tie; never holdout',
        optimizer='RMSprop lr0.0005 alpha0.95 eps0.01; Huber delta1; grad norm10',
        replay_capacity=100000, batch=64, warmup=1000, updates_per_transition=.25,
        target_copy='every 10000 collected P0 transitions (batch boundary)',
        epsilon='linear 0.3 to 0.1 over 500000 P0 transitions',
        gamma=1.0, reward='balance delta/base_score between P0 decisions; final interval through full settlement',
        adaptations=['multiplayer public histories, 8 observed actions per opponent',
                     '31 macroactions and legal TD mask, first legal kong tile as existing executor',
                     'finite-episode gamma1, linear reward and PyTorch optimizer',
                     'batched 8 CPU actors and one GPU learner'],
        environment='frozen legacy rules with known unresolved scoring/runner issues; not a rules-validated ranking',
        budget=f'{args.frames} P0 decisions; not equivalent to 5000000 four-seat DouZero samples; one train seed')
    model, target = DRON().to(args.device), DRON().to(args.device)
    target.load_state_dict(model.state_dict()); target.eval()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=.0005, alpha=.95, eps=.01)
    replay = Replay(100000 if not args.smoke else 2000)
    frames = games = updates = batches = 0
    next_dev, next_target, best = args.dev_every, 10000, None
    latest, actor = out/'latest.pt', out/'actor.pt'

    def snapshot(path):
        atomic_save({k: v.detach().cpu() for k,v in model.state_dict().items()}, path)

    def check_sources():
        if any(digest(ROOT/f) != h for f,h in hashes.items()):
            raise RuntimeError('Frozen source changed; stopped rather than mixing implementations')

    def status(stage, **extra):
        save(out/'status.json', dict(stage=stage, frames=frames, games=games, updates=updates,
                                    target_frames=args.frames, updated_at=time.time(), **extra))

    def checkpoint():
        sizes = {n: (out/n).stat().st_size if (out/n).exists() else 0 for n in ('training.jsonl','games.jsonl')}
        atomic_save(dict(model=model.state_dict(), target=target.state_dict(), optimizer=optimizer.state_dict(),
            replay=replay.state(), frames=frames, games=games, updates=updates, batches=batches,
            next_dev=next_dev, next_target=next_target, best=best, rng=rng.bit_generator.state, log_bytes=sizes), latest)

    if args.resume:
        old_protocol = json.loads((out/'PROTOCOL.json').read_text())
        old_config = dict(old_protocol.get('config', {}))
        new_config = dict(protocol.get('config', {}))
        for key in ('out', 'frames', 'dev_every'):
            old_config.pop(key, None)
            new_config.pop(key, None)
        old_protocol_cmp = dict(old_protocol)
        new_protocol_cmp = dict(protocol)
        old_protocol_cmp['config'] = old_config
        new_protocol_cmp['config'] = new_config
        # The continuation adds only resume/protocol bookkeeping; the learner,
        # rollout and model code must remain identical to the saved run.
        old_protocol_cmp.pop('budget', None)
        new_protocol_cmp.pop('budget', None)
        old_hashes = dict(old_protocol_cmp.get('hashes', {}))
        new_hashes = dict(new_protocol_cmp.get('hashes', {}))
        old_hashes.pop('dron_sichuan/train.py', None)
        new_hashes.pop('dron_sichuan/train.py', None)
        old_protocol_cmp['hashes'] = old_hashes
        new_protocol_cmp['hashes'] = new_hashes
        if old_protocol_cmp != new_protocol_cmp:
            raise RuntimeError('Resume protocol/source mismatch')
        if args.frames <= int(old_protocol.get('config', {}).get('frames', 0)):
            raise RuntimeError('Resume target must extend the previous frame budget')
        save(out/'PROTOCOL.json', protocol)
        data = torch.load(latest, map_location=args.device, weights_only=False)
        model.load_state_dict(data['model']); target.load_state_dict(data['target'])
        optimizer.load_state_dict(data['optimizer']); replay.restore(data['replay'])
        frames,games,updates,batches,next_dev,next_target,best = [data[k] for k in ('frames','games','updates','batches','next_dev','next_target','best')]
        rng.bit_generator.state = data['rng']
        for n,size in data['log_bytes'].items():
            if (out/n).exists():
                with (out/n).open('r+b') as f: f.truncate(size)
        if best is not None:
            copy_frozen(out/f"best_{best['frames']}.pt", out/'selected.pt')
        del data
    else:
        save(out/'PROTOCOL.json', protocol)
        for name in hashes:
            dest = out/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, dest)
        checkpoint()
    start, initial_frames = time.monotonic(), frames
    try:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
            while frames < args.frames:
                check_sources(); status('training')
                snapshot(actor)
                epsilon = .3 - .2 * min(1., frames/500000)
                jobs = [(str(actor), batches, 927310000000+games+i, epsilon, 'train', True) for i in range(32)]
                results = list(pool.map(worker, jobs))
                count = 0
                for rows, metrics in results:
                    replay.add(rows); count += len(rows)
                if count == 0: raise RuntimeError('Empty rollout batch')
                frames += count; games += len(results); batches += 1
                losses = []
                if replay.n >= (64 if args.smoke else 1000):
                    for _ in range(max(1, count//4)):
                        b = replay.sample(rng, args.device)
                        with torch.no_grad():
                            y = td_target(b['r'], b['done'], target(b['ns'],b['nh']), b['mask'])
                        q = model(b['s'],b['h']).gather(1,b['a'][:,None]).squeeze(1)
                        loss = F.smooth_l1_loss(q,y)
                        if not torch.isfinite(loss): raise RuntimeError('Nonfinite TD loss')
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
                        optimizer.step(); updates += 1; losses.append(float(loss.detach()))
                if frames >= next_target:
                    target.load_state_dict(model.state_dict())
                    next_target = (frames//10000+1)*10000
                metrics = measures([m for _,m in results])
                log = dict(frames=frames,games=games,updates=updates,epsilon=epsilon,replay=replay.n,
                    loss=float(np.mean(losses)) if losses else None,
                    frames_per_second=(frames-initial_frames)/(time.monotonic()-start), **metrics)
                with (out/'training.jsonl').open('a') as f: f.write(json.dumps(log)+'\n')
                with (out/'games.jsonl').open('a') as f:
                    for _,m in results: f.write(json.dumps(m)+'\n')
                print(json.dumps(log),flush=True)
                evaluated = frames >= next_dev or frames >= args.frames
                if evaluated:
                    status('development',target_games=args.dev_games); snapshot(actor)
                    rows = list(pool.map(worker,[(str(actor),batches,927390000000+i,0,'train',False) for i in range(args.dev_games)]))
                    dev = measures(rows); save(out/f'dev_{frames}.json',dict(metrics=dev,rows=rows))
                    if best is None or dev['avg_net'] > best['avg_net']:
                        best = dict(frames=frames,avg_net=dev['avg_net'])
                        snapshot(out/f'best_{frames}.pt')
                        copy_frozen(out/f'best_{frames}.pt', out/'selected.pt')
                    next_dev = (frames//args.dev_every+1)*args.dev_every
                if batches % 10 == 0 or evaluated: checkpoint()
            check_sources()
            selected = out/'selected.pt'; expected = digest(selected)
            save(out/'SELECTION.json',dict(best=best,sha256=expected,holdout_used_for_selection=False))
            holdout = out/'holdout'; holdout.mkdir(exist_ok=True)
            summaries = []
            for seed in protocol['outer_seeds'][:args.units]:
                check_sources()
                if digest(selected) != expected: raise RuntimeError('Checkpoint changed')
                status('holdout',outer_seed=seed,completed_units=len(summaries))
                folder = holdout/f'seed_{seed}'; folder.mkdir(exist_ok=True)
                path = folder/'games.jsonl'; metadata = folder/'summary.json'
                if metadata.exists():
                    old = json.loads(metadata.read_text())
                    if old['checkpoint_sha256'] != expected or old['games_sha256'] != digest(path):
                        raise RuntimeError('Existing holdout mismatch')
                    summaries.append(old['metrics']); continue
                rows = []
                for offset in range(0,args.holdout_games,100):
                    rows.extend(pool.map(worker,[(str(selected),expected,seed*1000000+i,0,'holdout',False)
                        for i in range(offset,min(args.holdout_games,offset+100))]))
                    save(folder/'progress.json',dict(completed=len(rows),target=args.holdout_games))
                simple = [dict(seed=m['seed'],net=m['net'][0],ron=m['ron'][0],dealt_in=m['dr'][0],hu=m['hu'][0],
                               max_steps_or_unsettled=0,four_seat_metrics=m) for m in rows]
                if [r['seed'] for r in simple] != [seed*1000000+i for i in range(args.holdout_games)]:
                    raise RuntimeError('CRN seed sequence mismatch')
                temporary = folder/'games.jsonl.tmp'
                with temporary.open('w') as f:
                    for r in simple: f.write(json.dumps(r)+'\n')
                os.replace(temporary,path)
                m = summarize(simple,seed) if args.holdout_games == 1000 else measures(rows)
                save(metadata,dict(metrics=m,games=len(rows),checkpoint_sha256=expected,games_sha256=digest(path)))
                summaries.append(m)
            if not args.smoke: compare(holdout,summaries)
            status('smoke_complete' if args.smoke else 'complete',holdout_games=args.units*args.holdout_games)
    except Exception as error:
        status('failed',error=repr(error)); raise


if __name__ == '__main__':
    main()
