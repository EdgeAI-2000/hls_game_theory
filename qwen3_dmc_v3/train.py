"""Qwen LoRA complete-return MC training, dev selection, then frozen P4 holdout."""
import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from unified_tjong_eval import ROOT, CONFIG, digest, save, metrics
from .model import BASE, load, pad_sequences
from .rollout import game_episode

ACTOR = TOKENIZER = VERSION = SETTINGS = None


def atomic_torch(value, path):
    tmp = str(path) + '.tmp'
    torch.save(value, tmp)
    os.replace(tmp, path)


def initialize_actor(settings):
    global ACTOR, TOKENIZER, SETTINGS
    torch.set_num_threads(1)
    SETTINGS = settings
    ACTOR, TOKENIZER = load(settings['base'], settings['actor_device'],
                            settings['rank'], settings['alpha'])


def actor_job(job):
    global VERSION
    snapshot, version, seed, mode = job
    if VERSION != (snapshot, version):
        saved = torch.load(snapshot, map_location='cpu', weights_only=False)
        ACTOR.load_weights(saved['weights'])
        ACTOR.eval()
        VERSION = (snapshot, version)
    samples, row = game_episode(ACTOR, TOKENIZER, seed, mode, SETTINGS['max_length'], SETTINGS['epsilon'])
    return (samples, row) if mode in ('selfplay', 'train_rules') else row


def source_hashes(base, warm_start=None):
    files = list((ROOT / 'qwen3_dmc_v3').glob('*.py'))
    files += [ROOT / name for name in ('game.py', 'tiles.py', 'player.py', 'prompt_builder.py',
              'run_gate1_experiments.py', 'opponent_curriculum.py', 'unified_tjong_eval.py',
              'rule_engine.py', 'policy_metrics.py', 'belief_oracle.py', 'mask_llm.py',
              'experiment_trace.py', 'public_belief_model.py',
              'douzero_sichuan/core.py', 'Opponent_Curriculum_formal/screened.json')]
    files.append(CONFIG)
    files += list(Path(base).glob('*.json')) + list(Path(base).glob('*.safetensors'))
    if warm_start:
        files.append(Path(warm_start))
    return {str(p.resolve()): digest(p) for p in files if p.exists()}


def verify(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise RuntimeError(f'Frozen source or model changed: {path}')


def regress(model, tokenizer, optimizer, samples, rng, micro_batch, batch_size):
    model.train()
    model.update_return_stats([r[2] for r in samples], optimizer)
    adapter = [p for p in model.backbone.parameters() if p.requires_grad]
    head = list(model.head.parameters())
    order = rng.permutation(len(samples))
    losses, raw_losses, norms, head_norms, grads = [], [], [], [], []
    for start in range(0, len(order), batch_size):
        group = order[start:start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        group_loss = 0.0
        for offset in range(0, len(group), micro_batch):
            rows = [samples[i] for i in group[offset:offset + micro_batch]]
            ids, mask = pad_sequences([r[0] for r in rows], tokenizer.pad_token_id,
                                      next(model.head.parameters()).device)
            actions = torch.as_tensor(np.asarray([r[1] for r in rows]), device=ids.device)
            target = torch.tensor([r[2] for r in rows], device=ids.device, dtype=torch.float32)
            structured = torch.as_tensor(np.asarray([r[3] for r in rows]),
                                         device=ids.device, dtype=torch.float32)
            prediction = model(ids, mask, actions, structured)
            loss = (prediction - model.normalize(target)).square().sum() / len(group)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite MC loss')
            loss.backward()
            group_loss += loss.detach().item()
            raw_losses.append((model.denormalize(prediction.detach()) - target).square().mean().item())
        adapter_grad = sum(p.grad.float().square().sum().item() for p in model.backbone.parameters()
                           if p.requires_grad and p.grad is not None)
        head_grad = sum(p.grad.square().sum().item() for p in model.head.parameters() if p.grad is not None)
        if not np.isfinite(adapter_grad + head_grad):
            raise FloatingPointError('Non-finite model gradients')
        norm = torch.nn.utils.clip_grad_norm_(adapter, 1.0, error_if_nonfinite=True)
        head_norm = torch.nn.utils.clip_grad_norm_(head, 10.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(group_loss)
        norms.append(float(norm))
        head_norms.append(float(head_norm))
        grads.append((adapter_grad, head_grad))
    return dict(mse=float(np.mean(losses)), raw_return_mse=float(np.mean(raw_losses)),
                grad_norm=float(np.mean(norms)), head_grad_norm=float(np.mean(head_norms)),
                adapter_clipped_fraction=float(np.mean(np.array(norms)>1)),
                head_clipped_fraction=float(np.mean(np.array(head_norms)>10)),
                return_mean=model.return_mean.item(), return_std=model.return_std.item(),
                adapter_grad_sq=float(np.mean([g[0] for g in grads])),
                head_grad_sq=float(np.mean([g[1] for g in grads])))


def development(pool, snapshot, version, frames, settings, out):
    jobs = [(str(snapshot), version, 2610029900000000 + i, 'dev') for i in range(settings['dev_games'])]
    rows = list(pool.map(actor_job, jobs))
    report = dict(frames=frames, metrics=metrics(rows),
                  hu_available=sum(r['qwen_hu_available'] for r in rows),
                  missed_hu=sum(r['qwen_missed_hu'] for r in rows),
                  checkpoint_sha256=digest(snapshot), seeds=[r['game_seed'] for r in rows])
    kinds = sorted({k for r in rows for k in r['settlement_breakdown']['net']})
    report['settlement_per_game_net'] = {
        k: float(np.mean([r['settlement_breakdown']['net'].get(k, 0) for r in rows]))
        for k in kinds}
    report['final_ready_rate'] = float(np.mean([r['agent_final_ready'] for r in rows]))
    report['missed_hu_fraction'] = report['missed_hu'] / max(1, report['hu_available'])
    log = out / f'dev_games_{frames}.jsonl'
    with log.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row)+'\n')
    report['games_sha256'] = digest(log)
    scores = np.array([r['agent_net'] for r in rows], dtype=float)
    from scipy.stats import t
    if len(scores) > 1:
        margin = t.ppf(.975, len(scores)-1) * scores.std(ddof=1) / np.sqrt(len(scores))
        report['avg_net_ci95'] = [float(scores.mean()-margin), float(scores.mean()+margin)]
    report['score_quantiles'] = np.quantile(scores, [0, .1, .5, .9, 1]).tolist()
    trim = len(scores)//10
    ordered = np.sort(scores)
    report['trimmed_mean10'] = float(ordered[trim:len(scores)-trim].mean())
    if frames:
        baseline = [json.loads(line) for line in (out / 'dev_games_0.jsonl').read_text().splitlines()]
        if [r['game_seed'] for r in baseline] != [r['game_seed'] for r in rows]:
            raise RuntimeError('Development seeds not paired with initial checkpoint')
        differences = scores - np.array([r['agent_net'] for r in baseline])
        report['paired_initial_avg_net_delta'] = float(differences.mean())
        if len(scores) > 1:
            margin = t.ppf(.975, len(scores)-1) * differences.std(ddof=1) / np.sqrt(len(scores))
            report['paired_initial_delta_ci95'] = [float(differences.mean()-margin),
                                                    float(differences.mean()+margin)]
    save(out / f'dev_{frames}.json', report)
    print('DEV ' + json.dumps(report['metrics'] | {'frames': frames}), flush=True)
    return report


def holdout(pool, out, snapshot, hashes):
    verify(hashes)
    checkpoint_sha = digest(snapshot)
    units = []
    for outer in range(2026082001, 2026082021):
        folder = out / 'holdout' / f'seed_{outer}'
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / 'games.jsonl'
        existing = []
        if target.exists():
            with target.open() as handle:
                for i, line in enumerate(handle):
                    r = json.loads(line)
                    if r['game_seed'] != outer * 1000000 + i or r['checkpoint_sha256'] != checkpoint_sha:
                        raise RuntimeError('Invalid existing holdout unit')
                    existing.append(r)
        if len(existing) > 1000:
            raise RuntimeError('Oversized holdout unit')
        with target.open('a') as handle:
            for start in range(len(existing), 1000, 32):
                jobs = [(str(snapshot), checkpoint_sha, outer * 1000000 + i, 'holdout')
                        for i in range(start, min(start + 32, 1000))]
                for row in pool.map(actor_job, jobs):
                    row['checkpoint_sha256'] = checkpoint_sha
                    handle.write(json.dumps(row) + '\n')
                    existing.append(row)
                handle.flush()
                print(f'HOLDOUT {outer}: {len(existing)}/1000', flush=True)
        seeds = [r['game_seed'] for r in existing]
        assert len(set(seeds)) == len(seeds) == 1000
        summary = dict(seed=outer, metrics=metrics(existing), checkpoint_sha256=checkpoint_sha,
                       games_sha256=digest(target), game_seeds_sha256=digest_seeds(seeds),
                       max_steps_or_unsettled=sum(r['end_reason'] == 'max_steps_or_unsettled' for r in existing))
        save(folder / 'summary.json', summary)
        units.append(summary)
    verify(hashes)
    assert digest(snapshot) == checkpoint_sha
    averages = {k: float(np.mean([u['metrics'][k] for u in units])) for k in units[0]['metrics'] if k != 'games'}
    save(out / 'HOLDOUT_SUMMARY.json', dict(status='COMPLETE', outer_seeds=20, games=20000,
         checkpoint_sha256=checkpoint_sha, units=units,
         metrics=averages,
         cvar_aggregation='mean of per-outer-seed CVaR-10%; statistical unit is outer seed'))
    from scipy.stats import t
    def confidence(key):
        values = np.array([u['metrics'][key] for u in units])
        delta = t.ppf(.975, 19) * values.std(ddof=1) / np.sqrt(20)
        return [float(values.mean() - delta), float(values.mean() + delta)]
    save(out / 'METRIC_CONFIDENCE.json', {k: confidence(k) for k in averages})
    report = ['# Qwen3-DMC-adapted frozen holdout', '',
              'Completed 20 outer seeds x 1,000 games; checkpoint selected exclusively on train-distribution dev.', '',
              '| Metric | Value |', '|---|---:|', f'| Mean net | {averages["avg_net"]:.4f} |',
              f'| Ron game rate | {averages["ron"] * 100:.3f}% |',
              f'| Win game rate | {averages["hu"] * 100:.3f}% |',
              f'| Deal-in game rate | {averages["dr"] * 100:.3f}% |',
              f'| Mean per-seed CVaR-10% | {averages["cvar10"]:.4f} |', '',
              'See PROTOCOL.json, SELECTION.json, METRIC_CONFIDENCE.json and per-seed logs for audit evidence.',
              'This is fixed-budget Qwen LoRA MC training, not a convergence or exact DouZero reproduction claim.',
              'Do not claim superiority over historical P4 or other baselines without paired executor/configuration auditing.']
    (out / 'REPORT.md').write_text('\n'.join(report) + '\n')


def digest_seeds(seeds):
    import hashlib
    return hashlib.sha256('\n'.join(map(str, seeds)).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--base', default=BASE)
    p.add_argument('--train-device', default='cuda:0')
    p.add_argument('--actor-device', default='cuda:1')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--frames', type=int, default=5000000)
    p.add_argument('--games-per-update', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--micro-batch', type=int, default=4)
    p.add_argument('--dev-every', type=int, default=500000)
    p.add_argument('--first-dev-after', type=int, default=50000)
    p.add_argument('--dev-games', type=int, default=1000)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--seed', type=int, default=26100401)
    p.add_argument('--epsilon', type=float, default=.05)
    p.add_argument('--warm-start')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--skip-holdout', action='store_true')
    args = p.parse_args()
    if min(args.frames, args.workers, args.games_per_update, args.batch_size,
           args.micro_batch, args.dev_every, args.dev_games, args.first_dev_after) <= 0:
        p.error('Budgets and batch sizes must be positive')
    if not 0 <= args.epsilon <= 1:
        p.error('epsilon must be between 0 and 1')
    if args.games_per_update < 2 or args.games_per_update % 2:
        p.error('games-per-update must be positive and even for the two training environments')
    settings = vars(args).copy()
    settings['base'] = str(Path(args.base).resolve())
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / 'PROTOCOL.json'
    if protocol_path.exists() and not args.resume:
        raise RuntimeError('Existing run: use --resume or a fresh directory')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    hashes = source_hashes(settings['base'], args.warm_start)
    if args.resume:
        protocol = json.loads(protocol_path.read_text())
        expected_settings = {k: v for k, v in protocol['settings'].items() if k != 'resume'}
        if expected_settings != {k: v for k, v in settings.items() if k != 'resume'} or hashes != protocol['hashes']:
            raise RuntimeError('Resume configuration or source mismatch')
    else:
        save(protocol_path, dict(settings=settings, hashes=hashes,
             algorithm='Qwen3-DMC-v3 structured-state complete return-to-go MC regression; gamma=1; PopArt normalized MSE',
             architecture=f'{Path(settings["base"]).name} + LoRA r{args.rank} alpha{args.alpha} all attention/MLP projections + exact DouZero STATE_DIM structured vector + action-conditioned five FC512 value head',
             data='alternating half shared four-seat selfplay, half P0 versus six TRAIN curriculum rules; own future net/base_score; epsilon from settings',
             optimizer='RMSprop: LoRA lr2e-6, head lr1e-4; alpha=.99 eps1e-5; separate grad clips adapter1/head10; PopArt without reward clipping',
             adaptation='Qwen text/history representation plus exact public structured state encoding; shared state encoding for all legal macro actions; no language-token likelihood loss',
             budget='fresh complete-game decisions since warm start, including forced actions, one regression pass; warm-start lineage accounted separately; no replay/new SFT/MASK',
             selection='fixed train-curriculum dev seeds; highest dev avg_net after training, earliest tie; no convergence claim',
             training_seed_start=args.seed * 10000000, dev_seed_start=2610029900000000,
             holdout='P4 macro executor, auto_hu off; frozen 3 holdout rules; 2026082001..2026082020, 1000 games each; same historical benchmark seeds',
             exchange_dingque='existing P4 deterministic heuristic, not trained',
             reference='https://proceedings.mlr.press/v139/zha21a.html'))
    model, tokenizer = load(settings['base'], args.train_device, args.rank, args.alpha, training=True)
    adapter = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.RMSprop([dict(params=adapter, lr=2e-6), dict(params=model.head.parameters(), lr=1e-4)],
                                     alpha=.99, eps=1e-5)
    frames = games = update = 0
    best_score, best_frames = -float('inf'), None
    latest, snapshot, selected = out / 'latest.pt', out / 'actor.pt', out / 'selected.pt'
    if args.warm_start and not args.resume:
        source = torch.load(args.warm_start, map_location='cpu', weights_only=False)
        model.load_text_weights(source['weights'])
        save(out / 'WARM_START.json', dict(path=str(Path(args.warm_start).resolve()),
                                         sha256=digest(args.warm_start), source_frames=source['frames'],
                                         optimizer='new optimizer; value-preserving input expansion',
                                         loaded_components=['adapter', 'head', 'return_stats'],
                                         new_components='zero-initialized structured input columns'))
    if args.resume:
        state = torch.load(latest, map_location='cpu', weights_only=False)
        model.load_weights(state['weights'])
        optimizer.load_state_dict(state['optimizer'])
        frames, games, update = state['frames'], state['games'], state['update']
        best_score, best_frames = state['best_score'], state['best_frames']
        rng.bit_generator.state = state['rng']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    start, initial_frames = time.monotonic(), frames
    next_dev = (frames // args.dev_every + 1) * args.dev_every
    if frames < args.first_dev_after:
        next_dev = min(next_dev, args.first_dev_after)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_actor, initargs=(settings,)) as pool:
        if not args.resume:
            atomic_torch(dict(weights=model.weights(), frames=0), snapshot)
            baseline = development(pool, snapshot, -1, 0, settings, out)
            if args.warm_start and source.get('dev', {}).get('seeds') == baseline['seeds']:
                matches = source['dev']['metrics'] == baseline['metrics']
                save(out / 'WARM_START_PARITY.json', dict(passed=matches,
                     comparison='identical fixed weights and dev seeds; aggregate metric equality',
                     original=source['dev']['metrics'], expanded_baseline=baseline['metrics']))
                if not matches:
                    raise RuntimeError('Warm-start fixed-checkpoint development parity failed')
            best_score, best_frames = baseline['metrics']['avg_net'], 0
            atomic_torch(dict(weights=model.weights(), frames=0, dev=baseline), selected)
            save(out / 'SELECTION.json', dict(frames=0, dev=baseline, checkpoint_sha256=digest(selected)))
            atomic_torch(dict(weights=model.weights(), optimizer=optimizer.state_dict(), frames=0,
                             games=0, update=0, best_score=best_score, best_frames=best_frames,
                             rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all()), latest)
        while frames < args.frames:
            save(out / 'STATUS.json', dict(phase='sampling', update=update, frames=frames, games=games,
                                          updated_unix=time.time(), workers=args.workers))
            atomic_torch(dict(weights=model.weights(), frames=frames), snapshot)
            jobs = [(str(snapshot), update, args.seed * 10000000 + games + i,
                     'selfplay' if (games+i)%2 == 0 else 'train_rules')
                    for i in range(args.games_per_update)]
            sampling_start = time.monotonic()
            batch = list(pool.map(actor_job, jobs))
            sampling_seconds = time.monotonic() - sampling_start
            samples = [s for ss, _ in batch for s in ss]
            if not samples:
                raise RuntimeError('Empty training batch')
            save(out / 'STATUS.json', dict(phase='learning', update=update, frames=frames, games=games,
                                          updated_unix=time.time(), pending_samples=len(samples)))
            learning_start = time.monotonic()
            train_stats = regress(model, tokenizer, optimizer, samples, rng, args.micro_batch, args.batch_size)
            learning_seconds = time.monotonic() - learning_start
            if update == 0 and (train_stats['adapter_grad_sq'] <= 0 or train_stats['head_grad_sq'] <= 0):
                raise RuntimeError('Qwen adapter or value head received no gradient')
            frames += len(samples)
            games += len(batch)
            update += 1
            elapsed = time.monotonic() - start
            rate = (frames - initial_frames) / elapsed
            report = dict(update=update, frames=frames, games=games, seconds=elapsed, frames_per_second=rate,
                          remaining_train_hours=max(0, args.frames - frames) / max(rate, 1e-9) / 3600,
                          target_min=min(s[2] for s in samples), target_max=max(s[2] for s in samples),
                          nonzero_target_rate=float(np.mean([s[2] != 0 for s in samples])),
                          training_p0=metrics([r for _, r in batch]),
                          hu_available=sum(r['qwen_hu_available'] for _, r in batch),
                          missed_hu=sum(r['qwen_missed_hu'] for _, r in batch),
                          max_tokens=max(len(s[0]) for s in samples),
                          sampling_seconds=sampling_seconds, learning_seconds=learning_seconds, **train_stats)
            report['past_settlement_sample_fraction'] = sum(r['decisions_with_past_settlement']
                for _, r in batch) / len(samples)
            report['training_environment'] = {mode: metrics([r for _,r in batch if r['sampling_mode']==mode])
                                               for mode in ('selfplay', 'train_rules')}
            with (out / 'training.jsonl').open('a') as handle:
                handle.write(json.dumps(report) + '\n')
            print('TRAIN ' + json.dumps(report), flush=True)
            if frames >= next_dev or frames >= args.frames:
                save(out / 'STATUS.json', dict(phase='dev', update=update, frames=frames, games=games,
                                              updated_unix=time.time()))
                verify(hashes)
                atomic_torch(dict(weights=model.weights(), frames=frames), snapshot)
                dev = development(pool, snapshot, update, frames, settings, out)
                if dev['metrics']['avg_net'] > best_score:
                    best_score, best_frames = dev['metrics']['avg_net'], frames
                    atomic_torch(dict(weights=model.weights(), frames=frames, dev=dev), selected)
                    save(out / 'SELECTION.json', dict(frames=frames, dev=dev, checkpoint_sha256=digest(selected)))
                next_dev = (frames // args.dev_every + 1) * args.dev_every
            atomic_torch(dict(weights=model.weights(), optimizer=optimizer.state_dict(), frames=frames,
                             games=games, update=update, best_score=best_score, best_frames=best_frames,
                             rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all()), latest)
            if (out / 'PAUSE_REQUESTED').exists():
                save(out / 'STATUS.json', dict(phase='paused', update=update, frames=frames, games=games,
                                              updated_unix=time.time(), checkpoint_sha256=digest(latest)))
                print('PAUSED after complete batch and checkpoint; formal holdout not started', flush=True)
                return
        save(out / 'TRAINING_COMPLETE.json', dict(frames=frames, games=games, updates=update,
                                                best_dev_score=best_score, best_frames=best_frames))
        if not args.skip_holdout:
            save(out / 'STATUS.json', dict(phase='holdout', frames=frames, updated_unix=time.time()))
            holdout(pool, out, selected, hashes)
    save(out / 'STATUS.json', dict(phase='complete', frames=frames, updated_unix=time.time()))
    print('PIPELINE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
