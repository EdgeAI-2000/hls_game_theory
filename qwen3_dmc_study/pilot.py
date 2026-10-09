"""One-pass offline MC vs relative-ranking control, followed by new-seed dev."""
import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from qwen3_dmc_v3.model import pad_sequences
from qwen3_dmc_v3.rollout import game_episode
from unified_tjong_eval import digest, save
from .common import (STUDENT, STUDENT_SHA, atomic_torch, holm, paired_report,
                     read_protocol, student_model, verify)

MODEL = TOKENIZER = OUT = VERSION = None


def relative_distribution(values, temperature):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    values = torch.as_tensor(values, dtype=torch.float32)
    return torch.softmax((values - values.mean()) / temperature, dim=-1)


def values_for_rows(model, tokenizer, rows):
    device = next(model.head.parameters()).device
    ids, mask = pad_sequences([r['ids'] for r in rows], tokenizer.pad_token_id, device)
    hidden = model.encode(ids, mask)
    lengths = [len(r['actions']) for r in rows]
    repetitions = torch.tensor(lengths, device=device)
    state = torch.tensor(np.stack([r['state'] for r in rows]), dtype=torch.float32, device=device)
    actions = torch.tensor(np.concatenate([r['actions'] for r in rows]), dtype=torch.float32, device=device)
    raw = model.head(torch.cat((hidden.repeat_interleave(repetitions, 0),
          state.repeat_interleave(repetitions, 0), actions), -1)).squeeze(-1)
    return raw.split(lengths)


def train_batch(model, tokenizer, optimizer, rows, arm, micro_batch, temperature):
    model.train()
    # Both arms update exactly the same PopArt statistics from the executed
    # returns. Relative distillation does not copy the teacher's Q baseline.
    model.update_return_stats([r['target'] for r in rows], optimizer)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for offset in range(0, len(rows), micro_batch):
        chunk = rows[offset:offset + micro_batch]
        values = values_for_rows(model, tokenizer, chunk)
        losses = []
        for row, prediction in zip(chunk, values):
            if arm == 'mc':
                target = torch.tensor(row['target'], device=prediction.device)
                losses.append((prediction[row['chosen']] - model.normalize(target)).square())
            elif arm == 'rank':
                teacher = relative_distribution(row['teacher_q'], temperature).to(prediction.device)
                logits = model.denormalize(prediction) / temperature
                losses.append(-(teacher * F.log_softmax(logits, dim=-1)).sum())
            else:
                raise ValueError(arm)
        loss = torch.stack(losses).sum() / len(rows)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite {arm} loss')
        loss.backward()
        total += loss.detach().item()
    adapter = [p for p in model.backbone.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(adapter, 1.0, error_if_nonfinite=True)
    head_norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), 10.0, error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=total, adapter_grad_norm=float(norm), head_grad_norm=float(head_norm),
                return_std=model.return_std.item())


def fit(out, arm, data):
    protocol = read_protocol(out)
    folder = out / arm
    folder.mkdir(exist_ok=True)
    selected_path = folder / 'trained.pt'
    if selected_path.exists():
        report = json.loads((folder / 'TRAINING_COMPLETE.json').read_text())
        if digest(selected_path) != report['checkpoint_sha256'] or digest(out / 'matched_samples.pt') != report['data_sha256']:
            raise RuntimeError('Completed control arm changed')
        return selected_path
    torch.manual_seed(protocol['seed'])
    model, tokenizer = student_model(protocol['device'], training=True)
    adapter = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.RMSprop([dict(params=adapter, lr=2e-6),
        dict(params=list(model.head.parameters()), lr=1e-4)], alpha=.99, eps=1e-5)
    order = np.random.default_rng(protocol['seed']).permutation(len(data)).tolist()
    order = order * protocol['epochs']
    batch = protocol['batch_size']
    last = folder / 'latest.pt'
    offset = updates = 0
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['data_sha256'] != digest(out / 'matched_samples.pt') or saved['arm'] != arm:
            raise RuntimeError('Invalid resumable control arm')
        model.load_weights(saved['weights'])
        optimizer.load_state_dict(saved['optimizer'])
        offset, updates = saved['offset'], saved['updates']
    started = time.monotonic()
    with (folder / 'training.jsonl').open('a') as handle:
        while offset < len(order):
            indices = order[offset:offset + batch]
            result = train_batch(model, tokenizer, optimizer, [data[i] for i in indices], arm,
                                 protocol['micro_batch'], protocol['temperature'])
            offset += len(indices)
            updates += 1
            result.update(samples=offset, updates=updates, arm=arm, elapsed_seconds=time.monotonic() - started)
            handle.write(json.dumps(result) + '\n')
            handle.flush()
            if updates % 10 == 0 or offset == len(order):
                print('TRAIN ' + json.dumps(result), flush=True)
                atomic_torch(dict(weights=model.weights(), optimizer=optimizer.state_dict(),
                    offset=offset, updates=updates, arm=arm,
                    data_sha256=digest(out / 'matched_samples.pt')), last)
    atomic_torch(dict(weights=model.weights(), arm=arm, samples=offset, initial_sha256=STUDENT_SHA), selected_path)
    save(folder / 'TRAINING_COMPLETE.json', dict(arm=arm, samples=offset, optimizer_updates=updates,
         tokens=sum(len(data[i]['ids']) for i in order),
         candidate_head_evaluations=sum(len(data[i]['actions']) for i in order),
         initial_checkpoint_sha256=STUDENT_SHA, data_sha256=digest(out / 'matched_samples.pt'),
         checkpoint_sha256=digest(selected_path), elapsed_seconds=time.monotonic() - started,
         objective='offline future MC normalized MSE' if arm == 'mc' else 'relative teacher action distribution cross-entropy'))
    del optimizer, model
    torch.cuda.empty_cache()
    return selected_path


def initialize(out):
    global MODEL, TOKENIZER, OUT
    torch.set_num_threads(1)
    OUT = Path(out)
    MODEL, TOKENIZER = student_model(read_protocol(OUT)['device'])


def dev_job(job):
    global VERSION
    snapshot, sha, seed = job
    if VERSION != sha:
        MODEL.load_weights(torch.load(snapshot, map_location='cpu', weights_only=False)['weights'])
        MODEL.eval()
        VERSION = sha
    row = game_episode(MODEL, TOKENIZER, seed, mode='dev')[1]
    row['checkpoint_sha256'] = sha
    return row


def development(pool, out, name, snapshot, count):
    protocol = read_protocol(out)
    path = out / f'dev_{name}.jsonl'
    sha = digest(snapshot)
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) > count or any(r['game_seed'] != protocol['dev_start'] + i or r['checkpoint_sha256'] != sha for i, r in enumerate(rows)):
            raise RuntimeError('Invalid resumable new-seed dev results')
    with path.open('a') as handle:
        for start in range(len(rows), count, 32):
            jobs = [(str(snapshot), sha, protocol['dev_start'] + i) for i in range(start, min(count, start + 32))]
            for row in pool.map(dev_job, jobs):
                handle.write(json.dumps(row) + '\n')
                rows.append(row)
            handle.flush()
            print(f'DEV {name}: {len(rows)}/{count}', flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    protocol = read_protocol(args.out)
    verify(protocol['hashes'])
    diagnostic = json.loads((args.out / 'ACTION_DIAGNOSTIC.json').read_text())
    if digest(args.out / 'matched_samples.pt') != diagnostic['matched_samples_sha256']:
        raise RuntimeError('Matched data changed')
    data = torch.load(args.out / 'matched_samples.pt', weights_only=False)
    if args.smoke:
        data = data[:64]
    if not data or any(len(r['actions']) < 2 or not 0 <= r['chosen'] < len(r['actions']) for r in data):
        raise RuntimeError('Invalid action-comparable data')
    torch.set_num_threads(2)
    snapshots = {'initial': STUDENT}
    for arm in ('mc', 'rank'):
        snapshots[arm] = fit(args.out, arm, data)
    budgets = [json.loads((args.out / arm / 'TRAINING_COMPLETE.json').read_text()) for arm in ('mc', 'rank')]
    for key in ('samples', 'tokens', 'candidate_head_evaluations', 'optimizer_updates', 'initial_checkpoint_sha256', 'data_sha256'):
        if budgets[0][key] != budgets[1][key]:
            raise RuntimeError(f'Unmatched training budget: {key}')
    count = 8 if args.smoke else protocol['dev_games']
    with ProcessPoolExecutor(8, mp_context=mp.get_context('spawn'), initializer=initialize, initargs=(str(args.out),)) as pool:
        rows = {name: development(pool, args.out, name, path, count) for name, path in snapshots.items()}
    reports = {name: paired_report(result, rows['initial']) for name, result in rows.items()}
    corrected = holm([reports[a]['p_t'] for a in ('mc', 'rank')])
    for arm, value in zip(('mc', 'rank'), corrected):
        reports[arm]['p_holm_vs_initial'] = float(value)
    report = dict(status='SMOKE_COMPLETE' if args.smoke else 'COMPLETE', budgets=budgets,
                  new_dev=reports, rank_minus_mc=paired_report(rows['rank'], rows['mc']),
                  scope='matched offline objective diagnostic; neither convergence nor formal holdout result',
                  continuation_gate='Do not expand training based on historic holdout; require positive new-dev paired evidence and inspect all primary/diagnostic metrics.')
    verify(protocol['hashes'])
    save(args.out / 'CONTROLLED_PILOT_SUMMARY.json', report)
    print('CONTROL_COMPLETE ' + json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
