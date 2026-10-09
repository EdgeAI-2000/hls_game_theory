import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from douzero_sichuan import core
from qwen3_dmc_v3.model import BASE, load
from qwen3_dmc_v3.train import atomic_torch, source_hashes, verify
from unified_baseline_eval import CHECKPOINTS, load_model
from unified_tjong_eval import ROOT, digest, kwargs, metrics, save

TEACHER = ROOT / CHECKPOINTS['douzero']
STUDENT = ROOT / 'Qwen3_DMC_20261008_Structured/pilot/selected.pt'
TEACHER_SHA = 'd6c23afc1fbbd2fc51df622c03919dddf7d23cf4359f4c42c7812d793d7e153d'
STUDENT_SHA = '302527d8cae323af83504d8fb5aaaf1b1f857aabc2cd0ba55b8ca041ef947f7f'


def read_protocol(out):
    return json.loads((Path(out) / 'PROTOCOL.json').read_text())


def freeze(out):
    out = Path(out)
    if (out / 'PROTOCOL.json').exists():
        verify(read_protocol(out)['hashes'])
        return
    out.mkdir(parents=True, exist_ok=True)
    if digest(TEACHER) != TEACHER_SHA or digest(STUDENT) != STUDENT_SHA:
        raise RuntimeError('Selected checkpoint changed')
    hashes = source_hashes(ROOT / BASE, STUDENT)
    for p in list((ROOT / 'qwen3_dmc_study').glob('*.py')) + [
            ROOT / 'unified_baseline_eval.py', TEACHER,
            ROOT / 'Qwen3_DMC_Controlled_20261009/run.sh',
            ROOT / 'Qwen3_DMC_Controlled_20261009/PROTOCOL.md']:
        hashes[str(p.resolve())] = digest(p)
    for p in (ROOT / BASE).iterdir():
        if p.is_file():
            hashes[str(p.resolve())] = digest(p)
    references = []
    for name in ('DouZero_Sichuan_Formal_20260926', 'Qwen3_DMC_20261004_Repair/formal'):
        for outer in range(2026082001, 2026082021):
            references.extend([ROOT / name / 'holdout' / f'seed_{outer}' / filename
                               for filename in ('summary.json', 'games.jsonl')])
    for p in references:
        hashes[str(p.resolve())] = digest(p)
    save(out / 'PROTOCOL.json', dict(
        hashes=hashes, teacher=dict(path=str(TEACHER), sha256=TEACHER_SHA),
        student=dict(path=str(STUDENT), sha256=STUDENT_SHA),
        executor='unchanged legacy P4 macro executor; auto_hu off; public-only history',
        formal_outer_seeds=list(range(2026082001, 2026082021)), formal_games_per_seed=1000,
        audit_start=2610029900000000, audit_games=128,
        collection_start=2610090100000000, collection_games=1024,
        dev_start=2610099900000000, dev_games=1000,
        sample_cap=20000, epochs=1, batch_size=64, micro_batch=4,
        seed=26100901, temperature=1.0, workers=8, device='cuda:0',
        selection='initial vs one trained checkpoint per arm on NEW train-distribution dev; no holdout tuning',
        objectives=['future MC normalized MSE', 'teacher relative action distribution cross-entropy'],
        controls='same public states, trajectories, initialization, sample order, optimizer, one pass and token budget; offline diagnostic, not online MC continuation',
        rules='Legacy P4 scoring caveats persist. Zero-sum ledger is not real-rule correctness.'))


def train_options(seed):
    options = kwargs(seed)
    options['defender_cfg']['curriculum_split'] = 'train'
    options['defender_cfg'].pop('curriculum_specs_by_pid', None)
    options['defender_cfg'].pop('curriculum_spec_id', None)
    return options


def teacher_model():
    return load_model('douzero', TEACHER_SHA)


def student_model(device='cuda:0', checkpoint=STUDENT, training=False):
    if Path(checkpoint) == STUDENT and digest(checkpoint) != STUDENT_SHA:
        raise RuntimeError('Initial student checkpoint changed')
    model, tokenizer = load(str(ROOT / BASE), device, 16, 32, training)
    weights = torch.load(checkpoint, map_location='cpu', weights_only=False)['weights']
    model.load_weights(weights)
    return model, tokenizer


@torch.inference_mode()
def teacher_values(model, state, history, candidates):
    # Keep the same forward path as the frozen teacher's greedy choose().
    _, (hidden, _) = model.lstm(torch.from_numpy(history[None]))
    x = torch.from_numpy(np.concatenate((np.repeat(state[None], len(candidates), axis=0), candidates), 1))
    result = model.mlp(torch.cat((x, hidden[-1].expand(len(candidates), -1)), 1)).squeeze(-1)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Non-finite teacher action values')
    return result.numpy()


def reconcile(game, capture, row):
    net = [p.balance - b for p, b in zip(game.players, capture['initial'])]
    ledger = [0] * 4
    for entry in game.settlement_ledger:
        ledger[entry['payer_id']] -= entry['score']
        ledger[entry['receiver_id']] += entry['score']
    if ledger != net or sum(net) or row['agent_net'] != net[0]:
        raise RuntimeError('Settlement ledger mismatch')
    if not game.is_game_over or not row['settlement_reconciled'] or row['end_reason'] == 'max_steps_or_unsettled':
        raise RuntimeError('Unsettled game')
    row.update(four_seat_net=net, evaluator='p4_macro_v1')


def seed_digest(seeds):
    return hashlib.sha256('\n'.join(map(str, seeds)).encode()).hexdigest()


def confidence(values):
    from scipy.stats import t
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    margin = t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return [float(values.mean() - margin), float(values.mean() + margin)]


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(len(pvalues), 0, -1)))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def paired_report(rows, reference):
    from scipy.stats import ttest_1samp
    if [r['game_seed'] for r in rows] != [r['game_seed'] for r in reference]:
        raise RuntimeError('Unpaired game seeds')
    difference = np.array([a['agent_net'] - b['agent_net'] for a, b in zip(rows, reference)])
    ordered = np.sort([r['agent_net'] for r in rows])
    trim = len(rows) // 10
    return dict(metrics=metrics(rows), mean_delta=float(difference.mean()),
                delta_ci95=confidence(difference),
                p_t=float(ttest_1samp(difference, 0).pvalue) if np.any(difference) else 1.0,
                trimmed_mean10=float(ordered[trim:len(rows) - trim].mean()),
                median_net=float(np.median(ordered)),
                hu_available=sum(r.get('qwen_hu_available', 0) for r in rows),
                hu_not_chosen=sum(r.get('qwen_missed_hu', 0) for r in rows),
                settlement_per_game_net={k: float(np.mean([r.get('settlement_breakdown', {}).get('net', {}).get(k, 0) for r in rows]))
                    for k in sorted({k for r in rows for k in r.get('settlement_breakdown', {}).get('net', {})})},
                seed_sha256=seed_digest([r['game_seed'] for r in rows]),
                inference_unit='independent game seed; fixed train opponents; diagnostic only')
