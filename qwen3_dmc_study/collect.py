"""Score teacher and student on identical public states, including student trajectories."""
import argparse
import hashlib
import json
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from douzero_sichuan import core
from qwen3_dmc_v3.model import tokens, action_vectors
from qwen3_dmc_v3.rollout import return_to_go
from rule_engine import ShantenCalculator
from unified_baseline_eval import Policy, runner
from unified_tjong_eval import digest, save
from .common import (atomic_torch, read_protocol, reconcile, student_model,
                     teacher_model, teacher_values, train_options, verify)

STUDENT_MODEL = TOKENIZER = TEACHER_MODEL = OUT = None


def discard_shanten(game, pid, actions):
    values = []
    player = game.players[pid]
    for action in actions:
        if not action.startswith('d '):
            values.append(None)
            continue
        hand = player.hand_tiles.copy()
        tile = core.parse_console_tile(action[2:])
        hand.remove(tile)
        values.append(int(ShantenCalculator.calculate_shanten(hand, player.missing_suit)))
    return values


class PairedPolicy(Policy):
    def __init__(self, teacher, student, tokenizer, seed, behavior, max_length=1024):
        super().__init__('douzero', teacher, seed)
        self.student, self.tokenizer = student, tokenizer
        self.behavior, self.max_length = behavior, max_length
        self.samples, self.diagnostics = [], []

    def decide(self, game, pid, actions, response):
        if pid != 0:
            raise RuntimeError('Only P0 is learned in this controlled study')
        state, history, candidates = core.encode(game, pid, actions, self.events, response)
        np.testing.assert_array_equal(candidates, action_vectors(game, pid, actions, response))
        if len(actions) == 1:
            teacher_choice = student_choice = choice = 0
            tq = sq = np.zeros(1, dtype=np.float32)
            ids = None
        else:
            ids = tokens(self.tokenizer, game, pid, actions, response, self.max_length)
            tq = teacher_values(self.model, state, history, candidates)
            sq = self.student.score(ids, candidates, state)
            teacher_choice, student_choice = int(tq.argmax()), int(sq.argmax())
            if not np.isfinite(sq).all():
                raise FloatingPointError('Non-finite student values')
            choice = teacher_choice if self.behavior == 'teacher' else student_choice
            # Symmetric behavior exploration. Both training arms later consume
            # exactly these executed actions and completed-game returns.
            if self.rng.random() < .05:
                choice = self.rng.randrange(len(actions))
        progress = discard_shanten(game, pid, actions)
        item = dict(response=response, legal=list(actions), teacher_index=teacher_choice,
                    student_index=student_choice, chosen_index=choice,
                    teacher_q=tq.tolist(), student_q=sq.tolist(), discard_shanten=progress,
                    behavior=self.behavior, decision_index=len(self.diagnostics), game_seed=game.seed if hasattr(game, 'seed') else None)
        self.diagnostics.append(item)
        if ids is not None:
            observation_sha = hashlib.sha256(np.asarray(ids, np.int32).tobytes() + state.tobytes() + history.tobytes()).hexdigest()
            self.samples.append(dict(ids=ids, state=state, actions=candidates,
                 teacher_q=tq, chosen=choice, balance_before=game.players[pid].balance,
                 observation_sha256=observation_sha, diagnostic=item))
        action = actions[choice]
        self.decisions.append(dict(action=action, response=response, legal=list(actions)))
        if response:
            self.record(game, pid, action, True)
        return action, dict(mode='exploit', reason='matched public-state diagnostic')


def paired_game(teacher, student, tokenizer, seed, behavior, max_length=1024):
    capture = {}
    policy = PairedPolicy(teacher, student, tokenizer, seed, behavior, max_length)
    row = runner(policy, capture)(**train_options(seed))[0]
    game = capture['game']
    reconcile(game, capture, row)
    for sample in policy.samples:
        sample['target'] = return_to_go(game.players[0].balance, sample.pop('balance_before'), game.base_score)
        sample['game_seed'] = seed
        sample['diagnostic']['game_seed'] = seed
    for item in policy.diagnostics:
        item['game_seed'] = seed
    row.update(game_seed=seed, behavior=behavior, train_samples=len(policy.samples),
               hu_available=sum('h' in r['legal'] for r in policy.diagnostics),
               missed_hu=sum('h' in r['legal'] and r['legal'][r['chosen_index']] != 'h' for r in policy.diagnostics),
               settlement_ledger=game.settlement_ledger)
    return policy.samples, policy.diagnostics, row


def initialize(out):
    global STUDENT_MODEL, TOKENIZER, TEACHER_MODEL, OUT
    torch.set_num_threads(1)
    OUT = Path(out)
    protocol = read_protocol(OUT)
    STUDENT_MODEL, TOKENIZER = student_model(protocol['device'])
    TEACHER_MODEL = teacher_model()


def job(index):
    protocol = read_protocol(OUT)
    seed = protocol['collection_start'] + index
    folder = OUT / 'dataset'
    path = folder / f'{seed}.pt'
    row_path = folder / f'{seed}.json'
    if path.exists() and row_path.exists():
        saved = json.loads(row_path.read_text())
        if digest(path) != saved['samples_sha256'] or saved['game_seed'] != seed:
            raise RuntimeError('Invalid cached collection game')
        return saved
    if path.exists() or row_path.exists():
        raise RuntimeError('Incomplete collection game: inspect before resuming')
    behavior = 'teacher' if index % 2 == 0 else 'student'
    samples, diagnostic, row = paired_game(TEACHER_MODEL, STUDENT_MODEL, TOKENIZER, seed, behavior)
    atomic_torch(dict(samples=samples, diagnostic=diagnostic, row=row), path)
    row.update(samples_sha256=digest(path))
    save(row_path, row)
    return row


def summarize(diagnostics):
    comparable = [r for r in diagnostics if len(r['legal']) > 1]
    if not comparable:
        raise RuntimeError('No multi-action diagnostic states')
    discards = [r for r in comparable if r['legal'][r['teacher_index']].startswith('d ') and
                r['legal'][r['student_index']].startswith('d ')]
    def shanten_gap(row, index):
        return row['discard_shanten'][index] - min(x for x in row['discard_shanten'] if x is not None)
    hu = [r for r in comparable if 'h' in r['legal']]
    return dict(multi_action_states=len(comparable), both_discard_states=len(discards), hu_opportunities=len(hu),
        top1_agreement=float(np.mean([r['teacher_index'] == r['student_index'] for r in comparable])),
        student_teacher_q_gap=float(np.mean([max(r['teacher_q']) - r['teacher_q'][r['student_index']] for r in comparable])),
        teacher_discard_shanten_loss=float(np.mean([shanten_gap(r, r['teacher_index']) for r in discards])) if discards else None,
        student_discard_shanten_loss=float(np.mean([shanten_gap(r, r['student_index']) for r in discards])) if discards else None,
        teacher_hu_not_chosen=sum(r['legal'][r['teacher_index']] != 'h' for r in hu),
        student_hu_not_chosen=sum(r['legal'][r['student_index']] != 'h' for r in hu),
        caveat='Teacher predicted Q and own-shanten diagnostics are not oracle action values or causal explanations.')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    protocol = read_protocol(args.out)
    verify(protocol['hashes'])
    (args.out / 'dataset').mkdir(exist_ok=True)
    count = 8 if args.smoke else protocol['collection_games']
    rows = []
    with ProcessPoolExecutor(8, mp_context=mp.get_context('spawn'), initializer=initialize, initargs=(str(args.out),)) as pool:
        for row in pool.map(job, range(count)):
            rows.append(row)
            if len(rows) % 8 == 0:
                print(f'COLLECT {len(rows)}/{count}; states={sum(r["train_samples"] for r in rows)}', flush=True)
    all_samples, diagnostics, manifest = [], [], []
    for row in rows:
        path = args.out / 'dataset' / f'{row["game_seed"]}.pt'
        data = torch.load(path, map_location='cpu', weights_only=False)
        all_samples.extend(data['samples'])
        diagnostics.extend(data['diagnostic'])
        manifest.append(dict(path=str(path.resolve()), sha256=digest(path), game_seed=row['game_seed']))
    if len({r['game_seed'] for r in rows}) != count:
        raise RuntimeError('Duplicate collection game seeds')
    rng = random.Random(protocol['seed'])
    order = rng.sample(range(len(all_samples)), min(protocol['sample_cap'], len(all_samples)))
    selected = [all_samples[i] for i in order]
    atomic_torch(selected, args.out / 'matched_samples.pt')
    report = dict(status='SMOKE_COMPLETE' if args.smoke else 'COMPLETE', games=count,
                  available_samples=len(all_samples), matched_samples=len(selected),
                  matched_tokens=sum(len(r['ids']) for r in selected),
                  teacher_behavior_samples=sum(r['diagnostic']['behavior'] == 'teacher' for r in selected),
                  student_behavior_samples=sum(r['diagnostic']['behavior'] == 'student' for r in selected),
                  matched_samples_sha256=digest(args.out / 'matched_samples.pt'), games_manifest=manifest,
                  diagnostic=summarize(diagnostics),
                  by_behavior={b: summarize([r for r in diagnostics if r['behavior'] == b]) for b in ('teacher', 'student')},
                  failures=0)
    verify(protocol['hashes'])
    save(args.out / 'ACTION_DIAGNOSTIC.json', report)
    print('DIAGNOSTIC ' + json.dumps(report['diagnostic']), flush=True)


if __name__ == '__main__':
    main()
