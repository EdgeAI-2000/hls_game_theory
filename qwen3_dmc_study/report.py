"""Report only completed evidence; incomplete stages remain explicitly pending."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

from unified_tjong_eval import ROOT, digest, save, metrics
from .common import confidence, read_protocol, verify


def reference_metrics(rows):
    return metrics([dict(agent_net=r.get('agent_net', r.get('net')),
        agent_hu=r.get('agent_hu', r.get('hu')),
        agent_dealin=r.get('agent_dealin', r.get('dealt_in')),
        agent_hu_by_discard=r.get('agent_hu_by_discard', r.get('ron'))) for r in rows])


def benchmark_comparison(out, teacher):
    protocol = read_protocol(out)
    differences = {'unified_minus_legacy_douzero': [], 'unified_douzero_minus_qwen': []}
    changed = 0
    for unit in teacher['units']:
        outer = unit['outer_seed']
        current = [json.loads(line) for line in (out / 'teacher_holdout' / f'seed_{outer}' / 'games.jsonl').read_text().splitlines()]
        for name, key in [('DouZero_Sichuan_Formal_20260926', 'unified_minus_legacy_douzero'),
                          ('Qwen3_DMC_20261004_Repair/formal', 'unified_douzero_minus_qwen')]:
            path = ROOT / name / 'holdout' / f'seed_{outer}' / 'games.jsonl'
            if digest(path) != protocol['hashes'][str(path.resolve())]:
                raise RuntimeError('Historical comparison input changed')
            reference = [json.loads(line) for line in path.read_text().splitlines()]
            if len(reference) != len(current) or [r['game_seed'] for r in current] != [r.get('game_seed', r['seed']) for r in reference]:
                raise RuntimeError('Historical comparison seeds mismatch')
            ref_metrics = reference_metrics(reference)
            differences[key].append({k: unit['metrics'][k] - ref_metrics[k] for k in ref_metrics if k != 'games'})
            if key == 'unified_minus_legacy_douzero':
                changed += sum(a['agent_net'] != b['net'] for a, b in zip(current, reference))
    report = {}
    for name, units in differences.items():
        report[name] = {k: dict(delta=float(np.mean([r[k] for r in units])),
             ci95=confidence([r[k] for r in units]),
             p_t=float(ttest_1samp([r[k] for r in units], 0).pvalue) if any(r[k] for r in units) else 1.)
             for k in units[0]}
    report.update(changed_douzero_net_games=changed, inference_unit='20 paired outer seeds',
        caveat='Teacher observation repair and executor change are combined; same legacy scoring, no attribution to representation.')
    save(out / 'BENCHMARK_COMPARISON.json', report)
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    verify(read_protocol(args.out)['hashes'])
    sections = ['# Frozen-teacher and matched-objective study', '']
    parity_path = args.out / 'LEGACY_PARITY_128.json'
    if parity_path.exists():
        parity = json.loads(parity_path.read_text())
        sections += [f'Archived train-dev audit: {parity["games"]} games; changed net {parity["changed_net_games"]}; changed outcomes {parity["changed_outcome_games"]}.', '']
    teacher_path = args.out / 'TEACHER_HOLDOUT_SUMMARY.json'
    if teacher_path.exists():
        teacher = json.loads(teacher_path.read_text())
        if teacher['status'] == 'COMPLETE':
            comparison = benchmark_comparison(args.out, teacher)
            m = teacher['metrics']
            sections += ['Frozen DouZero, unified P4: 20 outer seeds x 1,000 games.', '',
                '| Mean net | Win game rate | Ron game rate | Deal-in rate | Mean seed CVaR-10% |',
                '|---:|---:|---:|---:|---:|',
                f'| {m["avg_net"]:.4f} | {m["hu"]:.3%} | {m["ron"]:.3%} | {m["dr"]:.3%} | {m["cvar10"]:.4f} |', '',
                f'Unified teacher minus legacy teacher mean net: {comparison["unified_minus_legacy_douzero"]["avg_net"]}.',
                f'Unified teacher minus repaired Qwen mean net: {comparison["unified_douzero_minus_qwen"]["avg_net"]}.', '']
    else:
        sections += ['Frozen DouZero formal re-evaluation is pending.', '']
    diagnostic_path = args.out / 'ACTION_DIAGNOSTIC.json'
    if diagnostic_path.exists():
        d = json.loads(diagnostic_path.read_text())
        sections += [f'Paired-action collection: {d["games"]} games; {d["available_samples"]} comparable states; {d["matched_samples"]} shared training states.',
            f'Initial top-1 agreement: {d["diagnostic"]["top1_agreement"]:.3%}. This does not establish which policy is correct.', '']
    pilot_path = args.out / 'CONTROLLED_PILOT_SUMMARY.json'
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text())
        sections += ['New-seed offline objective pilot:', '',
            '| Arm | Mean net | Ron | DR | CVaR-10% | Mean delta vs initial (95% CI) | Holm p |',
            '|---|---:|---:|---:|---:|---|---:|']
        for name in ('initial', 'mc', 'rank'):
            result = pilot['new_dev'][name]
            m = result['metrics']
            sections.append(f'| {name} | {m["avg_net"]:.4f} | {m["ron"]:.3%} | {m["dr"]:.3%} | {m["cvar10"]:.4f} | {result["mean_delta"]:.4f} ({result["delta_ci95"]}) | {result.get("p_holm_vs_initial", 1):.6g} |')
        sections += ['', 'A one-pass matched offline pilot is not convergence or formal holdout evidence. Review paired uncertainty and revenue/behavior diagnostics before expanding training.', '']
    else:
        sections += ['Matched MC/ranking pilot is pending.', '']
    sections += ['All figures use unchanged legacy P4 scoring. Ledger reconciliation does not validate real-world fan rules. No causal claim that text encoding or distillation explains the full score gap is made.', '']
    (args.out / 'REPORT.md').write_text('\n'.join(sections))


if __name__ == '__main__':
    main()
