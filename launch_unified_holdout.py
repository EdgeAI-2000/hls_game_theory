"""Eight CPU workers shared across four frozen baseline reevaluations."""
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import stats

import unified_tjong_eval as t
import unified_baseline_eval as b

TJ = t.ROOT/'Tjong_Unified_P4_20260929'
BL = t.ROOT/'Baselines_Unified_P4_20260929'


def work(name, seed):
    if name == 'tjong':
        result = t.unit(str(TJ), seed, 1000, False)
        result['name'] = name
        return result
    return b.unit(str(BL), name, seed, 1000, False)


def main():
    os.chdir(t.ROOT)
    for out in (TJ, BL):
        assert (out/'PILOT_PASSED.json').is_file(), f'Pilot not passed: {out}'
        t.verify_sources(out)
    start = time.time()
    t.save(BL/'LAUNCH.json', dict(start=start, workers=8, device='cpu', units=80,
           games=80000, launcher_sha256=t.digest(__file__),
           sources={str(p.relative_to(t.ROOT)):t.digest(p) for p in
                    (TJ/'PROTOCOL.json',BL/'PROTOCOL.json')},
           note='Legacy shared rules, evaluation repair only, fixed checkpoints, no holdout tuning'))
    results = []
    try:
        with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn')) as pool:
            jobs = [pool.submit(work,name,seed) for seed in range(2026082001,2026082021)
                    for name in ('tjong','douzero','douzero_plus','dron')]
            for job in as_completed(jobs):
                results.append(job.result())
                t.save(BL/'STATUS.json',dict(stage='running',completed_units=len(results),total_units=80,
                                           elapsed_seconds=time.time()-start,updated_at=time.time()))
    except BaseException as exc:
        t.save(BL/'STATUS.json',dict(stage='failed',completed_units=len(results),error=repr(exc)))
        raise
    results.sort(key=lambda r:(r['name'],r['seed']))
    series = {name:[r for r in results if r['name']==name]
              for name in ('tjong','douzero','douzero_plus','dron')}
    for out in (TJ,BL): t.verify_sources(out)
    t.save(TJ/'FORMAL_COMPLETE.json',series['tjong'])
    t.save(BL/'FORMAL_COMPLETE.json',[r for r in results if r['name']!='tjong'])
    metrics = ('avg_net','ron','dr','hu','cvar10')
    means = {name:{m:float(np.mean([r['metrics'][m] for r in rows])) for m in metrics}
             for name,rows in series.items()}
    # Historical MASK rows are references, not newly replayed evidence.
    references = {}
    for name in ('L0','L2_no_deceive','L2_full'):
        rows=[]
        for seed in range(2026082001,2026082021):
            file=t.ROOT/f'Formal_E2_E3_P4_STRICT_FFR_20260823/formal_e3/{name}/seed_{seed}/gate1_summary.json'
            raw=json.loads(file.read_text())
            r=next(iter(raw['E2_ladder'].values()))
            rows.append(dict(avg_net=r['avg_net'],ron=r['ron_game_rate'],dr=r['dealin_rate'],
                             hu=r['hu_rate'],cvar10=r['cvar_10pct']))
        references[name]=rows
    paired={}
    for name,rows in series.items():
        paired[name]={}
        for m in metrics:
            delta=np.array([r['metrics'][m]-ref[m] for r,ref in zip(rows,references['L2_full'])])
            se=float(stats.sem(delta)); mean=float(delta.mean())
            p=float(stats.ttest_1samp(delta,0).pvalue) if se else float(mean==0)
            half=float(stats.t.ppf(.975,19))*se
            paired[name][m]=dict(difference=mean,ci95=[mean-half,mean+half],p_nominal=p)
    t.save(BL/'COMPARISON.json',dict(fresh_means=means,by_seed=series,
        historical_references=references,paired_vs_historical_MASK=paired,
        limitation='P4 executor reused, but historical MASK checkpoint outcomes not freshly replayed; rules remain legacy; p-values nominal.'))
    lines=['# Unified P4 executor reevaluation','',
           'Frozen checkpoints; 20 paired seeds x 1,000 games per method. Eight CPU workers.',
           'Legacy engine retained. This is an execution-consistency repair, not validation of all Sichuan rules.',
           'MASK values remain historical references; no new MASK replay was performed.', '',
           '| Method | Mean | Ron | DR | Hu | Mean seed CVaR-10% |',
           '|---|---:|---:|---:|---:|---:|']
    for name,r in means.items():
        lines.append(f"| {name} | {r['avg_net']:.4f} | {r['ron']:.3%} | {r['dr']:.3%} | {r['hu']:.3%} | {r['cvar10']:.4f} |")
    (BL/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    t.save(BL/'STATUS.json',dict(stage='complete',completed_units=80,elapsed_seconds=time.time()-start))


if __name__ == '__main__':
    main()
