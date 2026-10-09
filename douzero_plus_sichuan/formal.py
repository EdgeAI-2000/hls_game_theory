"""Frozen-budget DouZero+ adaptation with automatic development/holdout stages."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats
import torch

from douzero_sichuan import core
from douzero_sichuan.formal import summarize, read_rows, METRICS, SEEDS
from douzero_sichuan.train import atomic_save
from .models import PlusPolicy, Coach, losses, coach_target
from .rollout import collect

ROOT=Path(__file__).resolve().parents[1]
_policy=_coach=_version=None


def save(path,obj):
    temp=Path(str(path)+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(job):
    global _policy,_coach,_version
    path,version,seed,mode,beta=job
    if _policy is None:
        torch.set_num_threads(1)
        _policy,_coach=PlusPolicy().eval(),Coach().eval()
    if _version!=(path,version):
        data=torch.load(path,map_location='cpu',weights_only=True)
        _policy.load_state_dict(data['policy'])
        _coach.load_state_dict(data['coach'])
        _version=(path,version)
    if mode=='train':
        return collect(_policy,seed,_coach,beta)
    # Coach never sees or selects evaluation openings.
    if mode not in ('dev','holdout'):
        raise ValueError(mode)
    return core.episode(_policy,seed,0,True,'train' if mode=='dev' else 'holdout')[3]


def measures(rows):
    scores=np.array([r['net'][0] for r in rows])
    return dict(avg_net=float(scores.mean()),cvar10=float(np.sort(scores)[:max(1,len(scores)//10)].mean()),
                ron_game_rate=float(np.mean([r['ron'][0] for r in rows])),
                dealin_game_rate=float(np.mean([r['dr'][0] for r in rows])),
                hu_game_rate=float(np.mean([r['hu'][0] for r in rows])))


def comparison(out,plus):
    series={'DouZero+-adapted':plus}
    base=ROOT/'Formal_E2_E3_P4_STRICT_FFR_20260823/formal_e3'
    for name,variant in [('L0','L0'),('no-deceive','L2_no_deceive'),('MASK-full','L2_full')]:
        series[name]=[summarize(read_rows(base/variant/f'seed_{s}/gate1_games.jsonl'),s,True) for s in SEEDS]
    series['DouZero-adapted']=[summarize(read_rows(ROOT/'DouZero_Sichuan_Formal_20260926/holdout'/f'seed_{s}/games.jsonl'),s) for s in SEEDS]
    means={name:{m:float(np.mean([r[m] for r in rows])) for m in METRICS} for name,rows in series.items()}
    paired={}
    for name in list(series)[1:]:
        paired[name]={}
        for metric in METRICS:
            delta=np.array([a[metric]-b[metric] for a,b in zip(plus,series[name])])
            mean,se=float(delta.mean()),float(stats.sem(delta))
            half=float(stats.t.ppf(.975,19))*se
            p=float(stats.ttest_1samp(delta,0).pvalue) if se else float(mean==0)
            paired[name][metric]=dict(difference=mean,ci95=[mean-half,mean+half],p_nominal=p)
        running=0.
        for rank,m in enumerate(sorted(METRICS,key=lambda k:paired[name][k]['p_nominal'])):
            running=max(running,min(1.,(5-rank)*paired[name][m]['p_nominal']))
            paired[name][m]['p_holm']=running
    save(out/'comparison.json',dict(by_seed=series,means=means,paired_plus_minus_baseline=paired))
    lines=['# DouZero+-adapted: frozen legacy-environment comparison','',
           'WARNING: known shared scoring and runner boundary issues remain unresolved. This is not a rules-validated ranking.',
           'One training seed; model selected on TRAIN development data; 20 paired outer seeds x 1000 holdout games.',
           'Coach four-seat target and frame-based threshold schedule are explicit adaptations.','',
           '| System | Mean | Ron rate | DR | CVaR-10% | Hu rate | Excess-DIR vs no-deceive |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in means.items():
        excess=100*(r['ron_game_rate']-means['no-deceive']['ron_game_rate'])
        lines.append(f"| {name} | {r['avg_net']:.4f} | {r['ron_game_rate']:.3%} | {r['dealin_game_rate']:.3%} | {r['cvar10']:.4f} | {r['hu_game_rate']:.3%} | {excess:+.3f}pp |")
    lines+=['','Paired nominal and within-comparison Holm p-values and 95% CIs are in comparison.json.',
            'CVaR averages seed-level worst-decile means. Excess-DIR alone does not identify deception. Strict FFR is not evaluated.']
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='DouZeroPlus_Sichuan_Formal_20260927')
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--frames',type=int,default=5000000)
    p.add_argument('--dev-every',type=int,default=500000)
    p.add_argument('--dev-games',type=int,default=1000)
    p.add_argument('--holdout-games',type=int,default=1000)
    p.add_argument('--units',type=int,default=20)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not args.smoke and (args.frames,args.dev_every,args.dev_games,args.holdout_games,args.units)!=(5000000,500000,1000,1000,20):
        raise ValueError('Formal budget changes require an explicit new protocol; use --smoke for integration tests')
    os.chdir(ROOT)
    out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=args.resume)
    torch.set_num_threads(2)
    torch.manual_seed(92711)
    rng=np.random.default_rng(92711)
    files=list((ROOT/'douzero_plus_sichuan').glob('*.py'))+[ROOT/'douzero_sichuan/core.py',ROOT/'game.py',ROOT/'rule_engine.py',ROOT/'run_gate1_experiments.py',ROOT/'prompt_builder.py',ROOT/'tiles.py',ROOT/'opponent_curriculum.py',ROOT/'Opponent_Curriculum_formal/screened.json']
    hashes={str(f):digest(f) for f in files}
    config={k:v for k,v in vars(args).items() if k!='resume'}
    protocol=dict(config=config,hashes=hashes,training_seed=92711,train_seed_start=927110000000,dev_seed_start=927990000000,
                  outer_seeds=SEEDS[:args.units] if not args.smoke else [9279800001+i for i in range(args.units)],
                  selection='max raw mean on fixed TRAIN dev; ties earliest; excludes initialization',
                  coach='4-way highest-net seat target with tied mass; max_probability <= 1-beta',
                  beta='min(.30, .05*floor(completed_decisions/500000)); deliberate departure from original update-based schedule',
                  budget='played decision samples; attempts and rejected openings reported separately; abort after 20 attempts per requested game',
                  batch_size=3200,games_per_update=32,gradient_microbatch=128,
                  optimizer='RMSprop lr1e-4 alpha.99 eps1e-5; gradient norm40; CE + MC MSE; detached prediction into value',
                  environment='frozen legacy rules; known errors unresolved; same local impossible-kong mask as DouZero',
                  formal_evidence_rules_validated=False,official_commit='64a4dc29894082f752b43952a8f3a13868445c03')
    frames=games=attempts=update=0
    best=None
    next_dev=args.dev_every
    policy,coach=PlusPolicy().to(args.device),Coach().to(args.device)
    opt=torch.optim.RMSprop(policy.parameters(),lr=1e-4,alpha=.99,eps=1e-5)
    copt=torch.optim.RMSprop(coach.parameters(),lr=1e-4,alpha=.99,eps=1e-5)
    latest=out/'latest.pt'
    snapshot=out/'actor.pt'
    def state(stage,**extra):
        save(out/'status.json',dict(stage=stage,frames=frames,games=games,attempted_openings=attempts,updated_at=time.time(),**extra))
    def freeze_actor(path):
        atomic_save(dict(policy={k:v.detach().cpu() for k,v in policy.state_dict().items()},coach={k:v.detach().cpu() for k,v in coach.state_dict().items()}),path)
    def checkpoint():
        atomic_save(dict(policy=policy.state_dict(),coach=coach.state_dict(),optimizer=opt.state_dict(),coach_optimizer=copt.state_dict(),
                         frames=frames,games=games,attempts=attempts,update=update,next_dev=next_dev,best=best,rng=rng.bit_generator.state,
                         log_bytes={name:(out/name).stat().st_size if (out/name).exists() else 0 for name in ('training.jsonl','games.jsonl','opening_attempts.jsonl')}),latest)
    if args.resume:
        old=json.loads((out/'PROTOCOL.json').read_text())
        if old!=protocol:
            raise RuntimeError('Protocol or source changed; refusing resume')
        saved=torch.load(latest,map_location=args.device,weights_only=False)
        policy.load_state_dict(saved['policy']);coach.load_state_dict(saved['coach'])
        opt.load_state_dict(saved['optimizer']);copt.load_state_dict(saved['coach_optimizer'])
        frames,games,attempts,update,next_dev,best=[saved[k] for k in ('frames','games','attempts','update','next_dev','best')]
        rng.bit_generator.state=saved['rng']
        # Discard only this run's uncommitted log tail after an interrupted batch.
        for name,size in saved['log_bytes'].items():
            path=out/name
            if path.exists():
                with path.open('r+b') as stream:stream.truncate(size)
        if best is not None:
            import shutil
            shutil.copyfile(out/f"best_{best['frames']}.pt",out/'selected.pt')
    else:
        save(out/'PROTOCOL.json',protocol)
    start=time.monotonic();initial_frames=frames
    try:
        with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
            while frames<args.frames:
                state('training',target_frames=args.frames)
                freeze_actor(snapshot)
                beta=min(.30,.05*(frames//500000))
                batch=[];tried=0;rejected=0
                with (out/'opening_attempts.jsonl').open('a') as stream:
                    while len(batch)<32:
                        count=32-len(batch)
                        if tried+count>640:
                            raise RuntimeError('Coach rejection attempt budget exceeded')
                        jobs=[(str(snapshot),update,927110000000+attempts+i,'train',beta) for i in range(count)]
                        results=list(pool.map(worker,jobs))
                        for job,r in zip(jobs,results):
                            stream.write(json.dumps(dict(update=update,seed=job[2],beta=beta,accepted=r['accepted'],probabilities=r['probabilities']))+'\n')
                            if r['accepted']:batch.append(r)
                            else:rejected+=1
                        attempts+=count;tried+=count
                arrays={k:np.concatenate([r[k] for r in batch]) for k in ('x','z','y','labels')}
                order=rng.permutation(len(arrays['y']))
                loss_records=[]
                for offset in range(0,len(order),3200):
                    idx=order[offset:offset+3200]
                    opt.zero_grad(set_to_none=True)
                    # Same minibatch mean gradient, smaller memory footprint.
                    for j in range(0,len(idx),128):
                        sub=idx[j:j+128]
                        tensors=[torch.from_numpy(arrays[k][sub]).to(args.device) for k in ('x','z','y','labels')]
                        loss,parts=losses(policy,*tensors)
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite policy loss')
                        (loss*len(sub)/len(idx)).backward()
                        loss_records.append([float(parts[k].detach()) for k in ('value','prediction')])
                    torch.nn.utils.clip_grad_norm_(policy.parameters(),40,error_if_nonfinite=True)
                    opt.step()
                hands=torch.from_numpy(np.stack([r['hands'] for r in batch])).to(args.device)
                missing=torch.from_numpy(np.stack([r['missing'] for r in batch])).to(args.device)
                target=coach_target([r['metrics']['net'] for r in batch]).to(args.device)
                closs=-(target*coach(hands,missing).log_softmax(-1)).sum(-1).mean()
                if not torch.isfinite(closs):raise RuntimeError('Nonfinite Coach loss')
                copt.zero_grad(set_to_none=True);closs.backward()
                torch.nn.utils.clip_grad_norm_(coach.parameters(),40,error_if_nonfinite=True);copt.step()
                frames+=len(order);games+=len(batch);update+=1
                log=dict(update=update,frames=frames,games=games,attempts=attempts,beta=beta,rejected_batch=rejected,
                         value_loss=float(np.mean(loss_records,0)[0]),prediction_loss=float(np.mean(loss_records,0)[1]),
                         coach_loss=float(closs.detach()),frames_per_second=(frames-initial_frames)/(time.monotonic()-start))
                with (out/'training.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
                with (out/'games.jsonl').open('a') as f:
                    for r in batch:f.write(json.dumps(dict(r['metrics'],policy_version=update-1))+'\n')
                print(json.dumps(log),flush=True)
                if frames>=next_dev or frames>=args.frames:
                    freeze_actor(snapshot)
                    state('development',target_games=args.dev_games)
                    # Distinct mode routes to unfiltered TRAIN evaluation.
                    rows=list(pool.map(worker,[(str(snapshot),update,927990000000+i,'dev',0) for i in range(args.dev_games)]))
                    metrics=measures(rows)
                    save(out/f'dev_{frames}.json',dict(frames=frames,metrics=metrics,rows=rows))
                    if best is None or metrics['avg_net']>best['avg_net']:
                        best=dict(frames=frames,avg_net=metrics['avg_net'])
                        freeze_actor(out/f'best_{frames}.pt')
                        freeze_actor(out/'selected.pt')
                    next_dev+=args.dev_every
                checkpoint()
            selected=out/'selected.pt';expected=digest(selected)
            save(out/'SELECTION.json',dict(best=best,sha256=expected,holdout_seen=False))
            holdout=out/'holdout';holdout.mkdir(exist_ok=True)
            unit_metrics=[]
            for seed in protocol['outer_seeds']:
                if any(digest(f)!=h for f,h in hashes.items()):raise RuntimeError('Source changed')
                if digest(selected)!=expected:raise RuntimeError('Selected checkpoint changed')
                state('holdout',outer_seed=seed,completed_units=len(unit_metrics))
                folder=holdout/f'seed_{seed}';folder.mkdir(exist_ok=True)
                rows=[]
                for offset in range(0,args.holdout_games,100):
                    end=min(args.holdout_games,offset+100)
                    rows.extend(pool.map(worker,[(str(selected),expected,seed*1000000+i,'holdout',0) for i in range(offset,end)]))
                    save(folder/'progress.json',dict(completed=len(rows),target=args.holdout_games))
                simple=[dict(seed=r['seed'],net=r['net'][0],ron=r['ron'][0],dealt_in=r['dr'][0],hu=r['hu'][0],max_steps_or_unsettled=0,four_seat_metrics=r) for r in rows]
                path=folder/'games.jsonl';temp=folder/'games.jsonl.tmp'
                with temp.open('w') as f:
                    for r in simple:f.write(json.dumps(r)+'\n')
                os.replace(temp,path)
                metrics=summarize(simple,seed) if args.holdout_games==1000 else measures(rows)
                unit_metrics.append(metrics)
                save(folder/'summary.json',dict(metrics=metrics,checkpoint_sha256=expected,games_sha256=digest(path),games=len(rows)))
            if not args.smoke:comparison(holdout,unit_metrics)
            state('smoke_complete' if args.smoke else 'complete',holdout_games=args.units*args.holdout_games,checkpoint_sha256=expected)
    except Exception as exc:
        state('failed',error=repr(exc))
        raise


if __name__=='__main__':
    main()
