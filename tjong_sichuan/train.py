"""BC -> on-policy self-play PPO -> development selection -> paired holdout."""
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

from douzero_sichuan.train import atomic_save
from douzero_sichuan.formal import SEEDS, METRICS, summarize, read_rows
from .model import Tjong, tensor_batch
from .rollout import collect

ROOT=Path(__file__).resolve().parents[1]
_actor=_version=None


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,data):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def frozen_copy(source,target):
    temp=Path(str(target)+'.tmp');shutil.copyfile(source,temp);os.replace(temp,target)


def worker(job):
    global _actor,_version
    path,version,seed,mode,split=job
    torch.set_num_threads(1)
    if mode!='bc':
        if _actor is None:_actor=Tjong().eval()
        if _version!=(path,version):
            _actor.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));_version=(path,version)
    return collect(_actor,seed,mode,split)


def measures(rows):
    a=np.array([r['net'][0] for r in rows])
    return dict(avg_net=float(a.mean()),cvar10=float(np.sort(a)[:max(1,len(a)//10)].mean()),
        ron_game_rate=float(np.mean([r['ron'][0] for r in rows])),dealin_game_rate=float(np.mean([r['dr'][0] for r in rows])),
        hu_game_rate=float(np.mean([r['hu'][0] for r in rows])))


def advantages(rewards,done,values,gamma=1.,lam=.95):
    adv=np.zeros_like(rewards);carry=0.
    for i in range(len(rewards)-1,-1,-1):
        next_v=0. if done[i] else values[i+1]
        delta=rewards[i]+gamma*next_v-values[i]
        carry=delta+gamma*lam*(not done[i])*carry
        adv[i]=carry
    return adv,adv+values


def ppo_loss(logp,old_logp,adv,value,old_value,target,entropy):
    ratio=(logp-old_logp).clamp(-20,20).exp()
    policy=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
    clipped=old_value+(value-old_value).clamp(-.3,.3)
    critic=.5*torch.maximum((value-target).square(),(clipped-target).square()).mean()
    loss=policy+.5*critic-.01*entropy.mean()
    return loss,dict(policy=float(policy.detach()),critic=float(critic.detach()),
                     kl=float(((ratio-1)-(logp-old_logp)).mean().detach()),
                     clip_fraction=float(((ratio-1).abs()>.2).float().mean()))


def compare(out,series):
    data={'Tjong-adapted':series}
    base=ROOT/'Formal_E2_E3_P4_STRICT_FFR_20260823/formal_e3'
    for name,variant in [('MASK-full','L2_full'),('no-deceive','L2_no_deceive'),('L0','L0')]:
        data[name]=[summarize(read_rows(base/variant/f'seed_{s}/gate1_games.jsonl'),s,True) for s in SEEDS]
    for name,folder in [('DouZero-adapted','DouZero_Sichuan_Formal_20260926'),('DouZero+-adapted','DouZeroPlus_Sichuan_Formal_20260927'),('DRON-1M','DRON_MoE_Sichuan_Formal_20260927')]:
        data[name]=[summarize(read_rows(ROOT/folder/'holdout'/f'seed_{s}/games.jsonl'),s) for s in SEEDS]
    means={name:{k:float(np.mean([r[k] for r in rows])) for k in METRICS} for name,rows in data.items()}
    paired={}
    for name,rows in list(data.items())[1:]:
        paired[name]={}
        for k in METRICS:
            d=np.array([a[k]-b[k] for a,b in zip(series,rows)])
            se=float(stats.sem(d));half=float(stats.t.ppf(.975,19))*se
            p=float(stats.ttest_1samp(d,0).pvalue) if se else float(d.mean()==0)
            paired[name][k]=dict(delta=float(d.mean()),ci95=[float(d.mean()-half),float(d.mean()+half)],p_nominal=p)
        running=0.
        for i,k in enumerate(sorted(METRICS,key=lambda k:paired[name][k]['p_nominal'])):
            running=max(running,min(1.,(len(METRICS)-i)*paired[name][k]['p_nominal']));paired[name][k]['p_holm']=running
    save(out/'comparison.json',dict(by_seed=data,means=means,paired_tjong_minus_baseline=paired))
    lines=['# Tjong-adapted: legacy Sichuan benchmark','',
           'Paper-based adaptation, not original-author-code reproduction. Known rule defects unresolved.',
           'One training seed, fixed budget; training resources differ; no convergence guarantee.','',
           '| Method | Mean | Ron | DR | CVaR10 | Hu |','|---|---:|---:|---:|---:|---:|']
    for name,m in means.items():lines.append(f"| {name} | {m['avg_net']:.4f} | {m['ron_game_rate']:.3%} | {m['dealin_game_rate']:.3%} | {m['cvar10']:.4f} | {m['hu_game_rate']:.3%} |")
    lines+=['','Per-seed nominal/Holm statistics: comparison.json. Strict FFR not evaluated.',
            'BC teacher source, full tile-specific kong actions and fan mapping are explicit adaptations.']
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='Tjong_Sichuan_Formal_20260928')
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--bc-games',type=int,default=8000)
    p.add_argument('--bc-epochs',type=int,default=20)
    p.add_argument('--frames',type=int,default=2000000)
    p.add_argument('--dev-every',type=int,default=200000)
    p.add_argument('--dev-games',type=int,default=1000)
    p.add_argument('--holdout-games',type=int,default=1000)
    p.add_argument('--units',type=int,default=20)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not a.smoke and (a.bc_games,a.bc_epochs,a.frames,a.dev_every,a.dev_games,a.holdout_games,a.units)!=(8000,20,2000000,200000,1000,1000,20):
        raise ValueError('New budgets require a new formal protocol; smoke uses --smoke')
    if a.bc_games<32 or a.bc_games%32:raise ValueError('BC games must be divisible by32, with at least32')
    os.chdir(ROOT)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=a.resume)
    torch.set_num_threads(2);torch.manual_seed(92841)
    if a.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.25,torch.device(a.device))
    rng=np.random.default_rng(92841)
    files=list((ROOT/'tjong_sichuan').glob('*.*'))+[ROOT/f for f in (
        'douzero_sichuan/core.py','douzero_sichuan/train.py','douzero_sichuan/formal.py','game.py','rule_engine.py','tiles.py',
        'prompt_builder.py','run_gate1_experiments.py','belief_oracle.py','public_belief_model.py','policy_metrics.py',
        'mask_llm.py','ppo_features.py','experiment_trace.py','llm_backends.py','opponent_curriculum.py',
        'run_mix_ppo_formal_holdout.py','Opponent_Curriculum_formal/screened.json')]
    hashes={str(f.relative_to(ROOT)):digest(f) for f in files}
    model=Tjong().to(a.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    protocol=dict(config={k:v for k,v in vars(a).items() if k!='resume'},hashes=hashes,
        parameters=sum(p.numel() for p in model.parameters()),python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        paper='https://doi.org/10.1049/cit2.12298',author_code='https://github.com/750251985/Tjong (404 at review)',
        architecture='TIT: inner3 outer3 width448 heads8 FFN1792 memory4; action-conditioned second pass; 3 heads',
        privileged='TRAIN critic only: opponent hands and wall tile counts; policy excludes this channel',
        bc='8000 synthetic teacher games, seed split by game_index modulo10; max20epochs patience4 on joint validation NLL',
        bc_seed_start=928410000000,ppo_seed_start=928420000000,dev_seed_start=928490000000,
        teacher='P0 independent rule heuristic; P1-P3 frozen six train profiles; illegal-under-policy calls become pass in BC',
        rl='four-seat synchronous self-play; 32 games per PPO update; all seats contribute',
        optimization='Adam1e-4; minibatch1024/micro16; BC NLL; PPO3epochs; policyclip.2 valueclip.3 gradnorm.5',
        gamma=1.,gae_lambda=.95,entropy_coefficient=.01,value_coefficient=.5,fan_coefficient=.1,
        fan='Eq6 tile-pattern supports rescaled to capped payout with zero-fan base; Eq7 winner normalized by trajectory length; meld/ron giver negatives; sum across win events',
        environment='Legacy shared engine unchanged; known scoring defects; explicit tile-specific kong selection differs from previous macroaction baselines',
        selection='fixed TRAIN development raw mean, earliest tie; no holdout-based selection; includes BC baseline',
        holdout_seeds=SEEDS if not a.smoke else [9284800001+i for i in range(a.units)],
        runtime_memory='GPU allocator capped at25% total memory; other users untouched')
    latest,actor=out/'latest.pt',out/'actor.pt'
    phase='bc';epoch=frames=games=update=0
    bc_best=None;stale=0;best=None;next_dev=a.dev_every
    def status(stage,**kw):save(out/'status.json',dict(stage=stage,phase=phase,epoch=epoch,frames=frames,games=games,update=update,target_frames=a.frames,updated_at=time.time(),**kw))
    def snapshot(path):atomic_save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
    def check():
        if any(digest(ROOT/f)!=h for f,h in hashes.items()):raise RuntimeError('Frozen source changed')
    def checkpoint():
        sizes={name:(out/name).stat().st_size if (out/name).exists() else 0 for name in ('training.jsonl','ppo_games.jsonl')}
        atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),phase=phase,epoch=epoch,frames=frames,
            games=games,update=update,bc_best=bc_best,stale=stale,best=best,next_dev=next_dev,
            rng=rng.bit_generator.state,log_bytes=sizes),latest)
    if a.resume:
        if json.loads((out/'PROTOCOL.json').read_text())!=protocol:raise RuntimeError('Resume protocol mismatch')
        saved=torch.load(latest,map_location=a.device,weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        phase,epoch,frames,games,update,bc_best,stale,best,next_dev=[saved[k] for k in ('phase','epoch','frames','games','update','bc_best','stale','best','next_dev')]
        rng.bit_generator.state=saved['rng']
        for name,size in saved['log_bytes'].items():
            if (out/name).exists():
                with (out/name).open('r+b') as f:f.truncate(size)
        if best is not None:frozen_copy(out/f"best_{best['frames']}.pt",out/'selected.pt')
        del saved
    else:
        save(out/'PROTOCOL.json',protocol)
        for name in hashes:
            dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        checkpoint()
    start=time.monotonic();start_frames=frames
    try:
        with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
            shards=out/'bc_data';shards.mkdir(exist_ok=True)
            if phase=='bc':
                for offset in range(0,a.bc_games,32):
                    check();status('collecting_bc',completed_games=offset,target_games=a.bc_games)
                    marker=shards/f'{offset}.json'
                    if marker.exists():
                        manifest=json.loads(marker.read_text())
                        if any(digest(shards/name)!=h for name,h in manifest['files'].items()):raise RuntimeError('BC shard changed')
                        continue
                    results=list(pool.map(worker,[('',0,928410000000+i,'bc','train') for i in range(offset,offset+32)]))
                    manifest={'files':{},'games':[]}
                    for split in ('train','valid'):
                        chosen=[d for j,(d,m) in enumerate(results) if ((offset+j)%10==0)==(split=='valid')]
                        data={k:np.concatenate([d[k] for d in chosen]) for k in chosen[0]}
                        path=shards/f'{offset}_{split}.npz';temp=Path(str(path)+'.tmp')
                        with temp.open('wb') as f:np.savez_compressed(f,**data)
                        os.replace(temp,path);manifest['files'][path.name]=digest(path)
                    manifest['games']=[m for _,m in results];save(marker,manifest)
                train_files=sorted(shards.glob('*_train.npz'));val_files=sorted(shards.glob('*_valid.npz'))
                while epoch<a.bc_epochs and stale<4:
                    check();status('supervised_training',target_epochs=a.bc_epochs)
                    total=count=0
                    for file_index in rng.permutation(len(train_files)):
                        with np.load(train_files[file_index]) as z:data={k:z[k] for k in z.files}
                        order=rng.permutation(len(data['kind']))
                        for begin in range(0,len(order),1024):
                            ids=order[begin:begin+1024];optimizer.zero_grad(set_to_none=True)
                            for pos in range(0,len(ids),16):
                                idx=ids[pos:pos+16];b=tensor_batch({k:v[idx] for k,v in data.items()},a.device)
                                logp,_,_=model.evaluate(b);loss=-logp.mean()
                                if not torch.isfinite(loss):raise RuntimeError('Nonfinite BC loss')
                                (loss*len(idx)/len(ids)).backward();total+=float(loss.detach())*len(idx);count+=len(idx)
                            torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True);optimizer.step()
                    valid_total=valid_count=correct=0
                    with torch.no_grad():
                        for path in val_files:
                            with np.load(path) as z:data={k:z[k] for k in z.files}
                            for begin in range(0,len(data['kind']),16):
                                b=tensor_batch({k:v[begin:begin+16] for k,v in data.items()},a.device)
                                lp,_,_=model.evaluate(b);valid_total+=float(-lp.sum());valid_count+=len(lp)
                    epoch+=1;nll=valid_total/valid_count
                    if bc_best is None or nll<bc_best['nll']:
                        bc_best=dict(epoch=epoch,nll=nll);stale=0;snapshot(out/f'bc_best_{epoch}.pt')
                    else:stale+=1
                    log=dict(stage='bc',epoch=epoch,train_nll=total/count,valid_nll=nll,train_samples=count,valid_samples=valid_count)
                    with (out/'training.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
                    print(json.dumps(log),flush=True);checkpoint()
                model.load_state_dict(torch.load(out/f"bc_best_{bc_best['epoch']}.pt",map_location=a.device,weights_only=True))
                optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
                phase='ppo';checkpoint()

            def development():
                nonlocal best
                check();snapshot(actor);status('development',target_games=a.dev_games)
                rows=list(pool.map(worker,[(str(actor),f'dev_{frames}',928490000000+i,'eval','train') for i in range(a.dev_games)]))
                metrics=measures(rows);save(out/f'dev_{frames}.json',dict(frames=frames,metrics=metrics,rows=rows))
                if best is None or metrics['avg_net']>best['avg_net']:
                    best=dict(frames=frames,avg_net=metrics['avg_net']);snapshot(out/f'best_{frames}.pt');frozen_copy(out/f'best_{frames}.pt',out/'selected.pt')
                print('DEV '+json.dumps(dict(frames=frames,**metrics)),flush=True)
            if phase=='ppo' and best is None:development();checkpoint()
            while phase=='ppo' and frames<a.frames:
                check();snapshot(actor);status('selfplay_ppo')
                results=list(pool.map(worker,[(str(actor),update,928420000000+games+i,'ppo','train') for i in range(32)]))
                data={k:np.concatenate([d[k] for d,m in results]) for k in results[0][0]}
                n=len(data['kind']);values=[];difference=[]
                with torch.no_grad():
                    for begin in range(0,n,16):
                        b=tensor_batch({k:v[begin:begin+16] for k,v in data.items()},a.device)
                        lp,_,v=model.evaluate(b);values.extend(v.cpu().numpy());difference.extend((lp-b['old_logp']).abs().cpu().numpy())
                if max(difference)>1e-3:raise RuntimeError(f'On-policy likelihood mismatch: {max(difference)}')
                value=np.asarray(values,np.float32)
                reward=data['actual_reward']+.1*data['fan_reward']
                adv,returns=advantages(reward,data['done'],value)
                adv=(adv-adv.mean())/(adv.std()+1e-8)
                data.update(advantage=adv.astype(np.float32),returns=returns.astype(np.float32),old_value=value)
                parts=[]
                for _ in range(3):
                    order=rng.permutation(n)
                    for begin in range(0,n,1024):
                        ids=order[begin:begin+1024];optimizer.zero_grad(set_to_none=True)
                        for pos in range(0,len(ids),16):
                            idx=ids[pos:pos+16];b=tensor_batch({k:v[idx] for k,v in data.items()},a.device)
                            lp,ent,v=model.evaluate(b)
                            loss,part=ppo_loss(lp,b['old_logp'],b['advantage'],v,b['old_value'],b['returns'],ent)
                            if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
                            (loss*len(idx)/len(ids)).backward();parts.append(part)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True);optimizer.step()
                frames+=n;games+=len(results);update+=1
                log=dict(stage='ppo',frames=frames,games=games,update=update,
                    actual_reward_mean=float(data['actual_reward'].mean()),fan_reward_mean=float(data['fan_reward'].mean()),
                    policy_loss=float(np.mean([x['policy'] for x in parts])),critic_loss=float(np.mean([x['critic'] for x in parts])),
                    kl=float(np.mean([x['kl'] for x in parts])),frames_per_second=(frames-start_frames)/max(1,time.monotonic()-start))
                with (out/'training.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
                with (out/'ppo_games.jsonl').open('a') as f:
                    for _,m in results:f.write(json.dumps(m)+'\n')
                print(json.dumps(log),flush=True)
                if frames>=next_dev or frames>=a.frames:
                    development();next_dev=(frames//a.dev_every+1)*a.dev_every
                checkpoint()
            phase='holdout';checkpoint()
            selected=out/'selected.pt';expected=digest(selected)
            save(out/'SELECTION.json',dict(best=best,sha256=expected,holdout_used_for_selection=False))
            holdout=out/'holdout';holdout.mkdir(exist_ok=True);summaries=[]
            for outer in protocol['holdout_seeds'][:a.units]:
                check()
                if digest(selected)!=expected:raise RuntimeError('Selected checkpoint changed')
                folder=holdout/f'seed_{outer}';folder.mkdir(exist_ok=True)
                path=folder/'games.jsonl';marker=folder/'summary.json'
                if marker.exists():
                    m=json.loads(marker.read_text())
                    if m['checkpoint_sha256']!=expected or m['games_sha256']!=digest(path):raise RuntimeError('Existing holdout changed')
                    summaries.append(m['metrics']);continue
                status('holdout',outer_seed=outer,completed_units=len(summaries));rows=[]
                for offset in range(0,a.holdout_games,100):
                    rows.extend(pool.map(worker,[(str(selected),expected,outer*1000000+i,'eval','holdout') for i in range(offset,min(a.holdout_games,offset+100))]))
                    save(folder/'progress.json',dict(completed=len(rows),target=a.holdout_games))
                simple=[dict(seed=m['seed'],net=m['net'][0],ron=m['ron'][0],dealt_in=m['dr'][0],hu=m['hu'][0],max_steps_or_unsettled=0,four_seat_metrics=m) for m in rows]
                if [r['seed'] for r in simple]!=[outer*1000000+i for i in range(a.holdout_games)]:raise RuntimeError('Seed mismatch')
                temp=folder/'games.jsonl.tmp'
                with temp.open('w') as f:
                    for row in simple:f.write(json.dumps(row)+'\n')
                os.replace(temp,path)
                metrics=summarize(simple,outer) if a.holdout_games==1000 else measures(rows)
                save(marker,dict(metrics=metrics,checkpoint_sha256=expected,games_sha256=digest(path)));summaries.append(metrics)
            if not a.smoke:compare(holdout,summaries)
            status('smoke_complete' if a.smoke else 'complete',holdout_games=a.units*a.holdout_games)
    except Exception as exc:
        status('failed',error=repr(exc));raise


if __name__=='__main__':main()
