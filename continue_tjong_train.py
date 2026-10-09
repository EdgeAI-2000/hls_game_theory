"""Continue Tjong PPO from a frozen latest checkpoint on a new budget."""
import argparse, json, multiprocessing as mp, os, shutil, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch

from tjong_sichuan.model import Tjong, tensor_batch
from tjong_sichuan.train import worker, measures, advantages, ppo_loss, digest, save, frozen_copy

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--target-frames',type=int,default=6000000)
    p.add_argument('--dev-every',type=int,default=200000)
    p.add_argument('--dev-games',type=int,default=1000)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args(); a.source=a.source.resolve(); a.out=a.out.resolve()
    if a.out.exists():
        existing = list(a.out.iterdir())
        if set(x.name for x in existing) - {'latest.pt', 'run.log'}:
            raise RuntimeError(f'Output exists with unexpected files: {a.out}')
    else:
        a.out.mkdir(parents=True)
    latest=a.source/'latest.pt'
    if not (a.out/'latest.pt').exists(): shutil.copy2(latest,a.out/'latest.pt')
    torch.set_num_threads(2); torch.manual_seed(92841)
    rng=np.random.default_rng(92841)
    model=Tjong().to(a.device); optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    saved=torch.load(a.out/'latest.pt',map_location=a.device,weights_only=False)
    model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
    phase,epoch,frames,games,update,bc_best,stale,best,next_dev=[saved[k] for k in ('phase','epoch','frames','games','update','bc_best','stale','best','next_dev')]
    rng.bit_generator.state=saved['rng']; start=time.monotonic(); start_frames=frames
    protocol=dict(source=str(a.source), source_latest_sha256=digest(latest), target_frames=a.target_frames,
                  start_frames=frames, start_update=update, device=a.device, workers=a.workers,
                  train_seed=92841, dev_seed_start=928490000000, holdout_used_for_selection=False,
                  note='Continuation from frozen PPO latest.pt; development-only selection; no holdout tuning')
    save(a.out/'CONTINUATION_PROTOCOL.json',protocol)
    def snapshot(path): torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
    def checkpoint():
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),phase=phase,epoch=epoch,
                        frames=frames,games=games,update=update,bc_best=bc_best,stale=stale,best=best,
                        next_dev=next_dev,rng=rng.bit_generator.state,log_bytes={}),a.out/'latest.pt')
    def dev(pool):
        nonlocal best
        snapshot(a.out/'actor.pt')
        rows=list(pool.map(worker,[(str(a.out/'actor.pt'),f'dev_{frames}',928490000000+i,'eval','train') for i in range(a.dev_games)]))
        metric=measures(rows); save(a.out/f'dev_{frames}.json',dict(frames=frames,metrics=metric,rows=rows))
        if best is None or metric['avg_net']>best['avg_net']:
            best=dict(frames=frames,avg_net=metric['avg_net']); snapshot(a.out/f'best_{frames}.pt'); frozen_copy(a.out/f'best_{frames}.pt',a.out/'selected.pt')
        print('DEV '+json.dumps(dict(frames=frames,**metric)),flush=True)
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
        dev(pool)
        while frames<a.target_frames:
            snapshot(a.out/'actor.pt')
            results=list(pool.map(worker,[(str(a.out/'actor.pt'),update,928420000000+games+i,'ppo','train') for i in range(32)]))
            data={k:np.concatenate([d[k] for d,m in results]) for k in results[0][0]}; n=len(data['kind'])
            values=[]; differences=[]
            with torch.no_grad():
                for begin in range(0,n,16):
                    b=tensor_batch({k:v[begin:begin+16] for k,v in data.items()},a.device)
                    lp,_,v=model.evaluate(b); values.extend(v.cpu().numpy()); differences.extend((lp-b['old_logp']).abs().cpu().numpy())
            if max(differences)>1e-3: raise RuntimeError(f'Likelihood mismatch: {max(differences)}')
            value=np.asarray(values,np.float32); reward=data['actual_reward']+.1*data['fan_reward']
            adv,returns=advantages(reward,data['done'],value); adv=(adv-adv.mean())/(adv.std()+1e-8)
            data.update(advantage=adv.astype(np.float32),returns=returns.astype(np.float32),old_value=value)
            parts=[]
            for _ in range(3):
                for begin in range(0,n,1024):
                    ids=rng.permutation(n)[begin:begin+1024]; optimizer.zero_grad(set_to_none=True)
                    for pos in range(0,len(ids),16):
                        idx=ids[pos:pos+16]; b=tensor_batch({k:v[idx] for k,v in data.items()},a.device)
                        lp,ent,v=model.evaluate(b); loss,part=ppo_loss(lp,b['old_logp'],b['advantage'],v,b['old_value'],b['returns'],ent)
                        if not torch.isfinite(loss): raise RuntimeError('Nonfinite PPO loss')
                        (loss*len(idx)/len(ids)).backward(); parts.append(part)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True); optimizer.step()
            frames+=n; games+=len(results); update+=1
            log=dict(stage='ppo',frames=frames,games=games,update=update,actual_reward_mean=float(data['actual_reward'].mean()),fan_reward_mean=float(data['fan_reward'].mean()),policy_loss=float(np.mean([x['policy'] for x in parts])),critic_loss=float(np.mean([x['critic'] for x in parts])),kl=float(np.mean([x['kl'] for x in parts])),frames_per_second=(frames-start_frames)/max(1,time.monotonic()-start))
            with (a.out/'training.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            print(json.dumps(log),flush=True)
            if frames>=next_dev or frames>=a.target_frames:
                dev(pool); next_dev=(frames//a.dev_every+1)*a.dev_every
            checkpoint()
    save(a.out/'SELECTION.json',dict(best=best,holdout_used_for_selection=False,source_checkpoint=str(latest),target_frames=a.target_frames))
    save(a.out/'STATUS.json',dict(stage='training_complete',frames=frames,updates=update,best=best))

if __name__=='__main__': main()
