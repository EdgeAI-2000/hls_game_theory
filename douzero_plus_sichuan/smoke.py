"""Bounded integration validation only; not a formal learning experiment."""
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib

import numpy as np
import torch

from .models import PlusPolicy,Coach,losses,coach_target
from .rollout import collect


def actor(job):
    path,seed,beta=job
    torch.set_num_threads(1)
    data=torch.load(path,map_location='cpu',weights_only=True)
    policy=PlusPolicy().eval()
    policy.load_state_dict(data['policy'])
    coach=Coach().eval()
    coach.load_state_dict(data['coach'])
    return collect(policy,seed,coach,beta,epsilon=.01)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(92703)
    out=Path('DouZeroPlus_Sichuan_Prototype_20260927')
    out.mkdir(exist_ok=False)
    policy,coach=PlusPolicy(),Coach()
    optimizer=torch.optim.RMSprop(policy.parameters(),lr=1e-4,alpha=.99,eps=1e-5)
    coach_optimizer=torch.optim.RMSprop(coach.parameters(),lr=1e-4,alpha=.99,eps=1e-5)
    logs=[]
    with ProcessPoolExecutor(4,mp_context=mp.get_context('spawn')) as pool:
        for update in range(2):
            snapshot=out/f'actor_{update}.pt'
            torch.save(dict(policy=policy.state_dict(),coach=coach.state_dict()),snapshot)
            # beta=.3 on update 1 exercises the branch, NOT the formal schedule.
            beta=0 if update==0 else .3
            batches=list(pool.map(actor,[(str(snapshot),927030000000+update*8+i,beta) for i in range(8)]))
            accepted=[r for r in batches if r['accepted']]
            if not accepted:
                raise RuntimeError('All integration games rejected')
            values={k:torch.from_numpy(np.concatenate([r[k] for r in accepted])) for k in ('x','z','y','labels')}
            stats=[]
            for start in range(0,len(values['y']),64):
                args=[values[k][start:start+64] for k in ('x','z','y','labels')]
                total,parts=losses(policy,*args)
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(),40,error_if_nonfinite=True)
                optimizer.step()
                stats.append({k:float(v.detach()) for k,v in parts.items()})
            hands=torch.from_numpy(np.stack([r['hands'] for r in accepted]))
            missing=torch.from_numpy(np.stack([r['missing'] for r in accepted]))
            targets=coach_target([r['metrics']['net'] for r in accepted])
            coach_loss=-(targets*coach(hands,missing).log_softmax(-1)).sum(-1).mean()
            coach_optimizer.zero_grad(set_to_none=True)
            coach_loss.backward()
            torch.nn.utils.clip_grad_norm_(coach.parameters(),40,error_if_nonfinite=True)
            coach_optimizer.step()
            log=dict(update=update+1,attempted=8,accepted=len(accepted),beta_smoke_override=beta,
                     decisions=len(values['y']),mean_value_loss=float(np.mean([s['value'] for s in stats])),
                     mean_prediction_loss=float(np.mean([s['prediction'] for s in stats])),
                     coach_loss=float(coach_loss.detach()),games=[r['metrics'] for r in accepted])
            logs.append(log)
            print(json.dumps({k:v for k,v in log.items() if k!='games'}),flush=True)
    checkpoint=out/'integration_only.pt'
    torch.save(dict(policy=policy.state_dict(),coach=coach.state_dict()),checkpoint)
    reloaded=PlusPolicy().eval()
    reloaded.load_state_dict(torch.load(checkpoint,weights_only=True)['policy'])
    policy.eval()
    with torch.no_grad():
        a=policy(values['x'][:4],values['z'][:4])[0]
        b=reloaded(values['x'][:4],values['z'][:4])[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    report=dict(status='integration_passed_not_formal',updates=logs,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                official_commit='64a4dc29894082f752b43952a8f3a13868445c03',
                limits=['shared rules unresolved','16 attempted games only','beta overridden solely to exercise filtering','Coach target and four-seat sharing are adaptations'])
    (out/'REPORT.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
