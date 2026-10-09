"""GPU-only inference adapter; retain frozen P4 executor and Tjong weights."""
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

import unified_tjong_eval as shared

CHECKPOINT_PATH = Path(os.environ.get(
    'TJONG_CHECKPOINT',
    str(shared.ROOT/'Tjong_Sichuan_Formal_20260928_v2/selected.pt'),
)).resolve()
OUT = Path(os.environ.get(
    'TJONG_EVAL_OUT',
    str(shared.ROOT/'Tjong_Unified_P4_GPU1_20260929'),
)).resolve()
shared.CHECKPOINT = CHECKPOINT_PATH
shared.EXPECTED = shared.digest(CHECKPOINT_PATH)


class GPUActor:
    def __init__(self, model):
        self.model = model.to('cuda:0').eval()

    @torch.inference_mode()
    def choose(self, batch, rng, greedy=False):
        assert greedy, 'Evaluation only'
        batch = {k:v.to('cuda:0') for k,v in batch.items()}
        _, embeddings, action = self.model.first(batch)
        # Match the frozen CPU decoder's normalization and argmax semantics.
        probabilities = action.probs[0].cpu().numpy().astype(float)
        probabilities /= probabilities.sum()
        kind = int(probabilities.argmax())
        kd = torch.tensor([kind], device='cuda:0')
        tile_dist = self.model.second(batch, embeddings, kd)
        probabilities = tile_dist.probs[0].cpu().numpy().astype(float)
        probabilities /= probabilities.sum()
        tile = int(probabilities.argmax())
        logp = action.log_prob(kd) + tile_dist.log_prob(torch.tensor([tile],device='cuda:0'))
        return kind, tile, float(logp.item())


CPU_LOAD = shared.load_model


def gpu_load():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1', 'Physical GPU1 required'
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.04, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return GPUActor(CPU_LOAD())


def worker(seed):
    # Only this spawned worker's model factory changes; source files stay frozen.
    shared.load_model = gpu_load
    return shared.unit(str(OUT),seed,1000,False)


def main():
    os.chdir(shared.ROOT)
    OUT.mkdir(exist_ok=False)
    protocol = json.loads((shared.ROOT/'Tjong_Unified_P4_20260929/PROTOCOL.json').read_text())
    protocol.update(checkpoint=str(CHECKPOINT_PATH), checkpoint_sha256=shared.EXPECTED,
                    device='physical GPU1 / visible cuda:0', workers=8, precision='float32, TF32 off',
                    gpu_memory_fraction_per_worker=.04, scope='Tjong only, 20,000 fresh games',
                    prior_cpu_partial='retained separately, not mixed into this run')
    protocol['sources']['launch_tjong_gpu_eval.py'] = shared.digest(__file__)
    shared.save(OUT/'PROTOCOL.json',protocol)
    shared.verify_sources(OUT)
    cpu, gpu = CPU_LOAD(), gpu_load()
    checks=[]
    for seed in (2026082001000000,2026082002000000):
        a, b = shared.run_game(cpu,seed), shared.run_game(gpu,seed)
        assert shared.stable(a) == shared.stable(b), f'CPU/GPU trajectory mismatch: {seed}'
        checks.append(dict(seed=seed,net=b['agent_net'],decisions=len(b['policy_trace'])))
    shared.save(OUT/'GPU_PARITY_PASSED.json',checks)
    del cpu, gpu
    torch.cuda.empty_cache()
    start=time.time()
    shared.save(OUT/'STATUS.json',dict(stage='running',completed_units=0,total_units=20,started_at=start))
    results=[]
    try:
        with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(worker,seed) for seed in range(2026082001,2026082021)]
            for future in as_completed(futures):
                results.append(future.result())
                shared.save(OUT/'STATUS.json',dict(stage='running',completed_units=len(results),total_units=20,
                                                  elapsed_seconds=time.time()-start))
    except BaseException as exc:
        shared.save(OUT/'STATUS.json',dict(stage='failed',error=repr(exc),completed_units=len(results)))
        raise
    results.sort(key=lambda r:r['seed'])
    shared.verify_sources(OUT)
    shared.save(OUT/'FORMAL_COMPLETE.json',results)
    shared.save(OUT/'METRICS.json',{key:float(np.mean([r['metrics'][key] for r in results]))
                                  for key in ('avg_net','ron','dr','hu','cvar10')})
    shared.save(OUT/'STATUS.json',dict(stage='complete',completed_units=20,elapsed_seconds=time.time()-start))


if __name__ == '__main__':
    main()
