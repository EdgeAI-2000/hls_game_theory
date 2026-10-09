"""Full frozen payoff audit. Outputs new artifacts; never edits originals."""
import hashlib
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from . import core

ROOT = Path('DouZero_Sichuan_Formal_20260926')
OUT = ROOT / 'settlement_audit' / 'full'
MASK = Path('Formal_E2_E3_P4_STRICT_FFR_20260823/formal_e3/L2_full')
KINDS = ['hu_self_draw', 'hu_discard', 'da_jiao', 'hua_zhu']


def save(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temp, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    total = {side: Counter() for side in ['received', 'paid', 'net']}
    for row in rows:
        b = row['breakdown']
        assert abs(sum(b['net'].values()) - row['net']) < 1e-8
        assert all(abs(b['received'].get(k, 0)-b['paid'].get(k, 0)-b['net'].get(k, 0)) < 1e-8 for k in KINDS)
        for side in total:
            total[side].update(b[side])
    n = len(rows)
    return dict(games=n, avg_net=sum(r['net'] for r in rows)/n,
                total={side: dict(v) for side,v in total.items()},
                per_game={side: {k: v.get(k,0)/n for k in KINDS} for side,v in total.items()})


def extract_mask():
    rows = []
    files = {}
    for seed in range(2026082001, 2026082021):
        path = MASK/f'seed_{seed}'/'gate1_games.jsonl'
        unit = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(unit)==1000 and {r['seed'] for r in unit}=={seed*1000000+i for i in range(1000)}
        files[str(path)] = digest(path)
        for r in unit:
            assert r['settlement_reconciled'] and r['end_reason']!='max_steps'
            rows.append(dict(seed=r['seed'],net=r['agent_net'],breakdown=r['settlement_breakdown']))
    save(OUT/'mask_breakdown.json',dict(summary=summarize(rows),source_hashes=files))
    return rows


def worker(seed):
    torch.set_num_threads(1)
    target=OUT/f'seed_{seed}.jsonl'
    source=ROOT/'holdout'/f'seed_{seed}'/'games.jsonl'
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    if target.exists():
        cached=[json.loads(line) for line in target.read_text().splitlines()]
        if len(cached)==1000 and [r['seed'] for r in cached]==[r['seed'] for r in rows]:
            return seed
    original=core.init_game
    model=core.ValueNetwork().eval()
    model.load_state_dict(torch.load(ROOT/'selected.pt',map_location='cpu',weights_only=True))
    captured=[]
    counters=Counter()
    def capture(*args,**kwargs):
        result=original(*args,**kwargs)
        game=result[0]
        captured.append(result)
        handle=game._handle_liuju
        responses=game.check_responses
        def settle():
            counters['settlement_calls']+=1
            return handle()
        def check(tile,pid):
            res=responses(tile,pid)
            if game.deck.remaining_count()<=1:
                counters['blocked_response_gang_options']+=sum('gang' in acts for acts in res.values())
            return res
        game._handle_liuju=settle
        game.check_responses=check
        return result
    core.init_game=capture
    try:
        with target.with_suffix('.tmp').open('w') as stream:
            for index,row in enumerate(rows):
                captured.clear()
                counters.clear()
                metrics=core.episode(model,row['seed'],0,True,'holdout')[3]
                assert metrics==row['four_seat_metrics'], ('replay mismatch',row['seed'])
                game,_,defenders=captured[0]
                breakdown={side:Counter() for side in ['received','paid','net']}
                keys=Counter()
                for e in game.settlement_ledger:
                    keys[(e['kind'],e['payer_id'],e['receiver_id'])]+=1
                    if 'fan' in e:
                        assert e['score']==game.base_score*2**min(e['fan'],game.max_fan)
                    if e['receiver_id']==0:
                        breakdown['received'][e['kind']]+=e['score']
                        breakdown['net'][e['kind']]+=e['score']
                    if e['payer_id']==0:
                        breakdown['paid'][e['kind']]+=e['score']
                        breakdown['net'][e['kind']]-=e['score']
                counts=Counter((t.suit.value,t.number) for p in game.players for t in
                               p.hand_tiles+p.discarded_tiles+[t for meld in p.open_melds for t in meld])
                counts.update((t.suit.value,t.number) for t in game.deck.tiles)
                incomplete_hu=[p.player_id for p in game.players if p.is_hu and p.hu_win_tile is None]
                counts.update((p.hu_win_tile.suit.value,p.hu_win_tile.number) for p in game.players if p.is_hu and not p.hu_is_self_drawn and p.hu_win_tile is not None)
                assert len(counts)==27 and all(v==4 for v in counts.values()), ('tile conservation',row['seed'])
                assert counters['settlement_calls']<=1, ('repeated settlement',row['seed'])
                assert all(v==1 for v in keys.values()), ('duplicate transfer',row['seed'])
                assert sum(breakdown['net'].values())==row['net']
                item=dict(seed=row['seed'],net=row['net'],breakdown=breakdown,
                          incomplete_hu_players=incomplete_hu,
                          checks=dict(counters),ledger=game.settlement_ledger,
                          opponents={str(pid):d.curriculum_spec['id'] for pid,d in defenders.items()})
                stream.write(json.dumps(item,ensure_ascii=False)+'\n')
                if (index+1)%100==0:
                    stream.flush()
                    save(OUT/f'progress_{seed}.json',dict(completed=index+1,target=1000,updated=time.time()))
        os.replace(target.with_suffix('.tmp'),target)
    finally:
        core.init_game=original
    return seed


def report(mask_rows):
    rows=[json.loads(line) for p in sorted(OUT.glob('seed_*.jsonl')) for line in p.read_text().splitlines()]
    assert len(rows)==20000 and {r['seed'] for r in rows}=={r['seed'] for r in mask_rows}
    # Compare opponent identities at every seat, not merely the set of profiles.
    opponent_mismatches=0
    mapping={r['seed']:r for r in rows}
    for seed in range(2026082001,2026082021):
        for line in (MASK/f'seed_{seed}'/'gate1_games.jsonl').read_text().splitlines():
            r=json.loads(line)
            expected={str(pid):r['opponent_curriculum'][f'P{pid}']['id'] for pid in (1,2,3)}
            opponent_mismatches+=expected!=mapping[r['seed']]['opponents']
    assert opponent_mismatches==0
    d,m=summarize(rows),summarize(mask_rows)
    boundary=sum(r['checks'].get('blocked_response_gang_options',0)>0 for r in rows)
    incomplete=[dict(seed=r['seed'],players=r['incomplete_hu_players']) for r in rows if r.get('incomplete_hu_players')]
    output=dict(DouZero=d,MASK=m,opponent_mismatches=opponent_mismatches,
                games_with_blocked_response_gang=boundary,incomplete_hu_games=incomplete,all_replays_matched=True)
    save(OUT/'comparison.json',output)
    lines=['# 全量收益分解（各 20,000 局）','',
           'DouZero 全量确定性重放，四座位指标与冻结记录逐局一致；MASK 使用冻结日志的分类收支。',
           '两组种子及每个座位的对手 ID 完全配对。DouZero 每局金额公式、牌张守恒、单次流局结算及流水对账通过。','',
           '| 收支类型 | DouZero 局均收入 | 局均支出 | 局均净收益 | MASK 局均收入 | 局均支出 | 局均净收益 | 净收益差 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels=dict(hu_self_draw='自摸',hu_discard='荣和/放铳',da_jiao='查叫',hua_zhu='花猪')
    for kind in KINDS:
        values=[a['per_game'][side][kind] for a in (d,m) for side in ('received','paid','net')]
        lines.append('| '+labels[kind]+' | '+' | '.join(f'{v:.4f}' for v in values)+f" | {values[2]-values[5]:+.4f} |")
    lines += ['',f"总平均分：DouZero {d['avg_net']:.4f}；MASK {m['avg_net']:.4f}；差 {d['avg_net']-m['avg_net']:+.4f}。",'',
              f'发现 {boundary} 局出现牌墙末尾的不可执行响应杠选项；这只是适配差异触发次数，不是收益影响的反事实估计。',
              f'另有 {len(incomplete)} 局存在已胡但没有胡牌张元数据的玩家（起手天胡路径）；这类规则异常保存在 comparison.json，账面一致不等于胡牌结算完整。',
              '本审计验证了当前冻结实现下的收支，不证明全部规则正确。规则审查另见上级 RULE_REVIEW.md。']
    (OUT/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    return output


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        mask=extract_mask()
        save(OUT/'status.json',dict(stage='replaying',completed_units=0,total=20,mask_complete=True))
        done=0
        with ProcessPoolExecutor(8,mp_context=mp.get_context('spawn')) as pool:
            jobs={pool.submit(worker,s):s for s in range(2026082001,2026082021)}
            for future in as_completed(jobs):
                seed=future.result()
                done+=1
                save(OUT/'status.json',dict(stage='replaying',completed_units=done,last_seed=seed,total=20))
                print(f'Completed {done}/20 seed {seed}',flush=True)
        result=report(mask)
        save(OUT/'status.json',dict(stage='complete',games_per_system=20000,report=str(OUT/'COMPARISON.md')))
        print(json.dumps(result),flush=True)
    except Exception as exc:
        save(OUT/'status.json',dict(stage='failed',error=repr(exc)))
        raise


if __name__=='__main__':
    main()
