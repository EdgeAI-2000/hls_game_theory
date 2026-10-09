"""Read-only evidence audit of frozen Tjong results; writes a separate report."""
import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'Tjong_Sichuan_Formal_20260928_v2'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    protocol = json.loads((RUN/'PROTOCOL.json').read_text())
    selection = json.loads((RUN/'SELECTION.json').read_text())
    selected = sha(RUN/'selected.pt')
    assert selected == selection['sha256']
    errors = Counter()
    totals = Counter()
    money = Counter()
    seed_set = set()
    for outer in protocol['holdout_seeds']:
        folder = RUN/'holdout'/f'seed_{outer}'
        summary = json.loads((folder/'summary.json').read_text())
        assert summary['checkpoint_sha256'] == selected
        assert summary['games_sha256'] == sha(folder/'games.jsonl')
        rows = [json.loads(s) for s in (folder/'games.jsonl').read_text().splitlines()]
        assert [r['seed'] for r in rows] == [outer*1000000+i for i in range(1000)]
        # Independently regenerate the per-seed main metrics.
        scores = sorted(r['net'] for r in rows)
        regenerated = dict(avg_net=sum(scores)/1000, cvar10=sum(scores[:100])/100,
            ron_game_rate=sum(r['ron'] for r in rows)/1000,
            dealin_game_rate=sum(r['dealt_in'] for r in rows)/1000,
            hu_game_rate=sum(r['hu'] for r in rows)/1000)
        assert all(abs(v-summary['metrics'][k])<1e-8 for k,v in regenerated.items())
        for row in rows:
            m = row['four_seat_metrics']; seed_set.add(row['seed'])
            totals['games'] += 1
            totals['missed_hu'] += m['missed_hu'][0]
            totals['available_hu'] += m['missed_hu'][1]
            totals['initial_hu_without_win_tile'] += m['unsettled_initial_hu']
            ledger = [0]*4
            for e in m['settlement_ledger']:
                ledger[e['payer_id']] -= e['score']; ledger[e['receiver_id']] += e['score']
                money[e['kind']] += e['score']*((e['receiver_id']==0)-(e['payer_id']==0))
            errors['ledger_balance_mismatch'] += ledger != m['net']
            errors['nonzero_sum'] += sum(m['net']) != 0
            errors['unsettled'] += m['max_steps_or_unsettled']
            errors['wrong_opponent_set'] += set(m['opponent_ids'].values()) != {'beta_b2_NN_4','beta_b9_NN_3','delta_d3_Reader'}
    developments = [json.loads(p.read_text()) for p in RUN.glob('dev_*.json')]
    developments.sort(key=lambda d:d['frames'])
    best = max(developments, key=lambda d:d['metrics']['avg_net'])
    assert best['frames'] == selection['best']['frames']
    hashes = protocol['hashes']
    changed = [p for p,h in hashes.items() if sha(ROOT/p)!=h]
    result = dict(totals=dict(totals),unique_seeds=len(seed_set), errors=dict(errors),
        selected_sha256=selected, selected_frames=best['frames'],
        changed_sources=changed, net_by_source_per_game={k:v/totals['games'] for k,v in money.items()},
        development=[dict(frames=d['frames'],**d['metrics']) for d in developments],
        conclusion='Data integrity checked; cross-runner fairness NOT established.')
    (RUN/'IMPLEMENTATION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='development'},indent=2))

if __name__ == '__main__':
    main()
