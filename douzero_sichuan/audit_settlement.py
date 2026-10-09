"""Read-only replay audit; does not modify the frozen training/evaluation code."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from collections import Counter
import torch
from . import core

ROOT = Path('DouZero_Sichuan_Formal_20260926')


def replay(row):
    torch.set_num_threads(1)
    model = core.ValueNetwork().eval()
    model.load_state_dict(torch.load(ROOT/'selected.pt', map_location='cpu', weights_only=True))
    original = core.init_game
    captured = []
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result[0])
        return result
    core.init_game = capture
    try:
        result = core.episode(model, row['seed'], 0, True, 'holdout')[3]
    finally:
        core.init_game = original
    assert result == row['four_seat_metrics'], row['seed']
    game = captured[0]
    components = Counter()
    for event in game.settlement_ledger:
        if event['receiver_id'] == 0:
            components[event['kind']] += event['score']
        if event['payer_id'] == 0:
            components[event['kind']] -= event['score']
    players = []
    for p in game.players:
        ready, waits = p.is_ready_with_missing_suit()
        fan, types = p.calculate_potential_fan()
        players.append(dict(pid=p.player_id, hand=[str(t) for t in p.hand_tiles],
                            melds=[[str(t) for t in m] for m in p.open_melds],
                            hu=p.is_hu, ready=ready, waits=[str(t) for t in waits],
                            potential_fan=fan, fan_types=types, missing=str(p.missing_suit)))
    counts = Counter((t.suit.value,t.number) for p in game.players for t in
                     p.hand_tiles + p.discarded_tiles + [t for meld in p.open_melds for t in meld])
    counts.update((t.suit.value,t.number) for t in game.deck.tiles)
    # Ron removes the winning discard without appending it to the winner's hand.
    counts.update((p.hu_win_tile.suit.value,p.hu_win_tile.number) for p in game.players
                  if p.is_hu and not p.hu_is_self_drawn)
    return dict(seed=row['seed'], net=row['net'], components=dict(components), players=players,
                ledger=game.settlement_ledger, tile_counts_valid=all(n==4 for n in counts.values()) and len(counts)==27)


def main():
    rows = [json.loads(line) for path in sorted((ROOT/'holdout').glob('seed_*/games.jsonl')) for line in path.read_text().splitlines()]
    high = [r for r in rows if r['net']>1000]
    with ProcessPoolExecutor(8, mp_context=mp.get_context('spawn')) as pool:
        replayed = list(pool.map(replay, high))
    totals = Counter()
    for r in replayed:
        totals.update(r['components'])
    out = ROOT/'settlement_audit'
    out.mkdir(exist_ok=True)
    report = dict(games=len(rows), high_games=len(high), high_p0_hu=sum(r['hu'] for r in high),
                  high_net=sum(r['net'] for r in high), components=dict(totals),
                  tile_audit_failures=sum(not r['tile_counts_valid'] for r in replayed), replays=replayed)
    (out/'high_games.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k!='replays'},ensure_ascii=False))
    print(json.dumps(sorted(replayed,key=lambda r:r['net'],reverse=True)[:2],ensure_ascii=False))


if __name__ == '__main__':
    main()
