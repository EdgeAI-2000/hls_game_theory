"""Evidence fixtures for current-rule gaps, not tests of a corrected engine."""
import json
from pathlib import Path

from game import MahjongGame
from rule_engine import FanCalculator
from tiles import Tile, Suit


def wan(numbers):
    return [Tile(Suit.WAN, n) for n in numbers]


def main():
    out = {}
    game = MahjongGame('audit_idempotence', ['0','1','2','3'])
    game.deck.tiles = []
    p = game.players[0]
    p.hand_tiles = wan([6])
    p.open_melds = [wan([n]*3) for n in [1,3,5,7]]
    assert p.is_ready_with_missing_suit()[0]
    game.check_game_over()
    first = dict(net=p.balance-10000,events=len(game.settlement_ledger))
    game.check_game_over()
    second = dict(net=p.balance-10000,events=len(game.settlement_ledger))
    assert second['net']==2*first['net'] and first['net']>0
    out['non_idempotent_terminal_check'] = dict(first=first,second=second)

    hand=wan([7,8,9,9,9])
    melds=[wan([n]*3) for n in [1,2,3]]
    fan,types=FanCalculator.calculate_fan(hand,hand[-1],melds,is_self_drawn=False)
    assert any('带幺九' in name for name in types)
    # Exposed 222 and 333 cannot be rearranged into terminal-containing sequences.
    out['fixed_melds_rearranged_for_daiyao'] = dict(hand=[str(t) for t in hand],
        melds=[[str(t) for t in m] for m in melds],fan=fan,types=types,
        reason='Exposed 222 and 333 contain no terminal, but full_tiles can be rearranged into 123 sequences.')

    game=MahjongGame('audit_first_discard',['0','1','2','3'])
    game.players[0].hand_tiles=wan([5])
    game.players[1].hand_tiles=wan([1,1,1,2,2,2,3,3,3,4,4,4,5])
    assert game.first_discard
    assert game.discard_tile(0,wan([5])[0])
    fan,types=game.hu(1,wan([5])[0],False,0)
    assert game.players[1].is_hu and not any('地胡' in t for t in types)
    out['first_discard_flag_cleared_before_ron'] = dict(types=types,fan=fan,
        reason='discard_tile clears first_discard before hu checks it; di_hu branch unreachable through normal discard flow.')

    game=MahjongGame('audit_kong',['0','1','2','3'])
    game.players[0].hand_tiles=wan([1,1,1,1,2,2,3,3,4,4,5,5,6,6])
    success,_=game.gang(0,wan([1])[0])
    assert success
    assert not game.settlement_ledger
    out['kong_has_no_immediate_payment'] = dict(success=success,transfers=game.settlement_ledger)
    target=Path('DouZero_Sichuan_Formal_20260926/settlement_audit/rule_boundary_fixtures.json')
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
