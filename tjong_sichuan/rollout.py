import numpy as np
import torch

from douzero_sichuan.core import init_game, execute_action, parse_console_tile, tile_index, SUITS
from game import bot_decide_turn_action, bot_decide_response
from .model import options, public_state, privileged_state, tensor_batch, MEMORY, STATE_DIM
from .fan import counts, score_tiles, backward


def tile_object(index):
    from tiles import Tile
    return Tile(SUITS[index//9],index%9+1)


def collect(model,seed,mode='ppo',split='train'):
    if mode not in ('bc','ppo','eval'): raise ValueError(mode)
    game,opponents,defenders = init_game(seed,'curriculum',f'tjong_{seed}',
        {'curriculum_manifest':'Opponent_Curriculum_formal/screened.json','curriculum_split':split})
    rng = np.random.default_rng(seed+17)
    initial = [p.balance for p in game.players]
    records,context = [[] for _ in range(4)],[[] for _ in range(4)]
    previous_actions,previous_tiles = [7]*4,[28]*4
    previous_balances = initial.copy()
    wins,claims = [],[]
    ron,dr,missed = [False]*4,[False]*4,[0,0]
    decisions = [0]*4

    def decide(pid,acts=None):
        legal = options(game,pid,acts)
        teacher = mode=='bc' or (mode=='eval' and pid!=0)
        sample = mode!='eval' or pid==0
        observation = None
        if sample:
            s = public_state(game,pid,acts)
            reward = (game.players[pid].balance-previous_balances[pid])/game.base_score
            item = (s,previous_actions[pid],previous_tiles[pid],reward)
            seq = (context[pid]+[item])[-MEMORY:]
            states = np.zeros((MEMORY,STATE_DIM),np.float32)
            pa,pt,pr = np.full(MEMORY,7,np.int64),np.full(MEMORY,28,np.int64),np.zeros(MEMORY,np.float32)
            for j,(state,ak,tk,r) in enumerate(seq,MEMORY-len(seq)):
                states[j],pa[j],pt[j],pr[j] = state,ak,tk,r
            observation = dict(states=states,prev_actions=pa,prev_tiles=pt,prev_rewards=pr,legal=legal)
        if teacher:
            if pid==0:
                a = bot_decide_response(game.players[pid],acts) if acts is not None else bot_decide_turn_action(game.players[pid],game)
            else:
                a = defenders[pid].response(game.players[pid],acts,game) if acts is not None else opponents[pid][0](game.players[pid],game)
            if a.startswith('d '): kind,tile = 0,tile_index(parse_console_tile(a[2:]))
            elif a=='h': kind,tile = 1,27
            elif a=='p': kind,tile = 2,tile_index(game.last_discarded_tile)
            elif a in ('n','pass'):
                # The shared rule teacher uses both console spellings for
                # declining a response; TIT represents both as pass.
                kind,tile = 6,27
            elif a=='g':
                kinds = np.flatnonzero(legal[3:6].any(-1))+3
                if not len(kinds): kind,tile = 6,27
                else:
                    kind = int(kinds[0]); tile = int(np.flatnonzero(legal[kind])[0])
            else: raise RuntimeError(f'Unknown teacher action {a}')
            if not legal[kind,tile]:
                if mode=='eval':
                    # Keep frozen opponents' engine-legal calls while P0 uses
                    # the stricter missing-suit action filter, as in P4/DouZero.
                    if acts is None or a not in ('p','g'): raise RuntimeError('Illegal benchmark opponent')
                else:
                    if legal[6,27]: kind,tile=6,27
                    else: kind,tile=map(int,np.argwhere(legal)[0])
            logp = 0.
        else:
            batch = tensor_batch({k:v[None] for k,v in observation.items()})
            kind,tile,logp = model.choose(batch,rng,greedy=mode=='eval')
        decisions[pid] += 1
        if legal[1,27] and not teacher:
            missed[1]+=1;missed[0]+=kind!=1
        idx = None
        if sample:
            idx=len(records[pid])
            observation.update(kind=np.int64(kind),tile=np.int64(tile),old_logp=np.float32(logp),
                               privileged=privileged_state(game,pid) if mode=='ppo' else np.zeros(108,np.float32),
                               balance=game.players[pid].balance,post_counts=None)
            records[pid].append(observation)
            context[pid]=(context[pid]+[item])[-(MEMORY-1):]
            previous_actions[pid],previous_tiles[pid]=kind,tile
            previous_balances[pid]=game.players[pid].balance
        return kind,tile,idx

    def finish_record(pid,idx):
        if idx is not None:
            p=game.players[pid]
            tiles=p.hand_tiles+[t for m in p.open_melds for t in m]
            if p.is_hu and p.hu_win_tile is not None and not p.hu_is_self_drawn:tiles=tiles+[p.hu_win_tile]
            records[pid][idx]['post_counts']=counts(tiles)

    def record_win(pid,giver,discard_index,time):
        p=game.players[pid]
        if p.hu_win_tile is None: return
        concealed=p.hand_tiles.copy()
        if not p.hu_is_self_drawn: concealed.append(p.hu_win_tile)
        final=counts(concealed+[t for m in p.open_melds for t in m])
        wins.append(dict(pid=pid,giver=giver,discard_record=discard_index,tile=tile_index(p.hu_win_tile),
                         time=time,counts=final,score=score_tiles(final,counts(concealed),p.hu_fan_types,p.hu_fan,game.max_fan),
                         record_end=len(records[pid])))

    skip_draw=True
    for step in range(1,301):
        if game.is_game_over: break
        if game.deck.remaining_count()==0 or sum(p.is_hu for p in game.players)>=3:
            game.check_game_over();break
        pid=game.current_player_id
        if game.players[pid].is_hu:
            game.next_player();skip_draw=False;continue
        drawn=None
        if not skip_draw:
            drawn=game.draw_tile(pid)
            if drawn is None: game.check_game_over();break
        skip_draw=False
        kind,tile,idx=decide(pid)
        if kind in (4,5):
            success,_=game.gang(pid,tile_object(tile))
            if not success: raise RuntimeError('Legal own kong rejected')
            if kind==5:
                for claim in claims:
                    if claim['winner']==pid and claim['tile']==tile and claim['size']==3: claim['size']=4
            discarded=None
        else:
            a=f'd {tile_object(tile)}' if kind==0 else ('h' if kind==1 else 'n')
            discarded=execute_action(game,pid,a,drawn)
        finish_record(pid,idx)
        if kind==1: record_win(pid,None,None,step)
        if game.is_game_over: break
        if discarded is None:
            if kind in (4,5):skip_draw=True
            else:game.next_player()
            continue
        responded=False
        for rid,acts in game.check_responses(discarded,pid).items():
            if game.deck.remaining_count()<=1:acts=[a for a in acts if a!='gang']
            if not acts:continue
            rk,rt,ri=decide(rid,acts)
            if rk==1:
                game.hu(rid,discarded,False,pid)
                finish_record(rid,ri)
                record_win(rid,pid,idx,step)
                game.check_game_over()
                ron[rid],dr[pid],responded=True,True,True
            elif rk in (2,3):
                success=game.peng(rid,discarded,pid) if rk==2 else game.gang(rid,discarded,pid)[0]
                if not success:raise RuntimeError('Legal claim rejected')
                claims.append(dict(winner=rid,giver=pid,record=idx,tile=tile_index(discarded),size=3 if rk==2 else 4,time=step))
                game.current_player_id,responded=rid,True
            finish_record(rid,ri)
            if responded:break
        if game.is_game_over:break
        if responded:skip_draw=True
        else:game.next_player()
    if not game.is_game_over:raise RuntimeError(f'Unsettled game {seed}')
    net=[p.balance-initial[i] for i,p in enumerate(game.players)]
    ledger=[0]*4
    for e in game.settlement_ledger:
        ledger[e['payer_id']]-=e['score'];ledger[e['receiver_id']]+=e['score']
    if ledger!=net or sum(net):raise RuntimeError('Settlement mismatch')
    metrics=dict(seed=seed,net=net,ron=ron,dr=dr,hu=[p.is_hu for p in game.players],decisions=decisions,
                 missed_hu=missed,max_steps_or_unsettled=0,settlement_ledger=game.settlement_ledger,
                 opponent_ids={str(pid):d.curriculum_spec['id'] for pid,d in defenders.items()} if mode!='ppo' else {},
                 wins=len(wins),unsettled_initial_hu=sum(p.is_hu and p.hu_win_tile is None for p in game.players))
    if mode=='eval':return metrics
    shaped=backward(records,wins,claims)
    flat=[]
    for pid,rows in enumerate(records):
        for i,r in enumerate(rows):
            balance=game.players[pid].balance if i==len(rows)-1 else rows[i+1]['balance']
            actual=(balance-r['balance'])/game.base_score
            r.update(actual_reward=np.float32(actual),fan_reward=np.float32(shaped[pid][i]),done=np.bool_(i==len(rows)-1))
            del r['post_counts'];del r['balance']
            flat.append(r)
    if not flat: raise RuntimeError('No trajectory decisions')
    data={k:np.stack([r[k] for r in flat]) for k in flat[0]}
    metrics['actual_reward_sum']=float(data['actual_reward'].sum())
    metrics['fan_reward_sum']=float(data['fan_reward'].sum())
    return data,metrics
