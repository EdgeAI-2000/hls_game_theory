"""Single-game prototype bridge to the frozen engine, confined to one process.

Temporary observation hooks collect privileged labels outside the policy.
Not thread-safe; concurrent actors must use separate spawned processes.
"""
import numpy as np
import torch
from douzero_sichuan import core


def opening(game):
    hands=np.full((4,14),27,np.int64)
    missing=np.zeros((4,3),np.float32)
    for i,p in enumerate(game.players):
        hands[i,:len(p.hand_tiles)]=sorted(core.tile_index(t) for t in p.hand_tiles)
        if p.missing_suit is not None:
            missing[i,core.SUITS.index(p.missing_suit)]=1
    return hands,missing


def collect(policy,seed,coach=None,beta=0,epsilon=.01):
    original_init,original_encode=core.init_game,core.encode
    records=[[] for _ in range(4)]
    initial=[]
    rejected=[]
    opening_probabilities=[]
    class Rejected(Exception):
        pass
    def init(*args,**kwargs):
        result=original_init(*args,**kwargs)
        hands,missing=opening(result[0])
        initial.extend([hands,missing])
        if coach is not None:
            from .models import accept_opening
            with torch.no_grad():
                probabilities=coach(torch.from_numpy(hands[None]),torch.from_numpy(missing[None])).softmax(-1)
            opening_probabilities.extend(probabilities[0].tolist())
            if not bool(accept_opening(probabilities,beta)[0]):
                rejected.append(probabilities[0].tolist())
                raise Rejected()
        return result
    def encode(game,pid,actions,events,response=False):
        observation=original_encode(game,pid,actions,events,response)
        labels=np.zeros((3,27),np.int64)
        for relative in range(1,4):
            for tile in game.players[(pid+relative)%4].hand_tiles:
                labels[relative-1,core.tile_index(tile)]+=1
        records[pid].append(labels)
        return observation
    core.init_game,core.encode=init,encode
    try:
        try:
            x,z,y,metrics=core.episode(policy,seed,epsilon)
        except Rejected:
            return dict(accepted=False,seed=seed,probabilities=rejected[0])
    finally:
        core.init_game,core.encode=original_init,original_encode
    labels=np.stack([label for trajectory in records for label in trajectory])
    assert len(labels)==len(y)
    return dict(accepted=True,x=x,z=z,y=y,labels=labels,hands=initial[0],missing=initial[1],metrics=metrics,
                probabilities=opening_probabilities)
