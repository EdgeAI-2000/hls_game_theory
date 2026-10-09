import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from douzero_sichuan.core import STATE_DIM, ACTION_DIM, EVENT_DIM


def trunk(width, layers):
    blocks = []
    for _ in range(layers):
        blocks.extend([nn.Linear(width, 512), nn.ReLU()])
        width = 512
    return nn.Sequential(*blocks)


def public_count_mask(state):
    # Nine multihot planes: own hand, four rivers, four exposed-meld sets.
    # Called discards are removed from rivers by the engine. Hidden tiles and
    # deck contents are deliberately inaccessible. Untracked ron tiles make
    # this upper bound looser, never a privileged bound.
    visible = state[..., :972].reshape(*state.shape[:-1], 9, 4, 27).sum(dim=(-3, -2))
    remaining = (4-visible).clamp(0, 4)
    hand_counts = (state[..., 984:1000].reshape(*state.shape[:-1], 4, 4)[..., 1:, 0]*14).round()
    upper = torch.minimum(remaining.unsqueeze(-2), hand_counts.unsqueeze(-1))
    return torch.arange(5, device=state.device) <= upper.unsqueeze(-1)


class OpponentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(EVENT_DIM, 128, batch_first=True)
        self.trunk = trunk(STATE_DIM+128, 4)
        # Shared projection, with independent output parameters per seat/tile.
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,5)) for _ in range(81)])

    def forward(self, state, history):
        _,(h,_) = self.lstm(history)
        features = self.trunk(torch.cat([state,h[-1]],-1))
        logits = torch.stack([head(features) for head in self.heads],1).reshape(-1,3,27,5)
        logits = logits.masked_fill(~public_count_mask(state), -1e9)
        return logits, logits.softmax(-1)


class PlusPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.opponent = OpponentModel()
        self.history = nn.LSTM(EVENT_DIM,128,batch_first=True)
        self.value = nn.Sequential(trunk(STATE_DIM+ACTION_DIM+128+405,5),nn.Linear(512,1))

    def forward(self, state_action, history):
        logits, probability = self.opponent(state_action[:,:STATE_DIM],history)
        _,(h,_) = self.history(history)
        # Official code detaches the predicted hand before value regression.
        q = self.value(torch.cat([state_action,h[-1],probability.detach().flatten(1)],-1)).squeeze(-1)
        return q, logits

    @torch.no_grad()
    def choose(self,state,history,candidates,rng,epsilon):
        if rng.random()<epsilon:
            return rng.randrange(len(candidates))
        device=next(self.parameters()).device
        n=len(candidates)
        s=torch.as_tensor(state[None],device=device)
        z=torch.as_tensor(history[None],device=device)
        _,p=self.opponent(s,z)
        _,(h,_)=self.history(z)
        a=torch.as_tensor(candidates,device=device)
        q=self.value(torch.cat([s.expand(n,-1),a,h[-1].expand(n,-1),p.flatten(1).expand(n,-1)],-1))
        if not torch.isfinite(q).all():
            raise ValueError('Nonfinite actor values')
        return int(q.argmax())


def losses(policy, x, history, returns, labels):
    q, logits = policy(x,history)
    mask=public_count_mask(x[:,:STATE_DIM])
    if not mask.gather(-1,labels.unsqueeze(-1)).all():
        raise ValueError('Oracle label violates public count mask')
    value=F.mse_loss(q,returns)
    prediction=F.cross_entropy(logits.reshape(-1,5),labels.reshape(-1))
    return value+prediction, dict(value=value,prediction=prediction)


class Coach(nn.Module):
    """Training-only four-seat initial-hand model, never a policy input."""
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(28,256,padding_idx=27)
        self.seat=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.output=nn.Sequential(nn.Linear(128*4+12,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,4))

    def forward(self,hands,missing):
        valid=(hands!=27).unsqueeze(-1)
        embedded=(self.embed(hands)*valid).sum(-2)/valid.sum(-2).clamp_min(1)
        seats=self.seat(embedded).flatten(1)
        return self.output(torch.cat([seats,missing.flatten(1)],-1))


def coach_target(net):
    # No landlord/team winner in blood-battle. Predict highest final net seat;
    # ties share probability, including all-zero games. This is an adaptation.
    net=torch.as_tensor(net,dtype=torch.float32)
    winner=(net==net.max(-1,keepdim=True).values).float()
    return winner/winner.sum(-1,keepdim=True)


def accept_opening(probabilities,beta):
    if not 0<=beta<=.3:
        raise ValueError('beta must be within [0, .3]')
    # Four-way balance threshold, not the original binary [beta,1-beta] test.
    return probabilities.max(-1).values <= 1-beta


def beta_schedule(learner_updates):
    return min(.3,.05*(learner_updates//20000))
