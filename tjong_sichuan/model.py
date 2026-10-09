import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from douzero_sichuan.core import encode, tile_index, parse_console_tile, STATE_DIM
from prompt_builder import get_legal_actions

MEMORY = 4
WIDTH = 448
KINDS = ('discard', 'hu', 'peng', 'minggang', 'angang', 'bugang', 'pass')


def options(game, pid, responses=None):
    mask = np.zeros((7, 28), bool)
    legal = get_legal_actions(game, pid, responses)
    for a in legal:
        if a.startswith('d '): mask[0, tile_index(parse_console_tile(a[2:]))] = True
        elif a == 'h': mask[1, 27] = True
        elif a == 'n': mask[6, 27] = True
        elif a == 'p': mask[2, tile_index(game.last_discarded_tile)] = True
        elif a == 'g':
            if game.deck.remaining_count() <= 1: continue
            if responses is not None:
                mask[3, tile_index(game.last_discarded_tile)] = True
            else:
                info = game.can_self_gang(pid)
                for tile in info['gang_tiles']:
                    kind = info['gang_types'][f'{tile.number}{tile.suit.value}']
                    mask[4 if kind == 'an' else 5, tile_index(tile)] = True
    if not mask.any(): raise RuntimeError('Empty legal action set')
    return mask


def public_state(game, pid, responses=None):
    return encode(game, pid, get_legal_actions(game, pid, responses), [], responses is not None)[0]


def privileged_state(game, pid):
    # Counts only; this channel is never passed into the policy methods.
    rows = []
    for offset in (1, 2, 3):
        rows.append(np.bincount([tile_index(t) for t in game.players[(pid+offset)%4].hand_tiles], minlength=27)/4)
    rows.append(np.bincount([tile_index(t) for t in game.deck.tiles], minlength=27)/4)
    return np.asarray(rows, np.float32).ravel()


class TIT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tile = nn.Linear(9, WIDTH)
        self.meta = nn.Linear(STATE_DIM-972, WIDTH)
        self.spatial_position = nn.Parameter(torch.randn(1, 28, WIDTH)*.02)
        self.temporal_position = nn.Parameter(torch.randn(1, MEMORY, WIDTH)*.02)
        self.condition = nn.Embedding(8, WIDTH)
        self.previous_action = nn.Embedding(8, WIDTH)
        self.previous_tile = nn.Embedding(29, WIDTH)
        self.previous_reward = nn.Linear(1, WIDTH)
        def stack():
            layer = nn.TransformerEncoderLayer(WIDTH, 8, WIDTH*4, dropout=0,
                                                batch_first=True, norm_first=True, activation='gelu')
            return nn.TransformerEncoder(layer, 3, norm=nn.LayerNorm(WIDTH), enable_nested_tensor=False)
        self.inner, self.outer = stack(), stack()
        # TransformerEncoder clones its template; initialize each layer separately.
        for module in (self.inner, self.outer):
            for p in module.parameters():
                if p.ndim > 1: nn.init.xavier_uniform_(p)

    def spatial(self, state, condition=None):
        tile = state[..., :972].reshape(-1, 9, 4, 27).sum(2).transpose(1, 2)/4
        cls = self.meta(state[..., 972:]).reshape(-1, WIDTH)
        if condition is not None: cls = cls + self.condition(condition)
        tokens = torch.cat((cls[:,None], self.tile(tile)), 1) + self.spatial_position
        return self.inner(tokens)[:,0]

    def temporal(self, embeddings, actions, tiles, rewards):
        x = embeddings + self.previous_action(actions) + self.previous_tile(tiles)
        x = x + self.previous_reward(rewards[...,None]) + self.temporal_position
        causal = torch.ones(MEMORY, MEMORY, device=x.device, dtype=torch.bool).triu(1)
        return self.outer(x, mask=causal)[:,-1]

    def first(self, states, actions, tiles, rewards):
        embedded = self.spatial(states.flatten(0,1)).reshape(-1,MEMORY,WIDTH)
        return self.temporal(embedded,actions,tiles,rewards), embedded

    def conditioned(self, states, actions, tiles, rewards, embedded, kind):
        current = self.spatial(states[:,-1],kind)
        changed = torch.cat((embedded[:,:-1], current[:,None]),1)
        return self.temporal(changed,actions,tiles,rewards)


class Tjong(nn.Module):
    def __init__(self):
        super().__init__()
        self.tit = TIT()
        self.action = nn.Linear(WIDTH,7)
        self.discard = nn.Linear(WIDTH,28)
        self.claim = nn.Linear(WIDTH,28)
        self.critic = nn.Sequential(nn.Linear(WIDTH+108,512),nn.GELU(),nn.Linear(512,1))

    def first(self,b):
        h,e = self.tit.first(b['states'],b['prev_actions'],b['prev_tiles'],b['prev_rewards'])
        mask = b['legal'].any(-1)
        action = Categorical(logits=self.action(h).masked_fill(~mask,-1e9))
        return h,e,action

    def second(self,b,e,kinds):
        h = self.tit.conditioned(b['states'],b['prev_actions'],b['prev_tiles'],b['prev_rewards'],e,kinds)
        logits = torch.where((kinds==0)[:,None],self.discard(h),self.claim(h))
        mask = b['legal'][torch.arange(len(kinds),device=kinds.device),kinds]
        if (~mask.any(-1)).any(): raise RuntimeError('Illegal action type during decoding')
        return Categorical(logits=logits.masked_fill(~mask,-1e9))

    def evaluate(self,b):
        h,e,ad = self.first(b)
        td = self.second(b,e,b['kind'])
        logp = ad.log_prob(b['kind']) + td.log_prob(b['tile'])
        entropy = ad.entropy() + td.entropy()
        value = self.critic(torch.cat((h,b['privileged']),-1)).squeeze(-1)
        return logp,entropy,value

    @torch.no_grad()
    def choose(self,b,rng,greedy=False):
        h,e,ad = self.first(b)
        probs = ad.probs[0].numpy().astype(float); probs /= probs.sum()
        kind = int(probs.argmax()) if greedy else int(rng.choice(7,p=probs))
        kd = torch.tensor([kind])
        td = self.second(b,e,kd)
        probs = td.probs[0].numpy().astype(float); probs /= probs.sum()
        tile = int(probs.argmax()) if greedy else int(rng.choice(28,p=probs))
        return kind,tile,float(ad.log_prob(kd)+td.log_prob(torch.tensor([tile])))


def tensor_batch(data, device='cpu'):
    return {k:torch.from_numpy(v).to(device) for k,v in data.items() if isinstance(v,np.ndarray)}
