import json
from collections import Counter

import numpy as np
import torch
from torch import nn
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModel, AutoTokenizer

from douzero_sichuan.core import ACTION_DIM, KINDS, STATE_DIM, action_tile, tile_index

BASE = 'models/qwen/Qwen3-0.6B'
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def observation(game, pid, actions, response, history_length=16):
    # Only own concealed tiles and public fields enter the encoder. In
    # particular, neither other concealed hands nor wall contents are accessed.
    def counts(tiles):
        return [[str(t), n] for t, n in sorted(Counter(tiles).items(), key=lambda x: tile_index(x[0]))]

    players = [game.players[(pid + i) % 4] for i in range(4)]
    public = [dict(seat=i, discards=counts(p.discarded_tiles),
                   melds=[[str(t) for t in m] for m in p.open_melds],
                   missing=p.missing_suit.value if p.missing_suit else None,
                   tiles=len(p.hand_tiles), won=p.is_hu, balance=p.balance)
              for i, p in enumerate(players)]
    history = [[(e['pid'] - pid) % 4, e['act'], e['tile']]
               for e in game.history if e['act'] in ('discard', 'peng', 'gang', 'hu')]
    state = dict(own_hand=counts(players[0].hand_tiles), public=public,
                 current_seat=(game.current_player_id - pid) % 4,
                 wall_remaining=game.deck.remaining_count(), response=response,
                 last_discard=str(game.last_discarded_tile) if game.last_discarded_tile else None,
                 legal_actions=actions, history=history[-history_length:] if history_length else [])
    return 'Sichuan Mahjong. Seat 0 is self. Predict final net score.\n' + json.dumps(
        state, ensure_ascii=False, separators=(',', ':')) + '\nValue:'


def tokens(tokenizer, game, pid, actions, response, max_length):
    # Trim history only, never silently truncate the hand or legal action set.
    for n in (16, 8, 0):
        ids = tokenizer.encode(observation(game, pid, actions, response, n), add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        if len(ids) <= max_length:
            return ids
    raise ValueError(f'Essential observation exceeds token budget: {len(ids)} > {max_length}')


def action_vectors(game, pid, actions, response):
    candidates = np.zeros((len(actions), ACTION_DIM), np.float32)
    for i, action in enumerate(actions):
        candidates[i, KINDS.index(action[0])] = 1
        tile = action_tile(game, pid, action, response)
        if tile is not None:
            candidates[i, 5 + tile_index(tile)] = 1
    return candidates


def pad_sequences(sequences, pad_id, device):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, :len(seq)] = 1
    return ids, mask


class QwenValue(nn.Module):
    def __init__(self, backbone, lora_rank=16, lora_alpha=32):
        super().__init__()
        cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0,
                         target_modules=TARGET_MODULES, bias='none')
        self.backbone = get_peft_model(backbone, cfg)
        # Keep the language representation, but expose exact numeric Mahjong
        # geometry to the value head instead of asking the LM to infer it from
        # serialized text alone.
        layers, width = [], backbone.config.hidden_size + STATE_DIM + ACTION_DIM
        for _ in range(5):
            layers += [nn.Linear(width, 512), nn.ReLU()]
            width = 512
        self.head = nn.Sequential(*layers, nn.Linear(width, 1))
        self.register_buffer('return_mean', torch.tensor(0.0))
        self.register_buffer('return_second', torch.tensor(1.0))
        self.register_buffer('return_std', torch.tensor(1.0))
        self.register_buffer('return_updates', torch.tensor(0))

    def encode(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False, return_dict=True).last_hidden_state
        last = attention_mask.sum(1) - 1
        return hidden[torch.arange(len(last), device=hidden.device), last].float()

    def forward(self, input_ids, attention_mask, action, structured=None):
        hidden = self.encode(input_ids, attention_mask)
        if structured is None:
            structured = torch.zeros((len(hidden), STATE_DIM), device=hidden.device)
        return self.head(torch.cat((hidden, structured.float(), action.float()), -1)).squeeze(-1)

    def normalize(self, target):
        return (target - self.return_mean) / self.return_std

    def denormalize(self, value):
        return value * self.return_std + self.return_mean

    @torch.no_grad()
    def update_return_stats(self, targets, optimizer=None, decay=.99):
        """PopArt: change target scale without changing decoded action values."""
        targets = torch.as_tensor(targets, device=self.return_mean.device, dtype=torch.float32)
        old_mean, old_std = self.return_mean.clone(), self.return_std.clone()
        weight = 1.0 if self.return_updates.item() == 0 else 1.0 - decay
        self.return_mean.lerp_(targets.mean(), weight)
        self.return_second.lerp_(targets.square().mean(), weight)
        self.return_std.copy_((self.return_second - self.return_mean.square()).clamp_min(1.0).sqrt())
        ratio = old_std / self.return_std
        output = self.head[-1]
        output.weight.mul_(ratio)
        output.bias.copy_((old_std * output.bias + old_mean - self.return_mean) / self.return_std)
        if optimizer is not None:
            for parameter in (output.weight, output.bias):
                state = optimizer.state.get(parameter, {})
                if 'square_avg' in state:
                    state['square_avg'].mul_(ratio.square())
                for name in ('grad_avg', 'momentum_buffer'):
                    if name in state:
                        state[name].mul_(ratio)
        self.return_updates.add_(1)

    @torch.inference_mode()
    def score(self, ids, candidates, structured=None):
        device = next(self.head.parameters()).device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        hidden = self.encode(input_ids, torch.ones_like(input_ids))
        action = torch.as_tensor(candidates, dtype=torch.float32, device=device)
        if structured is None:
            structured = torch.zeros((len(action), STATE_DIM), device=device)
        else:
            structured = torch.as_tensor(structured, dtype=torch.float32, device=device)
            if structured.ndim == 1:
                structured = structured.unsqueeze(0).expand(len(action), -1)
        values = self.head(torch.cat((hidden.expand(len(action), -1), structured, action), -1)).squeeze(-1)
        if not torch.isfinite(values).all():
            raise FloatingPointError('Non-finite actor values')
        return self.denormalize(values).cpu().numpy()

    def weights(self):
        return dict(adapter={k: v.detach().cpu().clone() for k, v in
                             get_peft_model_state_dict(self.backbone).items()},
                    head={k: v.detach().cpu().clone() for k, v in self.head.state_dict().items()},
                    return_stats={k: getattr(self, k).detach().cpu().clone() for k in
                                  ('return_mean', 'return_second', 'return_std', 'return_updates')})

    def load_weights(self, weights):
        expected = get_peft_model_state_dict(self.backbone)
        if weights['adapter'].keys() != expected.keys() or any(
                weights['adapter'][k].shape != v.shape for k, v in expected.items()):
            raise RuntimeError('Incomplete or incompatible Qwen adapter checkpoint')
        result = set_peft_model_state_dict(self.backbone, weights['adapter'])
        if result.unexpected_keys:
            raise RuntimeError(result.unexpected_keys)
        self.head.load_state_dict(weights['head'], strict=True)
        if 'return_stats' in weights:
            expected_stats = {'return_mean', 'return_second', 'return_std', 'return_updates'}
            if weights['return_stats'].keys() != expected_stats:
                raise RuntimeError('Incomplete return normalization checkpoint')
            for key, value in weights['return_stats'].items():
                getattr(self, key).copy_(value)
        else:
            self.return_mean.zero_()
            self.return_second.fill_(1)
            self.return_std.fill_(1)
            self.return_updates.zero_()
        if not torch.isfinite(self.return_mean + self.return_std) or self.return_std < 1:
            raise RuntimeError('Invalid return normalization checkpoint')

    def load_text_weights(self, weights):
        """Expand a text-only head without changing its initial action values."""
        first = weights['head']['0.weight']
        hidden = self.head[0].in_features - STATE_DIM - ACTION_DIM
        if first.shape != (512, hidden + ACTION_DIM):
            raise RuntimeError('Expected a compatible text-only value head')
        expanded = first.new_zeros(self.head[0].weight.shape)
        expanded[:, :hidden] = first[:, :hidden]
        expanded[:, hidden + STATE_DIM:] = first[:, hidden:]
        migrated = dict(weights, head=dict(weights['head'], **{'0.weight': expanded}))
        self.load_weights(migrated)


def load(base, device, rank=16, alpha=32, training=False):
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(base, local_files_only=True, dtype=torch.bfloat16,
                                         attn_implementation='sdpa')
    model = QwenValue(backbone, rank, alpha).to(device)
    if training:
        model.backbone.enable_input_require_grads()
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
    else:
        model.eval()
    return model, tokenizer
