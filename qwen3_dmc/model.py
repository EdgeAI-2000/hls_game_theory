import json
from collections import Counter

import numpy as np
import torch
from torch import nn
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModel, AutoTokenizer

from douzero_sichuan.core import ACTION_DIM, KINDS, action_tile, tile_index

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
        layers, width = [], backbone.config.hidden_size + ACTION_DIM
        for _ in range(5):
            layers += [nn.Linear(width, 512), nn.ReLU()]
            width = 512
        self.head = nn.Sequential(*layers, nn.Linear(width, 1))

    def encode(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False, return_dict=True).last_hidden_state
        last = attention_mask.sum(1) - 1
        return hidden[torch.arange(len(last), device=hidden.device), last].float()

    def forward(self, input_ids, attention_mask, action):
        hidden = self.encode(input_ids, attention_mask)
        return self.head(torch.cat((hidden, action.float()), -1)).squeeze(-1)

    @torch.inference_mode()
    def score(self, ids, candidates):
        device = next(self.head.parameters()).device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        hidden = self.encode(input_ids, torch.ones_like(input_ids))
        action = torch.as_tensor(candidates, dtype=torch.float32, device=device)
        values = self.head(torch.cat((hidden.expand(len(action), -1), action), -1)).squeeze(-1)
        if not torch.isfinite(values).all():
            raise FloatingPointError('Non-finite actor values')
        return values.cpu().numpy()

    def weights(self):
        return dict(adapter={k: v.detach().cpu().clone() for k, v in
                             get_peft_model_state_dict(self.backbone).items()},
                    head={k: v.detach().cpu().clone() for k, v in self.head.state_dict().items()})

    def load_weights(self, weights):
        expected = get_peft_model_state_dict(self.backbone)
        if weights['adapter'].keys() != expected.keys() or any(
                weights['adapter'][k].shape != v.shape for k, v in expected.items()):
            raise RuntimeError('Incomplete or incompatible Qwen adapter checkpoint')
        result = set_peft_model_state_dict(self.backbone, weights['adapter'])
        if result.unexpected_keys:
            raise RuntimeError(result.unexpected_keys)
        self.head.load_state_dict(weights['head'], strict=True)


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
