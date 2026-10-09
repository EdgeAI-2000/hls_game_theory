"""Auditable Eq.(6)/(7) adaptation with bloodbattle win-event attribution."""
import re
import numpy as np

from douzero_sichuan.core import tile_index


def counts(tiles):
    return np.bincount([tile_index(t) for t in tiles],minlength=27).astype(np.float32)


def score_tiles(final_counts, concealed_counts, fan_types, fan, max_fan=8):
    present = final_counts > 0
    if not present.any(): raise ValueError('Empty winning hand')
    # A base component preserves a nonzero auxiliary signal for zero-fan wins.
    type_score = present.astype(float)/present.sum()
    parsed = 0
    for label in fan_types:
        match = re.search(r'\((\d+)番\)',label)
        if match is None: raise ValueError(f'Unrecognized fan label: {label}')
        points = int(match.group(1)); parsed += points
        if not points: continue
        if label.startswith('根'):
            involved = final_counts == 4
        elif label.startswith('金钩钓'):
            involved = concealed_counts > 0
        else:
            # Remaining local patterns constrain the entire hand or winning event.
            known = ('清双豪七对','豪华七对','清龙七对','龙七对','清七对','七对',
                     '清一色','碰碰胡','带幺九','自摸','天胡','地胡','杠上花','海底捞月','抢杠胡')
            if not label.startswith(known): raise ValueError(f'Unknown pattern support: {label}')
            involved = present
        if not involved.any(): raise ValueError(f'No supporting tiles: {label}')
        type_score += involved * (points/involved.sum())
    if parsed != fan: raise ValueError('Fan labels do not reconcile with declared fan')
    # Sichuan pays exponentially, unlike additive MCR fan. Preserve supports,
    # then normalize their total to one payer's actual capped multiplier.
    type_score *= (2**min(fan,max_fan))/type_score.sum()
    per_copy = np.divide(type_score,final_counts,out=np.zeros(27),where=present)
    return per_copy.astype(np.float32)


def backward(records, wins, claims):
    rewards = [np.zeros(len(r),np.float32) for r in records]
    for win in wins:
        pid,final,score = win['pid'],win['counts'],win['score']
        own = records[pid][:win['record_end']]
        # Length normalization is an explicit adaptation, preventing additional
        # turns from multiplying the entire winning value repeatedly.
        for i,r in enumerate(own):
            rewards[pid][i] += float(np.dot(final-np.abs(final-r['post_counts']),score))/max(1,len(own))
        for claim in claims:
            if claim['winner'] == pid and claim['time'] <= win['time']:
                giver, index = claim['giver'],claim['record']
                if index is not None:
                    rewards[giver][index] -= claim['size']*score[claim['tile']]
        if win['giver'] is not None and win['discard_record'] is not None:
            rewards[win['giver']][win['discard_record']] -= score[win['tile']]
    return rewards
