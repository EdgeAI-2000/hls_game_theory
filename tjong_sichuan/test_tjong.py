import copy
import unittest
import numpy as np
import torch

from douzero_sichuan.core import init_game,parse_console_tile
from .model import Tjong,options,public_state,privileged_state,tensor_batch,STATE_DIM
from .fan import score_tiles,backward
from .rollout import collect
from .train import advantages,ppo_loss


class TjongTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(12)
        cls.model=Tjong().eval()

    def batch(self):
        game,_,_=init_game(9284701,'random','test')
        states=np.zeros((1,4,STATE_DIM),np.float32);states[0,-1]=public_state(game,0)
        return game,dict(states=states,prev_actions=np.full((1,4),7,np.int64),prev_tiles=np.full((1,4),28,np.int64),
                         prev_rewards=np.zeros((1,4),np.float32),legal=options(game,0)[None],privileged=privileged_state(game,0)[None])

    def test_private_policy_isolation(self):
        game,b=self.batch();other=copy.deepcopy(game)
        for p in other.players[1:]:p.hand_tiles=[game.players[0].hand_tiles[0]]*len(p.hand_tiles)
        other.deck.tiles.reverse()
        np.testing.assert_array_equal(public_state(game,0),public_state(other,0))
        before=self.model.choose(tensor_batch(b),np.random.default_rng(8))
        b['privileged']=privileged_state(other,0)[None]*0
        after=self.model.choose(tensor_batch(b),np.random.default_rng(8))
        self.assertEqual(before,after)

    def test_hierarchy_log_probability_and_gradients(self):
        _,b=self.batch();kind,tile,lp=self.model.choose(tensor_batch(b),np.random.default_rng(3))
        self.assertTrue(b['legal'][0,kind,tile])
        b.update(kind=np.array([kind]),tile=np.array([tile]))
        logp,entropy,value=self.model.evaluate(tensor_batch(b))
        self.assertAlmostEqual(float(logp.detach()),lp,places=5)
        loss,parts=ppo_loss(logp,logp.detach(),torch.ones(1),value,value.detach(),torch.ones(1),entropy)
        self.model.zero_grad();loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
        self.assertAlmostEqual(parts['kl'],0,places=5)

    def test_hu_pass_and_multiple_kong_masks(self):
        game,_=self.batch()
        game.players[0].missing_suit=None
        game.players[0].hand_tiles=[parse_console_tile(t) for t in ['1万']*4+['2万']*4+['3万']*3+['4万']*3]
        mask=options(game,0)
        self.assertEqual(int(mask[4].sum()),2)
        legal=options(game,0,['hu'])
        self.assertTrue(legal[1,27] and legal[6,27])

    def test_fan_zero_and_pattern_support(self):
        final=np.zeros(27,np.float32);final[:4]=3;final[4]=2
        score=score_tiles(final,final,['平胡(0番)'],0)
        self.assertAlmostEqual(float(np.dot(final,score)),1,places=6)
        roots=final.copy();roots[0]=4
        s=score_tiles(roots,roots,['平胡(0番)','根×1(1番)'],1)
        self.assertGreater(s[0]*4,s[1]*3)
        self.assertAlmostEqual(float(np.dot(roots,s)),2,places=6)
        with self.assertRaises(ValueError):score_tiles(final,final,['unknown'],0)

    def test_fan_winner_meld_and_dealin_credit(self):
        final=np.zeros(27,np.float32);final[0]=3;final[1]=2
        score=np.zeros(27,np.float32);score[0]=2;score[1]=1
        records=[[{'post_counts':final.copy()}],[{'post_counts':final.copy()}],[{'post_counts':final.copy()}],[]]
        win=dict(pid=0,counts=final,score=score,record_end=1,time=8,giver=1,discard_record=0,tile=1)
        claims=[dict(winner=0,giver=2,record=0,time=2,size=3,tile=0)]
        r=backward(records,[win],claims)
        self.assertEqual(float(r[0][0]),8)
        self.assertEqual(float(r[1][0]),-1)
        self.assertEqual(float(r[2][0]),-6)

    def test_gae_does_not_cross_seats_or_games(self):
        rewards=np.array([1,2,100],np.float32);done=np.array([False,True,True]);values=np.zeros(3,np.float32)
        adv,ret=advantages(rewards,done,values,gamma=1,lam=1)
        np.testing.assert_array_equal(ret,[3,2,100])

    def test_real_teacher_game_reward_reconciliation(self):
        data,m=collect(None,928470000001,'bc')
        self.assertEqual(sum(m['net']),0)
        self.assertAlmostEqual(float(data['actual_reward'].sum()),0,places=5)
        self.assertEqual(data['states'].shape[1:],(4,STATE_DIM))
        self.assertTrue(data['legal'][np.arange(len(data['kind'])),data['kind'],data['tile']].all())


if __name__=='__main__':unittest.main()
