import copy
import random
import unittest
import numpy as np
import torch
from douzero_sichuan import core
from .models import PlusPolicy,Coach,losses,coach_target,accept_opening,beta_schedule,public_count_mask
from .rollout import collect


class PrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(927)
        cls.policy=PlusPolicy().eval()

    def test_privacy(self):
        game,_,_=core.init_game(927111,'random','privacy')
        pid=game.current_player_id
        legal=core.get_legal_actions(game,pid)
        a=core.encode(game,pid,legal,[])
        altered=copy.deepcopy(game)
        for p in altered.players:
            if p.player_id!=pid:
                p.hand_tiles=[game.players[pid].hand_tiles[0]]*len(p.hand_tiles)
        altered.deck.tiles.reverse()
        b=core.encode(altered,pid,legal,[])
        for left,right in zip(a,b):
            np.testing.assert_array_equal(left,right)
        self.assertEqual(self.policy.choose(*a,random.Random(1),0),self.policy.choose(*b,random.Random(1),0))

    def test_returns_labels_gradients(self):
        original=core.encode
        r=collect(self.policy,927010000001,epsilon=1)
        self.assertIs(core.encode,original)
        self.assertTrue(r['accepted'])
        cursor=0
        for pid,count in enumerate(r['metrics']['decisions']):
            np.testing.assert_array_equal(r['y'][cursor:cursor+count],np.full(count,r['metrics']['net'][pid]/10))
            cursor+=count
        x,z,y,labels=[torch.from_numpy(r[k][:8]) for k in ('x','z','y','labels')]
        total,parts=losses(self.policy,x,z,y,labels)
        self.assertTrue(torch.isfinite(total))
        self.policy.zero_grad()
        parts['value'].backward(retain_graph=True)
        self.assertTrue(all(p.grad is None for p in self.policy.opponent.parameters()))
        parts['prediction'].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in self.policy.opponent.parameters()))
        mask=public_count_mask(x[:,:core.STATE_DIM])
        self.assertTrue(mask.gather(-1,labels.unsqueeze(-1)).all())

    def test_coach(self):
        coach=Coach()
        hands=torch.full((2,4,14),27,dtype=torch.long)
        hands[:,:,:13]=1
        missing=torch.zeros(2,4,3)
        logits=coach(hands,missing)
        target=coach_target([[0,0,0,0],[5,-5,5,-5]])
        torch.testing.assert_close(target,torch.tensor([[.25]*4,[.5,0,.5,0]]))
        loss=-(target*logits.log_softmax(-1)).sum(-1).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(accept_opening(torch.tensor([[.99,.005,.003,.002]]),0).item())
        self.assertFalse(accept_opening(torch.tensor([[.99,.005,.003,.002]]),.3).item())
        self.assertEqual(beta_schedule(20000),.05)
        self.assertEqual(beta_schedule(200000),.3)


if __name__=='__main__':
    unittest.main()
