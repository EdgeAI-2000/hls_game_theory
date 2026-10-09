import copy
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from douzero_sichuan import core
from .model import DRON, observe, action_id, td_target, STATE_DIM, OPP_DIM
from .rollout import collect
from .train import Replay, copy_frozen, digest


class DRONTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_private_information_and_pass_isolation(self):
        game, _, _ = core.init_game(92737001,'random','privacy')
        legal = core.get_legal_actions(game,0)
        tile = game.players[0].hand_tiles[0]
        events = [(1,'d',tile,False)]
        before = observe(game,0,legal,events)
        other = copy.deepcopy(game)
        for p in other.players[1:]: p.hand_tiles = [tile] * len(p.hand_tiles)
        other.deck.tiles.reverse()
        after = observe(other,0,legal,events+[(2,'n',None,True)]*100)
        for a,b in zip(before,after): np.testing.assert_array_equal(a,b)

    def test_legal_and_terminal_td_targets(self):
        q = torch.tensor([[999.,3.,-2.],[999.,999.,999.]])
        mask = torch.tensor([[False,True,True],[False,False,False]])
        y = td_target(torch.tensor([1.,-1.]),torch.tensor([False,True]),q,mask)
        torch.testing.assert_close(y,torch.tensor([4.,-1.]))
        with self.assertRaises(RuntimeError):
            td_target(torch.zeros(2),torch.zeros(2,dtype=torch.bool),q,mask)

    def test_experts_and_gater_learn(self):
        torch.manual_seed(42)
        model = DRON()
        s,h = torch.randn(12,STATE_DIM),torch.randn(12,OPP_DIM)
        opt = torch.optim.RMSprop(model.parameters(),lr=.0005,alpha=.95,eps=.01)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        initial = float(F.smooth_l1_loss(model(s,h)[:,0],torch.ones(12)).detach())
        for _ in range(30):
            loss = F.smooth_l1_loss(model(s,h)[:,0],torch.ones(12))
            opt.zero_grad();loss.backward();opt.step()
        self.assertLess(float(loss.detach()),initial)
        for prefix in ('trunk','experts','gater'):
            self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith(prefix)))

    def test_complete_settlement_and_runner_parity(self):
        class Fixed:
            def choose(self,obs,rng,epsilon): return int(np.flatnonzero(obs[2])[0])
        class OriginalFixed:
            def choose(self,state,history,candidates,rng,epsilon):
                ids = []
                for a in candidates:
                    kind = int(a[:5].argmax())
                    ids.append(int(a[5:].argmax()) if kind == 0 else 26 + kind)
                return int(np.argmin(ids))
        for seed in range(927370000000,927370000004):
            rows,m = collect(Fixed(),seed)
            original = core.episode(OriginalFixed(),seed,0,True,'train')[3]
            for key in ('net','ron','dr','hu','steps'): self.assertEqual(m[key],original[key])
            self.assertAlmostEqual(sum(r[2] for r in rows),m['net'][0]/10)
            self.assertEqual(sum(r[4] for r in rows),1)
            self.assertFalse(rows[-1][3][2].any())

    def test_only_legal_actions_selected(self):
        game,_,_ = core.init_game(92737002,'random','legal')
        legal = core.get_legal_actions(game,0)
        obs = observe(game,0,legal,[])
        model = DRON().eval()
        for epsilon in (0,1):
            self.assertIn(model.choose(obs,random.Random(9),epsilon),[action_id(a) for a in legal])

    def test_checkpoint_bytes_and_replay_resume(self):
        with tempfile.TemporaryDirectory() as d:
            source, destination = Path(d)/'best.pt', Path(d)/'selected.pt'
            torch.save(DRON().state_dict(),source)
            copy_frozen(source,destination)
            self.assertEqual(digest(source),digest(destination))
            copy_frozen(source,destination)
            self.assertEqual(digest(source),digest(destination))
        replay = Replay(3)
        obs = (np.zeros(STATE_DIM,np.float32),np.zeros(OPP_DIM,np.uint8),np.ones(31,bool))
        replay.add([(obs,0,float(i),obs,False) for i in range(5)])
        restored = Replay(3); restored.restore(replay.state())
        a = replay.sample(np.random.default_rng(4),'cpu')
        b = restored.sample(np.random.default_rng(4),'cpu')
        for key in a: torch.testing.assert_close(a[key],b[key])


if __name__ == '__main__': unittest.main()
