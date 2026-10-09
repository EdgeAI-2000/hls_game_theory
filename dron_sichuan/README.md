# DRON-MoE-adapted (Sichuan)

Source: He et al., Opponent Modeling in Deep Reinforcement Learning, ICML 2016,
PMLR 48:1804-1813. Paper: https://proceedings.mlr.press/v48/he16.html
Author code: https://github.com/hhexiy/opponent
Reference commit: 5ab544ca150013cbe5549227107eed64b9928da8 (MIT).

This is an independent PyTorch adaptation, not Qwen fine-tuning and not a claim
of reproducing the original soccer/quizbowl results. The author soccer MoE
architecture is retained: two shared 50-wide ReLU layers, three linear Q
experts, and a 50-wide gate reading both state and opponent features.
Experts are learned components, not prescribed safe/exploit/deceive modes.

## Adaptation contract

- P0 trains against the six frozen TRAIN curriculum profiles. Three HOLDOUT
  profiles are reserved for final evaluation. No hidden opponent identity input.
- Own hand and public table use the existing 1006-dimensional encoding.
  Opponent features contain the last eight observed discard/hu/kong/peng events
  per relative opponent, 3 x 8 x 31 dimensions. Private draws, passes, hidden
  hands and future deck order are excluded. The actor is feedforward, not LSTM.
- There are 31 macroactions: 27 discards plus hu/kong/peng/pass. As in the shared
  baseline executor, kong selects the first legal kong tile. Both acting and TD
  bootstrapping mask illegal actions. This is not a full tile-specific kong space.
- Uniform replay, target network, epsilon-greedy and TD learning are retained.
  The original multitask variants are not claimed: no auxiliary risk/hand head.
- Reward is actual P0 balance difference / base score between P0 decisions.
  The final interval runs through full bloodbattle settlement even after P0 hu.
  Gamma=1 is a finite-episode adaptation; no artificial win reward or hu override.
- PyTorch RMSprop lr0.0005, alpha0.95, eps0.01; Huber threshold1 and gradient
  norm10; replay100000, batch64, warmup1000, approximately one update per four
  P0 decisions. Target copy every10000 decisions at batch boundaries. Epsilon
  declines0.3 to0.1 over500000 decisions. Optimizer details and scheduling are
  declared adaptations, not byte-for-byte Torch/Lua replication.

## Frozen first run

GPU0, eight CPU actors, 32 completed games per actor batch, seed92731,
1000000 P0 training decisions (last batch may overshoot).
This is NOT equivalent to 5000000 four-seat DouZero samples or matched compute.
One training seed does not measure training-seed robustness; fixed budget does
not establish convergence. No extra DQN or concat result is claimed.

Every100000 decisions: fixed1000 TRAIN development games starting927390000000.
Choose highest raw development mean, earliest tie. Freeze checkpoint before
holdout; no holdout-driven selection. Train seeds start927310000000.
Holdout: outer2026082001..2026082020,1000 games each, exact
outer_seed*1000000+game_index CRN plan. Eight actors also evaluate.
Report mean, ron, deal-in, CVaR10, hu and seed-level paired confidence intervals
and nominal/Holm p-values. FFR is not computed. Store full settlement ledgers.

The shared LEGACY engine is intentionally unchanged while other runs execute.
Known scoring and response-order/boundary issues remain; consult
DouZero_Sichuan_Formal_20260926/settlement_audit/RULE_REVIEW.md.
These results cannot establish a rules-validated ranking. Our runner follows
the existing DouZero impossible-kong mask, which differs from P4 at that edge.

## Run

```bash
py10/bin/python3 -m unittest dron_sichuan.test_dron -v
bash dron_sichuan/launch.sh
# Same options and sources, restoring replay, target, optimizer and RNG:
bash dron_sichuan/launch.sh --resume
```

All source hashes and protocol are frozen before training. Checkpoints retain
replay and log commit offsets; interrupted uncommitted batches are replayed on
resume. Holdout units are atomic and verified before reuse. Status, development
selection, logs and final comparisons live under the run output directory.
