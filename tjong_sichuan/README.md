# Tjong-adapted for Sichuan bloodbattle

Primary source: https://doi.org/10.1049/cit2.12298 (Li et al., 2024).
The paper's https://github.com/750251985/Tjong repository returned404 at review.
This is a paper-based implementation, NOT a reproduction using author code.

## Retained methods

TIT inner spatial Transformer3 layers and outer causal Transformer3 layers;
memory4; action decision followed by an action-conditioned second encoder pass
for tile selection; action/discard/claim heads; supervised initialization then
PPO self-play. Privileged hand and wall counts feed ONLY the training critic.
Both masked action and conditional tile probabilities enter PPO ratios.

Width448,8 heads, FFN1792, dropout0. Parameter count is in PROTOCOL.json.
Tile tokens and public features are adapted to27 Sichuan tiles. Memory stores
the last4 own decision states, previous actions and already observed monetary
rewards. Retrospective fan rewards never enter actor memory. Zero padding is
causal history padding, not future observations.

Seven action types: discard/hu/peng/minggang/angang/bugang/pass. Chow is removed;
all legal kong tiles are retained. Existing baselines use a first-kong-tile
macroaction: this difference must be disclosed. Hu/pass use a dummy tile index.
Initial exchange/missing-suit decisions retain shared environment helpers.

## Fan backward adaptation

Equation6: identify tile-type support for reported fan patterns, distribute
each component over supporting types, then divide by tile count. Roots support
quads, single-wait supports the concealed pair, other local patterns/event
bonuses support the full hand. Unknown labels fail. Whole-hand pattern support,
especially luxury pairs, is an explicit approximation because the engine does
not expose MCR-style decomposition traces.

Include a base component for zero-fan wins, then rescale type weights to the
Sichuan capped payout multiplier2^min(fan,8). Equation7 similarity is used for
the winner's post-action states, divided by its trajectory length to avoid
multiplying payout by turn count. Meld providers receive negative contribution
at the source discard. Ron providers receive the winning-tile penalty. Sum
across actual bloodbattle wins. These apply even to players who do not win.
Supplemented kongs retain the original meld's provider attribution.

Reward = actual interval net/base_score +0.1*fan_credit. Actual dajiao/huazhu
penalties are retained. No hypothetical winning hand is assigned to nonwinners.
Critic inputs and reward construction are inaccessible to action selection.
No policy-invariance claim is made for this retrospective shaping.

## Frozen budget

BC:8000 synthetic rule games, not Botzone human logs. Index modulo10==0 is
validation (800 games),7200 training games. P0 uses an independent shared rule
heuristic; other seats use the6 train profiles. Policy-mask-blocked responses
become pass in BC. All four seats teach. Max20 epochs, patience4 on validation
joint NLL, no holdout selection. Data and epochs differ from the original paper.

RL:2000000 four-seat samples; synchronous four-player self-play of a frozen
actor snapshot per32-game batch. PPO3 epochs, Adam1e-4, batch1024 accumulated in
microbatches16, clip.2/valueclip.3/gradnorm.5. Gamma1, GAE lambda.95, value
coefficient.5, entropy.01 are disclosed choices. Complete settlement required.

Development:1000 fixed train-opponent games after BC and each200000 PPO frames,
max raw mean, earliest tie. Supervised checkpoint can remain best. One training
seed and fixed budget do not establish convergence. BC seeds928410000000+,
RL928420000000+, dev928490000000+ are disjoint. Holdout2026082001..2026082020,
1000 games each,outer*1000000+index. Report mean/ron/DR/CVaR10/hu, paired20-seed
confidence intervals and Holm tests. Strict FFR is not computed. Save ledgers.

GPU1;8 CPU workers; allocator cap25% GPU memory,1 BLAS thread per worker.
Other users' processes remain untouched. Sources and protocol are hashed and
snapshotted. Shared engine remains unchanged, including known legacy defects.
This is NOT a rules-validated ranking.

```bash
py10/bin/python3 -m unittest tjong_sichuan.test_tjong -v
bash tjong_sichuan/launch.sh
bash tjong_sichuan/launch.sh --resume
```

Status/checkpoints: Tjong_Sichuan_Formal_20260928. Resume restores optimizer,
counters/RNG, verifies BC shards and reuses only checkpoint-verified holdout
units. Interrupted BC epochs/PPO batches restart from the last checkpoint.
