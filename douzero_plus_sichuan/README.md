# DouZero+-adapted: implementation prototype

The user subsequently requested formal-budget training on the frozen legacy
environment. Known shared rule issues remain unresolved and are explicitly
recorded in the run protocol; results are not a rules-validated final ranking.
This implementation is independent of Qwen and MASK. Smoke artifacts
are not performance evidence and must not be used as pretrained formal weights.

## Primary sources reviewed

- Zhao et al., DouZero+: Improving DouDizhu AI by Opponent Modeling and
  Coach-guided Learning, IEEE CoG 2022:
  https://ieee-cog.org/2022/assets/papers/paper_50.pdf
- Public code: https://github.com/submit-paper/Doudizhu_plus
- Reviewed commit: `64a4dc29894082f752b43952a8f3a13868445c03`
- `oppo_modeling/dmc/models.py`: LSTM128, four shared FC512 prediction layers,
  per-card FC256/5-class heads; decision network five FC512 layers.
- `oppo_modeling/dmc/dmc.py`: cross entropy prediction loss, MC value MSE,
  predictions detached before decision network input, summed losses.
- `coach_netwrok/dmc/models.py`: hand embedding256, per-seat FC128 stack,
  combined FC256/128/64/output. Public code and paper architecture descriptions
  differ in granularity; this prototype follows the code layer sizes.

## Preserved mechanism and explicit changes

| Component | Implementation | Adaptation |
|---|---|---|
| Deep MC | Each seat's complete final net / base_score, MSE | Shared four-seat policy instead of three landlord/farmer role networks |
| Opponent prediction | Public state/history -> 0..4 copies per tile, CE | Three relative opponents x 27 tiles x 5 classes (405 probabilities) |
| Legal-label mask | Upper bound from own hand, public rivers/melds, public hand lengths | Hidden wall/other hands never accessed for mask; untracked ron tiles leave a conservative loose bound |
| Value network | State + action + detached predicted probabilities | Mahjong encoding inherited from independent DouZero adapter |
| Coach | Training-only initial hands + missing suits -> four logits | Predict seat with highest final net; ties share target mass. Original predicts binary landlord victory |
| Opening filter | max predicted seat probability <= 1-beta | Four-way dominance criterion, not the original binary interval |
| Beta schedule | +.05 per 20,000 learner updates, cap .3 | Original update units preserved; short runs may never activate filtering |
| Coach hand encoder | Shared embedding and seat trunk, padding-aware average | Four exchangeable seats replace landlord/farmer-specific encoders |

The Coach target is a proposed adaptation, not a proven equivalent. It does not
measure payoff magnitude or guarantee learning value. Before full training,
validate on held-out TRAIN openings, report prediction calibration and acceptance
rates, set a bounded attempt budget, and define any exploration/bypass schedule
in advance. Never silently force acceptance after rejection. All rejected seeds
must be logged; compare both attempted deals and played-game training budgets.

## Information boundary

Policy inputs contain only own/private and public information. Simulator hidden
hands are read by the separate rollout hook ONLY for supervised labels. The value
loss cannot backpropagate into the opponent predictor through the prediction
features, matching the reviewed source. Coach is not attached to the deployment
policy and never filters development or holdout evaluation.

`rollout.collect` temporarily hooks the existing encoder/initializer in one
process, restoring them in finally. It is intentionally not thread-safe: actors
must use spawned processes. This bridge avoids editing frozen DouZero or Mix-PPO
files; a final common environment adapter should replace it after rule fixes.

## Validation

```
py10/bin/python3 -m unittest douzero_plus_sichuan.test_prototype -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 py10/bin/python3 -m douzero_plus_sichuan.smoke
```

Tests cover hidden-information invariance, legal count labels, correct four-seat
returns, CE/value gradient separation, Coach tie handling and filter boundaries.
Integration runs 4 actors, 16 attempted games, 2 joint updates and exact checkpoint
reload equivalence. The second smoke batch deliberately uses beta=.3 only to
exercise filtering; it is not a formal training schedule or a learned Coach claim.

## Next experiment, after rule agreement

Compare DouZero-adapted, DouZero+OM-adapted and DouZero+-adapted. A full name requires
both prediction and Coach components with validated training behavior. Freeze
rules, train/dev split, budgets and selection first; evaluate all methods with
P4 outer seeds 2026082001..2026082020 and 1000 games each. Keep training seed
families separate. Do not reuse the old scores after rules change as if they
were obtained in the corrected environment. This prototype does not replace
the original paper's three-player experimental results.

## Authorized formal-budget run, 2026-09-27

`bash douzero_plus_sichuan/launch_formal.sh` starts tmux-compatible training:
GPU1, eight CPU actors, 5,000,000 played decision samples, seed 92711, fresh
initialization. Value and opponent losses use RMSprop; 3200-sample batches are
gradient-accumulated in 128-sample chunks without changing the batch mean loss.
Coach is trained jointly from accepted opening outcomes. All attempted seeds,
predictions, accepted/rejected counts and played-game returns are logged.

**Explicit schedule adaptation:** beta increases .05 every 500,000 completed
decision samples, capped at .30. This replaces the prototype's original
20,000-update schedule so filtering can become active within the fixed budget.
No rejections occur during the first 500,000-sample warmup. A batch aborts
after 640 attempted openings rather than silently bypassing Coach rejection.
Acceptance and calibration have not been established by the short smoke test.

Every 500,000 samples, evaluate 1000 fixed TRAIN development games (no Coach).
Select highest raw development mean, ties earliest; never select on holdout.
Then evaluate outer seeds 2026082001..2026082020, 1000 games each, with eight
parallel game workers and no Coach filtering. Outputs include comparison tables,
paired CIs and nominal/Holm tests against frozen DouZero and P4 baselines.

The run directory is `DouZeroPlus_Sichuan_Formal_20260927`; check `status.json`.
`--resume` with identical arguments checks protocol and source hashes, restores
model/optimizers/RNG/counters and rolls back this run's uncommitted log tails.
Known legacy scoring issues and the legacy P4 impossible-kong runner difference
remain in all reports. Formal-budget execution is not a convergence claim.
