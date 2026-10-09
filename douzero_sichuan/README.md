# DouZero-adapted: Sichuan blood-battle

Independent implementation of the deep Monte Carlo recipe from DouZero
(ICML 2021), https://proceedings.mlr.press/v139/zha21a.html.
Reference code: https://github.com/kwai/DouZero at
`718a5c920bf3361e34178a38f3b80458e176b351`.
This is a game adaptation, not a bit-for-bit reproduction or Qwen fine-tuning.

## Preserved algorithm

- Score every legal state-action pair; epsilon-greedy with epsilon 0.01.
- LSTM hidden size 128; five 512-unit ReLU hidden layers; scalar Q output.
- Regress complete episode returns with MSE. No TD bootstrap, GAE, PPO,
  search, risk head, privileged observation, or MASK modules.
- RMSprop: lr 0.0001, alpha 0.99, eps 0.00001, momentum 0; gradient norm cap 40.
- Four CPU actor processes and a GPU learner; each game uses frozen weights.

## Explicit adaptations

- One shared relative-seat network replaces the three asymmetric DouDizhu roles.
- 27 tile types; own hand and public rivers, melds, missing suits, hand counts,
  balances, win flags, current seat, remaining wall count and response phase.
  History is the last 32 public executed decisions, never raw engine history
  (which can contain hidden draws). Public meld visibility follows this engine.
- Candidates are the existing `d tile/h/g/p/n` interface. Self-kong selects the
  engine's first gang tile. Exchange and missing suit use the common fixed
  heuristics and are not learned. These limitations must be disclosed.
- All four seats self-play. Each recorded action is labeled with its owner's
  FINAL net balance divided by base_score=10, including post-win settlements.
  There is no shaping, clipping, log transform, or payoff sign copying.
- Synchronous parallel game batches replace the original asynchronous queues.
  Fresh full-episode samples receive one shuffled regression pass, minibatches
  up to 3200; this replaces the original fixed unroll buffer layout.
- Preserve P4 response insertion order and first accepted response. Rule bots
  use engine response options, while learner seats use the stricter common
  P0 missing-suit filter. Remove response kongs at wall<=1 because the engine
  rejects them. This local boundary correction should be audited before a
  formal paired comparison; no shared engine file has been modified.

## First training stage

Run `bash douzero_sichuan/launch.sh` in tmux `douzero_sichuan_20260926`.
GPU1 learner, four CPU workers, 32 games/update, 300,000 decision-transition
budget (last full batch may overshoot). This is a learning pilot, NOT a
convergence claim or formal holdout experiment.

Evaluate initialization and every 50,000 decisions on 100 fixed development
games against TRAIN curriculum opponents. Never select using holdout.
Development mean, ron, DR, win rate and empirical CVaR are descriptive only.
Self-play mean across four seats is necessarily zero and is not a strength metric.
The training seed family is separate from development and P4 game seeds.

Every completed game checks zero-sum payoff and exact settlement ledger
reconciliation. Logs retain each seat's decisions, wins, ron, deal-in flags,
missed hu, seed, policy version and settlement totals. Checkpoints contain
optimizer state, RNG state and source hashes; `--resume` refuses changed sources.

Verification: `py10/bin/python3 -m unittest douzero_sichuan.test_core -v`.
The first smoke run exposed a rule-bot response-interface mismatch; its failed
artifacts are retained under `DouZero_Sichuan_20260926/smoke`. The corrected
parallel smoke completed 48 games / 3372 decisions under `smoke_fixed`.
Before using results in the paper, examine development learning curves, obtain
adequate training budgets/multiple training seeds, freeze a selection rule,
and run the same formal paired holdout protocol as other algorithms.

## Formal pipeline (2026-09-26)

`bash douzero_sichuan/launch_formal.sh` starts a separate run under
`DouZero_Sichuan_Formal_20260926`, tmux `douzero_formal_20260926`.
The pilot remains intact. Formal training starts from random initialization
with seed 92602, GPU1 and eight CPU actors, fixed 5,000,000 decision budget.
Every 500,000 decisions, evaluate 1,000 fixed TRAIN development games.
Select the checkpoint with highest development mean net; ties prefer the
earlier checkpoint. Initialization is excluded. Freeze SHA before holdout.

Holdout uses eight parallel workers, outer seeds 2026082001--2026082020,
1,000 games each, game seed = outer_seed*1000000+game_index. Training uses
a disjoint seed family. All five P4 baseline series are validated before
training. Automatically produce `holdout/COMPARISON.md` and `comparison.json`
with per-seed metrics, paired confidence intervals and nominal/Holm p-values.
The status file records training, holdout, complete or failed. This is one
training seed; 20 evaluation seeds do not measure training-seed variance.
The training worker's historical `PILOT_COMPLETE` marker also marks budget
completion here; the authoritative formal completion is `status.json`.
