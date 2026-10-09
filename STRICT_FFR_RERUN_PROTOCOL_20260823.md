# Strict FFR formal rerun protocol (2026-08-23)

This protocol is frozen before the P4 smoke and formal results are inspected. It
supersedes the FFR and discard-win labels used by the P3R result tables, without
changing the frozen policies, opponent curriculum, outer seeds, or CRN game-seed
schedule.

## 1. Metric definitions

### 1.1 Ron game rate

The former display label `discard_hu_rate` / "弃牌和率" is renamed to
`ron_game_rate` / "荣和对局率". Its numerator is the number of games in which P0
wins from an opponent discard, and its denominator is the number of games. It is
not a per-discard probability and must not be interpreted as the fraction of
discarded tiles that complete P0's hand.

The legacy JSON key `discard_hu_rate` remains as a compatibility alias. New tables
must display `ron_game_rate`.

### 1.2 Strict oracle FFR

For opponent `j` in game `g`, only the first turn satisfying every condition below
is a primary FFR opportunity:

1. The opponent's oracle hand shanten is at most 1.
2. P0's oracle shanten is greater than 0, so P0 is not actually tenpai.
3. Both the policy's push action and chosen action are discards with
   computable post-discard progress.

Let `(s_push, u_push)` and `(s_def, u_def)` be post-discard shanten and effective
remaining tile copies. The opportunity is a strict false fold only if perceived
public threat reaches the opponent's defensive threshold, the chosen action
differs from push, and either:

- `s_def > s_push`; or
- `s_def == s_push` and `u_def < u_push`.

The primary metric is:

```text
strict_FFR = strict_false_folds / first_eligible_opponent_game_opportunities
```

Each `(game_seed, opponent_pid)` contributes at most once. Later eligible turns are
retained only in `strict_FFR_event_diagnostic`; they cannot increase the primary
numerator or denominator. Declined self-gang actions are excluded and reported
separately.

### 1.3 Conditional defensive-switch rate

The former broad FFR is retained only as `conditional_defensive_switch_rate`. It is
a repeated turn-level diagnostic over opponent hands near completion while P0 is
not oracle-dangerous. A switch means that the defensive action differs from push;
it does not require measurable progress loss. It must not be called FFR in paper
tables or claims.

## 2. Frozen inputs

- Base model: `models/Qwen-Mahjong-V3-Merged`
- Curriculum: `Opponent_Curriculum_formal/screened.json`
- Curriculum SHA-256: `0dd20c523dfeedaf50a23bcf509d36ea8bce3bd94005b5b0e49bd018e55fe1f7`
- Public belief model: `Public_Belief_Model_v4_multifamily/public_belief_model.joblib`
- Belief SHA-256: `d5bd71f50633eb5dadae49e396ff9bbcb4c6899538891409ab7d7e80c01364c3`
- Full checkpoint SHA-256: `af105336a034e2865e54a0a56fcc22d75f469ae755cec1f40f65cd8e1aa2a3cb`
- No-z checkpoint SHA-256: `f10421312734df7115660515bbd28b0a202aabe8ac9c453004b6c081b13837ee`
- No-belief checkpoint SHA-256: `68965c2389dc47b00a5028d5a855f78e94a708bea8df35c60f2b1a7894362327`
- No-deceive checkpoint SHA-256: `0cc6b84d382eb22e84758dcf46d082c4d0f23d94f4e848c5f9856d653b7f139a`
- No-course checkpoint SHA-256: `2b9eb5f0f851ffa9a2f95663ea583de2205ec91bad7a6dae1c59d8850a41a468`
- Formal outer seeds: `2026082001` through `2026082020`
- Game seed: `outer_seed*1000000+game_index`
- Games per formal cell: 1,000
- `auto_hu=off`, oracle samples 60, beta 2.0, danger threshold 0

Only measurement and terminal-state correctness code changed. The frozen policies
are reused; no checkpoint is retrained or selected using P4 results.

## 3. Staged execution and gates

### Stage A: smoke preflight

Run all eight E2/E3 systems for two outer seeds and 100 games per seed, giving 200
games per system and 1,600 games total. The formal stages may start only if the
automated audit reports all of the following:

- exactly 100 games per cell and no duplicate game seeds;
- identical paired game-seed sets across all eight systems;
- zero `max_steps_or_unsettled` games;
- stable runtime artifact hashes and the frozen belief SHA;
- at most three primary FFR opportunities per game;
- no repeated `(game_seed, opponent_pid)` primary FFR key;
- every primary opportunity satisfies shanten, action, and opponent-ID constraints;
- every strict false fold reaches the threat threshold and has measurable shanten
  or effective-copy loss;
- event files and aggregate FFR numerators/denominators agree;
- opponent-level decomposition is present.

### Stage B: formal E2/E3

Run 8 systems x 20 seeds x 1,000 games = 160,000 games on the frozen holdout.
The Stage A audit is repeated at formal scale.

### Stage C: E4

Run full on the six training opponents for 20 seeds x 1,000 games = 20,000 games.
Reuse the new Stage B full holdout cells. Recompute E4 from those paired outputs.

### Stage D: E5

Run holdout `tight` and `loose` for 40,000 games, reusing Stage B endpoints for
`no_deceive`, `default`, and `constant_deceive`. Run all five settings against the
frozen anti-deception opponent for 100,000 games. Total new E5 simulation is
140,000 games.

### Stage E: E6

Reanalyze the new Stage B full traces. E6 does not launch additional games.

The total new formal simulation after smoke is 320,000 games. Any failed audit
stops the pipeline before the next stage. P3R FFR values and P3R display labels are
historical only after P4 completes.
