# Formal E2/E3 Protocol

E2 and E3 are separate claims but share one evaluation matrix. E2 reports the
system ladder; E3 reports strict module ablations. Every row uses the same
frozen curriculum holdout, the same 20 outer seeds, and the same games per seed.

Existing neural-opponent ablations remain supplementary because their FFR is a
behavioral proxy and they do not include `no_course`.

## 1. Generate And Screen The Course

```bash
CUDA_VISIBLE_DEVICES=0 py10/bin/python3 generate_opponent_curriculum.py \
  --output Opponent_Curriculum_formal/generated.json \
  --model-path models/qwen/Qwen2___5-1___5B-Instruct \
  --batches 10 --per-batch 4 --target-size 30 --min-distance 0.05

py10/bin/python3 screen_opponent_curriculum.py \
  --input Opponent_Curriculum_formal/generated.json \
  --output Opponent_Curriculum_formal/screened.json \
  --work-dir Opponent_Curriculum_formal/screen_runs \
  --games 100 --parallel-workers 8 \
  --min-inference-score 0.25 --min-distance 0.05
```

The screen has two independent parts:

- `100` full games per candidate check legality and the strength window.
- Paired low/high public-discard interventions measure actual threat change and
  defensive-policy switching under common random numbers.
- Diversity is the RMS distance between fixed behavioral response vectors, not
  JSON parameter distance. The final train/holdout split is assigned after
  screening by farthest-point selection in the same response space.

Freeze `screened.json` before training. Report its fingerprint, generation
model, screening thresholds, accepted count, and fixed train/holdout counts.

## 2. Run The Shared Matrix

```bash
py10/bin/python3 run_formal_e3_pipeline.py \
  --output-dir Formal_E2_E3 \
  --base-model models/Qwen-Mahjong-V3-Merged \
  --curriculum-manifest Opponent_Curriculum_formal/screened.json \
  --belief-calibration-summary H1_belief_results_v11_public_model_clustered_final/h1_summary.json \
  --belief-public-model Public_Belief_Model_v4_multifamily/public_belief_model.joblib \
  --mask-oracle-samples 60 --mask-oracle-beta 2.0 --mask-danger-threshold 0 \
  --target-states 900 --target-deceive-states 240 --crn-seeds 8 \
  --epochs 50 --min-examples 500 \
  --eval-seeds 2026082001 2026082002 2026082003 2026082004 2026082005 \
               2026082006 2026082007 2026082008 2026082009 2026082010 \
               2026082011 2026082012 2026082013 2026082014 2026082015 \
               2026082016 2026082017 2026082018 2026082019 2026082020 \
  --games-per-seed 1000 --parallel-workers 8 --auto-hu off --gpu 0
```

Each per-game RNG seed is encoded as
`outer_seed * 1000000 + game_index`. Before launching workers, the evaluator
writes `game_seed_audit.json` and refuses to run unless all 20,000 game seeds
are unique. Adjacent outer seeds are therefore valid and remain paired across
all variants without overlapping one another.

The pipeline trains `full`, `no_z`, `no_belief`, `no_deceive`, and
`no_course`. `no_risk_gate` reuses `full` weights and forces deceive whenever a
distinct legal deceive candidate exists. Runtime module inputs are disabled for
`no_z` and `no_belief`; these are not feature-zeroing-only ablations.

The single run evaluates eight unique systems:

- E2: `L0`, `L1`, `L2_no_deceive` (safe/exploit), `L2_full`.
- E3: `L2_full`, `L2_no_z`, `L2_no_belief`, `L2_no_deceive`,
  `L2_no_risk_gate`, `L2_no_course`.

## 3. Publishability Checks

- Use `e2_ladder_table` for E2 and `e3_ablation_table` for E3 in
  `Formal_E2_E3/formal_e3/e3_ablation_summary.json`.
- Require 20 completed outer seeds for every row.
- Report paired confidence intervals and paired tests, not only means.
- Use baseline-adjusted `excess_DIR_per_game` as formal DIR.
- Use oracle-grounded FFR only. Neural proxy FFR is supplementary.
- Report raw and calibrated Brier/ECE/BSE and paired `DeltaBelief` separately.
- Do not publish values from directories containing `smoke` in the name.
