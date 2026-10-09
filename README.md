# LLM Game Theory for Sichuan Mahjong

Core code for experiments on multi-agent belief shaping, opponent modeling, and action selection in Sichuan Mahjong.

The project studies how an LLM-based Mahjong agent can combine public-risk estimation, value-gated action candidates, and opponent curricula to improve induced deal-in outcomes while controlling self-risk.

## Contents

- `game.py`, `tiles.py`, and `rule_engine.py`: Sichuan Mahjong environment and rule utilities.
- `belief_oracle.py`, `public_belief_model.py`, and `ppo_features.py`: public-risk and state features.
- `gate_value_policy.py` and `mask_llm.py`: MASK action candidates and gating components.
- `dron_sichuan/`: DRON-MoE adapted baseline.
- `douzero_sichuan/`: DouZero adapted baseline.
- `douzero_plus_sichuan/`: opponent-aware DouZero+ adapted baseline.
- `tjong_sichuan/`: Tjong-style hierarchical Transformer adaptation.
- `qwen*_dmc/`: controlled decision-making studies.
- `FORMAL_RESULTS_SUMMARY_20260822.md`: formal experiment summary and limitations.
- `MASK_PAPER_DRAFT_P4_V4.md`: paper draft.

## Environment

- Python 3.10+
- PyTorch
- NumPy

Install the dependencies required by the selected experiment in your own environment. Large language model weights, generated datasets, checkpoints, and raw logs are intentionally excluded from this public source release.

## Reproducibility

The formal protocols record seeds, paired evaluation rules, artifact hashes, and metric definitions. Before running an experiment, inspect the corresponding protocol and adaptation README. The adaptations of DouZero, DRON, and Tjong are labeled as adapted methods because the original implementations or rule assumptions do not exactly match Sichuan Mahjong.

## License

The repository is released for research use. Add the group-approved license before publication if required by the project or venue.
