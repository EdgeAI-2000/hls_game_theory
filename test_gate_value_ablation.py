import json
import tempfile
import unittest
from pathlib import Path

from gate_value_policy import (
    GATE_VALUE_FEATURE_DIM,
    GATE_VALUE_FEATURE_NAMES,
    MODE_FEATURE_DIM,
    STATE_DIM,
    apply_gate_value_feature_ablation,
    apply_gate_value_mode_mask,
)
from run_e3_gate_value_ablation import audit_game_seed_schedule, completed_run
from run_gate1_experiments import (
    derive_game_seed,
    load_belief_calibration_steps,
    serializable_mask_config,
)
from run_h1_belief_experiment import clustered_paired_brier_test


class GateValueAblationTest(unittest.TestCase):
    def test_runtime_belief_model_is_replaced_by_artifact_identity(self):
        runtime_model = object()
        result = serializable_mask_config(
            {"belief_public_model": runtime_model, "oracle_samples": 60},
            belief_public_model_path="model.joblib",
            belief_public_model_sha256="abc123",
        )

        self.assertEqual(result["belief_public_model"], "model.joblib")
        self.assertEqual(result["belief_public_model_sha256"], "abc123")
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_h1_paired_test_clusters_repeated_rows_by_game_seed(self):
        rows = []
        for seed, b2, ref in [(1, 0.1, 0.7), (2, 0.2, 0.8), (3, 0.1, 0.6)]:
            for _ in range(10):
                rows.append({
                    "seed": seed,
                    "b_j_oracle": {"true_tenpai": False},
                    "B2": b2,
                    "REF": ref,
                })
        result = clustered_paired_brier_test(rows, "B2", "REF")
        self.assertEqual(result["n_clusters"], 3)
        self.assertEqual(result["n_rows"], 30)
        self.assertGreater(result["mean_brier_gain"], 0.0)
        self.assertLess(result["t_p_value"], 0.05)

    def test_calibration_artifact_must_match_runtime_estimator(self):
        artifact = {
            "calibration": {"B2_B_phi_steps": [[0.5, 0.2], [1.0, 0.8]]},
            "estimator_config": {
                "oracle_samples": 60,
                "oracle_beta": 2.0,
                "danger_threshold": 0,
                "b_phi_source": "mc",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h1_summary.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            steps = load_belief_calibration_steps(
                str(path), oracle_samples=60, oracle_beta=2.0, danger_threshold=0
            )
            self.assertEqual(steps, [(0.5, 0.2), (1.0, 0.8)])
            with self.assertRaisesRegex(ValueError, "danger_threshold"):
                load_belief_calibration_steps(
                    str(path), oracle_samples=60, oracle_beta=2.0, danger_threshold=1
                )

    def test_consecutive_outer_seeds_produce_disjoint_game_seeds(self):
        audit = audit_game_seed_schedule([2026082001, 2026082002], 1000)

        self.assertEqual(audit["total_game_seeds"], 2000)
        self.assertEqual(audit["unique_game_seeds"], 2000)
        self.assertEqual(audit["collision_count"], 0)
        self.assertNotEqual(
            derive_game_seed(2026082001, 999),
            derive_game_seed(2026082002, 0),
        )

    def test_seed_audit_rejects_duplicate_outer_seeds(self):
        with self.assertRaisesRegex(ValueError, "outer seeds must be unique"):
            audit_game_seed_schedule([7, 7], 10)

    def test_completed_run_rejects_summary_without_successful_trace_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "gate1_summary.json").write_text(
                json.dumps({"E2_ladder": {"llm_mask": {"games": 2}}}),
                encoding="utf-8",
            )
            (output / "gate1_games.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            (output / "run.log").write_text(
                "OSError: [Errno 122] Disk quota exceeded\n", encoding="utf-8"
            )
            self.assertFalse(completed_run(output, "llm_mask", 2))

            (output / "run.log").write_text(
                "Saved Gate1 outputs under: test\n", encoding="utf-8"
            )
            self.assertTrue(completed_run(output, "llm_mask", 2))

    def test_no_belief_reconstructs_public_gate_features(self):
        features = [0.0] * GATE_VALUE_FEATURE_DIM
        index = {name: i for i, name in enumerate(GATE_VALUE_FEATURE_NAMES)}
        features[index["own_shanten"]] = 1.0 / 6.0
        features[index["own_tell"]] = 0.3
        features[index["belief_avg"]] = 0.4
        features[index["belief_max"]] = 0.6
        features[index["belief_min"]] = 0.2
        features[index["belief_std"]] = 0.1
        features[index["z_drift_max"]] = 0.2
        features[index["risk_budget"]] = 0.7
        features[index["uncertainty"]] = 0.36

        result = apply_gate_value_feature_ablation(features, "no_belief")

        for name in ("belief_avg", "belief_max", "belief_min", "belief_std"):
            self.assertEqual(result[index[name]], 0.0)
        self.assertAlmostEqual(result[index["risk_budget"]], 0.49)
        self.assertAlmostEqual(result[index["uncertainty"]], 0.6)
        self.assertAlmostEqual(result[index["belief_tell_headroom"]], 0.3)
        self.assertEqual(result[index["near_tenpai_belief_headroom"]], 1.0)

    def test_no_deceive_zeros_only_deceive_action_block(self):
        features = [1.0] * GATE_VALUE_FEATURE_DIM
        result = apply_gate_value_mode_mask(features, ("exploit", "safe"))
        deceive_start = STATE_DIM + 2 * MODE_FEATURE_DIM

        self.assertTrue(all(value == 1.0 for value in result[:deceive_start]))
        self.assertTrue(
            all(value == 0.0 for value in result[deceive_start : deceive_start + MODE_FEATURE_DIM])
        )

    def test_no_z_reconstructs_belief_only_uncertainty(self):
        features = [0.0] * GATE_VALUE_FEATURE_DIM
        index = {name: i for i, name in enumerate(GATE_VALUE_FEATURE_NAMES)}
        features[index["belief_max"]] = 0.6
        features[index["z_drift_max"]] = 0.3
        features[index["z_cusum_max"]] = 0.5
        features[index["z_entropy_max"]] = 0.4
        features[index["z_opponent_count"]] = 1.0
        features[index["risk_budget"]] = 0.7
        features[index["uncertainty"]] = 0.46

        result = apply_gate_value_feature_ablation(features, "no_z")

        for name in (
            "z_drift_max", "z_cusum_max", "z_entropy_max", "z_opponent_count"
        ):
            self.assertEqual(result[index[name]], 0.0)
        self.assertAlmostEqual(result[index["uncertainty"]], 0.16)
        self.assertEqual(result[index["risk_budget"]], 0.7)


if __name__ == "__main__":
    unittest.main()
