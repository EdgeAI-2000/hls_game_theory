from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from opponent_curriculum import (
    assign_train_holdout,
    load_curriculum,
    normalize_spec,
    response_vector_distance,
)
from run_gate1_experiments import init_game
from screen_opponent_curriculum import assign_behavior_holdout, behavioral_inference_probe


def spec(index: int) -> dict:
    return {
        "id": f"beta_{index}",
        "name": f"opponent {index}",
        "archetype": "test",
        "threat_model": "discard_tell" if index % 2 else "blend",
        "threat_threshold": 0.28 + 0.06 * index,
        "oracle_samples": 10 + index,
        "beta": 1.0 + 0.2 * index,
        "danger_threshold": index % 3,
        "ffr_hand_shanten": index % 2,
        "tell_weight": 0.3 + 0.1 * index,
        "tell_window": 3 + index,
    }


class OpponentCurriculumTest(unittest.TestCase):
    def test_normalization_enforces_public_runtime_schema(self):
        value = normalize_spec({**spec(1), "threat_threshold": 99, "tell_window": -1})
        self.assertEqual(value["threat_threshold"], 0.70)
        self.assertEqual(value["tell_window"], 3)
        self.assertEqual(value["tell_weight"], 1.0)

    def test_fixed_split_and_runtime_ids(self):
        opponents = assign_train_holdout([spec(index) for index in range(8)], 0.25)
        self.assertEqual(sum(item["split"] == "holdout" for item in opponents), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "course.json"
            path.write_text(json.dumps({"opponents": opponents}), encoding="utf-8")
            manifest = load_curriculum(path, "holdout")
            self.assertEqual(len(manifest["opponents"]), 2)
            game, _, defenders = init_game(
                1234,
                "curriculum",
                "curriculum_test",
                {"curriculum_manifest": str(path), "curriculum_split": "holdout"},
            )
            self.assertTrue(game.players[0].hand_tiles)
            self.assertEqual(set(defenders), {1, 2, 3})
            self.assertTrue(all(defender.curriculum_spec["split"] == "holdout" for defender in defenders.values()))

    def test_behavioral_probe_uses_paired_public_signal_intervention(self):
        candidate = normalize_spec({
            **spec(1),
            "threat_model": "discard_tell",
            "threat_threshold": 0.4,
        })
        result = behavioral_inference_probe(candidate, [202608111], samples=2)

        self.assertGreater(result["mean_paired_threat_delta"], 0.0)
        self.assertGreater(result["inference_score"], 0.0)
        self.assertEqual(len(result["response_vector"]), 4)

    def test_response_distance_is_rms_over_behavior(self):
        self.assertAlmostEqual(response_vector_distance([0.0, 0.0], [1.0, 1.0]), 1.0)

    def test_final_split_is_assigned_after_behavioral_screening(self):
        opponents = []
        for index in range(6):
            item = normalize_spec(spec(index), index=index)
            item["screening"] = {"response_vector": [index / 5.0, float(index % 2)]}
            opponents.append(item)
        assigned = assign_behavior_holdout(opponents, 1.0 / 3.0)
        self.assertEqual(sum(item["split"] == "holdout" for item in assigned), 2)
        self.assertEqual(sum(item["split"] == "train" for item in assigned), 4)


if __name__ == "__main__":
    unittest.main()
