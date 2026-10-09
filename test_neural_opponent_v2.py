import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from neural_opponent_policy import (
    FEATURE_VERSION_V1,
    FEATURE_VERSION_V2,
    FEATURE_VERSION_V3,
    NeuralOpponentPolicy,
    RiskAwareNeuralOpponentNet,
    extract_policy_features,
)
from generate_neural_opponent_data import choose_independent_response
from prompt_builder import get_legal_actions
from run_gate1_experiments import init_game


class NeuralOpponentV2FeatureTest(unittest.TestCase):
    def test_runtime_progress_guard_rejects_regretful_discard(self):
        policy = NeuralOpponentPolicy.__new__(NeuralOpponentPolicy)
        policy.pid = 1
        policy.max_shanten_regret = 0
        policy.max_effective_copy_loss = 1
        progress = {
            "d best": {"shanten": 0, "effective_copies": 6},
            "d near": {"shanten": 0, "effective_copies": 5},
            "d bad": {"shanten": 1, "effective_copies": 9},
        }
        with patch(
            "run_gate1_experiments.discard_progress_metrics",
            side_effect=lambda game, pid, action: progress.get(action),
        ):
            self.assertEqual(
                policy._apply_progress_guard(
                    object(), list(progress), "d near"
                ),
                "d near",
            )
            self.assertEqual(
                policy._apply_progress_guard(
                    object(), list(progress), "d bad"
                ),
                "d best",
            )

    def test_independent_teacher_response_has_no_hidden_risk_input(self):
        legal = ["p", "n"]
        self.assertEqual(choose_independent_response("aggressive", legal), "p")
        self.assertEqual(choose_independent_response("balanced", legal), "p")
        self.assertEqual(choose_independent_response("cautious", legal), "n")
        self.assertEqual(
            choose_independent_response("cautious", ["h", "n"]), "h"
        )

    def test_v2_removes_only_engineered_p0_tell_summary(self):
        game, _opponent_funcs, _defenders = init_game(
            2026082601,
            "mixed",
            "NeuralOpponentV2FeatureTest",
        )
        legal = get_legal_actions(game, 1)
        v1 = extract_policy_features(
            game,
            1,
            legal_actions=legal,
            feature_version=FEATURE_VERSION_V1,
        )
        v2 = extract_policy_features(
            game,
            1,
            legal_actions=legal,
            feature_version=FEATURE_VERSION_V2,
        )

        self.assertEqual(len(v1), 377)
        self.assertEqual(len(v2), 373)
        self.assertEqual(v1[:-4], v2)

        v3 = extract_policy_features(
            game,
            1,
            legal_actions=legal,
            feature_version=FEATURE_VERSION_V3,
        )
        self.assertEqual(v3, v2)

    def test_risk_aware_model_exposes_risk_and_accepts_blind_override(self):
        model = RiskAwareNeuralOpponentNet(3, 4, hidden_dims=(4,))
        with torch.no_grad():
            model.risk_head.weight.zero_()
            model.risk_head.bias.fill_(torch.logit(torch.tensor(0.8)))
            model.action_head[0].weight.zero_()
            model.action_head[0].bias.zero_()
            model.action_head[0].weight[0, -2] = 1.0
            model.action_head[2].weight.zero_()
            model.action_head[2].bias.zero_()
            model.action_head[2].weight[0, 0] = 1.0
            model.action_head[2].weight[1, 0] = -1.0
        model.eval()
        x = torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float32)
        predicted_logits, risk_logits = model(x)
        blind_logits, blind_risk_logits = model(x, risk_override=0.25)

        self.assertEqual(tuple(predicted_logits.shape), (1, 4))
        self.assertEqual(tuple(risk_logits.shape), (1,))
        torch.testing.assert_close(risk_logits, blind_risk_logits)
        self.assertFalse(torch.equal(predicted_logits, blind_logits))

    def test_risk_aware_checkpoint_threat_is_public_prediction(self):
        game, _opponent_funcs, _defenders = init_game(
            2026082603,
            "mixed",
            "NeuralOpponentRiskHeadTest",
        )
        legal = get_legal_actions(game, 1)
        features = extract_policy_features(
            game, 1, legal_actions=legal, feature_version=FEATURE_VERSION_V3
        )
        model = RiskAwareNeuralOpponentNet(len(features), 31, hidden_dims=(8,))
        with torch.no_grad():
            model.risk_head.weight.zero_()
            model.risk_head.bias.fill_(torch.logit(torch.tensor(0.8)))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "risk_aware.pth"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": len(features),
                    "hidden_dims": (8,),
                    "actions": ["h", "g", "p", "n"]
                    + [
                        f"d {number}{suit}"
                        for suit in ("万", "条", "筒")
                        for number in range(1, 10)
                    ],
                    "feature_version": FEATURE_VERSION_V3,
                    "architecture": "risk_aware_multitask",
                    "risk_prevalence": 0.2,
                },
                checkpoint,
            )
            policy = NeuralOpponentPolicy(1, str(checkpoint), risk_mode="predicted")
            self.assertAlmostEqual(policy.threat(game), 0.8, places=5)

            blind = NeuralOpponentPolicy(1, str(checkpoint), risk_mode="blind")
            self.assertAlmostEqual(blind.threat(game), 0.8, places=5)

    def test_unknown_feature_version_is_rejected(self):
        game, _opponent_funcs, _defenders = init_game(
            2026082602,
            "mixed",
            "NeuralOpponentUnknownFeatureTest",
        )
        with self.assertRaises(ValueError):
            extract_policy_features(game, 1, feature_version="unknown")

    def test_neural_strict_ffr_uses_progress_loss_and_first_opportunity(self):
        policy = NeuralOpponentPolicy.__new__(NeuralOpponentPolicy)
        policy.pid = 1
        policy.ffr_hand_shanten = 1
        policy.danger_threshold = 0
        policy.ff_opportunities = 0
        policy.ff_false = 0
        policy.strict_ffr_event_opportunities = 0
        policy.strict_ffr_event_false = 0
        policy.strict_ffr_event_events = []
        policy.defensive_switch_opportunities = 0
        policy.defensive_switches = 0
        policy.defensive_switch_events = []
        policy.turn_folds = 0
        policy.ffr_events = []
        policy.false_fold_events = []
        player = SimpleNamespace(hand_tiles=[], missing_suit=None)
        game = SimpleNamespace(
            players=[SimpleNamespace(hand_tiles=[], missing_suit=None), player]
        )
        progress = {
            "d push": {"shanten": 1, "effective_copies": 8},
            "d safe": {"shanten": 1, "effective_copies": 4},
        }

        with (
            patch(
                "neural_opponent_policy.ShantenCalculator.calculate_shanten",
                side_effect=[1, 1, 1, 1],
            ),
            patch(
                "run_gate1_experiments.choose_min_shanten_action",
                return_value="d push",
            ),
            patch(
                "run_gate1_experiments.discard_progress_metrics",
                side_effect=lambda game, pid, action: progress.get(action),
            ),
            patch("neural_opponent_policy.discard_safety_score", return_value=0.0),
        ):
            for _ in range(2):
                policy._record_ffr_proxy(
                    game,
                    player,
                    ["d push", "d safe"],
                    "d safe",
                    step=1,
                    last_p0_state=None,
                    in_deceive_window=False,
                )

        self.assertEqual(policy.ff_opportunities, 1)
        self.assertEqual(policy.ff_false, 1)
        self.assertEqual(policy.strict_ffr_event_opportunities, 2)
        self.assertEqual(policy.strict_ffr_event_false, 2)
        self.assertEqual(policy.defensive_switch_opportunities, 2)
        self.assertEqual(policy.defensive_switches, 2)


if __name__ == "__main__":
    unittest.main()
